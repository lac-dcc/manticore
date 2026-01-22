#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h" 

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/SV/SVDialect.h"

using namespace mlir;
using namespace circt;

namespace {

struct SliceInfo {
    llvm::SetVector<Operation*> ops;
    llvm::SetVector<Value> inputs;
    Value rootOutput;
};

class Outliner {
public:
    explicit Outliner(hw::HWModuleOp module) : module(module) {}

    void run() {
        llvm::errs() << "[DEBUG] Outliner::run on module: " << module.getName() << "\n";

        if (module.getBody().empty()) {
            llvm::errs() << "[DEBUG] Module body is empty.\n";
            return;
        }

        Block &body = module.getBody().front();
        auto outputOp = dyn_cast<hw::OutputOp>(body.getTerminator());
        if (!outputOp) {
            llvm::errs() << "[DEBUG] No OutputOp found.\n";
            return;
        }

        OpBuilder builder(module.getContext());
        
        // --- PHASE 1: ANALYSIS ---
        llvm::errs() << "[DEBUG] Starting Analysis Phase...\n";
        SmallVector<SliceInfo, 4> slicesToExtract;
        
        int opIdx = 0;
        for (Value resultVal : outputOp.getOperands()) {
            llvm::errs() << "[DEBUG] Analyzing output operand " << opIdx++ << "\n";
            if (!shouldOutline(resultVal)) {
                llvm::errs() << "[DEBUG] -> Skipping (should not outline)\n";
                continue;
            }

            llvm::errs() << "[DEBUG] -> Getting Backward Slice...\n";
            SliceInfo slice = getBackwardSlice(resultVal);
            if (!slice.ops.empty()) {
                llvm::errs() << "[DEBUG] -> Found slice with " << slice.ops.size() << " operations.\n";
                slicesToExtract.push_back(slice);
            } else {
                llvm::errs() << "[DEBUG] -> Slice empty.\n";
            }
        }

        if (slicesToExtract.empty()) {
            llvm::errs() << "[DEBUG] No slices to extract. Exiting.\n";
            return;
        }

        // --- PHASE 2: TRANSFORMATION ---
        llvm::errs() << "[DEBUG] Starting Transformation Phase...\n";
        bool changed = false;
        int sliceIdx = 0;
        for (const auto& slice : slicesToExtract) {
            llvm::errs() << "[DEBUG] Processing Slice " << sliceIdx++ << "\n";
            
            llvm::errs() << "[DEBUG] -> Creating Extracted Module...\n";
            hw::HWModuleOp newModule = createExtractedModule(builder, slice);
            
            llvm::errs() << "[DEBUG] -> Instantiating Module...\n";
            instantiateModule(builder, newModule, slice);
            
            changed = true;
        }

        // --- PHASE 3: CLEANUP ---
        if (changed) {
            llvm::errs() << "[DEBUG] Starting Cleanup Phase...\n";
            cleanupDeadOps(body);
        }
        llvm::errs() << "[DEBUG] Outliner::run finished.\n";
    }

private:
    hw::HWModuleOp module;
    int moduleCounter = 0;

    bool shouldOutline(Value val) {
        Operation* op = val.getDefiningOp();
        if (!op) return false;
        if (isa<hw::ConstantOp>(op) || isa<BlockArgument>(val)) return false;
        return op->getDialect()->getNamespace() == "comb";
    }

    SliceInfo getBackwardSlice(Value root) {
        SliceInfo slice;
        slice.rootOutput = root;

        llvm::SmallVector<Value, 16> worklist;
        worklist.push_back(root);
        
        llvm::SmallPtrSet<Operation*, 16> visitedOps;

        while (!worklist.empty()) {
            Value current = worklist.pop_back_val();
            Operation *op = current.getDefiningOp();

            if (!op || isa<BlockArgument>(current)) {
                slice.inputs.insert(current);
                continue;
            }

            if (visitedOps.contains(op)) continue;

            bool isExtractable = isa<comb::CombDialect>(op->getDialect()) || isa<hw::ConstantOp>(op);
            
            if (!isExtractable) {
                slice.inputs.insert(current);
                continue;
            }

            visitedOps.insert(op);
            slice.ops.insert(op);

            for (Value operand : op->getOperands()) {
                worklist.push_back(operand);
            }
        }
        return slice;
    }

    hw::HWModuleOp createExtractedModule(OpBuilder &builder, const SliceInfo &slice) {
        std::string moduleName = module.getName().str() + "_extracted_" + std::to_string(moduleCounter++);
        llvm::errs() << "[DEBUG] ---- Creating module: " << moduleName << "\n";
        
        SmallVector<hw::PortInfo> ports;
        
        int inputIdx = 0;
        for (Value input : slice.inputs) {
            hw::PortInfo p;
            p.name = StringAttr::get(builder.getContext(), "in_" + std::to_string(inputIdx++));
            p.type = input.getType();
            p.dir = hw::ModulePort::Direction::Input;
            ports.push_back(p);
        }

        {
            hw::PortInfo p;
            p.name = StringAttr::get(builder.getContext(), "out");
            p.type = slice.rootOutput.getType();
            p.dir = hw::ModulePort::Direction::Output;
            ports.push_back(p);
        }

        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointAfter(module);
        
        auto newHWModule = builder.create<hw::HWModuleOp>(
            module.getLoc(),
            builder.getStringAttr(moduleName),
            ports,
            builder.getArrayAttr({}) 
        );

        Block *newBody = newHWModule.getBodyBlock();
        
        if (Operation* term = newBody->getTerminator()) {
            term->erase();
        }
        
        builder.setInsertionPointToStart(newBody);

        IRMapping mapper;
        
        for (size_t i = 0; i < slice.inputs.size(); ++i) {
            mapper.map(slice.inputs[i], newBody->getArgument(i));
        }

        for (Operation *op : llvm::reverse(slice.ops)) {
            builder.clone(*op, mapper);
        }

        Value resultInNewModule = mapper.lookup(slice.rootOutput);
        builder.create<hw::OutputOp>(module.getLoc(), resultInNewModule);

        return newHWModule;
    }

    void instantiateModule(OpBuilder &builder, hw::HWModuleOp newModule, const SliceInfo &slice) {
        llvm::errs() << "[DEBUG] ---- Instantiating " << newModule.getName() << "\n";
        Operation* rootOp = slice.rootOutput.getDefiningOp();
        
        if (rootOp && rootOp->getBlock() == &module.getBody().front()) {
            builder.setInsertionPointAfter(rootOp);
        } else {
            builder.setInsertionPointToStart(&module.getBody().front());
        }

        SmallVector<Value> instanceOperands;
        for (Value input : slice.inputs) {
            instanceOperands.push_back(input);
        }

        auto instance = builder.create<hw::InstanceOp>(
            rootOp ? rootOp->getLoc() : module.getLoc(),
            newModule,
            builder.getStringAttr("inst_" + newModule.getName()),
            instanceOperands,
            builder.getArrayAttr({})
        );

        llvm::errs() << "[DEBUG] ---- Replacing uses...\n";
        
        Value valToReplace = slice.rootOutput;
        valToReplace.replaceAllUsesWith(instance.getResult(0));
        
        llvm::errs() << "[DEBUG] ---- Uses replaced.\n";
    }

    void cleanupDeadOps(Block &block) {
        bool changed = true;
        int pass = 0;
        while (changed) {
            changed = false;
            llvm::SmallVector<Operation *, 16> deadOps;
            for (Operation &op : block) {
                if (op.use_empty() && !op.hasTrait<OpTrait::IsTerminator>() && !isa<hw::InstanceOp>(op)) {
                    deadOps.push_back(&op);
                }
            }
            if (!deadOps.empty()) {
                llvm::errs() << "[DEBUG] ---- Cleanup pass " << pass++ << ": removing " << deadOps.size() << " ops.\n";
                changed = true;
                for (Operation *op : deadOps) {
                    op->erase();
                }
            }
        }
    }
};

struct OutliningPass
    : public mlir::PassWrapper<OutliningPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    
    OutliningPass() = default;
    OutliningPass(const OutliningPass& pass) {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<circt::comb::CombDialect,
                        circt::hw::HWDialect,
                        circt::sv::SVDialect>();
    }

    void runOnOperation() override {
        llvm::errs() << "[DEBUG] Pass runOnOperation started.\n";
        mlir::ModuleOp mlir_module = getOperation();
        
        llvm::SmallVector<hw::HWModuleOp, 4> modulesToProcess;
        for (auto module : mlir_module.getOps<hw::HWModuleOp>()) {
            modulesToProcess.push_back(module);
        }
        
        llvm::errs() << "[DEBUG] Found " << modulesToProcess.size() << " modules to process.\n";

        for (auto module : modulesToProcess) {
            Outliner outliner(module);
            outliner.run();
        }
        llvm::errs() << "[DEBUG] Pass runOnOperation finished.\n";
    }

    mlir::StringRef getArgument() const override { return "hw-outlining"; }
    mlir::StringRef getDescription() const override { return "Extracts logic cones into new HW modules."; }
};

} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(OutliningPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(OutliningPass)

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "hw-outlining",
        LLVM_VERSION_STRING,
        []() {
            mlir::PassRegistration<OutliningPass>();
        }
    };
}