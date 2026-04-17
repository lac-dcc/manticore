#include "CareMaskPass.hpp"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h" 
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/RegionUtils.h" 

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "../include/Canonicalizer.hpp"

using namespace mlir;
using namespace circt;

namespace {

// Represents a captured cone of logic (operations + inputs + output).
struct SliceInfo {
    llvm::SetVector<Operation*> ops; 
    llvm::SetVector<Value> inputs;   
    Value rootOutput;                
};

class GraphComparator {
public:
    static bool isIsomorphic(const SliceInfo& candidateSlice,
                             const SliceInfo& referenceSlice,
                             hw::HWModuleOp module) {
        Block &body = module.getBody().front();

        // 1. Collect Module Operations (ignoring hw.output)
        llvm::SmallVector<Operation*> moduleOps;
        for (auto &op : body) {
            if (!isa<hw::OutputOp>(op)) {
                moduleOps.push_back(&op);
            }
        }

        // Quick size checks
        if (candidateSlice.ops.size() != moduleOps.size()) return false;
        if (candidateSlice.inputs.size() != referenceSlice.inputs.size()) return false;
        if (candidateSlice.inputs.size() != body.getNumArguments()) return false;

        // 2. Build the value map guided by the reference slice ordering.
        llvm::DenseMap<Value, Value> valueMap;
        for (size_t i = 0; i < candidateSlice.inputs.size(); ++i) {
            Value candInput = candidateSlice.inputs[i];
            Value moduleArg  = body.getArgument(i);

            // Types must match.
            if (candInput.getType() != moduleArg.getType()) return false;

            valueMap[candInput] = moduleArg;
        }

        // 3. Step-by-step comparison
        for (size_t i = 0; i < candidateSlice.ops.size(); ++i) {
            Operation* sliceOp  = candidateSlice.ops[i];
            Operation* moduleOp = moduleOps[i];

            // A. Op name
            if (sliceOp->getName() != moduleOp->getName()) return false;

            // B. Result types
            if (sliceOp->getNumResults() != moduleOp->getNumResults()) return false;
            for (size_t r = 0; r < sliceOp->getNumResults(); ++r) {
                if (sliceOp->getResult(r).getType() != moduleOp->getResult(r).getType())
                    return false;
            }

            // C. Attributes
            if (sliceOp->getAttrDictionary() != moduleOp->getAttrDictionary()) return false;

            // D. Operands (translated through the value map)
            if (sliceOp->getNumOperands() != moduleOp->getNumOperands()) return false;
            for (size_t k = 0; k < sliceOp->getNumOperands(); ++k) {
                Value sliceOperand  = sliceOp->getOperand(k);
                Value moduleOperand = moduleOp->getOperand(k);

                if (!valueMap.count(sliceOperand)) return false;
                if (valueMap[sliceOperand] != moduleOperand) return false;
            }

            // E. Update map with results
            for (size_t r = 0; r < sliceOp->getNumResults(); ++r) {
                valueMap[sliceOp->getResult(r)] = moduleOp->getResult(r);
            }
        }

        return true; 
    }
};

// Helper class to calculate structural hashes using Canonical Value Numbering.
class StructuralHasher {
public:
    // Hash a full module to build the catalog.
    static llvm::hash_code hashModule(hw::HWModuleOp module) {
        if (module.getBody().empty()) return llvm::hash_value(0);
        
        SliceInfo fullSlice;
        Block &body = module.getBody().front();
        
        for (auto arg : body.getArguments()) {
            fullSlice.inputs.insert(arg);
        }
        
        for (auto &op : body) {
            if (!isa<hw::OutputOp>(op)) {
                fullSlice.ops.insert(&op);
            }
        }
        
        // Hash the module signature (input port types in declaration order).
        llvm::hash_code sigHash = llvm::hash_value(0);
        auto moduleType = module.getModuleType();
        for (auto port : moduleType.getPorts()) {
            if (port.dir == hw::ModulePort::Direction::Input) {
                sigHash = llvm::hash_combine(sigHash, port.type.getAsOpaquePointer());
            }
        }

        return llvm::hash_combine(sigHash, hashSliceContent(fullSlice));
    }

    // Hash a specific logic cone (slice).
    static llvm::hash_code hashSlice(const SliceInfo& slice) {
        llvm::hash_code sigHash = llvm::hash_value(0);
        for (Value input : slice.inputs) {
            sigHash = llvm::hash_combine(sigHash, input.getType().getAsOpaquePointer());
        }
        return llvm::hash_combine(sigHash, hashSliceContent(slice));
    }

private:
    static llvm::hash_code hashSliceContent(const SliceInfo& slice) {
        llvm::hash_code code = llvm::hash_value(0);

        llvm::DenseMap<Value, unsigned> valueNumbering;
        unsigned nextValueIdx = 0;

        for (Value input : slice.inputs) {
            valueNumbering[input] = nextValueIdx++;
        }

        for (Operation *op : slice.ops) {
            code = llvm::hash_combine(code, op->getName().getStringRef());

            for (Value result : op->getResults()) {
                code = llvm::hash_combine(code, result.getType().getAsOpaquePointer());
                valueNumbering[result] = nextValueIdx++;
            }

            for (Value operand : op->getOperands()) {
                if (valueNumbering.count(operand)) {
                    code = llvm::hash_combine(code, valueNumbering[operand]);
                } else {
                    code = llvm::hash_combine(code, -1);
                }
            }
            code = llvm::hash_combine(code, hashAttributes(*op));
        }
        return code;
    }

    static llvm::hash_code hashAttributes(Operation &op) {
        llvm::hash_code code = llvm::hash_value(0);
        auto attrs = op.getAttrDictionary();
        for (auto namedAttr : attrs) {
            code = llvm::hash_combine(code, namedAttr.getName());
            code = llvm::hash_combine(code, namedAttr.getValue().getAsOpaquePointer());
        }
        return code;
    }
};

// Helper to extract backward slices from the IR.
class LogicAnalyzer {
public:
    static SliceInfo getBackwardSlice(Value root) {
        SliceInfo slice;
        slice.rootOutput = root;
        
        llvm::SmallVector<Value, 16> worklist;
        worklist.push_back(root);
        llvm::SmallPtrSet<Operation*, 16> visitedOps;

        llvm::SetVector<Operation*> opsFound;

        while (!worklist.empty()) {
            Value current = worklist.pop_back_val();
            Operation *op = current.getDefiningOp();

            // Stop at arguments, constants, or non-combinational ops.
            if (!op || isa<hw::ConstantOp>(op) || !isCombinational(op)) {
                slice.inputs.insert(current);
                continue;
            }

            if (visitedOps.contains(op)) continue;
            visitedOps.insert(op);
            opsFound.insert(op);

            for (Value operand : op->getOperands()) {
                worklist.push_back(operand);
            }
        }

        Block *parentBlock = root.getDefiningOp()->getBlock();
        for (Operation &op : parentBlock->getOperations()) {
            if (opsFound.contains(&op)) {
                slice.ops.insert(&op);
            }
        }

        llvm::DenseMap<Operation*, unsigned> opPosition;
        {
            unsigned pos = 0;
            for (Operation *op : slice.ops) {
                opPosition[op] = pos++;
            }
        }

        auto inputsVec = slice.inputs.takeVector();

        llvm::sort(inputsVec, [&opPosition](Value a, Value b) {
            auto argA = dyn_cast<BlockArgument>(a);
            auto argB = dyn_cast<BlockArgument>(b);

            // Both are block arguments: sort by argument number.
            if (argA && argB)
                return argA.getArgNumber() < argB.getArgNumber();

            // Block arguments before op results.
            if (argA) return true;
            if (argB) return false;

            // Both are op results: sort by the op's topological position,
            // then by result index within the op.
            auto resA = cast<OpResult>(a);
            auto resB = cast<OpResult>(b);

            Operation *opA = resA.getDefiningOp();
            Operation *opB = resB.getDefiningOp();

            unsigned posA = opPosition.count(opA) ? opPosition.lookup(opA) : UINT_MAX;
            unsigned posB = opPosition.count(opB) ? opPosition.lookup(opB) : UINT_MAX;

            if (posA != posB) return posA < posB;

            return resA.getResultNumber() < resB.getResultNumber();
        });

        // Re-insert in the now-deterministic canonical order.
        // takeVector() already emptied the SetVector, so insert is safe.
        slice.inputs.insert(inputsVec.begin(), inputsVec.end());

        return slice;
    }

    static bool isCombinational(Operation *op) {
        return op->getDialect()->getNamespace() == "comb";
    }
};

struct ExtractorStatistics {
    int numNewModules = 0;
    int numReplacedInstances = 0;
    int numOpsSaved = 0;
    int maxSliceSize = 0;    
    int maxSliceInputs = 0;

    void reset() {
        numNewModules = 0;
        numReplacedInstances = 0;
        numOpsSaved = 0;
        maxSliceSize = 0;
        maxSliceInputs = 0;
    }

    void printReport() {
        llvm::errs() << "=======================\n\n";
        llvm::errs() << "NewModules=" << numNewModules << "\n";
        llvm::errs() << "ReplacedInstances=" << numReplacedInstances << "\n";
        llvm::errs() << "OpsSaved=" << numOpsSaved << "\n";
        llvm::errs() << "MaxSliceSize=" << maxSliceSize << "\n";
        llvm::errs() << "MaxSliceInputs=" << maxSliceInputs << "\n";
        llvm::errs() << "=======================\n\n";
    }
};

struct SliceExtractorPass : public mlir::PassWrapper<SliceExtractorPass, mlir::OperationPass<mlir::ModuleOp>> {
    
    ExtractorStatistics stats;

    SliceExtractorPass() = default;
    SliceExtractorPass(const SliceExtractorPass& pass) : stats(pass.stats) {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<circt::comb::CombDialect, circt::hw::HWDialect>();
    }

    hw::HWModuleOp createNewModule(OpBuilder &builder, mlir::ModuleOp topModule, const SliceInfo &slice, std::string name) {
        
        builder.setInsertionPointToStart(topModule.getBody());
        
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

        auto newHWModule = builder.create<hw::HWModuleOp>(
            builder.getUnknownLoc(), builder.getStringAttr(name), ports, builder.getArrayAttr({}) 
        );

        Block *newBody = newHWModule.getBodyBlock();
        
        if (Operation *terminator = newBody->getTerminator()) {
            terminator->erase();
        }
        
        builder.setInsertionPointToStart(newBody);

        IRMapping mapper;
        for (size_t i = 0; i < slice.inputs.size(); ++i) {
            mapper.map(slice.inputs[i], newBody->getArgument(i));
        }

        for (Operation *op : slice.ops) {
            builder.clone(*op, mapper);
        }

        Value resultInNewModule = mapper.lookup(slice.rootOutput);
        builder.create<hw::OutputOp>(builder.getUnknownLoc(), resultInNewModule);
        
        return newHWModule;
    }

    void runOnOperation() override {
        stats.reset();

        mlir::ModuleOp topModule = getOperation();
        DontCareReducer reducer;
        reducer.apply_masks(topModule);

        llvm::DenseSet<llvm::StringRef> targetOps = {
         //"comb.add",
         "comb.mul",
         "comb.and",
         "comb.xor",
         "comb.or"
         };
         
        Canonicalizer canonicalizer(targetOps);
        canonicalizer.canonicalize(topModule);
        

        IRRewriter rewriter(topModule.getContext());
        
        // Step 1: Catalog Existing Modules
        llvm::DenseMap<llvm::hash_code, hw::HWModuleOp> moduleCatalog;
        
        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (module.getBody().empty()) continue;
            auto h = StructuralHasher::hashModule(module);
            if (moduleCatalog.count(h) == 0) {
                moduleCatalog[h] = module;
            }
        }

        // Step 2: Mining Frequent Patterns
        struct SliceGroup {
            SliceInfo referenceSlice;                      // the one used to create the module
            llvm::SmallVector<SliceInfo, 4> allSlices;     // all occurrences (including reference)
        };
        llvm::DenseMap<llvm::hash_code, SliceGroup> sliceHistogram;

        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (module.getBody().empty()) continue;
            Block &body = module.getBody().front();
            
            llvm::SmallVector<Operation*> opsToCheck;
            for (auto &op : body) opsToCheck.push_back(&op);

            for (Operation *op : opsToCheck) {
                if (op->getNumResults() == 0 || !LogicAnalyzer::isCombinational(op)) continue;
                
                SliceInfo slice = LogicAnalyzer::getBackwardSlice(op->getResult(0));
                
                if (slice.ops.size() < 2) continue;

                auto h = StructuralHasher::hashSlice(slice);

                auto &group = sliceHistogram[h];
                if (group.allSlices.empty()) {
                    // First occurrence becomes the reference.
                    group.referenceSlice = slice;
                }
                group.allSlices.push_back(slice);
            }
        }

        // Step 3: Create New Modules
        int extractedCounter = 0;
        
        for (auto &it : sliceHistogram) {
            llvm::hash_code h = it.first;
            auto &group = it.second;

            if (moduleCatalog.count(h) == 0 && group.allSlices.size() > 1) {
                
                std::string newName = "extracted_" + std::to_string(extractedCounter++);
                
                // Create module using the reference (first) slice as template.
                hw::HWModuleOp newModule = createNewModule(rewriter, topModule, group.referenceSlice, newName);
                
                moduleCatalog[h] = newModule;

                stats.numNewModules++;

                int currentOps    = group.referenceSlice.ops.size();
                int currentInputs = group.referenceSlice.inputs.size();
                
                if (currentOps    > stats.maxSliceSize)   stats.maxSliceSize   = currentOps;
                if (currentInputs > stats.maxSliceInputs) stats.maxSliceInputs = currentInputs;
            }
        }

        // Step 4: Instantiation
        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (module.getBody().empty()) continue;
            Block &body = module.getBody().front();
            llvm::SmallVector<Operation*> opsToCheck;
            for (auto &op : body) opsToCheck.push_back(&op);

            for (Operation *op : opsToCheck) {
                if (op->getNumResults() == 0 || !LogicAnalyzer::isCombinational(op)) continue;
                
                SliceInfo slice = LogicAnalyzer::getBackwardSlice(op->getResult(0));
                if (slice.ops.size() < 2) continue;

                auto h = StructuralHasher::hashSlice(slice);

                if (moduleCatalog.count(h)) {
                    hw::HWModuleOp target = moduleCatalog[h];
                    if (target == module) continue;

                    const SliceInfo *refSlice = nullptr;
                    if (sliceHistogram.count(h)) {
                        refSlice = &sliceHistogram[h].referenceSlice;
                    }

                    SliceInfo moduleBodySlice;
                    if (!refSlice) {
                        Block &targetBody = target.getBody().front();
                        for (auto arg : targetBody.getArguments())
                            moduleBodySlice.inputs.insert(arg);
                        for (auto &bodyOp : targetBody)
                            if (!isa<hw::OutputOp>(bodyOp))
                                moduleBodySlice.ops.insert(&bodyOp);
                        refSlice = &moduleBodySlice;
                    }

                    if (GraphComparator::isIsomorphic(slice, *refSlice, target)) {
                        instantiateAndReplace(rewriter, module, slice, target);

                        stats.numReplacedInstances++; 
                        stats.numOpsSaved += (slice.ops.size() - 1);
                    }
                }
            }
        }

        // Step 5: Cleanup
        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (!module.getBody().empty()) (void)mlir::runRegionDCE(rewriter, module.getBody());
        }

        llvm::SmallVector<hw::HWModuleOp> modulesToDelete;
        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (mlir::SymbolTable::symbolKnownUseEmpty(module, topModule)) {
                if (module.getName().starts_with("extracted_")) { 
                    modulesToDelete.push_back(module);
                }
            }
        }

        for (auto module : modulesToDelete) {
            module.erase();
        }

        stats.printReport();
    }

    void instantiateAndReplace(IRRewriter &rewriter, hw::HWModuleOp parentModule, 
                               const SliceInfo &slice, hw::HWModuleOp targetModule) {
        
        Operation *rootOp = slice.rootOutput.getDefiningOp();
        rewriter.setInsertionPointAfter(rootOp);

        SmallVector<Value> instanceOperands;
        for (Value input : slice.inputs) {
            instanceOperands.push_back(input);
        }

        auto instance = rewriter.create<hw::InstanceOp>(
            rootOp->getLoc(),
            targetModule,
            rewriter.getStringAttr("inst_" + targetModule.getName()),
            instanceOperands,
            rewriter.getArrayAttr({})
        );
        
        Value valToReplace = slice.rootOutput; 
        valToReplace.replaceAllUsesWith(instance.getResult(0));
    }

    mlir::StringRef getArgument() const override { return "slice-extractor"; }
    mlir::StringRef getDescription() const override { return "Extracts logic cones and deduplicates equal logic."; }
};

} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(SliceExtractorPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SliceExtractorPass)

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "slice-extractor",
        LLVM_VERSION_STRING,
        []() { mlir::PassRegistration<SliceExtractorPass>(); }
    };
}
