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
#include "circt/Dialect/HW/HWAttributes.h"
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
    static bool isIsomorphic(const SliceInfo& slice, hw::HWModuleOp module) {
        Block &body = module.getBody().front();
        
        // 1. Collect Module Operations (ignoring hw.output)
        // List of relevant ops to compare against the slice.
        llvm::SmallVector<Operation*> moduleOps;
        for (auto &op : body) {
            if (!isa<hw::OutputOp>(op)) {
                moduleOps.push_back(&op);
            }
        }

        // Quick Check: Size mismatch
        if (slice.ops.size() != moduleOps.size()) return false;
        if (slice.inputs.size() != body.getNumArguments()) return false;

        // 2. Initialize Map: Slice Value -> Module Value
        // Slice inputs to the module arguments (ports).
        llvm::DenseMap<Value, Value> valueMap;
        
        for (size_t i = 0; i < slice.inputs.size(); ++i) {
            valueMap[slice.inputs[i]] = body.getArgument(i);
        }

        // 3. Step-by-step Comparison
        for (size_t i = 0; i < slice.ops.size(); ++i) {
            Operation* sliceOp = slice.ops[i];
            Operation* moduleOp = moduleOps[i];

            // A. Check Op Name (e.g., comb.add vs comb.sub)
            if (sliceOp->getName() != moduleOp->getName()) return false;

            // B. Check Result Types (Bit width must match)
            if (sliceOp->getNumResults() != moduleOp->getNumResults()) return false;
            for (size_t r = 0; r < sliceOp->getNumResults(); ++r) {
                if (sliceOp->getResult(r).getType() != moduleOp->getResult(r).getType()) {
                    return false;
                }
            }

            // C. Check Attributes (Constants, names, parameters)
            if (sliceOp->getAttrDictionary() != moduleOp->getAttrDictionary()) return false;

            // D. Check Operands 
            // Check if the slice operands, when translated via the map,
            // match exactly the operands used in the module.
            if (sliceOp->getNumOperands() != moduleOp->getNumOperands()) return false;
            
            for (size_t k = 0; k < sliceOp->getNumOperands(); ++k) {
                Value sliceOperand = sliceOp->getOperand(k);
                Value moduleOperand = moduleOp->getOperand(k);

                // If the operand is not in the map, it means it's an external value
                // that wasn't captured or a logic error.
                if (!valueMap.count(sliceOperand)) return false;

                if (valueMap[sliceOperand] != moduleOperand) return false;
            }

            // E. Update Map
            // If ops are equal, map their results for future checks.
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
        
        // Treat the module body as a single large slice.
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
        
        // 1. Hash the Module Signature (Ports).
        // Only consider INPUT ports to ensure compatibility with extracted slices.
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
    // Maps values to sequential indices (0, 1, 2...) to ignore variable names.
    static llvm::hash_code hashSliceContent(const SliceInfo& slice) {
        llvm::hash_code code = llvm::hash_value(0);

        llvm::DenseMap<Value, unsigned> valueNumbering;
        unsigned nextValueIdx = 0;

        // 1. Map Inputs to sequential indices.
        for (Value input : slice.inputs) {
            valueNumbering[input] = nextValueIdx++;
        }

        // 2. Hash Operations in topological order.
        for (Operation *op : slice.ops) {
            code = llvm::hash_combine(code, op->getName().getStringRef());

            // Assign indices to results.
            for (Value result : op->getResults()) {
                code = llvm::hash_combine(code, result.getType().getAsOpaquePointer());
                valueNumbering[result] = nextValueIdx++;
            }

            // Hash operands using the canonical index.
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

        // Backward traversal to find the logic cone.
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

        // Restore topological order (Inputs -> Outputs).
        for (Operation *op : llvm::reverse(opsFound)) {
            slice.ops.insert(op);
        }

        // Sort inputs deterministically to match the original module's signature.
        // This ensures [z, x, y] becomes [x, y, z] if they correspond to arg0, arg1, arg2.
        auto inputsVec = slice.inputs.takeVector();
        
        llvm::sort(inputsVec, [](Value a, Value b) {
            auto argA = dyn_cast<BlockArgument>(a);
            auto argB = dyn_cast<BlockArgument>(b);
            
            if (argA && argB) {
                return argA.getArgNumber() < argB.getArgNumber();
            }
            if (argA) return true; 
            if (argB) return false;
            
            return a.getAsOpaquePointer() < b.getAsOpaquePointer();
        });

        slice.inputs.insert(inputsVec.begin(), inputsVec.end());

        return slice;
    }

    static bool isCombinational(Operation *op) {
        return op->getDialect()->getNamespace() == "comb";
    }
};

struct ExtractorStatistics {
   // Number of new modules created to encapsulate extracted logic
    int numNewModules = 0;
    
    // Number of times repeated logic was replaced by a module instance
    int numReplacedInstances = 0;
    
    // Estimated number of combinational operations removed from the design
    int numOpsSaved = 0;
    
    // Maximum number of operations found inside a single extracted slice
    int maxSliceSize = 0;    
    
    // Maximum number of input ports required by a single extracted slice
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

    // Helper to create a new module from a Slice
    hw::HWModuleOp createNewModule(OpBuilder &builder, mlir::ModuleOp topModule, const SliceInfo &slice, std::string name) {
        
        // Ensure we insert the module at the top of the file
        builder.setInsertionPointToStart(topModule.getBody());
        
        // Define Ports
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

        // Create the Module
        auto newHWModule = builder.create<hw::HWModuleOp>(
            builder.getUnknownLoc(), builder.getStringAttr(name), ports, builder.getArrayAttr({}) 
        );

        Block *newBody = newHWModule.getBodyBlock();
        
        // Remove the default terminator created by the builder to avoid duplication
        if (Operation *terminator = newBody->getTerminator()) {
            terminator->erase();
        }
        
        builder.setInsertionPointToStart(newBody);

        // Map inputs and clone operations
        IRMapping mapper;
        for (size_t i = 0; i < slice.inputs.size(); ++i) {
            mapper.map(slice.inputs[i], newBody->getArgument(i));
        }

        for (Operation *op : slice.ops) {
            builder.clone(*op, mapper);
        }

        // Create the final output
        Value resultInNewModule = mapper.lookup(slice.rootOutput);
        builder.create<hw::OutputOp>(builder.getUnknownLoc(), resultInNewModule);
        
        return newHWModule;
    }

    void runOnOperation() override {
        stats.reset();

        mlir::ModuleOp topModule = getOperation();

        llvm::DenseSet<llvm::StringRef> targetOps = {
         "comb.add",
         "comb.mul",
         "comb.and",
         "comb.xor",
         "comb.or"
         };
         
        Canonicalizer canonicalizer(targetOps);
        canonicalizer.canonicalize(topModule);
        

        IRRewriter rewriter(topModule.getContext());
        
        // Step 1: Catalog Existing Modules
        // Calculate hash for all modules to serve as potential "Masters".
        llvm::DenseMap<llvm::hash_code, hw::HWModuleOp> moduleCatalog;
        
        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (module.getBody().empty()) continue;
            auto h = StructuralHasher::hashModule(module);
            // Only add if not already present (preserve the first one).
            if (moduleCatalog.count(h) == 0) {
                moduleCatalog[h] = module;
            }
        }

        // Step 2: Mining Frequent Patterns 
        // Scan the code to find repeated logic that isn't a module yet.
        // Map: Hash -> List of Slices found
        llvm::DenseMap<llvm::hash_code, llvm::SmallVector<SliceInfo, 4>> sliceHistogram;

        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
            if (module.getBody().empty()) continue;
            Block &body = module.getBody().front();
            
            // Snapshot operations to iterate safely
            llvm::SmallVector<Operation*> opsToCheck;
            for (auto &op : body) opsToCheck.push_back(&op);

            for (Operation *op : opsToCheck) {
                if (op->getNumResults() == 0 || !LogicAnalyzer::isCombinational(op)) continue;
                
                SliceInfo slice = LogicAnalyzer::getBackwardSlice(op->getResult(0));
                
                // Filter: Ignore tiny slices (e.g., single operations)
                if (slice.ops.size() < 2) continue;

                auto h = StructuralHasher::hashSlice(slice);
                sliceHistogram[h].push_back(slice);
            }
        }

        // Step 3: Create New Modules 
        int extractedCounter = 0;
        
        for (auto &it : sliceHistogram) {
            llvm::hash_code h = it.first;
            auto &slices = it.second;

            // 1. If it's NOT in the catalog yet
            // 2. AND it appears more than once
            // 3. Create a new module
            if (moduleCatalog.count(h) == 0 && slices.size() > 1) {
                
                std::string newName = "extracted_" + std::to_string(extractedCounter++);
                
                // Create module using the first slice as a template
                hw::HWModuleOp newModule = createNewModule(rewriter, topModule, slices[0], newName);
                
                // Add to catalog so we can use it in Step 4
                moduleCatalog[h] = newModule;

                stats.numNewModules++;

                int currentOps = slices[0].ops.size();
                int currentInputs = slices[0].inputs.size();
                
                if (currentOps > stats.maxSliceSize) stats.maxSliceSize = currentOps;
                if (currentInputs > stats.maxSliceInputs) stats.maxSliceInputs = currentInputs;
            }
        }

        // Step 4: Instantiation 
        // Now we replace logic with instances from the Catalog (Originals + Extracted).
        for (auto module : topModule.getOps<hw::HWModuleOp>()) {
             if (module.getBody().empty()) continue;
             Block &body = module.getBody().front();
             llvm::SmallVector<Operation*> opsToCheck;
             for (auto &op : body) opsToCheck.push_back(&op);

             for (Operation *op : opsToCheck) {
                if (op->getNumResults() == 0 || !LogicAnalyzer::isCombinational(op)) continue;
                
                // Recalculate slice
                SliceInfo slice = LogicAnalyzer::getBackwardSlice(op->getResult(0));
                if (slice.ops.size() < 2) continue;

                // Recalculate hash
                auto h = StructuralHasher::hashSlice(slice);

                if (moduleCatalog.count(h)) {
                    hw::HWModuleOp target = moduleCatalog[h];
                    // Prevent recursion (don't instantiate a module inside itself)
                    if (target == module) continue; 

                    if (GraphComparator::isIsomorphic(slice, target)) {
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

        stats.printReport();
    }

    // Replaces the logic slice with an instance of the target module.
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
        
        // Replace uses of the old logic result with the instance result.
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
