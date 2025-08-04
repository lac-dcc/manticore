#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

#include "AssignmentBasedVectorization.h"
#include "StructuralPatternVectorization.h"
#include "VectorizationUtils.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"

namespace {

/**
 * @struct SimpleVectorizationPass
 * @brief The main MLIR pass for applying simple vectorization techniques.
 *
 * This pass operates on a `mlir::ModuleOp` and runs two distinct
 * vectorization strategies in sequence:
 * 1. Assignment-based vectorization (`processAssignTree`).
 * 2. Structural pattern-based vectorization (`processStructuralPatterns`).
 */
struct SimpleVectorizationPass
    : public mlir::PassWrapper<SimpleVectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    
    // An object to collect and report statistics on the optimizations performed.
    VectorizationStatistics stats;

    // Default constructor.
    SimpleVectorizationPass() = default;
    // Copy constructor is required for pass cloning.
    SimpleVectorizationPass(const SimpleVectorizationPass& pass) : stats(pass.stats) {}

    // Informs the pass manager which dialects may be created by this pass.
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<circt::comb::CombDialect,
                        circt::moore::MooreDialect,
                        circt::hw::HWDialect,
                        circt::sv::SVDialect>();
    }

    // The main entry point for the pass execution logic.
    void runOnOperation() override {
        // Clear statistics from any previous runs.
        stats.reset(); 
        // Get the top-level module operation to be processed.
        mlir::ModuleOp module = getOperation();
        
        // Run the first vectorization strategy: assignment-based.
        processAssignTree(module, stats);
        // Run the second vectorization strategy: structural pattern matching.
        processStructuralPatterns(module, stats);
        
        // After all transformations, print a summary of the results.
        stats.printReport();
    }

    // Returns the command-line argument used to invoke this pass.
    mlir::StringRef getArgument() const override { return "simple-vec"; }
    // Returns a human-readable description of what the pass does.
    mlir::StringRef getDescription() const override { return "A simple vectorization pass for Moore dialect operations."; }
};

} 

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "SimpleVec",
        LLVM_VERSION_STRING,
        []() {
            mlir::PassPipelineRegistration<>(
                "simple-vec", 
                "Runs the simple vectorization pass.",
                [](mlir::OpPassManager &pm) {
                    pm.addPass(std::make_unique<SimpleVectorizationPass>());
                });
        }
    };
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)