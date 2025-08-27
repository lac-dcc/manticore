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
struct HwVectorization
    : public mlir::PassWrapper<HwVectorization,
                               mlir::OperationPass<mlir::ModuleOp>> {
    
    // An object to collect and report statistics on the optimizations performed.
    VectorizationStatistics stats;

    // Default constructor.
    HwVectorization() = default;
    // Copy constructor is required for pass cloning.
    HwVectorization(const HwVectorization& pass) : stats(pass.stats) {}

    // Informs the pass manager which dialects may be created by this pass.
    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<circt::comb::CombDialect,
                        circt::moore::MooreDialect,
                        circt::hw::HWDialect,
                        circt::sv::SVDialect>();
    }

    // The main entry point for the pass execution logic.
    void runOnOperation() override {
        stats.reset(); 
        mlir::ModuleOp module = getOperation();
        
        processAssignTree(module, stats);

        // stats.printReport();
    }

    // Returns the command-line argument used to invoke this pass.
    mlir::StringRef getArgument() const override { return "hw-vec"; }
    // Returns a human-readable description of what the pass does.
    mlir::StringRef getDescription() const override { return "A simple vectorization pass for HW dialect operations."; }
};

} 

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "hw-vec",
        LLVM_VERSION_STRING,
        []() {
            mlir::PassPipelineRegistration<>(
                "hw-vec", 
                "Runs the simple vectorization pass.",
                [](mlir::OpPassManager &pm) {
                    pm.addPass(std::make_unique<HwVectorization>());
                });
        }
    };
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(HwVectorization)
MLIR_DEFINE_EXPLICIT_TYPE_ID(HwVectorization)
