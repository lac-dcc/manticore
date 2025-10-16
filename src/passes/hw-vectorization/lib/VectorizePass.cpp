#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/SV/SVDialect.h"

#include "../include/Vectorizer.h"
#include "../include/VectorizationUtils.h"

namespace {

struct VectorizationPass
    : public mlir::PassWrapper<VectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
    
    VectorizationStatistics stats;

    VectorizationPass() = default;
    VectorizationPass(const VectorizationPass& pass) : stats(pass.stats) {}

    void getDependentDialects(mlir::DialectRegistry &registry) const override {
        registry.insert<circt::comb::CombDialect,
                        circt::moore::MooreDialect,
                        circt::hw::HWDialect,
                        circt::sv::SVDialect>();
    }

    void runOnOperation() override {
        stats.reset();
        mlir::ModuleOp mlir_module = getOperation();

        SmallVector<hw::HWModuleOp, 8> hw_modules;
        for (auto module : mlir_module.getOps<hw::HWModuleOp>())
            hw_modules.push_back(module);

        for (auto module : hw_modules) {
            bool containsLLHD = false;

            module.walk([&](mlir::Operation *op) {
                if (op->getDialect()->getNamespace() == "llhd") {
                    containsLLHD = true;
                    return mlir::WalkResult::interrupt(); 
                }
                return mlir::WalkResult::advance();
            });

            if (containsLLHD) {
                // llvm::errs() << "AVISO: Pulando a vetorização do módulo '"
                //         << module.getName() 
                //         << "' porque ele contém operações do dialeto 'llhd', que não é suportado.\n";
                continue; 
            }

            vectorizer v(module);
            v.performInlining(stats);
            v.vectorize(stats);
        }
        stats.printReport();
    }

    mlir::StringRef getArgument() const override { return "hw-vectorization"; }
    mlir::StringRef getDescription() const override { return "A custom vectorization pass for HW."; }
};

} 

MLIR_DECLARE_EXPLICIT_TYPE_ID(VectorizationPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(VectorizationPass)

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "hw-vectorization",
        LLVM_VERSION_STRING,
        []() {
            mlir::PassRegistration<VectorizationPass>();
        }
    };
}
