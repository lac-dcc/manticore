#include "mlir/Pass/Pass.h"
#include "mlir/IR/PatternMatch.h"
#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace circt;
using namespace moore;

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Moore/MoorePasses.h.inc"
} // namespace

namespace {
struct MoorePrettyPrinterPass
    : public PassWrapper<MoorePrettyPrinterPass, OperationPass<ModuleOp>> {

  MoorePrettyPrinterPass() = default;
  MoorePrettyPrinterPass(const MoorePrettyPrinterPass& pass) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    llvm::outs() << "=== Pretty Printed Moore IR ===\n";

    module.walk([](Operation *op) {
      llvm::outs() << *op << "\n";  // Print each operation
    });

    llvm::outs() << "=== End of Pretty Printed IR ===\n";
  }

  StringRef getArgument() const override {
    return "moore-pretty-printer";
  }

  StringRef getDescription() const override {
    return "Formats Moore IR for readability.";
  }
};
} // namespace

// Explicit function to create the pass
std::unique_ptr<Pass> createMoorePrettyPrinterPass() {
  return std::make_unique<MoorePrettyPrinterPass>();
}

// Plugin entry point for MLIR
extern "C" void MLIRRegisterPassPlugin() {
  llvm::errs() << "Registering MoorePrettyPrinterPass plugin...\n";
  mlir::PassRegistration<MoorePrettyPrinterPass>();
}

