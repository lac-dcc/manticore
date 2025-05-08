#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;
using namespace circt;

namespace {

struct ScalarAssignGroup {
    moore::ExtractRefOp extractRef;
    moore::ExtractOp extract;
    moore::ContinuousAssignOp assign;
    int index;
};

struct SimpleVectorizationPass
    : public mlir::PassWrapper<SimpleVectorizationPass, mlir::OperationPass<mlir::ModuleOp>> {

    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SimpleVectorizationPass)

    void runOnOperation() override {
        auto module = getOperation();

        llvm::errs() << "Running SimpleVectorizationPass...\n";

        std::vector<ScalarAssignGroup> groups;

        module.walk([&](moore::ContinuousAssignOp assign) {
            auto lhs = assign.getDst();
            auto rhs = assign.getSrc();

            auto lhsOp = lhs.getDefiningOp();
            auto rhsOp = rhs.getDefiningOp();

            auto extractRef = dyn_cast_or_null<moore::ExtractRefOp>(lhsOp);
            auto extract = dyn_cast_or_null<moore::ExtractOp>(rhsOp);

            if (!extractRef || !extract)
                return;

            auto indexRefAttr = extractRef->getAttrOfType<mlir::IntegerAttr>("lowBit");
            auto indexAttr = extract->getAttrOfType<mlir::IntegerAttr>("lowBit");

            if (!indexRefAttr || !indexAttr)
                return;

            int indexRef = indexRefAttr.getInt();
            int indexVal = indexAttr.getInt();

            if (indexRef != indexVal)
                return;

            groups.push_back({extractRef, extract, assign, indexRef});
            llvm::errs() << "Found vectorizable assign at bit " << indexRef << "\n";
        });

        llvm::errs() << "Total vectorizable assigns: " << groups.size() << "\n";

        if (groups.empty())
            return;

        std::sort(groups.begin(), groups.end(), [](const ScalarAssignGroup &a, const ScalarAssignGroup &b) {
            return a.index < b.index;
        });

        bool allCompatible = true;
        auto baseDst = groups.front().extractRef.getOperand();
        auto baseSrc = groups.front().extract.getOperand();

        for (size_t i = 0; i < groups.size(); ++i) {
            if (groups[i].extractRef.getOperand() != baseDst ||
                groups[i].extract.getOperand() != baseSrc ||
                groups[i].index != static_cast<int>(i)) {
                allCompatible = false;
                break;
            }
        }

        if (!allCompatible) {
            llvm::errs() << "Incompatible assigns found; skipping vectorization.\n";
            return;
        }

        auto builder = OpBuilder(groups.front().assign.getContext());
        builder.setInsertionPoint(groups.front().assign);

        auto vectorizedAssign = builder.create<moore::ContinuousAssignOp>(
            groups.front().assign.getLoc(),
            baseDst,
            baseSrc
        );

        llvm::errs() << "Vectorized assign created: full assign from source to dest\n";

        for (auto &group : groups) {
            group.assign.erase();
            group.extractRef.erase();
            group.extract.erase();
        }
    }

    StringRef getArgument() const override { return "simple-vec"; }

    StringRef getDescription() const override {
        return "Simple Vectorization Pass";
    }
};
} 

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "SimpleVec",
        LLVM_VERSION_STRING,
        []() {
            PassPipelineRegistration<>(
                "simple-vec",
                "Simple Vectorization Pass",
                [](OpPassManager &pm) {
                    pm.addPass(std::make_unique<SimpleVectorizationPass>());
                });
        }
    };
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)