#include "AssignmentBasedVectorization.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/Comb/CombOps.h"

#include <map>
#include <vector>

using namespace mlir;
// using namespace HW;
using namespace circt;
using namespace moore;
using namespace comb;


void processAssignTree(mlir::ModuleOp module, VectorizationStatistics &stats) {
  module.walk([&](mlir::Operation *op) -> mlir::WalkResult {
    llvm::errs() << "op: " << op->getName() << "\n";
    return mlir::WalkResult::advance();   // keep going
  });
}
