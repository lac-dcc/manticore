#ifndef MANTICORE_ASSIGNMENT_BASED_VECTORIZATION_H
#define MANTICORE_ASSIGNMENT_BASED_VECTORIZATION_H

#include "VectorizationUtils.h"

namespace mlir {
class ModuleOp;
class OpBuilder;
} // namespace mlir


void processAssignTree(mlir::ModuleOp module, VectorizationStatistics &stats);

#endif // MANTICORE_ASSIGNMENT_BASED_VECTORIZATION_H