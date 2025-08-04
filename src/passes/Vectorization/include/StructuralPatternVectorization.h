#ifndef MANTICORE_STRUCTURAL_PATTERN_VECTORIZATION_H
#define MANTICORE_STRUCTURAL_PATTERN_VECTORIZATION_H

#include "VectorizationUtils.h"

namespace mlir {
class ModuleOp;
} // namespace mlir

/**
 * @brief Performs vectorization based on structural pattern detection.
 *
 * This function traverses the module, searching for operations that form
 * repeating patterns on consecutive vector bits, and replaces them with
 * equivalent vector operations.
 *
 * @param module The MLIR module to be processed.
 * @param stats An object for collecting statistics on the applied optimizations.
 */
void processStructuralPatterns(mlir::ModuleOp module, VectorizationStatistics &stats);

#endif // MANTICORE_STRUCTURAL_PATTERN_VECTORIZATION_H