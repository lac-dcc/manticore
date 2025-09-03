#ifndef VECTORIZER_H
#define VECTORIZER_H

#include "AssignmentBasedVectorization.h"
#include "../include/BitArray.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/Comb/CombOps.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>

#include "../include/BitArray.h"


using namespace mlir;
using namespace circt;
using namespace comb;
using namespace hw;

class vectorizer {
public:
  vectorizer(mlir::ModuleOp module);      

  mlir::ModuleOp module;
  llvm::DenseMap<mlir::Value, bit_array> bit_arrays;

  void process_extract_ops();
  void process_concat_ops();

  void process_or_op(comb::OrOp op);
  void process_and_op(comb::AndOp op);
  void process_logical_ops();

  void vectorize();

  void apply_linear_vectorization(hw::HWModuleOp hw_module);
  bool linear_vectorization_detected();
};


#endif
