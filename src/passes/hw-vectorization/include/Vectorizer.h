#ifndef VECTORIZER_H
#define VECTORIZER_H

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
  vectorizer(hw::HWModuleOp module);      

  hw::HWModuleOp module;
  llvm::DenseMap<mlir::Value, bit_array> bit_arrays;
  Block& body;
  OpBuilder builder;
  Location loc;

  void process_extract_ops();
  void process_concat_ops();

  void process_or_op(comb::OrOp op);
  void process_and_op(comb::AndOp op);
  void process_logical_ops();

  void vectorize();

  void apply_linear_vectorization();
  bool linear_vectorization_detected();

  void apply_reverse_linear_vectorization();
  bool reverse_linear_vectorization_detected();

  void apply_mixed_vectorization();

  void apply_vectorizations();
  mlir::Value vectorize_bit_array(bit_array& array, int size);

  void clean_hw_module(Block& body, OpBuilder& op_builder, Location& loc);
};


#endif
