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

  mlir::Value findBitSource(mlir::Value vectorVal, unsigned bitIndex);
  mlir::Value vectorizeSubgraph(OpBuilder &builder, mlir::Value slice0Val, unsigned vectorWidth,
                                          llvm::DenseMap<mlir::Value, mlir::Value> &vectorizedMap);

  bool can_vectorize_structurally(mlir::Value output);
  bool areSubgraphsEquivalent(mlir::Value slice0Val, mlir::Value sliceNVal, unsigned sliceIndex,
                                        llvm::DenseMap<mlir::Value, mlir::Value> &slice0ToNMap);
  bool isValidPermutation(const std::vector<unsigned> &perm, unsigned bitWidth);

  void process_extract_ops();
  void process_concat_ops();

  void process_or_op(comb::OrOp op);
  void process_and_op(comb::AndOp op);
  void process_logical_ops();

  void vectorize();

  void apply_linear_vectorization(mlir::Value oldOutputVal, mlir::Value sourceInput);
  void apply_reverse_vectorization(mlir::OpBuilder &builder, mlir::Value oldOutputVal, mlir::Value sourceInput);
  void apply_mix_vectorization(mlir::OpBuilder &builder, mlir::Value oldOutputVal, mlir::Value sourceInput, const std::vector<unsigned> &map);
  void apply_structural_vectorization(OpBuilder &builder, mlir::Value oldOutputVal);

  void clean_hw_module(Block& body, OpBuilder& op_builder, Location& loc);
  void cleanup_dead_ops(Block& body);
};


#endif
