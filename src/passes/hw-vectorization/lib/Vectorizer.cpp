#include "../include/Vectorizer.h"

vectorizer::vectorizer(mlir::ModuleOp module): module(module) {}

void vectorizer::vectorize() {
  process_extract_ops();
  process_concat_ops();
  process_logical_ops();

  // TODO: arrumar isso aqui. Fazer com que o vetorizador vetorize apenas um modulo
  SmallVector<hw::HWModuleOp, 8> mods;
  for (auto m : module.getOps<hw::HWModuleOp>()) mods.push_back(m);

  if(linear_vectorization_detected()) {
    apply_linear_vectorization(mods[0]);
  }
}

void vectorizer::apply_linear_vectorization(hw::HWModuleOp hw_module) {
  Block &body = hw_module.getBody().front();
  OpBuilder b(hw_module.getContext());
  Location loc = hw_module.getLoc();

  BlockArgument in0 = body.getArgument(0);

  for (Operation &op : body) {
    for (Value res : op.getResults()) {
      if (res != in0)               
        res.replaceAllUsesWith(in0);
    }
  }

  if (Operation *term = body.getTerminator()) term->erase();

  while (!body.empty()) body.back().erase();

  b.setInsertionPointToEnd(&body);
  b.create<hw::OutputOp>(loc, ValueRange{in0}); 
}


bool vectorizer::linear_vectorization_detected() {
  bool linear_vectorization;

  module.walk([&](hw::OutputOp op) {
    mlir::Value lhs = op.getOutputs()[0]; 
    unsigned bit_width = llvm::cast<mlir::IntegerType>(lhs.getType()).getWidth();
    
    linear_vectorization = bit_arrays[lhs].is_contiguous(bit_width);
  });

  return linear_vectorization;
}


void vectorizer::process_extract_ops() {
  module.walk([&](comb::ExtractOp op) {
    mlir::Value input = op.getInput();

    mlir::Value result = op.getResult();
    int index = op.getLowBit();

    llvm::DenseSet<bit> bit_dense_set;
    bit_dense_set.insert(bit(input, index));

    bit_array bits(bit_dense_set);
    bit_arrays.insert({result, bits});
  });

}

void vectorizer::process_concat_ops() {
  module.walk([&](comb::ConcatOp op) {
    mlir::Value result = op.getResult();

    for(auto input : op.getInputs()) {
      if(bit_arrays.contains(input)) {
        bit_arrays.insert({result, bit_arrays[input]}); 
      }
    }
  });
}

void vectorizer::process_or_op(comb::OrOp op) {
  mlir::Value result = op.getResult();
  mlir::Value lhs = op.getInputs()[0];
  mlir::Value rhs = op.getInputs()[1];

  bit_arrays.insert({result, bit_array::unite(bit_arrays[lhs], bit_arrays[rhs])});
} 

void vectorizer::process_and_op(comb::AndOp op) {
  mlir::Value result = op.getResult();
  mlir::Value lhs = op.getInputs()[0];

  bit_arrays.insert({result, bit_arrays[lhs]});
} 

void vectorizer::process_logical_ops() {
  module.walk([&](mlir::Operation* op) {
    if(llvm::isa<comb::OrOp, comb::AndOp>(op)) {
      if(auto or_op = llvm::dyn_cast<comb::OrOp>(op)) {
        process_or_op(or_op);
      }
      else {
        auto and_op = llvm::dyn_cast<comb::AndOp>(op);
        process_and_op(and_op);
      }
    }
  });
}
