#include "../include/Vectorizer.h"

vectorizer::vectorizer(hw::HWModuleOp module): module(module) {}

void vectorizer::vectorize() {
  process_extract_ops();
  process_concat_ops();
  process_logical_ops();

  if(linear_vectorization_detected()) {
    apply_linear_vectorization();
  }
  if(reverse_linear_vectorization_detected()) {
    apply_reverse_linear_vectorization();    
  }
}

void vectorizer::apply_linear_vectorization() {
  Block &body = module.getBody().front();
  OpBuilder builder(module.getContext());
  Location loc = module.getLoc();
  
  clean_hw_module(body, builder, loc);

  // concertar ISSO
  BlockArgument in0 = body.getArgument(0);

  builder.setInsertionPointToEnd(&body);
  builder.create<hw::OutputOp>(loc, ValueRange{in0}); 
}

void vectorizer::apply_reverse_linear_vectorization() {
  Block &body = module.getBody().front();
  OpBuilder builder(module.getContext());
  Location loc = module.getLoc();
  
  clean_hw_module(body, builder, loc);
  

  BlockArgument in0 = body.getArgument(0);

  builder.setInsertionPointToEnd(&body);
  mlir::Value reversed = builder.create<comb::ReverseOp>(loc, in0);
  builder.create<hw::OutputOp>(loc, ValueRange{reversed});
}

void vectorizer::clean_hw_module(Block& body, OpBuilder& op_builder, Location& loc) {
  for(auto input :  body.getArguments()) {
    for (Operation &op : body) {
      for (Value res : op.getResults()) {
        if (res != input)               
          res.replaceAllUsesWith(input);
      }
    }
  }

  if (Operation *term = body.getTerminator()) term->erase();
  while (!body.empty()) body.back().erase();
}


bool vectorizer::linear_vectorization_detected() {
  bool linear_vectorization;

  module.walk([&](hw::OutputOp op) {
    mlir::Value lhs = op.getOutputs()[0]; 
    unsigned bit_width = llvm::cast<mlir::IntegerType>(lhs.getType()).getWidth();
    
    linear_vectorization = bit_arrays[lhs].is_linear(bit_width);
  });

  return linear_vectorization;
}

bool vectorizer::reverse_linear_vectorization_detected() {
  bool reverse_vectorization;

  module.walk([&](hw::OutputOp op) {
    mlir::Value lhs = op.getOutputs()[0]; 
    unsigned bit_width = llvm::cast<mlir::IntegerType>(lhs.getType()).getWidth();
    
    reverse_vectorization = bit_arrays[lhs].is_reverse_and_linear(bit_width);
  });

  return reverse_vectorization;
}


void vectorizer::process_extract_ops() {
  module.walk([&](comb::ExtractOp op) {
    mlir::Value input = op.getInput();


    mlir::Value result = op.getResult();
    int index = op.getLowBit();

    llvm::DenseMap<int,bit> bit_dense_map;
    bit_dense_map.insert({0, bit(input, index)});

    bit_array bits(bit_dense_map);
    bit_arrays.insert({result, bits});
  });

}

void vectorizer::process_concat_ops() {
  module.walk([&](comb::ConcatOp op) {
    mlir::Value result = op.getResult();

    unsigned vector_size = llvm::cast<mlir::IntegerType>(result.getType()).getWidth();

    int index = 0;
    for(auto [i, value] : llvm::enumerate(op.getInputs())) {
      unsigned bit_width = llvm::cast<mlir::IntegerType>(value.getType()).getWidth();

      if(bit_arrays.contains(value)) {
        llvm::DenseMap<int,bit> array;
        //TODO REVISAR ESSA CONTA, PODE ESTAR ERRADO
        array.insert({vector_size - index - 1, bit_arrays[value].get_bit(0)});

        bit_arrays.insert({result, array});
      }
      index += bit_width;
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
