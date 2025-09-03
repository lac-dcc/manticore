#include "AssignmentBasedVectorization.h"
#include "../include/BitArray.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/Comb/CombOps.h"

#include "../include/Vectorizer.h"

#include <llvm/ADT/DenseMap.h>
#include <llvm/Support/Casting.h>
#include <mlir/IR/MLIRContext.h>
#include <set>
#include <utility>

using namespace mlir;
using namespace circt;
using namespace moore;
using namespace comb;
using namespace hw;


llvm::DenseMap<mlir::Value, std::pair<int,int>> get_assignments(mlir::ModuleOp module) {
    llvm::DenseMap<mlir::Value, std::pair<int,int>> assignments;
  
    module.walk([&](comb::ExtractOp op) {
        mlir::Value input = op.getInput();
        BlockArgument block_arg_input = mlir::cast<BlockArgument>(input);
        int arg_number = block_arg_input.getArgNumber();

        mlir::Value result = op.getResult();
        int variable_index = op.getLowBit();

        if(!assignments.contains(result)) {
          assignments.insert({result, {arg_number, variable_index}}); 
        }
    }); 

    return assignments;
}

// llvm::DenseMap<mlir::Value, bit_array> get_bit_arrays(mlir::ModuleOp module) {
//   llvm::DenseMap<mlir::Value, bit_array> bit_arrays;
//
//   module.walk([&](comb::ExtractOp op) {
//     mlir::Value input = op.getInput();
//
//     mlir::Value result = op.getResult();
//     int index = op.getLowBit();
//
//     llvm::DenseSet<bit> bit_dense_set;
//     bit_dense_set.insert(bit(input, index));
//
//     bit_array bits(bit_dense_set, result);
//     bit_arrays.insert({result, bits});
//   });
//
//   return bit_arrays;
// }

llvm::DenseMap<mlir::Value, std::pair<mlir::Value, int>> get_bit_vectors(mlir::ModuleOp module, llvm::DenseMap<mlir::Value, std::pair<int,int>>& extracted_bits) {
  llvm::DenseMap<mlir::Value, std::pair<mlir::Value, int>> concatenations;

  module.walk([&](comb::ConcatOp op) {
    mlir::Value result = op.getResult();

    int index = 0;
    for(auto [i, value] : llvm::enumerate(op.getInputs())) {
      unsigned bit_width = llvm::cast<mlir::IntegerType>(value.getType()).getWidth();

      if(extracted_bits.contains(value)) {
        concatenations.insert({result, {value, index}}); 
        break;
      }

      index += bit_width;
    }
  });


  return concatenations;
}

void process_or(llvm::DenseMap<mlir::Value, std::vector<std::pair<mlir::Value, int>>>& bit_vectors, comb::OrOp op) {
  mlir::Value result = op.getResult();

  mlir::Value lhs = op.getInputs()[0];
  mlir::Value rhs = op.getInputs()[1];
  
  std::vector<std::pair<mlir::Value,int>> bits; 

  for(auto bit : bit_vectors[lhs]) bits.push_back(bit);
  for(auto bit : bit_vectors[rhs]) bits.push_back(bit);

  bit_vectors.insert({result, bits});
}

void process_and(llvm::DenseMap<mlir::Value, std::vector<std::pair<mlir::Value, int>>>& bit_vectors, comb::AndOp op) {
  mlir::Value result = op.getResult();
  mlir::Value lhs = op.getInputs()[0];

  bit_vectors.insert({result, bit_vectors[lhs]});
}

llvm::DenseMap<mlir::Value, std::vector<std::pair<mlir::Value,int>>> compute_ors_ands(mlir::ModuleOp module, llvm::DenseMap<mlir::Value, std::pair<mlir::Value,int>>& concatenations) {
  llvm::DenseMap<mlir::Value, std::vector<std::pair<mlir::Value, int>>> bit_vectors;
  
  for(auto [variable, bit] : concatenations) {
    bit_vectors.insert({variable, {bit}});
  }


  module.walk([&](mlir::Operation* op) {
    if(llvm::isa<comb::OrOp, comb::AndOp>(op)) {
      if(auto orOp = llvm::dyn_cast<comb::OrOp>(op)) {
        process_or(bit_vectors, orOp);
      }
      else {
        auto andOp = llvm::dyn_cast<comb::AndOp>(op);
        process_and(bit_vectors, andOp);
      }
    }
  });


  return bit_vectors;
}

void vectorize(hw::HWModuleOp m) {
  Block &body = m.getBody().front();
  OpBuilder b(m.getContext());
  Location loc = m.getLoc();

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

void apply_vectorization(mlir::ModuleOp module) {
  SmallVector<hw::HWModuleOp, 8> mods;
  for (auto m : module.getOps<hw::HWModuleOp>()) mods.push_back(m);

  for (auto m : mods) vectorize(m);    
}

bool linear_vectorization_detected(mlir::ModuleOp module, llvm::DenseMap<mlir::Value, std::vector<std::pair<mlir::Value,int>>>& bit_vectors) {
  bool linear_vectorization = true;

  module.walk([&](hw::OutputOp op) {
    mlir::Value lhs = op.getOutputs()[0]; 
    unsigned bit_width = llvm::cast<mlir::IntegerType>(lhs.getType()).getWidth();

    std::set<int> assigned_bits;
    for(int i = 0; i < bit_width; i++) assigned_bits.insert(i);
    
    std::vector<std::pair<mlir::Value,int>> bit_vector = bit_vectors[lhs];

    for(auto [value, index] : bit_vector) {
      if(assigned_bits.find(index) != assigned_bits.end()) {
        assigned_bits.erase(index);
      }
    }

    if(assigned_bits.size() != 0) linear_vectorization = false;

  });

  return linear_vectorization;
}




void performVectorization(mlir::ModuleOp module, VectorizationStatistics &stats) {
  // llvm::DenseMap<mlir::Value, std::pair<int,int>> extracted_bits = get_assignments(module);
  // llvm::DenseMap<mlir::Value, std::pair<mlir::Value,int>> concatenations = get_bit_vectors(module, extracted_bits);
  //
  // llvm::DenseMap<mlir::Value, std::vector<std::pair<mlir::Value,int>>> final = compute_ors_ands(module, concatenations);
  //
  // if(linear_vectorization_detected(module, final)) {
  //   apply_vectorization(module);
  // }
  //

  vectorizer v(module);
  v.vectorize(); 
}

