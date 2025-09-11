#include "../include/Vectorizer.h"
#include "../include/BitArray.h"

#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SmallVector.h"
#include <algorithm>
#include <vector>

vectorizer::vectorizer(hw::HWModuleOp module): module(module) {}

void vectorizer::vectorize() {
    process_extract_ops();
    process_concat_ops();
    process_logical_ops();

    Block &block = module.getBody().front();
    auto outputOp = dyn_cast<hw::OutputOp>(block.getTerminator());
    if (!outputOp) return;

    OpBuilder builder(module.getContext());
    bool changed = false;

    for (Value oldOutputVal : outputOp->getOperands()) {
        unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
        if (bit_arrays.find(oldOutputVal) == bit_arrays.end())
            continue;
        
        bit_array &arr = bit_arrays[oldOutputVal];

        if (arr.size() != bitWidth) {
            continue; 
        }

        Value sourceInput = arr.getSingleSourceValue();
        if (!sourceInput)
            continue;
        
        if (arr.is_linear(bitWidth, sourceInput)) {
            apply_linear_vectorization(oldOutputVal, sourceInput);
            changed = true;
        } 
        else if (arr.is_reverse_and_linear(bitWidth, sourceInput)) {
            apply_reverse_vectorization(builder, oldOutputVal, sourceInput);
            changed = true;
        } 
        else {
            std::vector<unsigned> currentPermutationMap;
            currentPermutationMap.reserve(bitWidth);
            for (unsigned i = 0; i < bitWidth; ++i)
                currentPermutationMap.push_back(arr.get_bit(i).index);
            
            std::vector<bool> seen(bitWidth, false);
            bool isFullPermutation = true;
            for (unsigned idx : currentPermutationMap) {
                if (idx >= bitWidth || seen[idx]) {
                    isFullPermutation = false;
                    break;
                }
                seen[idx] = true;
            }

            if (isFullPermutation) {
                apply_mix_vectorization(builder, oldOutputVal, sourceInput, currentPermutationMap);
                changed = true;
            }
        }
    }

    if (changed) {
        cleanup_dead_ops(block);
    }
}

void vectorizer::apply_linear_vectorization(Value oldOutputVal, Value sourceInput) {
    oldOutputVal.replaceAllUsesWith(sourceInput);
}

void vectorizer::apply_reverse_vectorization(OpBuilder &builder, Value oldOutputVal, Value sourceInput) {
    builder.setInsertionPoint(*oldOutputVal.getUsers().begin());
    Location loc = sourceInput.getLoc();

    Value reversedInput = builder.create<comb::ReverseOp>(loc, sourceInput);
    oldOutputVal.replaceAllUsesWith(reversedInput);
}

void vectorizer::apply_mix_vectorization(OpBuilder &builder, Value oldOutputVal, Value sourceInput, const std::vector<unsigned> &map) {
    unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
    Location loc = sourceInput.getLoc();
    builder.setInsertionPoint(*oldOutputVal.getUsers().begin());

    std::vector<Value> extractedChunks;
    unsigned i = 0;
    while (i < bitWidth) {
        unsigned startBit = map[i];
        unsigned len = 1;
        while ((i + len < bitWidth) && (map[i + len] == startBit + len)) {
            len++;
        }

        Value chunk = builder.create<comb::ExtractOp>(loc, builder.getIntegerType(len),
                                                      sourceInput, builder.getI32IntegerAttr(startBit));
        extractedChunks.push_back(chunk);
        i += len;
    }

    Value newOutputVal;
    if (extractedChunks.size() == 1) {
        newOutputVal = extractedChunks[0];
    } else {
        std::reverse(extractedChunks.begin(), extractedChunks.end());
        newOutputVal = builder.create<comb::ConcatOp>(loc, extractedChunks);
    }
    
    oldOutputVal.replaceAllUsesWith(newOutputVal);
}

void vectorizer::cleanup_dead_ops(Block &block) {
    bool changed = true;
    while (changed) {
        changed = false;

        llvm::SmallVector<Operation *, 16> deadOps;
        for (Operation &op : block) {
            if (op.use_empty() && !op.hasTrait<mlir::OpTrait::IsTerminator>()) {
                deadOps.push_back(&op);
            }
        }
        if (!deadOps.empty()) {
            changed = true;
            for (Operation *op : deadOps) {
                op->erase();
            }
        }
    }
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
        bit_array concatenatedArray; 

        unsigned currentBitOffset = 0;

        for (Value operand : llvm::reverse(op.getInputs())) {
            unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();

            if (bit_arrays.count(operand)) {
                bit_array &operandArray = bit_arrays[operand];
                for (auto const& [bitIndex, bitInfo] : operandArray.bits) {
                    concatenatedArray.bits[bitIndex + currentBitOffset] = bitInfo;
                }
            }
            currentBitOffset += operandWidth;
        }
        bit_arrays.insert({result, concatenatedArray});
    });
}

void vectorizer::process_or_op(comb::OrOp op) {
    mlir::Value result = op.getResult();
    mlir::Value lhs = op.getInputs()[0];
    mlir::Value rhs = op.getInputs()[1];

    bit_array lhs_array = bit_arrays.count(lhs) ? bit_arrays[lhs] : bit_array();
    bit_array rhs_array = bit_arrays.count(rhs) ? bit_arrays[rhs] : bit_array();

    bit_arrays.insert({result, bit_array::unite(lhs_array, rhs_array)});
}

void vectorizer::process_and_op(comb::AndOp op) {
    mlir::Value result = op.getResult();
    mlir::Value lhs = op.getInputs()[0];
    mlir::Value rhs = op.getInputs()[1];

    if (bit_arrays.count(lhs) && !bit_arrays.count(rhs)) {
        bit_arrays.insert({result, bit_arrays[lhs]});
        return;
    }
    if (bit_arrays.count(rhs) && !bit_arrays.count(lhs)) {
        bit_arrays.insert({result, bit_arrays[rhs]});
        return;
    }
    
    bit_arrays.insert({result, bit_array()});
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
