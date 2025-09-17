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
        bool transformed = false;
        unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();

        if (bit_arrays.count(oldOutputVal)) {
            bit_array &arr = bit_arrays[oldOutputVal];
            if (arr.size() == bitWidth) {
                Value sourceInput = arr.getSingleSourceValue();
                if (sourceInput) {
                    std::vector<unsigned> currentPermutationMap;
                    llvm::SmallDenseSet<unsigned> seen;
                    bool hasDuplicate = false;

                    for (unsigned i = 0; i < bitWidth; ++i) {
                        unsigned idx = arr.get_bit(i).index;
                        if (!seen.insert(idx).second) {
                            hasDuplicate = true;
                            break; 
                        }
                        currentPermutationMap.push_back(idx);
                    }

                    if (hasDuplicate) continue;

                    if (isValidPermutation(currentPermutationMap, bitWidth)) {
                        if (arr.is_linear(bitWidth, sourceInput)) {
                            apply_linear_vectorization(oldOutputVal, sourceInput);
                            transformed = true;
                        } else if (arr.is_reverse_and_linear(bitWidth, sourceInput)) {
                            apply_reverse_vectorization(builder, oldOutputVal, sourceInput);
                            transformed = true;
                        } else {
                            apply_mix_vectorization(builder, oldOutputVal, sourceInput, currentPermutationMap);
                            transformed = true;
                        }
                    }
                }
            }
        }

        if (!transformed) {
            if (can_vectorize_structurally(oldOutputVal)) {
                apply_structural_vectorization(builder, oldOutputVal);
                transformed = true;
            }
        }

        if (transformed) changed = true;
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

bool vectorizer::isValidPermutation(const std::vector<unsigned> &perm, unsigned bitWidth) {
    if (perm.size() != bitWidth) return false;
    std::vector<bool> seen(bitWidth, false);

    for (unsigned idx : perm) {
        if (idx >= bitWidth) return false;
        if (seen[idx]) return false; 
        seen[idx] = true;
    }
    return true;
}

bool vectorizer::can_vectorize_structurally(mlir::Value output) {
    unsigned bitWidth = cast<IntegerType>(output.getType()).getWidth();
    if (bitWidth <= 1) return false;

    Value slice0Val = findBitSource(output, 0);
    if (!slice0Val || !slice0Val.getDefiningOp()) return false;

    for (unsigned i = 1; i < bitWidth; ++i) {
        Value sliceNVal = findBitSource(output, i);
        if (!sliceNVal || !sliceNVal.getDefiningOp()) return false;

        llvm::DenseMap<mlir::Value, mlir::Value> map;
        if (!areSubgraphsEquivalent(slice0Val, sliceNVal, i, map)) {
            return false;
        }
    }
    return true;
}

mlir::Value vectorizer::vectorizeSubgraph(OpBuilder &builder, mlir::Value slice0Val, unsigned vectorWidth,
                                          llvm::DenseMap<mlir::Value, mlir::Value> &vectorizedMap) {
    if (vectorizedMap.count(slice0Val))
        return vectorizedMap[slice0Val];

    if (auto extractOp = dyn_cast_or_null<comb::ExtractOp>(slice0Val.getDefiningOp())) {
        Value vector = extractOp.getInput();
        vectorizedMap[slice0Val] = vector;
        return vector;
    }

    if (mlir::isa<BlockArgument>(slice0Val) || mlir::isa<hw::ConstantOp>(slice0Val.getDefiningOp())) {
        unsigned scalarWidth = cast<IntegerType>(slice0Val.getType()).getWidth();
        if (scalarWidth == 1) {
            return builder.create<comb::ReplicateOp>(slice0Val.getLoc(), builder.getIntegerType(vectorWidth), slice0Val);
        }
        return slice0Val;
    }
    
    Operation *op0 = slice0Val.getDefiningOp();
    if (!op0) return nullptr;
    Location loc = op0->getLoc();

    SmallVector<Value> vectorizedOperands;
    for (Value operand : op0->getOperands()) {
        Value vectorizedOperand = vectorizeSubgraph(builder, operand, vectorWidth, vectorizedMap);
        if (!vectorizedOperand) return nullptr;
        vectorizedOperands.push_back(vectorizedOperand);
    }

    Type resultType = builder.getIntegerType(vectorWidth);
    Value vectorizedResult;

    if (dyn_cast<comb::AndOp>(op0)) {
        vectorizedResult = builder.create<comb::AndOp>(loc, resultType, vectorizedOperands);
    } else if (dyn_cast<comb::OrOp>(op0)) {
        vectorizedResult = builder.create<comb::OrOp>(loc, resultType, vectorizedOperands);
    } else if (dyn_cast<comb::XorOp>(op0)) {
        vectorizedResult = builder.create<comb::XorOp>(loc, resultType, vectorizedOperands);
    } else if (dyn_cast<comb::AddOp>(op0)) {
        vectorizedResult = builder.create<comb::AddOp>(loc, resultType, vectorizedOperands);
    } else if (dyn_cast<comb::MuxOp>(op0)) {
        Value sel = vectorizedOperands[0];
        if (cast<IntegerType>(sel.getType()).getWidth() != 1) {
           sel = builder.create<comb::ExtractOp>(loc, builder.getI1Type(), sel, 0);
        }
        Value replicatedSel = builder.create<comb::ReplicateOp>(loc, resultType, sel);
        vectorizedResult = builder.create<comb::MuxOp>(loc, replicatedSel, vectorizedOperands[1], vectorizedOperands[2]);
    } else {
        return nullptr;
    }
    
    vectorizedMap[slice0Val] = vectorizedResult;
    return vectorizedResult;
}

void vectorizer::apply_structural_vectorization(OpBuilder &builder, mlir::Value oldOutputVal) {
    unsigned bitWidth = cast<IntegerType>(oldOutputVal.getType()).getWidth();
    Value slice0Val = findBitSource(oldOutputVal, 0);
    if (!slice0Val) return;

    llvm::DenseMap<mlir::Value, mlir::Value> vectorizedMap;
    builder.setInsertionPoint(*oldOutputVal.getUsers().begin());

    Value newOutputVal = vectorizeSubgraph(builder, slice0Val, bitWidth, vectorizedMap);
    if (!newOutputVal) return;
    
    oldOutputVal.replaceAllUsesWith(newOutputVal);
}


bool vectorizer::areSubgraphsEquivalent(mlir::Value slice0Val, mlir::Value sliceNVal, unsigned sliceIndex,
                                        llvm::DenseMap<mlir::Value, mlir::Value> &slice0ToNMap) {
    if (slice0ToNMap.count(slice0Val))
        return slice0ToNMap[slice0Val] == sliceNVal;

    Operation *op0 = slice0Val.getDefiningOp();
    Operation *opN = sliceNVal.getDefiningOp();

    if (auto extract0 = dyn_cast_or_null<comb::ExtractOp>(op0)) {
        auto extractN = dyn_cast_or_null<comb::ExtractOp>(opN);
        if (extractN && extract0.getInput() == extractN.getInput() &&
            extractN.getLowBit() == extract0.getLowBit() + sliceIndex) {
            slice0ToNMap[slice0Val] = sliceNVal;
            return true;
        }
    }

    if (slice0Val == sliceNVal && (mlir::isa<BlockArgument>(slice0Val) || mlir::isa<hw::ConstantOp>(op0))) {
        slice0ToNMap[slice0Val] = sliceNVal;
        return true;
    }

    if (!op0 || !opN || op0->getName() != opN->getName() || op0->getNumOperands() != opN->getNumOperands())
        return false;

    for (unsigned i = 0; i < op0->getNumOperands(); ++i) {
        if (!areSubgraphsEquivalent(op0->getOperand(i), opN->getOperand(i), sliceIndex, slice0ToNMap))
            return false;
    }

    slice0ToNMap[slice0Val] = sliceNVal;
    return true;
}

mlir::Value vectorizer::findBitSource(mlir::Value vectorVal, unsigned bitIndex) {
    Operation *op = vectorVal.getDefiningOp();
    if (!op) return nullptr;

    if (op->getNumResults() == 1 && op->getResult(0).getType().isInteger(1)) {
        return op->getResult(0);
    }

    if (auto concat = dyn_cast<comb::ConcatOp>(op)) {
        unsigned currentBit = cast<IntegerType>(vectorVal.getType()).getWidth();
        for (Value operand : concat.getInputs()) {
            unsigned operandWidth = cast<IntegerType>(operand.getType()).getWidth();
            currentBit -= operandWidth;
            if (bitIndex >= currentBit && bitIndex < currentBit + operandWidth) {
                return findBitSource(operand, bitIndex - currentBit);
            }
        }
    } else if (auto orOp = dyn_cast<comb::OrOp>(op)) {
        if (auto source = findBitSource(orOp.getInputs()[1], bitIndex)) {
             if (!mlir::isa<hw::ConstantOp>(source.getDefiningOp()) || 
                 !cast<hw::ConstantOp>(source.getDefiningOp()).getValue().isZero())
                return source;
        }
        return findBitSource(orOp.getInputs()[0], bitIndex);
    } else if (auto andOp = dyn_cast<comb::AndOp>(op)) {
        mlir::Value lhs = andOp.getInputs()[0];
        mlir::Value rhs = andOp.getInputs()[1];
        if (mlir::isa<hw::ConstantOp>(rhs.getDefiningOp()))
            return findBitSource(lhs, bitIndex);
        if (mlir::isa<hw::ConstantOp>(lhs.getDefiningOp()))
            return findBitSource(rhs, bitIndex);
    }
    return nullptr;
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