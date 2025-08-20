#include "StructuralPatternVectorization.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using namespace mlir;
using namespace circt;
using namespace moore;

namespace {

// Struct for a sliced input from a vector.
struct SlicedInputInfo {
    mlir::Value vector;
    int bitIndex;
};

// Information about a found structural pattern.
struct StructuralPatternInfo {
  moore::ContinuousAssignOp assignOp;
  int bitIndex;
  llvm::SmallVector<SlicedInputInfo> slicedInputs;
  llvm::SmallVector<mlir::Value> uniformOperands;
};

// A unique signature that represents a structural pattern.
struct PatternSignature {
  std::string structure;
  llvm::SmallVector<mlir::Value> uniformOperands;
  llvm::SmallVector<mlir::Type> slicedOperandTypes;

  bool operator<(const PatternSignature &other) const {
    if (structure != other.structure) return structure < other.structure;
    if (uniformOperands != other.uniformOperands) {
      return std::lexicographical_compare(
          uniformOperands.begin(), uniformOperands.end(),
          other.uniformOperands.begin(), other.uniformOperands.end(),
          ValueComparator());
    }
    return std::lexicographical_compare(
        slicedOperandTypes.begin(), slicedOperandTypes.end(),
        other.slicedOperandTypes.begin(), other.slicedOperandTypes.end(),
        [](mlir::Type lhs, mlir::Type rhs) {
            return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
        });
  }
};

// Maps a signature to a list of occurrences of that pattern.
using PatternGroupMap =
    std::map<PatternSignature, std::vector<StructuralPatternInfo>>;


/**
 * @brief Recursively generates a signature string for an operation tree.
 *
 * Walks the operations starting from 'root', normalizing the inputs to
 * create a canonical representation of the computational structure.
 */
void generateSignature(Value root, std::stringstream &ss,
                       llvm::DenseMap<Value, std::string> &memo,
                       StructuralPatternInfo &info) {
  if (memo.count(root)) {
    ss << memo[root];
    return;
  }

  Operation *op = root.getDefiningOp();
  if (!op) {
    auto it = std::find(info.uniformOperands.begin(), info.uniformOperands.end(), root);
    size_t index = std::distance(info.uniformOperands.begin(), it);
    if (it == info.uniformOperands.end()) {
      index = info.uniformOperands.size();
      info.uniformOperands.push_back(root);
    }
    std::string repr = "U<" + std::to_string(index) + ">";
    ss << repr;
    memo[root] = repr;
    return;
  }

  if (auto extractOp = dyn_cast<moore::ExtractOp>(op)) {
    Value vector = extractOp.getInput();
    int bitIndex = extractOp.getLowBit();
    size_t index = info.slicedInputs.size();
    info.slicedInputs.push_back({vector, bitIndex});
    std::string repr = "S<" + std::to_string(index) + ">";
    ss << repr;
    memo[root] = repr;
    return;
  }

  ss << op->getName().getStringRef().str() << "(";
  for (unsigned i = 0; i < op->getNumOperands(); ++i) {
    generateSignature(op->getOperand(i), ss, memo, info);
    if (i < op->getNumOperands() - 1) ss << ",";
  }
  ss << ")";
}

/**
 * @brief Walks the module to find and group structural patterns.
 */
void collectAndGroupPatterns(ModuleOp module, PatternGroupMap &groups) {
  module.walk([&](moore::ContinuousAssignOp assign) {
    auto dst = assign.getDst();
    auto src = assign.getSrc();
    auto extractRef = dyn_cast_or_null<ExtractRefOp>(dst.getDefiningOp());
    if (!extractRef) return;
    
    // Ignore direct bit assignments (d[i] = s[j]), as they are handled by other logic.
    if (dyn_cast_or_null<ExtractOp>(src.getDefiningOp())) return;

    StructuralPatternInfo info;
    info.assignOp = assign;
    info.bitIndex = extractRef.getLowBit();

    std::stringstream ss;
    llvm::DenseMap<Value, std::string> memo;
    generateSignature(src, ss, memo, info);

    PatternSignature sig;
    sig.structure = ss.str();
    sig.uniformOperands = info.uniformOperands;
    for (const auto& slicedInfo : info.slicedInputs) {
      sig.slicedOperandTypes.push_back(slicedInfo.vector.getType());
    }
    groups[sig].push_back(info);
  });
}

/**
 * @brief Checks if a group of patterns is contiguous and can be vectorized.
 */
bool isValidForVectorization(std::vector<StructuralPatternInfo> &group) {
    if (group.size() < 2)
        return false;

    // Sort by the position of the destination bit (descending)
    llvm::sort(group, [](const StructuralPatternInfo &a, const StructuralPatternInfo &b) {
        return a.bitIndex > b.bitIndex;
    });

    // Checks if destination bits are contiguous
    for (size_t i = 1; i < group.size(); ++i) {
        if (group[i].bitIndex != group[i - 1].bitIndex - 1)
            return false;
    }

    // Checks contiguity in slicedInputs (if any)
    size_t numSlices = group.front().slicedInputs.size();
    for (size_t j = 0; j < numSlices; ++j) {
        for (size_t i = 1; i < group.size(); ++i) {
            if (group[i].slicedInputs[j].vector != group[0].slicedInputs[j].vector)
                return false;
            if (group[i].slicedInputs[j].bitIndex != group[i - 1].slicedInputs[j].bitIndex - 1)
                return false;
        }
    }

    return true;
}

/**
 * @brief Vectorizes a generic group of patterns.
 *
 * Reconstructs the operation graph using vector operands.
 */
void vectorizePatternGroup(std::vector<StructuralPatternInfo> &group, OpBuilder &builder) {
    if (group.empty()) return;

    Location loc = group.front().assignOp.getLoc();
    Value dstVector = cast<ExtractRefOp>(group.front().assignOp.getDst().getDefiningOp()).getInput();
    Type dstValueType = cast<RefType>(dstVector.getType()).getNestedType();

    llvm::DenseMap<Value, Value> scalarToVectorMap;

    for (Value uniform : group.front().uniformOperands) {
        scalarToVectorMap[uniform] = uniform;
    }
    for (const auto& slicedInfo : group.front().slicedInputs) {
        scalarToVectorMap[slicedInfo.vector] = slicedInfo.vector;
    }
    
    std::function<Value(Value)> recreate = 
        [&](Value scalarVal) -> Value {
        if (scalarToVectorMap.count(scalarVal))
            return scalarToVectorMap[scalarVal];

        Operation *op = scalarVal.getDefiningOp();
        assert(op && "Value with no defining operation found during recreation");
        
        if (auto extractOp = dyn_cast<moore::ExtractOp>(op)) {
            Value vector = extractOp.getInput();
            assert(scalarToVectorMap.count(vector) && "Sliced vector not found in map");
            return scalarToVectorMap[vector];
        }

        SmallVector<Value> newOperands;
        for (Value operand : op->getOperands()) {
            newOperands.push_back(recreate(operand));
        }

        Operation *newOp = builder.clone(*op);
        
        // Broadcasting logic: If an operand is a single bit (wire), replicate it to
        // match the destination vector's width.
        Type dominantType = dstValueType; 
        for (size_t i = 0; i < newOperands.size(); ++i) {
            if (newOperands[i].getType() != dominantType) {
                if (auto intType = dyn_cast<IntType>(newOperands[i].getType())) {
                    if (intType.getWidth() == 1) { 
                        Value scalar = newOperands[i];
                        int targetWidth = cast<IntType>(dominantType).getWidth();
                        SmallVector<Value> replicatedBits(targetWidth, scalar);
                        newOperands[i] = builder.create<moore::ConcatOp>(loc, dominantType, replicatedBits);
                    }
                }
            }
        }
        
        newOp->setOperands(newOperands);
        if (newOp->getNumResults() > 0) {
            newOp->getResult(0).setType(dominantType);
        }
        scalarToVectorMap[scalarVal] = newOp->getResult(0);
        return newOp->getResult(0);
    };

    Value finalVectorResult = recreate(group.front().assignOp.getSrc());
    builder.create<moore::ContinuousAssignOp>(loc, dstVector, finalVectorResult);

    // Clean up the old operations
    for (auto &info : group) {
        Operation* assign = info.assignOp;
        Value dst = assign->getOperand(0); // The ExtractRefOp
        assign->erase();
        if (dst.getDefiningOp() && dst.getDefiningOp()->use_empty()) {
            dst.getDefiningOp()->erase();
        }
    }
}
}

void processStructuralPatterns(mlir::ModuleOp module, VectorizationStatistics &stats) {
    PatternGroupMap patternGroups;
    collectAndGroupPatterns(module, patternGroups);

    OpBuilder builder(module.getContext());

    for (auto &[sig, infos] : patternGroups) {
        std::map<Value, std::vector<StructuralPatternInfo>, ValueComparator> subGroups;
        for (auto& info : infos) {
            Value dstVector = cast<ExtractRefOp>(info.assignOp.getDst().getDefiningOp()).getInput();
            subGroups[dstVector].push_back(info);
        }

        for (auto &[dstVec, subGroup] : subGroups) {
            // Safety lock: must cover the entire vector
            auto vecType = cast<RefType>(dstVec.getType()).getNestedType();
            if (!isa<IntType>(vecType)) continue;

            int totalWidth = cast<IntType>(vecType).getWidth();
            if ((int)subGroup.size() != totalWidth) continue;

            // Semantic validation of the pattern
            if (!isValidForVectorization(subGroup)) continue;

            builder.setInsertionPoint(subGroup.front().assignOp);
            vectorizePatternGroup(subGroup, builder);
            stats.increment("STRUCTURAL");
        }
    }
}
