#include "AssignmentBasedVectorization.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/Comb/CombOps.h"

#include <map>
#include <vector>

using namespace mlir;
using namespace circt;
using namespace moore;
using namespace comb;

namespace {

// Represents a single scalar assignment: d[dstIndex] = s[srcIndex]
struct ScalarAssignGroup {
  moore::ExtractRefOp       extractRef; // Operation that extracts the reference from the destination vector.
  moore::ExtractOp          extract;    // Operation that extracts the bit from the source vector.
  moore::ContinuousAssignOp assign;     // The assignment operation.
  int dstIndex;                         // The bit index in the destination.
  int srcIndex;                         // The bit index in the source.
};

// Maps a bit index to a scalar assignment.
using IndexedGroupMap = std::map<int, ScalarAssignGroup>;
// Maps a source vector to an index map.
using SourceGroupMap  = std::map<mlir::Value, IndexedGroupMap, ValueComparator>;
// Maps a destination vector to a source vector map. This is the main data structure.
using AssignTree      = std::map<mlir::Value, SourceGroupMap,  ValueComparator>;


// --- Pattern Detection Functions ---

// Checks if the group represents a linear assignment (d[i] = s[i]).
static bool isLinear(const std::vector<ScalarAssignGroup> &group) {
    if (group.empty()) return false;
    for (const auto &g : group) {
        if (g.dstIndex != g.srcIndex) return false;
    }
    return true;
}

// Checks if the group represents a reverse assignment (d[i] = s[N-1-i]).
static bool isReverse(const std::vector<ScalarAssignGroup> &group) {
    if (group.size() < 2) return false;
    // Sort by destination index to ensure a consistent stride check.
    std::vector<ScalarAssignGroup> sortedGroup = group;
    llvm::sort(sortedGroup, [](const auto& a, const auto& b){
        return a.dstIndex < b.dstIndex;
    });

    int dstStride = sortedGroup[1].dstIndex - sortedGroup[0].dstIndex;
    int srcStride = sortedGroup[1].srcIndex - sortedGroup[0].srcIndex;
    // The destination stride must be +1 and the source stride -1.
    return dstStride == 1 && srcStride == -1;
}

// Checks if it's a bit mix that is neither linear nor reverse.
bool isBitMixGroup(const std::vector<ScalarAssignGroup> &group) {
    if (group.empty() || isLinear(group) || isReverse(group)) return false;
    return true;
}

// --- Vectorization Functions (Actions) ---

// Replaces a group of linear assignments with a single vector assignment.
static void vectorizeLinearGroup(std::vector<ScalarAssignGroup> &group, OpBuilder &builder){
    if (group.empty()) return;
    builder.setInsertionPoint(group.front().assign);
    Value dst = group.front().extractRef.getInput();
    Value src = group.front().extract.getInput();

    // Create the new vector assignment: assign dst, src
    builder.create<moore::ContinuousAssignOp>(group.front().assign.getLoc(), dst, src);

    // Remove the old scalar operations.
    for (auto &g : group) {
        g.assign.erase();
        if(g.extractRef.use_empty()) g.extractRef.erase();
        // The `extract` operation is not erased as it might be used elsewhere.
    }
}

// Vectorizes a group of reverse-order assignments.
static void vectorizeReverseGroup(std::vector<ScalarAssignGroup> &group, OpBuilder &builder) {
    if (group.empty()) return;
    builder.setInsertionPoint(group.front().assign);
    Location loc = group.front().assign.getLoc();
    Value dst = group.front().extractRef.getInput();
    Value src = group.front().extract.getInput();
    Type sliceType = cast<RefType>(dst.getType()).getNestedType();
    Value sampleBit = group.front().extract.getResult();
    auto bitType = sampleBit.getType();

    // Extract the bits from the source in the order specified by the permutation.
    SmallVector<Value> bits(group.size());
    for (auto &g : group) {
        // Place the extracted bit in the correct destination position.
        bits[group.size() - 1 - g.dstIndex] = builder.create<moore::ExtractOp>(loc, bitType, src, builder.getI32IntegerAttr(g.srcIndex));
    }
    // Concatenate the bits to form the new vector.
    Value newVec = builder.create<moore::ConcatOp>(loc, sliceType, bits);
    builder.create<moore::ContinuousAssignOp>(loc, dst, newVec);

    // Remove the old operations.
    for (auto &g : group) {
        g.assign.erase();
        if(g.extractRef.use_empty()) g.extractRef.erase();
    }
}

// Vectorizes a group of mixed assignments (permutation).
static void vectorizeMixGroup(std::vector<ScalarAssignGroup> &group, OpBuilder &builder) {
    if (group.empty()) return;
    builder.setInsertionPoint(group.front().assign);
    Location loc = group.front().assign.getLoc();
    Value dst = group.front().extractRef.getInput();
    Value src = group.front().extract.getInput();
    Type sliceType = cast<RefType>(dst.getType()).getNestedType();
    Value sampleBit = group.front().extract.getResult();
    auto bitType = sampleBit.getType();

    // Extract the bits from the source in the order specified by the permutation.
    SmallVector<Value> bits(group.size());
    for (auto &g : group) {
        // Place the extracted bit in the correct destination position.
        bits[group.size() - 1 - g.dstIndex] = builder.create<moore::ExtractOp>(loc, bitType, src, builder.getI32IntegerAttr(g.srcIndex));
    }
    // Concatenate the bits to form the new vector.
    Value newVec = builder.create<moore::ConcatOp>(loc, sliceType, bits);
    builder.create<moore::ContinuousAssignOp>(loc, dst, newVec);

    // Remove the old operations.
    for (auto &g : group) {
        g.assign.erase();
        if(g.extractRef.use_empty()) g.extractRef.erase();
    }
}

// Walks the module to find all scalar assignments and populates the AssignTree.
void collectAssigns(mlir::ModuleOp module, AssignTree &assignTree) {
  module.walk([&](moore::ContinuousAssignOp assign) {
    auto lhs = assign.getDst();
    auto rhs = assign.getSrc();

    // Check if the assignment is of the form: d[i] = s[j]
    auto extractRef = dyn_cast_or_null<moore::ExtractRefOp>(lhs.getDefiningOp());
    auto extract = dyn_cast_or_null<moore::ExtractOp>(rhs.getDefiningOp());

    if (!extractRef || !extract) return;

    auto dstAttr = extractRef->getAttrOfType<mlir::IntegerAttr>("lowBit");
    auto srcAttr = extract->getAttrOfType<mlir::IntegerAttr>("lowBit");

    if (!dstAttr || !srcAttr) return;

    int dstIndex = dstAttr.getInt();
    int srcIndex = srcAttr.getInt();

    // Add the information to the tree: [dst_vector][src_vector][dst_index] = info
    assignTree[extractRef.getInput()][extract.getInput()][dstIndex] =
        {extractRef, extract, assign, dstIndex, srcIndex};
  });
}

// Processes the AssignTree, identifies patterns, and vectorizes them.
void processAssignTreeInternal(AssignTree &assignTree, VectorizationStatistics &stats, OpBuilder &builder) {
    if (assignTree.empty()) {
        return;
    }
    for (auto &[dst, srcMap] : assignTree) {
        for (auto &[src, indexMap] : srcMap) {
            std::vector<ScalarAssignGroup> group;
            for (const auto &[dstIndex, g] : indexMap) {
                group.push_back(g);
            }

            if (group.size() > 1) {
                auto dstType = dyn_cast<IntType>(cast<RefType>(dst.getType()).getNestedType());
                if (!dstType) continue;

                // Vectorization only makes sense if it covers the entire destination vector.
                bool isFullWidth = (group.size() == dstType.getWidth());
                if (!isFullWidth) continue;

                if (isLinear(group)) {
                    stats.increment("LINEAR");
                    vectorizeLinearGroup(group, builder);
                } else if (isReverse(group)) {
                    stats.increment("REVERSE");
                    vectorizeReverseGroup(group, builder);
                } else if (isBitMixGroup(group)) {
                    stats.increment("MIX");
                    vectorizeMixGroup(group, builder);
                }
            }
        }
    }
}

} // End of anonymous namespace

// orchestrates the collection and processing of assignment patterns.
void processAssignTree(mlir::ModuleOp module, VectorizationStatistics &stats) {
    AssignTree assignTree;
    collectAssigns(module, assignTree);

    // If the tree is empty, there is nothing to do.
    if (assignTree.empty()) {
        return;
    }

    OpBuilder builder(module.getContext());
    processAssignTreeInternal(assignTree, stats, builder);
}