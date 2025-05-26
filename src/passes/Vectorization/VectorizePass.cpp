#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/IndentedOstream.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Config/llvm-config.h"

using namespace mlir;
using namespace circt;
using namespace moore;
using namespace hw;

namespace {

struct ScalarAssignGroup {
  moore::ExtractRefOp     extractRef;
  moore::ExtractOp        extract;
  moore::ContinuousAssignOp assign;
  int dstIndex;
  int srcIndex;
};

struct ValueComparator {
  bool operator()(mlir::Value lhs, mlir::Value rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

// dst-wire  → src-wire → bit-dst → grupo
using IndexedGroupMap = std::map<int, ScalarAssignGroup>;
using SourceGroupMap  = std::map<mlir::Value, IndexedGroupMap, ValueComparator>;
using AssignTree      = std::map<mlir::Value, SourceGroupMap,  ValueComparator>;

///  Returns true if the group represents a reverse pattern:
///  dst indices go up by 1 (…,2,1,0)
///  src indices go down by 1 (0,1,2,… in the opposite direction)
static bool isReverse(const std::vector<ScalarAssignGroup> &grp) {
  if (grp.size() < 2)
    return false;
  int dstStride = grp[1].dstIndex - grp[0].dstIndex; // expected  +1
  int srcStride = grp[1].srcIndex - grp[0].srcIndex; // expected  -1
  return dstStride == 1 && srcStride == -1;
}

//  contiguous slice
void vectorizeGroup(std::vector<ScalarAssignGroup> &group){
    if (group.empty()) return;

    // llvm::errs() << "Vectorizing group of size: " << group.size() << "\n";
    // for (size_t i = 0; i < group.size(); ++i) {
    //     llvm::errs() << "  Element " << i 
    //                  << ": index = " << group[i].index
    //                  << ", dst base = " << group[i].extractRef.getOperand()
    //                  << ", src base = " << group[i].extract.getOperand() << "\n";
    // }

    // create builder at first assign's context
    auto builder = OpBuilder(group.front().assign.getContext());

    // set insertion point at the first assign
    builder.setInsertionPoint(group.front().assign);

    // get base destination operand (outA)
    auto dst = group.front().extractRef.getOperand();

    // get base source operand (inA)
    auto src = group.front().extract.getOperand();

    // create vectorized assign operation (assign whole vector at once) -> moore.continuous.assign %outA, %inA
    auto vectorizedAssign = builder.create<moore::ContinuousAssignOp>(
        group.front().assign.getLoc(),
        dst,
        src
    );

    // erase old scalar assign ops and extracts
    for (auto &g : group) {
        g.assign.erase();
        g.extractRef.erase();
        g.extract.erase();
    }
}

// vectorizes a reverse pattern group where
// dst indices increase and src indices decrease
static void vectorizeReverseGroup(std::vector<ScalarAssignGroup> &group) {
  if (group.empty())
    return;

  OpBuilder builder(group.front().assign.getContext());
  builder.setInsertionPoint(group.front().assign);

  Location loc = group.front().assign.getLoc();
  Value dst    = group.front().extractRef.getOperand(); // destination wire
  Value src    = group.front().extract.getOperand();    // source wire

  // source bit range 
  int highBit = group.front().srcIndex; // 3
  int lowBit  = group.back().srcIndex;  //0
  int width   = highBit - lowBit + 1;   // 4

  // type info 
  auto srcIntTy = mlir::cast<moore::IntType>(src.getType());
  auto domain   = srcIntTy.getDomain();  // TwoValued / FourValued domain

  // bit and vector types for extraction and concatenation
  auto bitType   = moore::IntType::get(builder.getContext(), /*width=*/1, domain);
  auto sliceType = moore::IntType::get(builder.getContext(), /*width=*/width, domain);

  // extract bits in ascending order 
  SmallVector<Value> bits;
  for (int i = lowBit; i <= highBit; ++i)
    bits.push_back(builder.create<moore::ExtractOp>(
        loc, bitType, src, builder.getI32IntegerAttr(i)));

  // concatenate bits into a reversed vector 
  Value reversedVec = builder.create<moore::ConcatOp>(loc, sliceType, bits);

  // create assign operation to destination 
  builder.create<moore::ContinuousAssignOp>(loc, dst, reversedVec);

  // erase old scalar assign ops and extracts
  for (auto &g : group) {
    g.assign.erase();
    g.extractRef.erase();
    g.extract.erase();
  }
}

struct SimpleVectorizationPass
    : public mlir::PassWrapper<SimpleVectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    auto module = getOperation();
    llvm::errs() << "Running SimpleVectorizationPass...\n";

    AssignTree assignTree;

    // Collects vectorizable assigns and organizes them in the tree
    module.walk([&](moore::ContinuousAssignOp assign) {
      auto lhs = assign.getDst(); // outA[0]
      auto rhs = assign.getSrc(); //inA[0]

      auto extractRef =
          dyn_cast_or_null<moore::ExtractRefOp>(lhs.getDefiningOp()); //ExtractRefOp
      auto extract = dyn_cast_or_null<moore::ExtractOp>(rhs.getDefiningOp()); //ExtractOp
      if (!extractRef || !extract) //verify if they exist
        return;

      auto dstAttr =
          extractRef->getAttrOfType<mlir::IntegerAttr>("lowBit"); 
      auto srcAttr = extract->getAttrOfType<mlir::IntegerAttr>("lowBit");
      if (!dstAttr || !srcAttr)
        return;

      int dstIndex = dstAttr.getInt();
      int srcIndex = srcAttr.getInt();

      assignTree[extractRef.getOperand()][extract.getOperand()][dstIndex] =
          {extractRef, extract, assign, dstIndex, srcIndex};

      llvm::errs() << "Found assign: dst[" << dstIndex
                   << "] = src[" << srcIndex << "]\n";
    });

    for (auto &[dst, srcMap] : assignTree) { //traverse the tree
      for (auto &[src, indexMap] : srcMap) { //traverse the tree
        std::vector<int> sortedDstIndices;
        for (const auto &[dstIndex, _] : indexMap) // extract the indexes
          sortedDstIndices.push_back(dstIndex); // add the indexes to sortedDstIndices vector
        std::sort(sortedDstIndices.begin(), sortedDstIndices.end()); // sort the indexes 

        std::vector<ScalarAssignGroup> group;  //temporary group
        for (size_t i = 0; i < sortedDstIndices.size(); ++i) {
          auto current = indexMap[sortedDstIndices[i]];

          if (!group.empty()) {
            auto prev = group.back();
            int dstStride = current.dstIndex - prev.dstIndex;
            int srcStride = current.srcIndex - prev.srcIndex;

            if (dstStride != 1 || (srcStride != 1 && srcStride != -1)) {
                if (group.size() > 1) {
                    if (isReverse(group)) {
                        llvm::errs() << ">> Detected REVERSE group (" << group.size()
                                    << " bits) between " << src << " -> " << dst << "\n";
                        vectorizeReverseGroup(group);
                    } else {
                        vectorizeGroup(group);
                    }
                }
              group.clear();
            }
          }
          group.push_back(current);
        }

        if (group.size() > 1) {
          if (isReverse(group)) {
            llvm::errs() << ">> Detected REVERSE group (" << group.size()
                         << " bits) between " << src << " -> " << dst << "\n";
            vectorizeReverseGroup(group);
          } else {
            vectorizeGroup(group);
          }
        }
      }
    }

    llvm::errs() << "SimpleVectorizationPass completed.\n";
  }

  StringRef getArgument() const override { return "simple-vec"; }

  StringRef getDescription() const override {
    return "Simple Vectorization Pass – passo 1: detecção de grupos reversos";
  }
};
} 

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
    return {
        MLIR_PLUGIN_API_VERSION,
        "SimpleVec",
        LLVM_VERSION_STRING,
        []() {
            PassPipelineRegistration<>(
                "simple-vec",
                "Simple Vectorization Pass",
                [](OpPassManager &pm) {
                    pm.addPass(std::make_unique<SimpleVectorizationPass>());
                });
        }
    };
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)