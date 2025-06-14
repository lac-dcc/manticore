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

static bool isReverse(const std::vector<ScalarAssignGroup> &grp) {
  if (grp.size() < 2)
    return false;
  int dstStride = grp[1].dstIndex - grp[0].dstIndex; // expected  +1
  int srcStride = grp[1].srcIndex - grp[0].srcIndex; // expected  -1
  return dstStride == 1 && srcStride == -1;
}

bool isBitMixGroup(const std::vector<ScalarAssignGroup> &group) {
  int n = group.size();
  if (n < 2)
    return false;

  bool isLinear = true;
  for (int i = 0; i < n; ++i) {
    if (group[i].srcIndex != group[i].dstIndex) {
      isLinear = false;
      break;
    }
  }
  if (isLinear)
    return false;

  if (isReverse(group))
    return false;

  std::set<int> srcIndices, dstIndices;
  for (const auto &g : group) {
    srcIndices.insert(g.srcIndex);
    dstIndices.insert(g.dstIndex);
  }

  if (srcIndices.size() != n || dstIndices.size() != n)
    return false;

  int srcMin = *srcIndices.begin();
  int dstMin = *dstIndices.begin();

  for (int i = 0; i < n; ++i) {
    if (!srcIndices.count(srcMin + i) || !dstIndices.count(dstMin + i))
      return false;
  }

  return true;
}

static void vectorizeGroup(std::vector<ScalarAssignGroup> &group){
    if (group.empty()) return;

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

static void vectorizeMixGroup(std::vector<ScalarAssignGroup> &group) {
  if (group.empty())
    return;

  llvm::sort(group, [](const ScalarAssignGroup &a, const ScalarAssignGroup &b) {
    return a.dstIndex < b.dstIndex;
  });

  OpBuilder builder(group.front().assign.getContext());
  builder.setInsertionPoint(group.front().assign);

  Location loc = group.front().assign.getLoc();
  Value dst    = group.front().extractRef.getOperand(); // !moore.ref<lN>
  Value src    = group.front().extract.getOperand();    // !moore.lN

  int width = group.size();

  auto srcIntTy = mlir::cast<moore::IntType>(src.getType());
  auto domain   = srcIntTy.getDomain();

  auto bitType   = moore::IntType::get(builder.getContext(), /*width=*/1, domain);
  auto sliceType = moore::IntType::get(builder.getContext(), /*width=*/width, domain);

  SmallVector<Value> bits;
  for (auto &g : group) {
    auto bit = builder.create<moore::ExtractOp>(
        loc, bitType, src, builder.getI32IntegerAttr(g.srcIndex));
    bits.push_back(bit);
  }

  std::reverse(bits.begin(), bits.end());

  Value vec = builder.create<moore::ConcatOp>(loc, sliceType, bits);

  builder.create<moore::ContinuousAssignOp>(loc, dst, vec);

  for (auto &g : group) {
    g.assign.erase();
    g.extractRef.erase();
    g.extract.erase();
  }
}

void collectAssigns(mlir::ModuleOp module, AssignTree &assignTree) {
  module.walk([&](moore::ContinuousAssignOp assign) {
    auto lhs = assign.getDst();
    auto rhs = assign.getSrc();

    auto extractRef = dyn_cast_or_null<moore::ExtractRefOp>(lhs.getDefiningOp());
    auto extract = dyn_cast_or_null<moore::ExtractOp>(rhs.getDefiningOp());
    if (!extractRef || !extract)
      return;

    auto dstAttr = extractRef->getAttrOfType<mlir::IntegerAttr>("lowBit");
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
}

void processAssignTree(AssignTree &assignTree) {
  for (auto &[dst, srcMap] : assignTree) {
    for (auto &[src, indexMap] : srcMap) {

      std::vector<int> sortedDstIndices;
      for (const auto &[dstIndex, _] : indexMap)
        sortedDstIndices.push_back(dstIndex);
      std::sort(sortedDstIndices.begin(), sortedDstIndices.end());

      std::vector<ScalarAssignGroup> group;
      for (int dstIndex : sortedDstIndices) {
        group.push_back(indexMap[dstIndex]);
      }

      if (group.size() > 1) {
        if (isReverse(group)) {
          llvm::errs() << ">> Detected REVERSE group (" << group.size()
                       << " bits) between " << src << " -> " << dst << "\n";
          vectorizeReverseGroup(group);
        } else if (isBitMixGroup(group)) {
          llvm::errs() << ">> Detected MIX group (" << group.size()
                       << " bits) between " << src << " -> " << dst << "\n";
          vectorizeMixGroup(group);
        } else {
          llvm::errs() << ">> Detected LINEAR group (" << group.size()
                       << " bits) between " << src << " -> " << dst << "\n";
          vectorizeGroup(group);
        }
      }
    }
  }
}

struct SimpleVectorizationPass
    : public mlir::PassWrapper<SimpleVectorizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
  auto module = getOperation();
  llvm::errs() << "Running SimpleVectorizationPass...\n";

  AssignTree assignTree;

  collectAssigns(module, assignTree);
  processAssignTree(assignTree);

  llvm::errs() << "SimpleVectorizationPass completed.\n";
}

  StringRef getArgument() const override { return "simple-vec"; }

  StringRef getDescription() const override {
    return "Simple Vectorization Pass";
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