//===- VectorizationPass.cpp - Vectorization Pass ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Vectorization pass.
// This pass identifies sequences of `moore.continuous_assign` operations that
// are effectively scalarized vector assignments (i.e., assigning individual
// bits from one vector to another). It merges these contiguous bit-wise
// assignments into a single, more efficient vector-level assignment.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Builders.h"

#include "mlir/Interfaces/FunctionInterfaces.h"

#include "mlir/IR/Operation.h"

#include "mlir/Pass/Pass.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "mlir/IR/Builders.h"

#include "mlir/Pass/PassManager.h"

#include "mlir/Pass/PassRegistry.h"

#include "mlir/IR/BuiltinOps.h"

#include "mlir/Support/IndentedOstream.h"

#include "mlir/Tools/Plugins/PassPlugin.h"

#include "llvm/Config/llvm-config.h"

#include "llvm/ADT/DenseMap.h"

#include "llvm/ADT/SmallVector.h"

#include "llvm/Support/raw_ostream.h"

#include "llvm/ADT/ArrayRef.h"

#include "llvm/ADT/SmallVector.h"

#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MoorePasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h" 
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/ArrayRef.h"   
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace circt;
using namespace moore;

namespace {

struct ScalarAssignGroup {
  moore::ExtractRefOp extractRef;
  moore::ExtractOp extract;
  moore::ContinuousAssignOp assign;
  int index;
};

using IndexedGroupMap = llvm::DenseMap<int, ScalarAssignGroup>;
using SourceGroupMap = llvm::DenseMap<mlir::Value, IndexedGroupMap>;
using AssignTree = llvm::DenseMap<mlir::Value, SourceGroupMap>;

void populateAssignTree(ModuleOp moduleOp, AssignTree &assignTree) {
  moduleOp.walk([&](moore::ContinuousAssignOp assign) {
    auto lhs = assign.getDst();
    auto rhs = assign.getSrc();

    auto extractRef = lhs.getDefiningOp<moore::ExtractRefOp>();
    auto extract = rhs.getDefiningOp<moore::ExtractOp>();
    if (!extractRef || !extract)
      return;

    auto indexRefAttr = extractRef.getLowBitAttr();
    auto indexAttr = extract.getLowBitAttr();
    if (!indexRefAttr || !indexAttr)
      return;

    int index = indexRefAttr.getInt();
    if (index != indexAttr.getInt())
      return;

    assignTree[extractRef.getOperand()][extract.getOperand()][index] =
        {extractRef, extract, assign, index};
  });
}

void vectorizeGroup(llvm::MutableArrayRef<ScalarAssignGroup> group,
                    mlir::Value dstVec, mlir::Value srcVec) {
  if (group.empty())
    return;

  for (auto &g : group) {

    if (!g.extract.getResult().hasOneUse()) {
      return;
    }

    if (!g.extractRef.getResult().hasOneUse()) {
      return;
    }
  }
  auto builder = OpBuilder(group.front().assign.getContext());
  builder.setInsertionPoint(group.front().assign);

  builder.create<moore::ContinuousAssignOp>(group.front().assign.getLoc(),
                                            dstVec, srcVec);

  for (auto &g : group) {
    g.assign.erase();
    g.extractRef.erase();
    g.extract.erase();
  }
}

void processIndexMap(IndexedGroupMap &indexMap, mlir::Value dst, mlir::Value src) {
  llvm::SmallVector<int, 32> sortedIndices;
  for (const auto &pair : indexMap) {
    sortedIndices.push_back(pair.getFirst());
  }
  std::sort(sortedIndices.begin(), sortedIndices.end());

  llvm::SmallVector<ScalarAssignGroup, 32> group;
  for (size_t i = 0; i < sortedIndices.size(); ++i) {
    if (!group.empty() && sortedIndices[i] != sortedIndices[i - 1] + 1) {
      if (group.size() > 1)j
        vectorizeGroup(group, dst, src);
      group.clear();
    }
    group.push_back(indexMap.at(sortedIndices[i]));
  }
  if (group.size() > 1) {
    vectorizeGroup(group, dst, src);
  }
}

struct SimpleVectorizationPass
    : public mlir::PassWrapper<SimpleVectorizationPass, mlir::OperationPass<mlir::ModuleOp>> {

  void runOnOperation() override;

  StringRef getArgument() const override { return "simple-vec"; }

  StringRef getDescription() const override {
    return "Simple Vectorization Pass using tree structure";
  }
};
} 

void SimpleVectorizationPass::runOnOperation(){
    ModuleOp moduleOp = getOperation();
    AssignTree assignTree;

    populateAssignTree(moduleOp, assignTree);

    llvm::SmallVector<mlir::Value> dstKeys;
    for (const auto &pair : assignTree) {
        dstKeys.push_back(pair.getFirst());
    }
    std::sort(dstKeys.begin(), dstKeys.end(), [](mlir::Value a, mlir::Value b) {
        return a.getDefiningOp()->isBeforeInBlock(b.getDefiningOp());
    });

    for (mlir::Value dst : dstKeys) {
        auto &srcMap = assignTree[dst];

        llvm::SmallVector<mlir::Value> srcKeys;
        for (const auto &pair : srcMap) {
        srcKeys.push_back(pair.getFirst());
        }
        std::sort(srcKeys.begin(), srcKeys.end(),
                [](mlir::Value a, mlir::Value b) {
                    return a.getDefiningOp()->isBeforeInBlock(b.getDefiningOp());
                });

        for (mlir::Value src : srcKeys) {
        processIndexMap(srcMap[src], dst, src);
        }
    }
}

extern "C" ::mlir::PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return {
      MLIR_PLUGIN_API_VERSION,
      "SimpleVec",
      LLVM_VERSION_STRING,
      []() {
        PassPipelineRegistration<>(
            "simple-vec", "Simple Vectorization Pass",
            [](OpPassManager &pm) {
              pm.addPass(std::make_unique<SimpleVectorizationPass>());
            });
      }};
}

MLIR_DECLARE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SimpleVectorizationPass)