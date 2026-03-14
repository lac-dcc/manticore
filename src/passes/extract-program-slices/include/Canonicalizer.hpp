#ifndef CANONICALIZER_HPP
#define CANONICALIZER_HPP

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h" 
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Transforms/RegionUtils.h" 

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"

using ValueStack = llvm::SmallVector<mlir::Value>;

class Canonicalizer {

public:

   Canonicalizer(llvm::DenseSet<llvm::StringRef> targetOps);
   void canonicalize(circt::hw::HWModuleOp* module);

private:

   llvm::DenseSet<llvm::StringRef> targetOps;

   bool is_associative(mlir::Operation* op);
   bool is_commutative(mlir::Operation* op);

   void flatten(circt::hw::HWModuleOp* module);
   void sort(circt::hw::HWModuleOp* module);
   void reduce(circt::hw::HWModuleOp* module);
   std::unique_ptr<ValueStack> get_topological_ordering(circt::hw::HWModuleOp* module, llvm::DenseSet<mlir::Operation*> invalid_operations);
   
};

#endif

