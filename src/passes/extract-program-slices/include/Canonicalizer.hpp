#ifndef CANONICALIZER_HPP
#define CANONICALIZER_HPP

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/HW/HWOps.h"

using ValueStack = llvm::SmallVector<mlir::Value>;

class Canonicalizer {

public:

   Canonicalizer(llvm::DenseSet<llvm::StringRef> targetOps);
   void canonicalize(mlir::ModuleOp topModule);

private:

   llvm::DenseSet<llvm::StringRef> targetOps;

   bool is_associative(mlir::Operation* op);
   bool is_commutative(mlir::Operation* op);

   void flatten(circt::hw::HWModuleOp module);
   void sort(circt::hw::HWModuleOp module);
   void reduce(circt::hw::HWModuleOp module);
   std::unique_ptr<ValueStack> get_top_ord(circt::hw::HWModuleOp module);
   
};

#endif

