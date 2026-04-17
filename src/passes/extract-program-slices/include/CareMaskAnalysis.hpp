#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

struct CareMaskValue{

   llvm::APInt mask;
   bool isUnitialized = true;
   CareMaskValue(llvm::APInt v);
   CareMaskValue();
   bool operator==(const CareMaskValue& rhs) const;
   void print(llvm::raw_ostream &os) const;

   static CareMaskValue join(const CareMaskValue &lhs, const CareMaskValue &rhs) {
      if (lhs.isUnitialized) return rhs;
      if (rhs.isUnitialized) return lhs;
      return CareMaskValue(lhs.mask | rhs.mask);
   }
};

class CareMaskLattice : public mlir::dataflow::Lattice<CareMaskValue>{

public:

   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CareMaskLattice);
   explicit CareMaskLattice(mlir::Value value) 
       : mlir::dataflow::Lattice<CareMaskValue>(value) {}

   mlir::ChangeResult meet(const CareMaskValue& rhs);
};

class CareMaskAnalysis : public mlir::dataflow::SparseBackwardDataFlowAnalysis<CareMaskLattice>{

public:

   explicit CareMaskAnalysis(mlir::DataFlowSolver &solver, mlir::SymbolTableCollection &symbolTable) 
       : mlir::dataflow::SparseBackwardDataFlowAnalysis<CareMaskLattice>(solver, symbolTable) {} 

   void setToExitState(CareMaskLattice *lattice) override;
   void visitBranchOperand(mlir::OpOperand &operand) override {}
   void visitCallOperand(mlir::OpOperand &operand) override {}

   void visitNonControlFlowArguments(mlir::RegionSuccessor &successor, 
                                     llvm::ArrayRef<mlir::BlockArgument> argLattices) override {}
   mlir::LogicalResult visitOperation(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results) override;
   mlir::LogicalResult visitExtOp(circt::comb::ExtractOp op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results);
   mlir::LogicalResult visitAddOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results);
   mlir::LogicalResult visitMuxOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results);
   mlir::LogicalResult visitAndOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results);
   mlir::LogicalResult visitConcatOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results);
   mlir::LogicalResult visitInst(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results);
};
