#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

struct CareMaskValue{

   llvm::APInt mask;
   bool isUnitialized = true;
   CareMaskValue(llvm::APInt v);
   bool operator==(const CareMaskValue& rhs) const;
};

class CareMaskLattice : public mlir::dataflow::Lattice<CareMaskValue>{

public:
   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CareMaskLattice);
   mlir::ChangeResult meet(const CareMaskValue &rhs);
};

class CareMaskAnalysis : public mlir::dataflow::SparseBackwardDataFlowAnalysis<CareMaskLattice>{

   mlir::LogicalResult initialize(mlir::Operation *top) override;

   mlir::LogicalResult visitOperation(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results) override;

};
