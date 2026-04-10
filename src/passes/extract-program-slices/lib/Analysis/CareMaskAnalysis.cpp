#include "../../include/CareMaskAnalysis.hpp"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"

CareMaskValue::CareMaskValue(llvm::APInt v) : mask{v} {};
bool CareMaskValue::operator==(const CareMaskValue& rhs) const{
   if(rhs.isUnitialized != this->isUnitialized) return false;
   if(rhs.mask == this->mask) return true;
   return this->mask == rhs.mask;
}

mlir::ChangeResult CareMaskLattice::meet(const CareMaskValue &rhs){

   auto& local_lattice_element = getValue(); 

   if(rhs.isUnitialized) return mlir::ChangeResult::NoChange;

   if(rhs == local_lattice_element) return mlir::ChangeResult::NoChange;

   if(local_lattice_element.isUnitialized){
      local_lattice_element.isUnitialized = false;
      local_lattice_element.mask = rhs.mask;
      return mlir::ChangeResult::Change;
   }

   assert(rhs.mask.getBitWidth() == local_lattice_element.mask.getBitWidth());

   auto old_mask = local_lattice_element.mask;
   local_lattice_element.mask = local_lattice_element.mask | rhs.mask;
   if(old_mask != local_lattice_element.mask) return mlir::ChangeResult::Change;
   return mlir::ChangeResult::NoChange;
} 

mlir::LogicalResult CareMaskAnalysis::initialize(mlir::Operation *top){

   top->walk([&](circt::hw::OutputOp outputOp) {
      for(mlir::Value operand : outputOp.getOperands()){

         auto bitwidth = operand.getType().getIntOrFloatBitWidth();
         llvm::APInt allOnes = llvm::APInt::getAllOnes(bitwidth);
         CareMaskValue topValue(allOnes);
         CareMaskLattice* lattice = getLatticeElement(operand);
         mlir::ChangeResult changed = lattice->meet(topValue);
         propagateIfChanged(lattice, changed);

      }
   });

   return mlir::dataflow::SparseBackwardDataFlowAnalysis<CareMaskLattice>::initialize(top);
}

mlir::LogicalResult CareMaskAnalysis::visitOperation(mlir::Operation* op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   if(results.empty()) return mlir::success();
   const CareMaskValue &mOutVal = results[0]->getValue();

   if(mOutVal.isUnitialized) return mlir::success();
   
   llvm::APInt mOut = mOutVal.mask;

   if (auto extractOp = mlir::dyn_cast<circt::comb::ExtractOp>(op)) {
        unsigned inputWidth = extractOp.getInput().getType().getIntOrFloatBitWidth();
        uint32_t lowBit = extractOp.getLowBit();
        
        llvm::APInt mIn = mOut.zext(inputWidth).shl(lowBit);
        
        mlir::ChangeResult changed = operands[0]->meet(CareMaskValue(mIn));
        propagateIfChanged(operands[0], changed);
        
        return mlir::success(); 
    }

   else if (auto concatOp = mlir::dyn_cast<circt::comb::ConcatOp>(op)) {
        unsigned currentOffset = 0;
        
        for (int i = operands.size() - 1; i >= 0; --i) {
            unsigned opWidth = concatOp.getOperand(i).getType().getIntOrFloatBitWidth();
            llvm::APInt partMask = mOut.extractBits(opWidth, currentOffset);
            currentOffset += opWidth;
            
            mlir::ChangeResult changed = operands[i]->meet(CareMaskValue(partMask));
            propagateIfChanged(operands[i], changed);
        }
        return mlir::success();
    }

   else if (auto muxOp = mlir::dyn_cast<circt::comb::MuxOp>(op)) {
        llvm::APInt selTop = llvm::APInt::getAllOnes(1);
        propagateIfChanged(operands[0], operands[0]->meet(CareMaskValue(selTop)));
        propagateIfChanged(operands[1], operands[1]->meet(CareMaskValue(mOut)));
        propagateIfChanged(operands[2], operands[2]->meet(CareMaskValue(mOut)));
        return mlir::success();
    }

   else if (auto addOp = mlir::dyn_cast<circt::comb::AddOp>(op)) {
        unsigned activeBits = mOut.getActiveBits();
        llvm::APInt addMask = llvm::APInt::getLowBitsSet(mOut.getBitWidth(), activeBits);
        
        for (size_t i = 0; i < operands.size(); ++i) {
            propagateIfChanged(operands[i], operands[i]->meet(CareMaskValue(addMask)));
        }
        return mlir::success();
    }

   else if (auto andOp = mlir::dyn_cast<circt::comb::AndOp>(op)) {
        llvm::APInt constMask = llvm::APInt::getAllOnes(mOut.getBitWidth());
        
        for (mlir::Value val : andOp.getOperands()) {
            if (auto constantOp = val.getDefiningOp<circt::hw::ConstantOp>()) {
                constMask &= constantOp.getValue();
            }
        }
        
        llvm::APInt propagatedMask = mOut & constMask;
        
        for (size_t i = 0; i < operands.size(); ++i) {
            if (!andOp.getOperand(i).getDefiningOp<circt::hw::ConstantOp>()) {
                propagateIfChanged(operands[i], operands[i]->meet(CareMaskValue(propagatedMask)));
            }
        }
        return mlir::success();
   }

   else if (auto instanceOp = mlir::dyn_cast<circt::hw::InstanceOp>(op)) {
        
        for (size_t i = 0; i < operands.size(); ++i) {
            
            unsigned bitWidth = instanceOp.getOperand(i).getType().getIntOrFloatBitWidth();
            llvm::APInt topMask = llvm::APInt::getAllOnes(bitWidth);
            
            mlir::ChangeResult changed = operands[i]->meet(CareMaskValue(topMask));
            propagateIfChanged(operands[i], changed);
        }
        
        return mlir::success();
    }

    return mlir::success();
}


