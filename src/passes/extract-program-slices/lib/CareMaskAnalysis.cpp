#include "../include/CareMaskAnalysis.hpp"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Casting.h"

CareMaskValue::CareMaskValue(llvm::APInt v) : mask{v} {
   this->isUnitialized = false;
};

CareMaskValue::CareMaskValue() : isUnitialized(true) {}; 

void CareMaskValue::print(llvm::raw_ostream &os) const {
        if (isUnitialized) os << "uninitialized";
        else os << "mask(" << mask << ")";
    }

bool CareMaskValue::operator==(const CareMaskValue& rhs) const{
   if(rhs.isUnitialized != this->isUnitialized) return false;
   if(this->isUnitialized) return true;
   return this->mask == rhs.mask;
}


mlir::ChangeResult CareMaskLattice::meet(const CareMaskValue &rhs){

   auto& lval1 = this->getValue();
   auto& lval2 = rhs;

   if(lval1 == lval2) return mlir::ChangeResult::NoChange; 

   else if(lval1.isUnitialized) {
      lval1.isUnitialized = false;
      lval1.mask = lval2.mask;
      return mlir::ChangeResult::Change;
   }

   else if(lval2.isUnitialized) {
      return mlir::ChangeResult::NoChange;
   }

   else{
      auto& old_val = lval1.mask;
      lval1.mask = lval1.mask | lval2.mask;
      return old_val == lval1.mask ? mlir::ChangeResult::NoChange : mlir::ChangeResult::Change;
   }
} 


void CareMaskAnalysis::setToExitState(CareMaskLattice *lattice){

   mlir::Value value = lattice->getAnchor();
   if(!value.getType().isIntOrFloat()) return;

   auto block = value.getParentBlock();
   auto parentOp = block->getParentOp();

   if(auto moduleOp = llvm::dyn_cast<circt::hw::HWModuleOp>(parentOp)){

      if(moduleOp.isPrivate()) return;

      llvm::APInt mask = llvm::APInt::getAllOnes(value.getType().getIntOrFloatBitWidth());
      auto meetValue = CareMaskValue(mask);
      propagateIfChanged(lattice,lattice->meet(meetValue));
   }
}

mlir::LogicalResult CareMaskAnalysis::visitOperation(mlir::Operation* op,
                                                     llvm::ArrayRef<CareMaskLattice *> operands, 
                                                     llvm::ArrayRef<const CareMaskLattice *> results){

   if(results.empty() || results[0]->getValue().isUnitialized) return mlir::success();
   if(auto extOp = llvm::dyn_cast<circt::comb::ExtractOp>(op)) return visitExtOp(extOp, operands, results);
   if(auto addOp = llvm::dyn_cast<circt::comb::AddOp>(op)) return visitAddOp(addOp, operands, results);
   if(auto muxOp = llvm::dyn_cast<circt::comb::MuxOp>(op)) return visitMuxOp(muxOp, operands, results);
   if(auto conOp = llvm::dyn_cast<circt::comb::ConcatOp>(op)) return visitConcatOp(conOp, operands, results);
   if(auto andOp = llvm::dyn_cast<circt::comb::AndOp>(op)) return visitAndOp(andOp, operands, results);
   if(auto instanceOp = llvm::dyn_cast<circt::hw::InstanceOp>(op)) return visitInst(instanceOp, operands, results);
   return mlir::success();
  
}


mlir::LogicalResult CareMaskAnalysis::visitExtOp(circt::comb::ExtractOp op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

    auto mOut = results[0]->getValue().mask;

    auto inputWidth = op.getInput().getType().getIntOrFloatBitWidth();
    auto lowBit = op.getLowBit();

    auto newMask = mOut.zext(inputWidth).shl(lowBit);

    auto changed = operands[0]->meet(CareMaskValue(newMask));
    propagateIfChanged(operands[0], changed);

    return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitAddOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){
    
   auto mOut = results[0]->getValue().mask;
   unsigned activeBits = mOut.getActiveBits();
    
   llvm::APInt addMask = llvm::APInt::getLowBitsSet(mOut.getBitWidth(), activeBits);
    
   for (auto& operand : operands){
      propagateIfChanged(operand, operand->meet(CareMaskValue(addMask)));
   }

   return mlir::success();
}


mlir::LogicalResult CareMaskAnalysis::visitMuxOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto resultLattice = results[0]->getValue();
   auto mOut = resultLattice.mask; 

   auto muxVal = CareMaskValue(llvm::APInt::getAllOnes(op->getOperand(0).getType().getIntOrFloatBitWidth()));

   propagateIfChanged(operands[0], operands[0]->meet(muxVal));
   propagateIfChanged(operands[1], operands[1]->meet(resultLattice));
   propagateIfChanged(operands[2], operands[2]->meet(resultLattice));

   return mlir::success();
}


mlir::LogicalResult CareMaskAnalysis::visitAndOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto mOut = results[0]->getValue().mask;
   auto constMask = llvm::APInt::getAllOnes(mOut.getBitWidth());

   for(auto operand : op->getOperands()){
      if(auto constOp = operand.getDefiningOp<circt::hw::ConstantOp>()){
         auto cval = constOp.getValue().zextOrTrunc(mOut.getBitWidth());
         constMask &= cval; 
      }
   }

   CareMaskValue mOutAndConst = CareMaskValue(constMask & mOut);

   for(auto i = 0 ; i < operands.size() ; ++i){
      if(auto constOp = op->getOperand(i).getDefiningOp<circt::hw::ConstantOp>()) continue;
      else{propagateIfChanged(operands[i], operands[i]->meet(mOutAndConst));}
   }
   return mlir::success();
}


mlir::LogicalResult CareMaskAnalysis::visitConcatOp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto mOut = results[0]->getValue().mask;

   uint64_t bot_bit = 0;
   size_t index = op->getNumOperands() - 1;

   for(auto arg : llvm::reverse(op->getOperands())){

      auto opWidth = arg.getType().getIntOrFloatBitWidth();

      auto meetVal = CareMaskValue(mOut.extractBits(opWidth, bot_bit));
      propagateIfChanged(operands[index], operands[index]->meet(meetVal));

      bot_bit += opWidth;
      index--;
   }
   return mlir::success();
}


mlir::LogicalResult CareMaskAnalysis::visitInst(mlir::Operation* op, 
                                                
    llvm::ArrayRef<CareMaskLattice *> operands, 
    llvm::ArrayRef<const CareMaskLattice *> results) {
    
    auto instOp = llvm::cast<circt::hw::InstanceOp>(op);
    auto module = mlir::SymbolTable::lookupNearestSymbolFrom<circt::hw::HWModuleOp>(
        op, instOp.getModuleNameAttr());

    if (!module) return mlir::success(); 

    auto outputOp = cast<circt::hw::OutputOp>(module.getBodyBlock()->getTerminator());
    for (size_t i = 0; i < results.size(); ++i) {
        auto *internalSignalLattice = getLatticeElement(outputOp.getOperand(i));
        propagateIfChanged(internalSignalLattice, internalSignalLattice->meet(results[i]->getValue()));
    }

    auto *body = module.getBodyBlock();
    for (size_t i = 0; i < operands.size(); ++i) {
        auto *argLattice = getLatticeElement(body->getArgument(i));
        propagateIfChanged(operands[i], operands[i]->meet(argLattice->getValue()));
    }

    return mlir::success();
}
