#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
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
      auto old_val = lval1.mask;
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


mlir::LogicalResult CareMaskAnalysis::initialize(mlir::Operation *top) {


    top->walk([&](mlir::Block *block) {
        auto *exec = this->getOrCreate<mlir::dataflow::Executable>(getProgramPointBefore(block));
        propagateIfChanged(exec, exec->setToLive());
    });

    if (failed(mlir::dataflow::SparseBackwardDataFlowAnalysis<CareMaskLattice>::initialize(top))) {
        return mlir::failure();
    }
        top->walk([&](circt::hw::OutputOp op) {
            for (mlir::Value operand : op.getOperands()) {
               auto *lattice = getLatticeElement(operand);

               setToExitState(lattice);
            }
        });

   return mlir::success();
}


mlir::LogicalResult CareMaskAnalysis::visitOperation(mlir::Operation* op,
                                                     llvm::ArrayRef<CareMaskLattice *> operands, 
                                                     llvm::ArrayRef<const CareMaskLattice *> results){

   if(results.empty() || results[0]->getValue().isUnitialized) return mlir::success();
   if(auto extOp = llvm::dyn_cast<circt::comb::ExtractOp>(op)) return visitExtOp(extOp, operands, results);
   else if(auto addOp = llvm::dyn_cast<circt::comb::AddOp>(op)) return visitAddOp(addOp, operands, results);
   else if(auto muxOp = llvm::dyn_cast<circt::comb::MuxOp>(op)) return visitMuxOp(muxOp, operands, results);
   else if(auto conOp = llvm::dyn_cast<circt::comb::ConcatOp>(op)) return visitConcatOp(conOp, operands, results);
   else if(auto andOp = llvm::dyn_cast<circt::comb::AndOp>(op)) return visitAndOp(andOp, operands, results);
   else if(auto instanceOp = llvm::dyn_cast<circt::hw::InstanceOp>(op)) return visitInst(instanceOp, operands, results);
   else if(auto divUOp = llvm::dyn_cast<circt::comb::DivUOp>(op)) return visitInst(divUOp, operands, results);
   else if(auto divSOp = llvm::dyn_cast<circt::comb::DivSOp>(op)) return visitInst(divSOp, operands, results);
   else if(auto iCmpOp = llvm::dyn_cast<circt::comb::ICmpOp>(op)) return visitInst(iCmpOp, operands, results);
   else if(auto modSOp = llvm::dyn_cast<circt::comb::ModSOp>(op)) return visitInst(modSOp, operands, results);
   else if(auto modUOp = llvm::dyn_cast<circt::comb::ModUOp>(op)) return visitInst(modUOp, operands, results);
   else if(auto orOp = llvm::dyn_cast<circt::comb::OrOp>(op)) return visitInst(orOp, operands, results);
   else if(auto parityOp = llvm::dyn_cast<circt::comb::ParityOp>(op)) return visitInst(parityOp, operands, results);
   else if(auto replicateOp = llvm::dyn_cast<circt::comb::ReplicateOp>(op)) return visitInst(replicateOp, operands, results);
   else if(auto reverseOp = llvm::dyn_cast<circt::comb::ReverseOp>(op)) return visitInst(reverseOp, operands, results);
   else if(auto shlOp = llvm::dyn_cast<circt::comb::ShlOp>(op)) return visitInst(shlOp, operands, results);
   else if(auto shrsOp = llvm::dyn_cast<circt::comb::ShrSOp>(op)) return visitInst(shrsOp, operands, results);
   else if(auto shruOp = llvm::dyn_cast<circt::comb::ShrUOp>(op)) return visitInst(shruOp, operands, results);
   else if(auto subOp = llvm::dyn_cast<circt::comb::SubOp>(op)) return visitInst(subOp, operands, results);
   else if(auto truthTableOp = llvm::dyn_cast<circt::comb::TruthTableOp>(op)) return visitInst(truthTableOp, operands, results);
   else if(auto mulOp = llvm::dyn_cast<circt::comb::MulOp>(op)) return visitInst(mulOp, operands, results);
   else if(auto xorOp = llvm::dyn_cast<circt::comb::XorOp>(op)) return visitXor(xorOp, operands, results);
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



mlir::LogicalResult CareMaskAnalysis::visitMul(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){
    
   auto mOut = results[0]->getValue().mask;
   unsigned activeBits = mOut.getActiveBits();
    
   llvm::APInt mulMask = llvm::APInt::getLowBitsSet(mOut.getBitWidth(), activeBits);
    
   for (auto& operand : operands){
      propagateIfChanged(operand, operand->meet(CareMaskValue(mulMask)));
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitSub(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){
    
   auto mOut = results[0]->getValue().mask;
   unsigned activeBits = mOut.getActiveBits();
    
   llvm::APInt subMask = llvm::APInt::getLowBitsSet(mOut.getBitWidth(), activeBits);
    
   for (auto& operand : operands){
      propagateIfChanged(operand, operand->meet(CareMaskValue(subMask)));
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


mlir::LogicalResult CareMaskAnalysis::visitIcmp(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto resultLattice = results[0]->getValue();
   auto mOut = resultLattice.mask; 
   unsigned operandBitWidth = op->getOperand(0).getType().getIntOrFloatBitWidth();

   if(mOut.isZero()){
      CareMaskValue nullValue = CareMaskValue(llvm::APInt::getZero(operandBitWidth));
      for(auto operand : operands){
         propagateIfChanged(operand, operand->meet(nullValue));
      }
   }
   else{
      CareMaskValue trueValue = CareMaskValue(llvm::APInt::getAllOnes(operandBitWidth));
      for(auto operand : operands){
         propagateIfChanged(operand, operand->meet(trueValue));
      }
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitTruthTable(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto resultLattice = results[0]->getValue();
   auto mOut = resultLattice.mask; 
   unsigned operandBitWidth = op->getOperand(0).getType().getIntOrFloatBitWidth();

   if(mOut.isZero()){
      CareMaskValue nullValue = CareMaskValue(llvm::APInt::getZero(operandBitWidth));
      for(auto operand : operands){
         propagateIfChanged(operand, operand->meet(nullValue));
      }
   }
   else{
      CareMaskValue trueValue = CareMaskValue(llvm::APInt::getAllOnes(operandBitWidth));
      for(auto operand : operands){
         propagateIfChanged(operand, operand->meet(trueValue));
      }
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitParity(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto resultLattice = results[0]->getValue();
   auto mOut = resultLattice.mask; 
   unsigned operandBitWidth = op->getOperand(0).getType().getIntOrFloatBitWidth();

   if(mOut.isZero()){
      CareMaskValue nullValue = CareMaskValue(llvm::APInt::getZero(operandBitWidth));
      for(auto operand : operands){
         propagateIfChanged(operand, operand->meet(nullValue));
      }
   }
   else{
      CareMaskValue trueValue = CareMaskValue(llvm::APInt::getAllOnes(operandBitWidth));
      for(auto operand : operands){
         propagateIfChanged(operand, operand->meet(trueValue));
      }
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitShrs(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   llvm::APInt mOut = results[0]->getValue().mask;
   auto amount = op->getOperand(1);

   //Talvez isso aqui dependa se 'amount' é uma constante ou não?  
   CareMaskValue amountMask = CareMaskValue(llvm::APInt::getAllOnes(amount.getType().getIntOrFloatBitWidth()));
   propagateIfChanged(operands[1], operands[1]->meet(amountMask));

   if(auto constant = llvm::dyn_cast<circt::hw::ConstantOp>(amount)){
      llvm::APInt cval = constant.getValue();
      auto newMOut = mOut << cval;
      newMOut.setSignBit();
      auto shiftMask = CareMaskValue(newMOut);
      propagateIfChanged(operands[0], operands[0]->meet(shiftMask));
   }
   else{
      auto bitWidthLogicalOperand = op->getOperand(0).getType().getIntOrFloatBitWidth();
      CareMaskValue operandMask = CareMaskValue(llvm::APInt::getAllOnes(bitWidthLogicalOperand));
      propagateIfChanged(operands[0], operands[0]->meet(operandMask));
   }
   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitShru(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   llvm::APInt mOut = results[0]->getValue().mask;
   auto amount = op->getOperand(1);

   //Talvez isso aqui dependa se 'amount' é uma constante ou não?  
   CareMaskValue amountMask = CareMaskValue(llvm::APInt::getAllOnes(amount.getType().getIntOrFloatBitWidth()));
   propagateIfChanged(operands[1], operands[1]->meet(amountMask));

   if(auto constant = llvm::dyn_cast<circt::hw::ConstantOp>(amount)){
      llvm::APInt cval = constant.getValue();
      auto newMOut = mOut << cval;
      auto shiftMask = CareMaskValue(newMOut);
      propagateIfChanged(operands[0], operands[0]->meet(shiftMask));
   }
   else{
      auto bitWidthLogicalOperand = op->getOperand(0).getType().getIntOrFloatBitWidth();
      CareMaskValue operandMask = CareMaskValue(llvm::APInt::getAllOnes(bitWidthLogicalOperand));
      propagateIfChanged(operands[0], operands[0]->meet(operandMask));
   }
   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitShl(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   llvm::APInt mOut = results[0]->getValue().mask;
   auto amount = op->getOperand(1);

   //Talvez isso aqui dependa se 'amount' é uma constante ou não?  
   CareMaskValue amountMask = CareMaskValue(llvm::APInt::getAllOnes(amount.getType().getIntOrFloatBitWidth()));
   propagateIfChanged(operands[1], operands[1]->meet(amountMask));

   if(auto constant = llvm::dyn_cast<circt::hw::ConstantOp>(amount)){
      llvm::APInt cval = constant.getValue();
      auto newMOut = mOut.lshr(cval);
      auto shiftMask = CareMaskValue(mOut);
      propagateIfChanged(operands[0], operands[0]->meet(shiftMask));
   }
   else{
      auto bitWidthLogicalOperand = op->getOperand(0).getType().getIntOrFloatBitWidth();
      CareMaskValue operandMask = CareMaskValue(llvm::APInt::getAllOnes(bitWidthLogicalOperand));
      propagateIfChanged(operands[0], operands[0]->meet(operandMask));
   }
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



mlir::LogicalResult CareMaskAnalysis::visitOr(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto mOutValue = results[0]->getValue();
   for(auto i = 0 ; i < operands.size() ; ++i){
      if(auto constOp = op->getOperand(i).getDefiningOp<circt::hw::ConstantOp>()) continue;
      else{propagateIfChanged(operands[i], operands[i]->meet(mOutValue));}
   }
   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitXor(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   auto mOutValue = results[0]->getValue();
   for(auto i = 0 ; i < operands.size() ; ++i){
      if(auto constOp = op->getOperand(i).getDefiningOp<circt::hw::ConstantOp>()) continue;
      else{propagateIfChanged(operands[i], operands[i]->meet(mOutValue));}
   }
   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitReverse(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   llvm::APInt mOut = results[0]->getValue().mask;
   CareMaskValue revMout = CareMaskValue(mOut.reverseBits()); 
   for(auto operand : operands){
      propagateIfChanged(operand, operand->meet(revMout));
   }
   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitReplicate(mlir::Operation *op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results){

   llvm::APInt mOut = results[0]->getValue().mask;
   unsigned operandBitWidth = op->getOperands()[0].getType().getIntOrFloatBitWidth();
   CareMaskValue repMout = CareMaskValue(mOut.extractBits(operandBitWidth, 0));
   for(auto operand : operands){
      propagateIfChanged(operand, operand->meet(repMout));
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
    circt::hw::HWModuleOp module = mlir::SymbolTable::lookupNearestSymbolFrom<circt::hw::HWModuleOp>(
        op, instOp.getModuleNameAttr());

    if (!module) return mlir::success(); 

    auto outputOp = llvm::cast<circt::hw::OutputOp>(module.getBodyBlock()->getTerminator());
    for (size_t i = 0; i < results.size(); ++i) {
        auto *internalSignalLattice = getLatticeElement(outputOp.getOperand(i));

        propagateIfChanged(internalSignalLattice, internalSignalLattice->meet(results[i]->getValue()));
    }

    auto *body = module.getBodyBlock();
    for (size_t i = 0; i < operands.size(); ++i) {
        auto *argLattice = getLatticeElement(body->getArgument(i));

        addDependency(argLattice, getProgramPointAfter(op));
        propagateIfChanged(operands[i], operands[i]->meet(argLattice->getValue()));
    }

    return mlir::success();
}

mlir::LogicalResult CareMaskAnalysis::visitModU(mlir::Operation* op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results) {

   for(size_t i = 0 ; i < operands.size() ; i++){
      auto currOpBitWidth = op->getOperand(i).getType().getIntOrFloatBitWidth();
      auto completeMask = CareMaskValue(llvm::APInt::getAllOnes(currOpBitWidth));
      propagateIfChanged(operands[i], operands[i]->meet(completeMask));
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitModS(mlir::Operation* op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results) {

   for(size_t i = 0 ; i < operands.size() ; i++){
      auto currOpBitWidth = op->getOperand(i).getType().getIntOrFloatBitWidth();
      auto completeMask = CareMaskValue(llvm::APInt::getAllOnes(currOpBitWidth));
      propagateIfChanged(operands[i], operands[i]->meet(completeMask));
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitDivS(mlir::Operation* op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results) {

   for(size_t i = 0 ; i < operands.size() ; i++){
      auto currOpBitWidth = op->getOperand(i).getType().getIntOrFloatBitWidth();
      auto completeMask = CareMaskValue(llvm::APInt::getAllOnes(currOpBitWidth));
      propagateIfChanged(operands[i], operands[i]->meet(completeMask));
   }

   return mlir::success();
}



mlir::LogicalResult CareMaskAnalysis::visitDivU(mlir::Operation* op, llvm::ArrayRef<CareMaskLattice *> operands, llvm::ArrayRef<const CareMaskLattice *> results) {

   for(size_t i = 0 ; i < operands.size() ; i++){
      auto currOpBitWidth = op->getOperand(i).getType().getIntOrFloatBitWidth();
      auto completeMask = CareMaskValue(llvm::APInt::getAllOnes(currOpBitWidth));
      propagateIfChanged(operands[i], operands[i]->meet(completeMask));
   }

   return mlir::success();
}
