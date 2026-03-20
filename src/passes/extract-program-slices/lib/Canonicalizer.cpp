#include "../include/Canonicalizer.hpp"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/APInt.h"
using namespace circt;
using namespace mlir;

Canonicalizer::Canonicalizer(llvm::DenseSet<llvm::StringRef> targetOps){
   this->targetOps = targetOps;
}

void Canonicalizer::canonicalize(mlir::ModuleOp top_module){

   for (auto module : top_module.getOps<hw::HWModuleOp>()) {
      if(module.getBody().empty()) continue;
      flatten(module);
      //this->sort(&module);
      //this->reduce(&module);
   }
} 

bool Canonicalizer::is_associative(mlir::Operation* op) {

   if(!op) return false;
   llvm::StringRef name = op->getName().getStringRef();
   return targetOps.count(name);

}

bool Canonicalizer::is_commutative(mlir::Operation* op) {

   if(!op) return false;
   return op->hasTrait<mlir::OpTrait::IsCommutative>();

}

void Canonicalizer::reduce(circt::hw::HWModuleOp module){
   
   auto block = module.getBodyBlock();
   mlir::IRRewriter rewriter(module->getContext());

   for(auto& operation : llvm::make_early_inc_range(block->getOperations())){

      llvm::SmallVector<Value, 4> new_operands; 
      auto op_name = operation.getName().getStringRef();

      llvm::SmallVector<mlir::Value, 4> operands(operation.getOperands().begin(), operation.getOperands().end());

      bool has_changed = false;

      if(operands.empty()) continue;

      if(op_name == "comb.add"){
         for(auto operand : operands){
            auto option_const = operand.getDefiningOp<circt::hw::ConstantOp>();
            if(!(option_const && option_const.getValue().isZero())){
               new_operands.push_back(operand);
            }
         }
      }
      else if(op_name == "comb.and"){
         for(auto operand : operands){
            auto option_const = operand.getDefiningOp<circt::hw::ConstantOp>();
            if(option_const && option_const.getValue().isZero()){
               rewriter.replaceOp(&operation, operand);
               new_operands.clear();
               has_changed = true;
               break;
            }
            if(!(option_const && option_const.getValue().isAllOnes())){
               if(!llvm::is_contained(new_operands, operand)) new_operands.push_back(operand);
            }
         }
      }
      else if(op_name == "comb.mul"){
         for(auto operand : operands){
            auto option_const = operand.getDefiningOp<circt::hw::ConstantOp>();
            if(option_const && option_const.getValue().isZero()){
               rewriter.replaceOp(&operation, operand);
               has_changed = true;
               new_operands.clear(); 
               break;
            }
            if(!(option_const && option_const.getValue().isOne())){
               new_operands.push_back(operand);
            }
         }
      }
      else if(op_name == "comb.or"){
         for(auto operand : operands){
            auto option_const = operand.getDefiningOp<circt::hw::ConstantOp>();
            if(option_const && option_const.getValue().isAllOnes()){
               rewriter.replaceOp(&operation, operand);
               has_changed = true;
               new_operands.clear();
               break;
            }
            if(!(option_const && option_const.getValue().isZero())){
               if(!llvm::is_contained(new_operands, operand)) new_operands.push_back(operand);
            }
         }
      }
      else{
         has_changed = true;
      }

      if(has_changed) continue;


      if (new_operands.empty()) rewriter.replaceOp(&operation, operands[0]);
      else if(new_operands.size() == 1) rewriter.replaceOp(&operation, new_operands[0]);
      else if(new_operands.size() < operands.size()){

         rewriter.modifyOpInPlace(&operation, [&](){
            operation.setOperands(new_operands);
         });
      }
   }
}

std::unique_ptr<ValueStack> Canonicalizer::get_top_ord(circt::hw::HWModuleOp module){

   auto block = module.getBodyBlock();
   if(!block) return nullptr; 
   
   ValueStack stack;
   stack.reserve(16);

   mlir::Operation* terminator = block->getTerminator();
   if(!terminator) return nullptr;


   size_t total_values = block->getNumArguments();
   for(mlir::Operation &op : *block){
      total_values += op.getNumResults();
   }

   llvm::DenseSet<mlir::Value> visited(total_values);
   llvm::DenseSet<mlir::Value> in_stack;
   std::unique_ptr<ValueStack> top_ordering = std::make_unique<ValueStack>();

   if(auto outputOp = llvm::dyn_cast<circt::hw::OutputOp>(terminator)){
      for(mlir::Value v : outputOp.getOperands()) {
         stack.push_back(v);
      }
   }

   while(!stack.empty()){

      auto current = stack.back();


      if(!current) {
         stack.pop_back();
         continue;
      }

      if(visited.count(current)){
         stack.pop_back();
         continue;
      }

      bool has_unvisited_operands = false;

      if(auto defOp = current.getDefiningOp()){
         if(defOp){
            for(auto operand : defOp->getOperands()){
               if(!visited.count(operand)) {
                  stack.push_back(operand);
                  in_stack.insert(operand);
                  has_unvisited_operands = true;
               }
            }
         }
      }

      if(!has_unvisited_operands){
         stack.pop_back();
         in_stack.erase(current);
         visited.insert(current);
         top_ordering->push_back(current);
      }
   }

   return top_ordering;
}


void Canonicalizer::flatten(circt::hw::HWModuleOp module){

   auto top_ordering = get_top_ord(module);
   if(!top_ordering) return;
   mlir::IRRewriter rewriter(module->getContext());
   llvm::SmallVector<mlir::Operation*> ops_to_erase;

   for(auto current_val : *top_ordering){

      if(current_val.use_empty()) continue;

      bool needs_flattening = false;
      llvm::SmallVector<mlir::Value, 8> new_operands;
      auto defining_op = current_val.getDefiningOp();

      if(defining_op && is_associative(defining_op)){

         for(auto args : defining_op->getOperands()){
            auto arg_defign_op = args.getDefiningOp();
            if(arg_defign_op && arg_defign_op->getName() == defining_op->getName()){
               needs_flattening = true;
               new_operands.append(arg_defign_op->operand_begin(), arg_defign_op->operand_end());
            }
            else{
               new_operands.push_back(args);
            }
         }
      } 

      if(needs_flattening){

         rewriter.setInsertionPoint(defining_op);

         mlir::OperationState new_state(defining_op->getLoc(), defining_op->getName());

         new_state.addOperands(new_operands);
         new_state.addTypes(defining_op->getResultTypes());
         new_state.addAttributes(defining_op->getAttrs());

         mlir::Operation* new_operation = rewriter.create(new_state);
         mlir::Value new_value = new_operation->getResult(0);

         rewriter.replaceAllUsesWith(current_val, new_value);
         ops_to_erase.push_back(defining_op);
      }
   }

   for(auto* op : ops_to_erase){
      rewriter.eraseOp(op);
   }
}


void Canonicalizer::sort(circt::hw::HWModuleOp module){

   auto block = module.getBodyBlock();
   mlir::IRRewriter rewriter(module->getContext());

   for(auto& operation : block->getOperations()){

      if(!is_commutative(&operation)) continue;

      llvm::SmallVector<mlir::Value> new_operands;

      for(auto operand : operation.getOperands()){
         new_operands.push_back(operand);
      }

      auto lexicographical_sort = [this](mlir::Value a, mlir::Value b) -> bool{

         if(a == b) return false;

         //Constant sorting
         auto const_op_a = a.getDefiningOp<circt::hw::ConstantOp>();
         auto const_op_b = b.getDefiningOp<circt::hw::ConstantOp>();

         bool a_is_const = const_op_a != nullptr;
         bool b_is_const = const_op_b != nullptr; 

         if(a_is_const != b_is_const) return a_is_const < b_is_const;

         else if(a_is_const && b_is_const){

            auto val_a = const_op_a.getValue();
            auto val_b = const_op_b.getValue();
            if(val_a.getBitWidth() != val_b.getBitWidth()){
               return val_a.getBitWidth() < val_b.getBitWidth();
            }
            return val_a.ult(val_b);
         } 

         //Block Argument Sorting
         auto a_optional_arg = llvm::dyn_cast<mlir::BlockArgument>(a);
         auto b_optional_arg = llvm::dyn_cast<mlir::BlockArgument>(b);
         bool a_is_arg = a_optional_arg != nullptr;
         bool b_is_arg = b_optional_arg != nullptr;

         if(a_is_arg != b_is_arg) return a_is_arg > b_is_arg;
         else if(a_is_arg && b_is_arg) return a_optional_arg.getArgNumber() < b_optional_arg.getArgNumber();

         //Operand sorting 
         mlir::Operation* a_op = a.getDefiningOp();
         mlir::Operation* b_op = b.getDefiningOp();

         if(a_op->getBlock() == b_op->getBlock()){
            return a_op->isBeforeInBlock(b_op);
         }

         return a.getAsOpaquePointer() < b.getAsOpaquePointer();
      };

      llvm::sort(new_operands, lexicographical_sort);
      rewriter.modifyOpInPlace(&operation, [&] () {
         operation.setOperands(new_operands);
      });
   }
}




