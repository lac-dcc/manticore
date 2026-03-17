#include "../include/Canonicalizer.hpp"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "llvm/ADT/STLExtras.h"
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

      };

      llvm::sort(new_operands.begin(), new_operands.end(), lexicographical_sort);
      //IMPLEMENT LAMBDA FUNCTION THAT FINDS PRIMARY KEYS TO EACH DIFFERENT mlir::Value
      //llvm::sort(new_operands, function_should_go_here); 
      operation.setOperands(new_operands);
   }
}




