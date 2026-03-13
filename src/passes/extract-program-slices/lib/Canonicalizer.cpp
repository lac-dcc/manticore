#include "../include/Canonicalizer.hpp"
#include "mlir/IR/OpDefinition.h"
using namespace circt;
using namespace mlir;

Canonicalizer::Canonicalizer(llvm::DenseSet<llvm::StringRef> targetOps){
   this->targetOps = targetOps;
}

void Canonicalizer::canonicalize(circt::hw::HWModuleOp* module){

   this->flatten(module);
   this->sort(module);
   this->reduce(module);

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

std::unique_ptr<ValueStack> Canonicalizer::get_reverse_topological_ordering(circt::hw::HWModuleOp* module){

   auto block = module->getBodyBlock();
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
   std::unique_ptr<ValueStack> reverse_top_ordering = std::make_unique<ValueStack>();

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
         for(auto operand : defOp->getOperands()){
            if(operand && !visited.count(operand)) {
               stack.push_back(operand);
               has_unvisited_operands = true;
            }
         }
      }

      if(!has_unvisited_operands){
         stack.pop_back();
         visited.insert(current);
         reverse_top_ordering->push_back(current);
      }
   }

   std::reverse(reverse_top_ordering->begin(), reverse_top_ordering->end());

   return reverse_top_ordering;
}





