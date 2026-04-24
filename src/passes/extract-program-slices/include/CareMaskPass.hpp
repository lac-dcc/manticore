#include "llvm/Support/Timer.h"
#include "CareMaskAnalysis.hpp"
#include "llvm/Support/float128.h"
#include <cstdint>

class DontCareReducer{

private:

   uint64_t uselessBits; 
   uint64_t completelyUselessModules;
   uint64_t totalOutputModuleBits; 
   llvm::float128 meanUselessBits;
   llvm::Timer passTimer;


public:

   DontCareReducer();
   void apply_masks(mlir::ModuleOp topModule);
   void gather_statistics(mlir::ModuleOp topModule);
   void print_statistics();

};
