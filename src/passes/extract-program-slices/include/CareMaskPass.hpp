#include "CareMaskAnalysis.hpp"
#include "llvm/Support/float128.h"
#include <cstdint>

class DontCareReducer{

private:

   uint64_t uselessBits = 0; 
   uint64_t completelyUselessModules = 0;
   uint64_t totalOutputModuleBits = 0; 
   llvm::float128 meanUselessBits = 0;


public:

   DontCareReducer() = default;
   void apply_masks(mlir::ModuleOp topModule);
   void gather_statistics(mlir::ModuleOp topModule);
   void print_statistics();

};
