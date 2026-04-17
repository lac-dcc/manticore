#include "CareMaskAnalysis.hpp"

class DontCareReducer{

public:

   DontCareReducer() = default;
   void apply_masks(mlir::ModuleOp topModule);

};
