#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "../include/CareMaskPass.hpp"

void DontCareReducer::apply_masks(mlir::ModuleOp topModule) {

   mlir::SymbolTableCollection symbolTables;
   mlir::DataFlowSolver solver;
   auto analysisResult = solver.load<CareMaskAnalysis>(symbolTables);
   if(failed(solver.initializeAndRun(topModule))) return;

   mlir::OpBuilder builder(topModule.getContext()); 

   auto masking_policy = [&](mlir::Operation *op) {

      llvm::TypeSwitch<mlir::Operation *>(op)
           .Case<circt::hw::OutputOp, circt::hw::InstanceOp>([&](auto op) {
            builder.setInsertionPoint(op);
            auto loc = op.getLoc();

            for (auto &operand : op->getOpOperands()) {
                mlir::Value val = operand.get();

                auto *lattice = solver.lookupState<CareMaskLattice>(val);
                if (!lattice || lattice->getValue().isUnitialized)
                    continue;

                llvm::APInt mask = lattice->getValue().mask;
                if (mask.isAllOnes())
                    continue;

                auto maskConst = circt::hw::ConstantOp::create(builder, loc, val.getType(), mask);
                auto maskedVal = circt::comb::AndOp::create(builder, loc, val.getType(), {val, maskConst});

                operand.set(maskedVal);
            }
        });
   };

   topModule.walk(masking_policy);
}; 


