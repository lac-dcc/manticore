#!/bin/bash

# Base file name (without extension)
BASE_NAME="test"

# Step 1: Convert Verilog to IR Moore
echo "Converting Verilog to IR Moore..."
circt-verilog --ir-hw ${BASE_NAME}.sv -o ${BASE_NAME}.hw.mlir

# Step 2: Run the vectorization pass on IR Moore
echo "Running vectorization pass..."
circt-opt ${BASE_NAME}.hw.mlir \
  --pass-pipeline="builtin.module(simple-vec)" \
  --load-pass-plugin=./build/VectorizePass.so \
  -o ${BASE_NAME}.after_pass.mlir

# Step 3: (optional) Convert IR Moore to IR Core
# echo "Converting IR Moore to IR Core..."
# ~/projects/circt/build/bin/circt-opt --convert-moore-to-core ${BASE_NAME}.after_pass.mlir -o ${BASE_NAME}.core.mlir

# Step 4: Clean the IR with LLHD passes and optimizations
echo "Cleaning and optimizing IR..."
circt-opt \
  --llhd-desequentialize \
  --llhd-hoist-signals \
  --llhd-sig2reg \
  --llhd-mem2reg \
  --llhd-process-lowering \
  --cse \
  --canonicalize \
  --hw-cleanup    \
  ${BASE_NAME}.after_pass.mlir -o ${BASE_NAME}.cleaned.mlir

# Step 5: Generate final Verilog
echo "Generating final Verilog..."
firtool ${BASE_NAME}.cleaned.mlir --verilog -o ${BASE_NAME}_final.v

echo "Pipeline completed! Verilog generated in ${BASE_NAME}_final.v"
