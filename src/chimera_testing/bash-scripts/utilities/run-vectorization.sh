#!/bin/bash

# Base file name (without extension)

file=$1
file_base_name=$(basename $file)

# echo "Converting Verilog to IR Moore..."
circt-verilog --ir-hw $file -o ${file_base_name}.hw.mlir

# echo "Running vectorization pass..."
circt-opt ${file_base_name}.hw.mlir \
  --pass-pipeline="builtin.module(simple-vec)" \
  --load-pass-plugin=./../passes/hw-vectorization/build/VectorizePass.so \
  -o ${file_base_name}.after_pass.mlir


# echo "Cleaning and optimizing IR..."
circt-opt \
  --llhd-desequentialize \
  --llhd-hoist-signals \
  --llhd-sig2reg \
  --llhd-mem2reg \
  --llhd-process-lowering \
  --cse \
  --canonicalize \
  --hw-cleanup    \
  ${file_base_name}.after_pass.mlir -o ${file_base_name}.cleaned.mlir

# Step 5: Generate final Verilog
# echo "Generating final Verilog..."

firtool ${file_base_name}.cleaned.mlir --verilog -o $file

# echo "Pipeline completed! Verilog generated in ${BASE_NAME}_final.v"

rm *.mlir
