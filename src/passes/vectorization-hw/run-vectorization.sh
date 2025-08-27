#!/bin/bash

# Base file name (without extension)
file_path=$1
file_name=$(basename "$file_path")

circt-verilog --ir-moore ${file_path} -o ${file_name}.moore.mlir


circt-opt ${file_name}.moore.mlir \
  --pass-pipeline="builtin.module(simple-vec)" \
  --load-pass-plugin=./../passes/Vectorization/VectorizePass.so \
  -o ${file_name}.after_pass.mlir

circt-opt --convert-moore-to-core ${file_name}.after_pass.mlir -o ${file_name}.core.mlir

circt-opt \
  --llhd-desequentialize \
  --llhd-hoist-signals \
  --llhd-sig2reg \
  --llhd-mem2reg \
  --llhd-process-lowering \
  --cse \
  --canonicalize \
  ${file_name}.core.mlir -o ${file_name}.cleaned.mlir




firtool ${file_name}.cleaned.mlir --verilog -o "/home/ullas/manticore/manticore/scripts/vectorizable-designs/vectorized/${file_name}"

