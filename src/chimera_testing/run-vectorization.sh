#!/bin/bash

# Base file name (without extension)
file_path=$1
placeholder="file"

circt-verilog --ir-moore ${file_path} -o ${placeholder}.moore.mlir


circt-opt ${placeholder}.moore.mlir \
  --pass-pipeline="builtin.module(simple-vec)" \
  --load-pass-plugin=./../passes/Vectorization/VectorizePass.so \
  -o ${placeholder}.after_pass.mlir

circt-opt --convert-moore-to-core ${placeholder}.after_pass.mlir -o ${placeholder}.core.mlir

circt-opt \
  --llhd-desequentialize \
  --llhd-hoist-signals \
  --llhd-sig2reg \
  --llhd-mem2reg \
  --llhd-process-lowering \
  --cse \
  --canonicalize \
  ${placeholder}.core.mlir -o ${placeholder}.cleaned.mlir


file_name=$(basename "$file_path")


firtool ${placeholder}.cleaned.mlir --verilog -o "/home/ullas/manticore/manticore/scripts/vectorizable-designs/vectorized/${file_name}"

