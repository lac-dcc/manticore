#!/bin/bash

BASE_NAME="test"

echo "Converting Verilog to HW..."
circt-verilog --ir-hw ${BASE_NAME}.sv -o ${BASE_NAME}.hw.mlir

echo "Running vectorization pass..."
~/projects/circt/build/bin/circt-opt ${BASE_NAME}.hw.mlir \
  --pass-pipeline="builtin.module(hw-vectorization)" \
  --load-pass-plugin=./build/VectorizePass.so \
  -o ${BASE_NAME}.after_pass.mlir

echo "Clean the IR and optimizations..."
~/projects/circt/build/bin/circt-opt ${BASE_NAME}.after_pass.mlir \
  --canonicalize \
  --cse \
  --prettify-verilog \
  --symbol-dce \
  -o ${BASE_NAME}.cleaned.mlir

echo "Generating final Verilog..."
~/projects/circt/build/bin/firtool ${BASE_NAME}.cleaned.mlir --verilog -o ${BASE_NAME}_final.v

echo "Pipeline completed! Verilog generated in ${BASE_NAME}_final.v"