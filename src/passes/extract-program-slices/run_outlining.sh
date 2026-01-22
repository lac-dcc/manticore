#!/bin/bash

BASE_NAME="test_outlining"
PLUGIN_DIR="./build" 
CIRCT_BIN=~/projects/circt/build/bin

echo "Converting Verilog to HW..."
circt-verilog --ir-hw ${BASE_NAME}.v -o ${BASE_NAME}.hw.mlir

echo "Running HW Outlining pass..."
$CIRCT_BIN/circt-opt ${BASE_NAME}.hw.mlir \
  --pass-pipeline="builtin.module(hw-outlining)" \
  --load-pass-plugin=${PLUGIN_DIR}/HWOutliningPass.so \
  -o ${BASE_NAME}.outlined.mlir

echo "Clean the IR and optimizations..."
$CIRCT_BIN/circt-opt ${BASE_NAME}.outlined.mlir \
  --canonicalize \
  --cse \
  --prettify-verilog \
  --symbol-dce \
  -o ${BASE_NAME}.cleaned.mlir

echo "Generating final Verilog..."
$CIRCT_BIN/firtool ${BASE_NAME}.cleaned.mlir \
  --verilog \
  --disable-all-randomization \
  -o ${BASE_NAME}_final.v

echo "Pipeline completed! Verilog generated in ${BASE_NAME}_final.v"