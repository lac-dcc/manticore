#!/bin/bash

BASE_NAME="test_extractor"
PLUGIN_DIR="./build" 
CIRCT_BIN=~/projects/circt/build/bin

echo "Converting Verilog to HW"
circt-verilog --ir-hw ${BASE_NAME}.v -o ${BASE_NAME}.hw.mlir

echo "Running HW Slice Extractor pass"
$CIRCT_BIN/circt-opt ${BASE_NAME}.hw.mlir \
  --pass-pipeline="builtin.module(slice-extractor)" \
  --load-pass-plugin=${PLUGIN_DIR}/SliceExtractorPass.so \
  -o ${BASE_NAME}.outlined.mlir

$CIRCT_BIN/circt-opt ${BASE_NAME}.outlined.mlir \
  --canonicalize \
  --cse \
  --prettify-verilog \
  --symbol-dce \
  -o ${BASE_NAME}.cleaned.mlir

echo "Generating final Verilog"
$CIRCT_BIN/firtool ${BASE_NAME}.cleaned.mlir \
  --verilog \
  --disable-all-randomization \
  -o ${BASE_NAME}_final.v

echo "Verilog generated in ${BASE_NAME}_final.v"