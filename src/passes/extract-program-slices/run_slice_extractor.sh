#!/bin/bash

INPUT_DIR="./benchmarking/input_verilog/test_extractor"
TMP_MLIR_DIR="./benchmarking/tmp_mlir/test_extractor"
OUTPUT_DIR="./benchmarking/generated_verilog/test_extractor"
PLUGIN_DIR="./build" 
# Correcting for general path 
CIRCT_BIN="$HOME/circt/build/bin"

echo "Converting Verilog to HW"

circt-verilog --ir-hw ${INPUT_DIR}.v -o ${TMP_MLIR_DIR}.hw.mlir
echo "Running HW Slice Extractor pass"
$CIRCT_BIN/circt-opt ${TMP_MLIR_DIR}.hw.mlir \
  --load-pass-plugin=${PLUGIN_DIR}/SliceExtractorPass.so \
  --pass-pipeline="builtin.module(slice-extractor)" \
  -o ${TMP_MLIR_DIR}.outlined.mlir

echo "Cleaning up MLIR"
$CIRCT_BIN/circt-opt ${TMP_MLIR_DIR}.outlined.mlir \
  --convert-moore-to-core \
  --canonicalize \
  --cse \
  --prettify-verilog \
  --symbol-dce \
  -o ${TMP_MLIR_DIR}.cleaned.mlir

echo "Generating final Verilog"

$CIRCT_BIN/firtool ${TMP_MLIR_DIR}.cleaned.mlir \
  --verilog \
  --disable-all-randomization \
  -o ${OUTPUT_DIR}_final.v

echo "Verilog generated in ${OUTPUT_DIR}_final.v"

