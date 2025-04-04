#!/bin/bash
# Converts Verilog file into an intermediate "core" representation
# Then converts it back to SystemVerilog using firtool

set -e  

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <input_file.v>"
    exit 1
fi

INPUT_FILE="$1"
INTERMEDIATE_FILE="${INPUT_FILE%.v}.core"
OUTPUT_FILE="${INPUT_FILE%.v}.sv"

echo "Generating hw file..."
circt-verilog --ir-hw "$INPUT_FILE" -o "$INTERMEDIATE_FILE"
echo "Generated hw file: $INTERMEDIATE_FILE"

echo "Generating sv..."
firtool --verilog "$INTERMEDIATE_FILE" -format=mlir -o "$OUTPUT_FILE"
echo "Generated sv file: $OUTPUT_FILE"

echo "Process completed successfully!"