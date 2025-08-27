BASE_NAME="test"

# Step 1: Verilog -> Moore
echo "Step 1: Verilog -> Moore"
circt-verilog --ir-hw ${BASE_NAME}.sv -o ${BASE_NAME}.hw.mlir

# Step 2: Run the custom vectorization pass
echo "Step 2: Vectorization"
circt-opt ${BASE_NAME}.hw.mlir \
  --pass-pipeline="builtin.module(hw-vec)" \
  --load-pass-plugin=./VectorizePass.so \
  -o ${BASE_NAME}.after_pass.mlir

# # Step 3: Moore -> Core
# echo "Step 3: Moore -> Core"
# circt-opt --convert-moore-to-core ${BASE_NAME}.after_pass.mlir \
#   -o ${BASE_NAME}.core.mlir
#
# # Step 4: Fix LLHD ops, then clean up and optimize the IR (NEW APPROACH)
# echo "Step 4: Fixing LLHD and cleaning the IR"
# circt-opt ${BASE_NAME}.core.mlir \
#   --llhd-sig2reg \
#   --hw-cleanup \
#   --cse \
#   --canonicalize \
#   --hw-legalize-modules \
#   --prettify-verilog \
#   -o ${BASE_NAME}.cleaned.mlir
#
# # Step 5: Generate Verilog
# echo "Step 5: Generating Final Verilog"
# firtool ${BASE_NAME}.cleaned.mlir --verilog -o ${BASE_NAME}_final.v

echo "Pipeline completed! Verilog generated in ${BASE_NAME}_final.v"
