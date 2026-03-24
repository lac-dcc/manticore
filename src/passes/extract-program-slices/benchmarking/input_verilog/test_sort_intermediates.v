module test_sort_intermediates(
    input [31:0] arg0,
    input [31:0] arg1,
    input [31:0] arg2,
    output [31:0] out_add
);
    // These define the operations in the block in a specific order (1, then 2, then 3)
    wire [31:0] op1 = arg0 & 32'hF;
    wire [31:0] op2 = arg1 | 32'hA;
    wire [31:0] op3 = arg2 ^ 32'h5;

    // Scrambled addition of intermediate operations.
    // Expected behavior: The pass should reorder this to op1 + op2 + op3
    assign out_add = op3 + op1 + op2;

endmodule
