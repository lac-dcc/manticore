module test_sorting(
    input [31:0] arg0,
    input [31:0] arg1,
    output [31:0] out_add,
    output [31:0] out_mul
);
    wire [31:0] interm = arg0 ^ 32'd2;

    // Scrambled Add (Commutative)
    assign out_add = 32'd5 + arg1 + interm + arg0 + 32'd2;

    // Scrambled Mul (Commutative)
    assign out_mul = arg1 * arg0 * 32'd2;

endmodule
