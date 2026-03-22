module test_reduce_annihilators(
    input [31:0] a,
    input [31:0] b,
    input [31:0] c,
    output [31:0] out_mul,
    output [31:0] out_and,
    output [31:0] out_or
);
    // Expected: Entire operation is replaced by 0
    assign out_mul = a * b * 32'd0 * c;

    // Expected: Entire operation is replaced by 0
    assign out_and = a & b & 32'd0 & c;

    // Expected: Entire operation is replaced by 32'hFFFFFFFF (-1)
    assign out_or  = a | b | 32'hFFFFFFFF | c;

endmodule
