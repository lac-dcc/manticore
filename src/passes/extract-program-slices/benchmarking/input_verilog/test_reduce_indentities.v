module test_reduce_identities(
    input [31:0] a,
    input [31:0] b,
    output [31:0] out_add,
    output [31:0] out_mul,
    output [31:0] out_or,
    output [31:0] out_and
);
    // Expected: a + b (drops the 0)
    assign out_add = a + 32'd0 + b;

    // Expected: a * b (drops the 1)
    assign out_mul = a * 32'd1 * b;

    // Expected: a | b (drops the 0)
    assign out_or  = a | 32'd0 | b;

    // Expected: a & b (drops the all-ones/FFFFFFFF)
    assign out_and = a & 32'hFFFFFFFF & b;

endmodule
