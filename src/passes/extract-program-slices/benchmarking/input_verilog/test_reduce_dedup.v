module test_reduce_dedup(
    input [31:0] a,
    input [31:0] b,
    output [31:0] out_and_dedup,
    output [31:0] out_or_single
);
    // Expected: a & b (Drops the duplicate 'a' and 'b')
    // Note: Since 'a' and 'b' are block arguments, they are the exact same mlir::Value pointer,
    // so your llvm::is_contained logic should successfully catch them!
    assign out_and_dedup = a & b & a & b;

    // Expected: Just 'a'
    // The duplicate 'a' is dropped. The 0 is dropped as an identity. 
    // Only one 'a' remains, triggering your single-operand replacement fallback.
    assign out_or_single = a | 32'd0 | a;

endmodule
