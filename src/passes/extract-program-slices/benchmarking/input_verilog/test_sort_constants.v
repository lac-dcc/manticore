module test_sort_constants(
    input [31:0] arg0,
    output [31:0] out_add
);
    // Expected behavior: arg0 should be pulled to the front.
    // The constants should be pushed to the back and sorted: 1, 5, 42, 100.
    assign out_add = 32'd100 + 32'd5 + arg0 + 32'd42 + 32'd1;

endmodule
