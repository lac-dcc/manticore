module test_sort_args(
    input a, // Arg 0
    input b, // Arg 1
    input c, // Arg 2
    input d, // Arg 3
    input e, // Arg 4
    output out_xor
);
    // Expected behavior: The pass should reorder this to a ^ b ^ c ^ d ^ e
    assign out_xor = e ^ b ^ d ^ a ^ c;

endmodule
