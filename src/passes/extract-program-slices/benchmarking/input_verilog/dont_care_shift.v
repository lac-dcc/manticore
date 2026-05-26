module test_shift(
    input wire [7:0] val,
    output wire [3:0] out
);
    // Shift left by 3
    wire [7:0] shifted = val << 3;
    
    // Extract the top 4 bits of the shifted result
    assign out = shifted[7:4];

endmodule
