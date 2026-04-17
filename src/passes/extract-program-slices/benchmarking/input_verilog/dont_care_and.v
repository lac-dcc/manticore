module test_and(
    input wire [7:0] in_val,
    output wire [7:0] out
);
    // We do complex math to the input...
    wire [7:0] complex_math = in_val ^ 8'b10101010;
    
    // ...but then we AND it with a constant (12 = 0b00001100)
    assign out = complex_math & 8'b00001100;

endmodule
