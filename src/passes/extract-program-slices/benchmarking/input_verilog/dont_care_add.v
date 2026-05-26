module test_add(
    input wire [7:0] a,
    input wire [7:0] b,
    output wire [1:0] out
);
    // 8-bit addition
    wire [7:0] sum = a + b;
    
    // Extracting bits 4 and 5
    assign out = sum[5:4];

endmodule
