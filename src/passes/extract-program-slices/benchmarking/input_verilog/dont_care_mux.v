module test_mux(
    input wire cond,
    input wire [7:0] a,
    input wire [7:0] b,
    output wire [3:0] out
);
    // 8-bit multiplexer
    wire [7:0] mux_result = cond ? a : b;
    
    // 4-bit extraction (truncation)
    assign out = mux_result[3:0];

endmodule
