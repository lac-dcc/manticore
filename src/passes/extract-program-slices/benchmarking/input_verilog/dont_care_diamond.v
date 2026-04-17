module test_diamond(
    input wire [7:0] x,
    output wire [3:0] out
);
    // Path A: Only cares about the bottom 4 bits
    wire [7:0] path_a = x & 8'h0F; 
    
    // Path B: Only cares about the top 4 bits
    wire [7:0] path_b = x & 8'hF0; 
    
    // Re-convergence
    wire [7:0] combined = path_a | path_b;
    
    // The ultimate sink: We only want the bottom 4 bits of the combined result
    assign out = combined[3:0];

endmodule
