// test_basic_add.v
module test_basic_add(
    input  [31:0] B, C, E,
    output [31:0] A, D
);
    // Operação interna
    assign A = B + C;
    
    // Cascata que deve ser "achatada" (flattened)
    assign D = A + E; 
endmodule
