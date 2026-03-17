// test_mul_cascade.v
module test_mul_cascade(
    input  [15:0] a, b, c, d,
    output [15:0] prod_ab, prod_abc, prod_abcd
);
    // Como estamos exportando os valores intermediários, 
    // o frontend NÃO pode achatar isso automaticamente.
    assign prod_ab   = a * b;
    assign prod_abc  = prod_ab * c;
    assign prod_abcd = prod_abc * d;
endmodule
