// test_heterogeneous_tree.v
module test_heterogeneous_tree(
    input  [31:0] a, b, c, d, e, f,
    output [31:0] out_mixed, out_final
);
    // Cadeia 1: Adição (deve ser achatada para a + b + c)
    wire [31:0] add1 = a + b;
    wire [31:0] add2 = add1 + c;

    // Cadeia 2: Multiplicação (deve ser achatada para d * e * f)
    wire [31:0] mul1 = d * e;
    wire [31:0] mul2 = mul1 * f;

    // Barreira: O XOR quebra a associatividade das cadeias anteriores
    // O pass NÃO pode puxar a,b,c,d,e,f diretamente para cá.
    wire [31:0] mixed = add2 ^ mul2;
    assign out_mixed = mixed;

    // Nova cadeia de adição começando após a barreira
    // Deve virar: mixed + a + b
    assign out_final = mixed + a + b;
endmodule
