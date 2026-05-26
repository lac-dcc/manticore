// 1. Definições dos Módulos Base
module status_a(input a, b, output [7:0] out);
    assign out = {6'b000000, b, a};
endmodule


module status_b(input x, y, output [7:0] out);
    assign out = {6'b111111, y, x};
endmodule

// 2. Módulo Top-Level (O Ambiente de Teste Formal)
module top_original(
    input in_a, in_b,
    output [1:0] display_en, motor_en
);
    wire [7:0] status_a_out, status_b_out;

    // Instanciação posicional para economizar linhas
    status_a inst_a (in_a, in_b, status_a_out);
    status_b inst_b (in_a, in_b, status_b_out);

    // O Downstream Context
    assign display_en = status_a_out[1:0];
    assign motor_en   = status_b_out[1:0];

endmodule

