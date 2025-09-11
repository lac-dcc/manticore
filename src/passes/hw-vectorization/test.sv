module simple_vectorization(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = in[0];
endmodule

module reverse_endianess_vectorization(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[0];
    assign out[2] = in[1];
    assign out[1] = in[2];
    assign out[0] = in[3];
endmodule

module bit_mixing_vectorization(output wire [3:0] out2, output wire [7:0] out, input wire [3:0] in2,  input wire [7:0] in);
    assign out2[3] = in2[0];
    assign out2[1] = in2[2];
    assign out2[2] = in2[3];
    assign out2[0] = in2[1];

    assign out[7] = in[7];
    assign out[6] = in[6];
    assign out[5] = in[4];
    assign out[4] = in[5];
    assign out[3] = in[0];
    assign out[2] = in[3];
    assign out[1] = in[2];
    assign out[0] = in[1];
endmodule

module mix_bit2(output wire [7:0] out, input wire [7:0] in);
    assign out[7] = in[7];
    assign out[6] = in[6];
    assign out[5] = in[4];
    assign out[4] = in[5];
    assign out[3] = in[0];
    assign out[2] = in[3];
    assign out[1] = in[2];
    assign out[0] = in[1];
endmodule

module linear_and_reverse(output wire [7:0] out, output wire [3:0] out2, input wire [7:0] in, input wire [3:0] in2);
    assign out[7] = in[7];
    assign out[6] = in[6];
    assign out[5] = in[5];
    assign out[4] = in[4];
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = in[0];

    assign out2[3] = in2[0];
    assign out2[2] = in2[1]; 
    assign out2[1] = in2[2];
    assign out2[0] = in2[3];
endmodule

module bit_drop(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = 1'b0; // Este bit não vem da entrada 'in'
endmodule

module bit_duplicate(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[0]; // Bit in[0] usado aqui
    assign out[0] = in[0]; // E aqui também
endmodule

module mixed_sources(
    output wire [7:0] out,
    input wire [3:0] in1,
    input wire [3:0] in2
);
    // A saída 'out' depende de duas fontes diferentes!
    assign out[7:4] = in1;
    assign out[3:0] = in2;
endmodule

module with_logic_gate(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = in[1] ^ in[0]; // Uma porta XOR
endmodule
