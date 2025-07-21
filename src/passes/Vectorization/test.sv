module md1(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = in[0];
    //assign out = in;
endmodule

module reverse(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[0];
    assign out[2] = in[1];
    assign out[1] = in[2];
    assign out[0] = in[3];
    //assign out = {in[0], in[1}, in[2], in[3]};
    //assign out = {<<{in}};
endmodule

module mix_bit(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[0];
    assign out[2] = in[3];
    assign out[1] = in[2];
    assign out[0] = in[1];
    //assign out = in[0], in[3:1]};
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
    //assign out = {in[7:6], in[4], in[5], in[0], in[3:1]};
endmodule

module pattern_recognition(output wire [3:0] result, input wire [3:0] a, b, input wire sel);
    assign result[3] = (a[3] & sel) | (b[3] & ~sel);
    assign result[2] = (a[2] & sel) | (b[2] & ~sel);
    assign result[1] = (a[1] & sel) | (b[1] & ~sel);
    assign result[0] = (a[0] & sel) | (b[0] & ~sel);
    //assign result = sel ? a : b;
endmodule
