module md1(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = in[0];
    //assign out = in;
endmodule

module md2(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[0];
    assign out[2] = in[1];
    assign out[1] = in[2];
    assign out[0] = in[3];
    //assign out = {in[0], in[1}, in[2], in[3]};
    //assign out = {<<{in}};
endmodule

module md3(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[0];
    assign out[1] = in[2];
    assign out[2] = in[3];
    assign out[0] = in[1];
    //assign out = in[0], in[3:1]};
endmodule
