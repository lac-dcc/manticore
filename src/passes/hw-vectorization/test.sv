module linear(output wire [3:0] out, output wire [3:0] out2, input wire [3:0] in, input wire [3:0] in2, input wire [3:0] in3, input wire [3:0] in4);
  assign out[0] = in[3];
  assign out[1] = in2[0];
  assign out[2] = in2[1];
  assign out[3] = in4[2];

  assign out2[0] = in[3];
  assign out2[1] = in2[0];
  assign out2[2] = in2[1];
  assign out2[3] = in4[2];

endmodule


