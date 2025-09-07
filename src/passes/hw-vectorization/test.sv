module linear(output wire [3:0] out, input wire [3:0] in, input wire [3:0] in2);
  assign out = {in[0], in[2:0]};
endmodule

