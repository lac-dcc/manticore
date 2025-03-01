module top (
  input wire [3:0] a, b, c, d,
  output wire [3:0] out1, out2
);
  wire [3:0] temp1, temp2;

  // Complex redundant logic #1
  assign temp1 = (a & b) | (~a & c);

  // Complex redundant logic #2
  assign temp2 = (a & b) | (~a & c);

  // Outputs
  assign out1 = temp1 ^ d;
  assign out2 = temp2 | d;
endmodule
