module pattern_recognition(output wire [3:0] result, input wire [3:0] a, b, input wire sel);
  assign result[3] = (a[3] & sel) | (b[3] & ~sel);
  assign result[2] = (a[2] & sel) | (b[2] & ~sel);
  assign result[1] = (a[1] & sel) | (b[1] & ~sel);
  assign result[0] = (a[0] & sel) | (b[0] & ~sel);
endmodule
