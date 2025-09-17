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

module test_mux(output wire [3:0] result, input wire [3:0] a, b, input wire sel);
    assign result[3] = (a[3] & sel) | (b[3] & ~sel);
    assign result[2] = (a[2] & sel) | (b[2] & ~sel);
    assign result[1] = (a[1] & sel) | (b[1] & ~sel);
    assign result[0] = (a[0] & sel) | (b[0] & ~sel);
endmodule

module test_and_enable(output wire [3:0] o, input wire [3:0] a, input wire enable);
  assign o[3] = a[3] & enable;
  assign o[2] = a[2] & enable;
  assign o[1] = a[1] & enable;
  assign o[0] = a[0] & enable;
endmodule

module test_multiple_patterns(
  output wire [3:0] out_xor,
  output wire [3:0] out_and,
  input wire [3:0] a, b, c
);
  assign out_xor[3] = a[3] ^ b[3];
  assign out_xor[2] = a[2] ^ b[2];
  assign out_xor[1] = a[1] ^ b[1];
  assign out_xor[0] = a[0] ^ b[0];

  assign out_and[3] = a[3] & c[3];
  assign out_and[2] = a[2] & c[2];
  assign out_and[1] = a[1] & c[1];
  assign out_and[0] = a[0] & c[0];
endmodule

module test_add(output wire [3:0] o, input wire [3:0] a, b);
  assign o[3] = a[3] + b[3];
  assign o[2] = a[2] + b[2];
  assign o[1] = a[1] + b[1];
  assign o[0] = a[0] + b[0];
endmodule

module CustomLogic (
  output wire [7:0] out,
  input  wire [7:0] a,
  input  wire [7:0] b
);

  assign out[0] = (a[0] & b[0]) | ~a[0];
  assign out[1] = (a[1] & b[1]) | ~a[1];
  assign out[2] = (a[2] & b[2]) | ~a[2];
  assign out[3] = (a[3] & b[3]) | ~a[3];
  assign out[4] = (a[4] & b[4]) | ~a[4];
  assign out[5] = (a[5] & b[5]) | ~a[5];
  assign out[6] = (a[6] & b[6]) | ~a[6];
  assign out[7] = (a[7] & b[7]) | ~a[7];

endmodule

module GatedXOR (
  output wire [3:0] out,
  input  wire [3:0] a,
  input  wire [3:0] b,
  input  wire       enable
);

  assign out[0] = (a[0] ^ b[0]) & enable;
  assign out[1] = (a[1] ^ b[1]) & enable;
  assign out[2] = (a[2] ^ b[2]) & enable;
  assign out[3] = (a[3] ^ b[3]) & enable;

endmodule

module with_logic_gate(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = in[1] ^ in[0]; 
endmodule

module bit_drop(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[1];
    assign out[0] = 1'b0; 
endmodule

module bit_duplicate(output wire [3:0] out, input wire [3:0] in);
    assign out[3] = in[3];
    assign out[2] = in[2];
    assign out[1] = in[0]; 
    assign out[0] = in[0]; 
endmodule

module ShuffledXOR(
  output wire [3:0] out,
  input  wire [3:0] a,
  input  wire [3:0] b
);

  wire [3:0] temp;
  assign temp = a ^ b; 

  assign out[3] = temp[0];
  assign out[2] = temp[2];
  assign out[1] = temp[3];
  assign out[0] = temp[1];

endmodule


module LogicalShiftRightBy2(
  output wire [7:0] out,
  input  wire [7:0] in
);

  assign out[7] = 1'b0;
  assign out[6] = 1'b0;
  assign out[5] = in[7];
  assign out[4] = in[6];
  assign out[3] = in[5];
  assign out[2] = in[4];
  assign out[1] = in[3];
  assign out[0] = in[2];

endmodule

module VectorizedEnable(
  output wire [3:0] o,
  input wire [3:0] a,
  input wire [3:0] enable
);

  assign o[3] = a[3] & enable[3];
  assign o[2] = a[2] & enable[2];
  assign o[1] = a[1] & enable[1];
  assign o[0] = a[0] & enable[0];
  
endmodule

// Should not vectorize

module mixed_sources(
    output wire [7:0] out,
    input wire [3:0] in1,
    input wire [3:0] in2
);
    assign out[7:4] = in1;
    assign out[3:0] = in2;
endmodule

module InconsistentLogic(
  output wire [3:0] out,
  input  wire [3:0] a,
  input  wire [3:0] b
);

  assign out[3] = a[3] & b[3]; 
  assign out[2] = a[2] | b[2]; 
  assign out[1] = a[1] ^ b[1]; 
  assign out[0] = ~a[0];       

endmodule

module CarryChainAdder(
  output wire [3:0] sum,
  input  wire [3:0] a,
  input  wire [3:0] b
);

  wire [3:0] carry;

  assign sum[0] = a[0] ^ b[0];
  assign carry[0] = a[0] & b[0];

  assign sum[1] = a[1] ^ b[1] ^ carry[0]; 
  assign carry[1] = (a[1] & b[1]) | (a[1] & carry[0]) | (b[1] & carry[0]);

  assign sum[2] = a[2] ^ b[2] ^ carry[1]; 
  assign carry[2] = (a[2] & b[2]) | (a[2] & carry[1]) | (b[2] & carry[1]);
  
  assign sum[3] = a[3] ^ b[3] ^ carry[2]; 

endmodule

module ShiftAndXOR(
  output wire [3:0] out,
  input  wire [3:0] a,
  input  wire [3:0] b
);

  assign out[3] = a[3] ^ b[2]; 
  assign out[2] = a[2] ^ b[1];
  assign out[1] = a[1] ^ b[0];
  assign out[0] = a[0] ^ 1'b0; 

endmodule

module VectorizedSubtraction(
  output wire [7:0] o,
  input  wire [7:0] a, b
);

  assign o[7] = a[7] - b[7];
  assign o[6] = a[6] - b[6];
  assign o[5] = a[5] - b[5];
  assign o[4] = a[4] - b[4];
  assign o[3] = a[3] - b[3];
  assign o[2] = a[2] - b[2];
  assign o[1] = a[1] - b[1];
  assign o[0] = a[0] - b[0];
  
endmodule
