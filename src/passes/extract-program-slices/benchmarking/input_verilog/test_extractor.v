module find_max_ab(
   input logic [7:0] a,
   input logic [7:0] b,
   output logic [7:0] max_val
);
   assign max_val = (a > b) ? a : b;
endmodule

module find_max_xy(
   input logic [7:0] x,
   input logic [7:0] y,
   output logic [7:0] max_val
);
   assign max_val = (x > y) ? x : y;
endmodule

module find_min_and_sum(
   input logic [7:0] a,
   input logic [7:0] b,
   output logic [7:0] min_val,
   output logic [8:0] sum
);

   assign min_val = (a < b) ? a : b;

   assign sum = {1'b0, a} + {1'b0, b};
endmodule

module find_min_and_diff(
   input logic [7:0] a,
   input logic [7:0] b,
   output logic [7:0] min_val,
   output logic [7:0] diff
);

   assign min_val = (a < b) ? a : b;

   assign diff = (a > b) ? (a - b) : (b - a);
endmodule

module bit_reverse(
   input  logic [7:0] in,
   output logic [7:0] out
);
   assign out = {<<{in}};
endmodule

module top_module(
   // find_max_ab
   input logic [7:0] max_ab_in1, max_ab_in2,
   output logic [7:0] max_ab_out,
   // find_max_xy
   input logic [7:0] max_xy_in1, max_xy_in2,
   output logic [7:0] max_xy_out,
   // find_min_and_sum
   input logic [7:0] min_sum_in1, min_sum_in2,
   output logic [7:0] min_sum_out,
   output logic [8:0] min_sum_sum,
   // find_min_and_diff
   input logic [7:0] min_diff_in1, min_diff_in2,
   output logic [7:0] min_diff_out, min_diff_diff,
   // bit_reverse
   input logic [7:0] reverse_in,
   output logic [7:0] reverse_out
);

   find_max_ab u_max_ab(
      .a(max_ab_in1), .b(max_ab_in2), .max_val(max_ab_out)
   );

   find_max_xy u_max_xy(
      .x(max_xy_in1), .y(max_xy_in2), .max_val(max_xy_out)
   );

   find_min_and_sum u_min_and_sum(
      .a(min_sum_in1), .b(min_sum_in2), .min_val(min_sum_out), .sum(min_sum_sum)
   );

   find_min_and_diff u_min_and_diff(
      .a(min_diff_in1), .b(min_diff_in2), .min_val(min_diff_out), .diff(min_diff_diff)
   );

   bit_reverse u_rev(
      .in(reverse_in), .out(reverse_out)
   );
endmodule
