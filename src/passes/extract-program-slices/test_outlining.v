module original(input [7:0] a, b, c, output [7:0] out);
    assign out = (a + b) - c;
endmodule

module copy_original(input [7:0] x, y, z, output [7:0] res);
    assign res = (x + y) - z;
endmodule

module different_logic(input [7:0] i, j, k, output [7:0] val);
    assign val = (i - j) + k;
endmodule