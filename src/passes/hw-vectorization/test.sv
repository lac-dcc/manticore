module linear(output wire [3:0] out, input wire [3:0] in);

    assign out = {2'b11, in[1:0]};

endmodule

