module test_contiguous_out_of_order(
    input  wire [3:0] inA,
    output wire [3:0] outA
);
    assign outA[2] = inA[2];
    assign outA[0] = inA[0];
    assign outA[3] = inA[3];
    assign outA[1] = inA[1];
endmodule

module test_interleaved_groups(
    input  wire [1:0] inA,
    input  wire [1:0] inB,
    output wire [1:0] outA,
    output wire [1:0] outB
);
    assign outA[1] = inA[1];
    assign outB[0] = inB[0];
    assign outA[0] = inA[0];
    assign outB[1] = inB[1];
endmodule


module test_contiguous_eight_bits(
    input  wire [7:0] inA,
    output wire [7:0] outA
);
    assign outA[7] = inA[7];
    assign outA[5] = inA[5];
    assign outA[6] = inA[6];
    assign outA[4] = inA[4];
endmodule

module test_multiple_groups_mixed(
    input  wire [1:0] inA,
    input  wire [1:0] inB,
    input  wire [3:0] inC,
    output wire [1:0] outA,
    output wire [1:0] outB,
    output wire [3:0] outC
);
    assign outA[0] = inA[0];
    assign outC[1] = inC[1];
    assign outC[0] = inC[0];
    assign outB[1] = inB[1];
    assign outC[3] = inC[3];
    assign outA[1] = inA[1];
    assign outB[0] = inB[0];
    assign outC[2] = inC[2];
endmodule
