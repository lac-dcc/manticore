module TestFlatAnd (
    input  wire a,
    input  wire b,
    output wire out
);
    assign out = a & b & a & b;
endmodule

module TestTreeAnd (
    input  wire a,
    input  wire b,
    output wire out
);
    wire t1 = a & b;
    wire t2 = a & b;
    assign out = t1 & t2;
endmodule
