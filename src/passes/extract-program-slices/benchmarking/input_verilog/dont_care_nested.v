module inner_core(
    input wire [7:0] data_in,
    output wire [7:0] data_out
);
    // Some complex logic
    assign data_out = ~data_in ^ 8'hAA;
endmodule

module middle_layer(
    input wire [7:0] mid_in,
    output wire [7:0] mid_out
);
    // Instantiates the core
    inner_core core_inst (
        .data_in(mid_in),
        .data_out(mid_out)
    );
endmodule

module test_nested(
    input wire [7:0] top_in,
    output wire [1:0] top_out
);
    wire [7:0] internal_wire;
    
    // Instantiates the middle layer
    middle_layer mid_inst (
        .mid_in(top_in),
        .mid_out(internal_wire)
    );
    
    // Extract only the bottom 2 bits
    assign top_out = internal_wire[1:0];

endmodule
