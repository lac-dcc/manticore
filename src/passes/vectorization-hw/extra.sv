module bintogrey(
  input  [3:0] bin_in,
  output [3:0] grey_out
);

  // Combina a parte vetorizada (bits 3 a 1) com a parte escalar (bit 0)
  assign grey_out = { (bin_in[3:1] ^ bin_in[2:0]), bin_in[0] };

endmodule
