module {
  hw.module @test_reverse(in %in: i8, out out: i8) {
    %reversed = comb.reverse %in : i8
    hw.output %reversed : i8
  }
}
