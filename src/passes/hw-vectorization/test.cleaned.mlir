module {
  hw.module @linear(in %in : i4, in %in2 : i4, out out : i4) {
    %0 = comb.reverse %in : i4
    hw.output %0 : i4
  }
}

