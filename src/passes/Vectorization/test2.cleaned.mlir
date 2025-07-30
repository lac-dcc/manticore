module {
  hw.module @pattern_recognition(in %a : i4, in %b : i4, in %sel : i1, out result : i4) {
    %0 = comb.mux %sel, %a, %b : i4
    hw.output %0 : i4
  }
}

