module {
  hw.module @linear(in %in : i4, out out : i4) {
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %1, %c0_i3 : i1, i3
    %1 = comb.extract %in from 3 : (i4) -> i1
    hw.output %0 : i4
  }
}

