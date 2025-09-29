module {
  hw.module @linear(in %in : i3, out out : i3) {
    %false = hw.constant false
    %c3_i3 = hw.constant 3 : i3
    %c0_i2 = hw.constant 0 : i2
    %0 = comb.concat %c0_i2, %6 : i2, i1
    %1 = comb.concat %false, %7, %false : i1, i1, i1
    %2 = comb.or %1, %0 : i3
    %3 = comb.and %2, %c3_i3 : i3
    %4 = comb.concat %8, %c0_i2 : i1, i2
    %5 = comb.or %4, %3 : i3
    %6 = comb.extract %in from 0 : (i3) -> i1
    %7 = comb.extract %in from 1 : (i3) -> i1
    %8 = comb.extract %in from 2 : (i3) -> i1
    hw.output %5 : i3
  }
}
