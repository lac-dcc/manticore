module {
  hw.module @linear(in %in : i4, in %in2 : i4, in %in3 : i4, in %in4 : i4, out out : i4, out out2 : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %9 : i3, i1
    %1 = comb.concat %c0_i2, %10, %false : i2, i1, i1
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c-5_i4 : i4
    %4 = comb.concat %false, %11, %c0_i2 : i1, i1, i2
    %5 = comb.or %4, %3 : i4
    %6 = comb.and %5, %c7_i4 : i4
    %7 = comb.concat %12, %c0_i3 : i1, i3
    %8 = comb.or %7, %6 : i4
    %9 = comb.extract %in from 3 : (i4) -> i1
    %10 = comb.extract %in2 from 0 : (i4) -> i1
    %11 = comb.extract %in2 from 1 : (i4) -> i1
    %12 = comb.extract %in4 from 2 : (i4) -> i1
    hw.output %8, %8 : i4, i4
  }
}
