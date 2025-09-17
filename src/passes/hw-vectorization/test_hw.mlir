module {
  hw.module @linear(in %in : i4, out out : i4) {
    %c-1_i2 = hw.constant -1 : i2
    %0 = comb.extract %in from 0 : (i4) -> i2
    %1 = comb.concat %c-1_i2, %0 : i2, i2
    hw.output %1 : i4
  }
}
