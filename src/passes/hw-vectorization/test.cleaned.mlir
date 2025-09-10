module {
  hw.module @linear(in %in : i4, in %in2 : i4, in %in3 : i4, in %in4 : i4, out out : i4) {
    %0 = comb.extract %in from 3 : (i4) -> i1
    %1 = comb.extract %in2 from 0 : (i4) -> i2
    %2 = comb.reverse %1 : i2
    %3 = comb.extract %in4 from 2 : (i4) -> i1
    %4 = comb.concat %0, %2, %3 : i1, i2, i1
    hw.output %4 : i4
  }
}

