module {
  hw.module @linear(in %in : i4, in %in2 : i4, out out : i4) {
    %0 = comb.extract %in from 0 : (i4) -> i1
    %1 = comb.extract %in from 1 : (i4) -> i1
    %2 = comb.extract %in from 2 : (i4) -> i1
    %3 = comb.extract %in from 3 : (i4) -> i1
    %4 = comb.concat %0, %1, %2, %3 : i1, i1, i1, i1
    hw.output %4 : i4
  }
}
