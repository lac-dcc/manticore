module {
  hw.module @bit_mixing(in %in : i4, out out : i4) {
    %0 = comb.extract %in from 0 : (i4) -> i1
    %1 = comb.extract %in from 1 : (i4) -> i3
    %2 = comb.concat %0, %1 : i1, i3
    hw.output %2 : i4
  }
}
