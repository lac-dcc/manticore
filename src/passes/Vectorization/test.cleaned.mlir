module {
  hw.module @md1(in %in : i4, out out : i4) {
    hw.output %in : i4
  }
  hw.module @md2(in %in : i4, out out : i4) {
    %0 = comb.extract %in from 0 : (i4) -> i1
    %1 = comb.extract %in from 1 : (i4) -> i1
    %2 = comb.extract %in from 2 : (i4) -> i1
    %3 = comb.extract %in from 3 : (i4) -> i1
    %4 = comb.concat %0, %1, %2, %3 : i1, i1, i1, i1
    hw.output %4 : i4
  }
  hw.module @md3(in %in : i4, out out : i4) {
    %0 = comb.extract %in from 0 : (i4) -> i1
    %1 = comb.extract %in from 1 : (i4) -> i3
    %2 = comb.concat %0, %1 : i1, i3
    hw.output %2 : i4
  }
  hw.module @md4(in %in : i8, out out : i8) {
    %0 = comb.extract %in from 0 : (i8) -> i1
    %1 = comb.extract %in from 5 : (i8) -> i1
    %2 = comb.extract %in from 4 : (i8) -> i1
    %3 = comb.extract %in from 6 : (i8) -> i2
    %4 = comb.extract %in from 1 : (i8) -> i3
    %5 = comb.concat %3, %2, %1, %0, %4 : i2, i1, i1, i1, i3
    hw.output %5 : i8
  }
}

