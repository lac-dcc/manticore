module {
  hw.module @md1(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %0 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out, %in after %0 : !hw.inout<i4>
    %1 = llhd.prb %out : !hw.inout<i4>
    hw.output %1 : i4
  }
  hw.module @md2(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %0 = comb.extract %in from 0 : (i4) -> i1
    %1 = comb.extract %in from 1 : (i4) -> i1
    %2 = comb.extract %in from 2 : (i4) -> i1
    %3 = comb.extract %in from 3 : (i4) -> i1
    %4 = comb.concat %0, %1, %2, %3 : i1, i1, i1, i1
    %5 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out, %4 after %5 : !hw.inout<i4>
    %6 = llhd.prb %out : !hw.inout<i4>
    hw.output %6 : i4
  }
  hw.module @md3(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %0 = comb.extract %in from 1 : (i4) -> i1
    %1 = comb.extract %in from 2 : (i4) -> i1
    %2 = comb.extract %in from 3 : (i4) -> i1
    %3 = comb.extract %in from 0 : (i4) -> i1
    %4 = comb.concat %3, %2, %1, %0 : i1, i1, i1, i1
    %5 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out, %4 after %5 : !hw.inout<i4>
    %6 = llhd.prb %out : !hw.inout<i4>
    hw.output %6 : i4
  }
  hw.module @md4(in %in : i8, out out : i8) {
    %c0_i8 = hw.constant 0 : i8
    %out = llhd.sig %c0_i8 : i8
    %0 = comb.extract %in from 1 : (i8) -> i1
    %1 = comb.extract %in from 2 : (i8) -> i1
    %2 = comb.extract %in from 3 : (i8) -> i1
    %3 = comb.extract %in from 0 : (i8) -> i1
    %4 = comb.extract %in from 5 : (i8) -> i1
    %5 = comb.extract %in from 4 : (i8) -> i1
    %6 = comb.extract %in from 6 : (i8) -> i1
    %7 = comb.extract %in from 7 : (i8) -> i1
    %8 = comb.concat %7, %6, %5, %4, %3, %2, %1, %0 : i1, i1, i1, i1, i1, i1, i1, i1
    %9 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out, %8 after %9 : !hw.inout<i8>
    %10 = llhd.prb %out : !hw.inout<i8>
    hw.output %10 : i8
  }
}

