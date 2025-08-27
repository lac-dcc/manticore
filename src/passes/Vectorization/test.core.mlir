module {
  hw.module @linear(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %c-1_i2 = hw.constant -1 : i2
    %0 = llhd.sig.extract %out from %c-1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %1 = comb.extract %in from 3 : (i4) -> i1
    %2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %0, %1 after %2 : !hw.inout<i1>
    %3 = llhd.prb %out : !hw.inout<i4>
    hw.output %3 : i4
  }
}

