module {
  hw.module @pattern_recognition(in %a : i4, in %b : i4, in %sel : i1, out result : i4) {
    %c0_i4 = hw.constant 0 : i4
    %result = llhd.sig %c0_i4 : i4
    %true = hw.constant true
    %0 = comb.xor %sel, %true : i1
    %1 = comb.mux %sel, %a, %b : i4
    %2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %result, %1 after %2 : !hw.inout<i4>
    %3 = llhd.prb %result : !hw.inout<i4>
    hw.output %3 : i4
  }
}

