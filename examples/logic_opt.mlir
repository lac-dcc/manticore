module {
  moore.module @top(in %a : !moore.l4, in %b : !moore.l4, in %c : !moore.l4, in %d : !moore.l4, out out1 : !moore.l4, out out2 : !moore.l4) {
    %temp1 = moore.assigned_variable %3 : l4
    %temp2 = moore.assigned_variable %3 : l4
    %0 = moore.and %a, %b : l4
    %1 = moore.not %a : l4
    %2 = moore.and %1, %c : l4
    %3 = moore.or %0, %2 : l4
    %4 = moore.xor %temp1, %d : l4
    %5 = moore.or %temp2, %d : l4
    moore.output %4, %5 : !moore.l4, !moore.l4
  }
}

