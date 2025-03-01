module {
  moore.module @top(in %a : !moore.l4, in %b : !moore.l4, in %c : !moore.l4, in %d : !moore.l4, out out1 : !moore.l4, out out2 : !moore.l4) {
    %a_0 = moore.net name "a" wire : <l4>
    %b_1 = moore.net name "b" wire : <l4>
    %c_2 = moore.net name "c" wire : <l4>
    %d_3 = moore.net name "d" wire : <l4>
    %out1 = moore.net wire : <l4>
    %out2 = moore.net wire : <l4>
    %temp1 = moore.net wire : <l4>
    %temp2 = moore.net wire : <l4>
    %0 = moore.read %a_0 : <l4>
    %1 = moore.read %b_1 : <l4>
    %2 = moore.and %0, %1 : l4
    %3 = moore.read %a_0 : <l4>
    %4 = moore.not %3 : l4
    %5 = moore.read %c_2 : <l4>
    %6 = moore.and %4, %5 : l4
    %7 = moore.or %2, %6 : l4
    moore.assign %temp1, %7 : l4
    %8 = moore.read %a_0 : <l4>
    %9 = moore.read %b_1 : <l4>
    %10 = moore.and %8, %9 : l4
    %11 = moore.read %a_0 : <l4>
    %12 = moore.not %11 : l4
    %13 = moore.read %c_2 : <l4>
    %14 = moore.and %12, %13 : l4
    %15 = moore.or %10, %14 : l4
    moore.assign %temp2, %15 : l4
    %16 = moore.read %temp1 : <l4>
    %17 = moore.read %d_3 : <l4>
    %18 = moore.xor %16, %17 : l4
    moore.assign %out1, %18 : l4
    %19 = moore.read %temp2 : <l4>
    %20 = moore.read %d_3 : <l4>
    %21 = moore.or %19, %20 : l4
    moore.assign %out2, %21 : l4
    moore.assign %a_0, %a : l4
    moore.assign %b_1, %b : l4
    moore.assign %c_2, %c : l4
    moore.assign %d_3, %d : l4
    %22 = moore.read %out1 : <l4>
    %23 = moore.read %out2 : <l4>
    moore.output %22, %23 : !moore.l4, !moore.l4
  }
}
