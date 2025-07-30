module {
  moore.module @pattern_recognition(out result : !moore.l4, in %a : !moore.l4, in %b : !moore.l4, in %sel : !moore.l1) {
    %result = moore.net wire : <l4>
    %0 = moore.extract_ref %result from 3 : <l4> -> <l1>
    %1 = moore.extract %a from 3 : l4 -> l1
    %2 = moore.and %1, %sel : l1
    %3 = moore.extract %b from 3 : l4 -> l1
    %4 = moore.not %sel : l1
    %5 = moore.and %3, %4 : l1
    %6 = moore.or %2, %5 : l1
    moore.assign %0, %6 : l1
    %7 = moore.extract_ref %result from 2 : <l4> -> <l1>
    %8 = moore.extract %a from 2 : l4 -> l1
    %9 = moore.and %8, %sel : l1
    %10 = moore.extract %b from 2 : l4 -> l1
    %11 = moore.and %10, %4 : l1
    %12 = moore.or %9, %11 : l1
    moore.assign %7, %12 : l1
    %13 = moore.extract_ref %result from 1 : <l4> -> <l1>
    %14 = moore.extract %a from 1 : l4 -> l1
    %15 = moore.and %14, %sel : l1
    %16 = moore.extract %b from 1 : l4 -> l1
    %17 = moore.and %16, %4 : l1
    %18 = moore.or %15, %17 : l1
    moore.assign %13, %18 : l1
    %19 = moore.extract_ref %result from 0 : <l4> -> <l1>
    %20 = moore.extract %a from 0 : l4 -> l1
    %21 = moore.and %20, %sel : l1
    %22 = moore.extract %b from 0 : l4 -> l1
    %23 = moore.and %22, %4 : l1
    %24 = moore.or %21, %23 : l1
    moore.assign %19, %24 : l1
    %25 = moore.read %result : <l4>
    moore.output %25 : !moore.l4
  }
}
