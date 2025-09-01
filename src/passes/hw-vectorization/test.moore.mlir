module {
  moore.module @linear(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 3 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 2 : l4 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 1 : l4 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 0 : l4 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
  moore.module @reverse(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 0 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 1 : l4 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 2 : l4 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 3 : l4 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
  moore.module @mix_bit(out out : !moore.l4, in %in : !moore.l4) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %1 = moore.extract %in from 0 : l4 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %3 = moore.extract %in from 3 : l4 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %5 = moore.extract %in from 2 : l4 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %7 = moore.extract %in from 1 : l4 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.read %out : <l4>
    moore.output %8 : !moore.l4
  }
  moore.module @mix_bit2(out out : !moore.l8, in %in : !moore.l8) {
    %out = moore.net wire : <l8>
    %0 = moore.extract_ref %out from 7 : <l8> -> <l1>
    %1 = moore.extract %in from 7 : l8 -> l1
    moore.assign %0, %1 : l1
    %2 = moore.extract_ref %out from 6 : <l8> -> <l1>
    %3 = moore.extract %in from 6 : l8 -> l1
    moore.assign %2, %3 : l1
    %4 = moore.extract_ref %out from 5 : <l8> -> <l1>
    %5 = moore.extract %in from 4 : l8 -> l1
    moore.assign %4, %5 : l1
    %6 = moore.extract_ref %out from 4 : <l8> -> <l1>
    %7 = moore.extract %in from 5 : l8 -> l1
    moore.assign %6, %7 : l1
    %8 = moore.extract_ref %out from 3 : <l8> -> <l1>
    %9 = moore.extract %in from 0 : l8 -> l1
    moore.assign %8, %9 : l1
    %10 = moore.extract_ref %out from 2 : <l8> -> <l1>
    %11 = moore.extract %in from 3 : l8 -> l1
    moore.assign %10, %11 : l1
    %12 = moore.extract_ref %out from 1 : <l8> -> <l1>
    %13 = moore.extract %in from 2 : l8 -> l1
    moore.assign %12, %13 : l1
    %14 = moore.extract_ref %out from 0 : <l8> -> <l1>
    %15 = moore.extract %in from 1 : l8 -> l1
    moore.assign %14, %15 : l1
    %16 = moore.read %out : <l8>
    moore.output %16 : !moore.l8
  }
  moore.module @test_mux(out result : !moore.l4, in %a : !moore.l4, in %b : !moore.l4, in %sel : !moore.l1) {
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
  moore.module @test_and_enable(out o : !moore.l4, in %a : !moore.l4, in %enable : !moore.l1) {
    %o = moore.net wire : <l4>
    %0 = moore.extract_ref %o from 3 : <l4> -> <l1>
    %1 = moore.extract %a from 3 : l4 -> l1
    %2 = moore.and %1, %enable : l1
    moore.assign %0, %2 : l1
    %3 = moore.extract_ref %o from 2 : <l4> -> <l1>
    %4 = moore.extract %a from 2 : l4 -> l1
    %5 = moore.and %4, %enable : l1
    moore.assign %3, %5 : l1
    %6 = moore.extract_ref %o from 1 : <l4> -> <l1>
    %7 = moore.extract %a from 1 : l4 -> l1
    %8 = moore.and %7, %enable : l1
    moore.assign %6, %8 : l1
    %9 = moore.extract_ref %o from 0 : <l4> -> <l1>
    %10 = moore.extract %a from 0 : l4 -> l1
    %11 = moore.and %10, %enable : l1
    moore.assign %9, %11 : l1
    %12 = moore.read %o : <l4>
    moore.output %12 : !moore.l4
  }
  moore.module @test_multiple_patterns(out out_xor : !moore.l4, out out_and : !moore.l4, in %a : !moore.l4, in %b : !moore.l4, in %c : !moore.l4) {
    %out_xor = moore.net wire : <l4>
    %out_and = moore.net wire : <l4>
    %0 = moore.extract_ref %out_xor from 3 : <l4> -> <l1>
    %1 = moore.extract %a from 3 : l4 -> l1
    %2 = moore.extract %b from 3 : l4 -> l1
    %3 = moore.xor %1, %2 : l1
    moore.assign %0, %3 : l1
    %4 = moore.extract_ref %out_xor from 2 : <l4> -> <l1>
    %5 = moore.extract %a from 2 : l4 -> l1
    %6 = moore.extract %b from 2 : l4 -> l1
    %7 = moore.xor %5, %6 : l1
    moore.assign %4, %7 : l1
    %8 = moore.extract_ref %out_xor from 1 : <l4> -> <l1>
    %9 = moore.extract %a from 1 : l4 -> l1
    %10 = moore.extract %b from 1 : l4 -> l1
    %11 = moore.xor %9, %10 : l1
    moore.assign %8, %11 : l1
    %12 = moore.extract_ref %out_xor from 0 : <l4> -> <l1>
    %13 = moore.extract %a from 0 : l4 -> l1
    %14 = moore.extract %b from 0 : l4 -> l1
    %15 = moore.xor %13, %14 : l1
    moore.assign %12, %15 : l1
    %16 = moore.extract_ref %out_and from 3 : <l4> -> <l1>
    %17 = moore.extract %c from 3 : l4 -> l1
    %18 = moore.and %1, %17 : l1
    moore.assign %16, %18 : l1
    %19 = moore.extract_ref %out_and from 2 : <l4> -> <l1>
    %20 = moore.extract %c from 2 : l4 -> l1
    %21 = moore.and %5, %20 : l1
    moore.assign %19, %21 : l1
    %22 = moore.extract_ref %out_and from 1 : <l4> -> <l1>
    %23 = moore.extract %c from 1 : l4 -> l1
    %24 = moore.and %9, %23 : l1
    moore.assign %22, %24 : l1
    %25 = moore.extract_ref %out_and from 0 : <l4> -> <l1>
    %26 = moore.extract %c from 0 : l4 -> l1
    %27 = moore.and %13, %26 : l1
    moore.assign %25, %27 : l1
    %28 = moore.read %out_xor : <l4>
    %29 = moore.read %out_and : <l4>
    moore.output %28, %29 : !moore.l4, !moore.l4
  }
  moore.module @test_add(out o : !moore.l4, in %a : !moore.l4, in %b : !moore.l4) {
    %o = moore.net wire : <l4>
    %0 = moore.extract_ref %o from 3 : <l4> -> <l1>
    %1 = moore.extract %a from 3 : l4 -> l1
    %2 = moore.extract %b from 3 : l4 -> l1
    %3 = moore.add %1, %2 : l1
    moore.assign %0, %3 : l1
    %4 = moore.extract_ref %o from 2 : <l4> -> <l1>
    %5 = moore.extract %a from 2 : l4 -> l1
    %6 = moore.extract %b from 2 : l4 -> l1
    %7 = moore.add %5, %6 : l1
    moore.assign %4, %7 : l1
    %8 = moore.extract_ref %o from 1 : <l4> -> <l1>
    %9 = moore.extract %a from 1 : l4 -> l1
    %10 = moore.extract %b from 1 : l4 -> l1
    %11 = moore.add %9, %10 : l1
    moore.assign %8, %11 : l1
    %12 = moore.extract_ref %o from 0 : <l4> -> <l1>
    %13 = moore.extract %a from 0 : l4 -> l1
    %14 = moore.extract %b from 0 : l4 -> l1
    %15 = moore.add %13, %14 : l1
    moore.assign %12, %15 : l1
    %16 = moore.read %o : <l4>
    moore.output %16 : !moore.l4
  }
  moore.module @CustomLogic(out out : !moore.l8, in %a : !moore.l8, in %b : !moore.l8) {
    %out = moore.net wire : <l8>
    %0 = moore.extract_ref %out from 0 : <l8> -> <l1>
    %1 = moore.extract %a from 0 : l8 -> l1
    %2 = moore.extract %b from 0 : l8 -> l1
    %3 = moore.and %1, %2 : l1
    %4 = moore.not %1 : l1
    %5 = moore.or %3, %4 : l1
    moore.assign %0, %5 : l1
    %6 = moore.extract_ref %out from 1 : <l8> -> <l1>
    %7 = moore.extract %a from 1 : l8 -> l1
    %8 = moore.extract %b from 1 : l8 -> l1
    %9 = moore.and %7, %8 : l1
    %10 = moore.not %7 : l1
    %11 = moore.or %9, %10 : l1
    moore.assign %6, %11 : l1
    %12 = moore.extract_ref %out from 2 : <l8> -> <l1>
    %13 = moore.extract %a from 2 : l8 -> l1
    %14 = moore.extract %b from 2 : l8 -> l1
    %15 = moore.and %13, %14 : l1
    %16 = moore.not %13 : l1
    %17 = moore.or %15, %16 : l1
    moore.assign %12, %17 : l1
    %18 = moore.extract_ref %out from 3 : <l8> -> <l1>
    %19 = moore.extract %a from 3 : l8 -> l1
    %20 = moore.extract %b from 3 : l8 -> l1
    %21 = moore.and %19, %20 : l1
    %22 = moore.not %19 : l1
    %23 = moore.or %21, %22 : l1
    moore.assign %18, %23 : l1
    %24 = moore.extract_ref %out from 4 : <l8> -> <l1>
    %25 = moore.extract %a from 4 : l8 -> l1
    %26 = moore.extract %b from 4 : l8 -> l1
    %27 = moore.and %25, %26 : l1
    %28 = moore.not %25 : l1
    %29 = moore.or %27, %28 : l1
    moore.assign %24, %29 : l1
    %30 = moore.extract_ref %out from 5 : <l8> -> <l1>
    %31 = moore.extract %a from 5 : l8 -> l1
    %32 = moore.extract %b from 5 : l8 -> l1
    %33 = moore.and %31, %32 : l1
    %34 = moore.not %31 : l1
    %35 = moore.or %33, %34 : l1
    moore.assign %30, %35 : l1
    %36 = moore.extract_ref %out from 6 : <l8> -> <l1>
    %37 = moore.extract %a from 6 : l8 -> l1
    %38 = moore.extract %b from 6 : l8 -> l1
    %39 = moore.and %37, %38 : l1
    %40 = moore.not %37 : l1
    %41 = moore.or %39, %40 : l1
    moore.assign %36, %41 : l1
    %42 = moore.extract_ref %out from 7 : <l8> -> <l1>
    %43 = moore.extract %a from 7 : l8 -> l1
    %44 = moore.extract %b from 7 : l8 -> l1
    %45 = moore.and %43, %44 : l1
    %46 = moore.not %43 : l1
    %47 = moore.or %45, %46 : l1
    moore.assign %42, %47 : l1
    %48 = moore.read %out : <l8>
    moore.output %48 : !moore.l8
  }
  moore.module @GatedXOR(out out : !moore.l4, in %a : !moore.l4, in %b : !moore.l4, in %enable : !moore.l1) {
    %out = moore.net wire : <l4>
    %0 = moore.extract_ref %out from 0 : <l4> -> <l1>
    %1 = moore.extract %a from 0 : l4 -> l1
    %2 = moore.extract %b from 0 : l4 -> l1
    %3 = moore.xor %1, %2 : l1
    %4 = moore.and %3, %enable : l1
    moore.assign %0, %4 : l1
    %5 = moore.extract_ref %out from 1 : <l4> -> <l1>
    %6 = moore.extract %a from 1 : l4 -> l1
    %7 = moore.extract %b from 1 : l4 -> l1
    %8 = moore.xor %6, %7 : l1
    %9 = moore.and %8, %enable : l1
    moore.assign %5, %9 : l1
    %10 = moore.extract_ref %out from 2 : <l4> -> <l1>
    %11 = moore.extract %a from 2 : l4 -> l1
    %12 = moore.extract %b from 2 : l4 -> l1
    %13 = moore.xor %11, %12 : l1
    %14 = moore.and %13, %enable : l1
    moore.assign %10, %14 : l1
    %15 = moore.extract_ref %out from 3 : <l4> -> <l1>
    %16 = moore.extract %a from 3 : l4 -> l1
    %17 = moore.extract %b from 3 : l4 -> l1
    %18 = moore.xor %16, %17 : l1
    %19 = moore.and %18, %enable : l1
    moore.assign %15, %19 : l1
    %20 = moore.read %out : <l4>
    moore.output %20 : !moore.l4
  }
}
