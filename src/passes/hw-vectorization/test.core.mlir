module {
  hw.module @linear(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %c-1_i2 = hw.constant -1 : i2
    %0 = llhd.sig.extract %out from %c-1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %1 = comb.extract %in from 3 : (i4) -> i1
    %2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %0, %1 after %2 : !hw.inout<i1>
    %c-2_i2 = hw.constant -2 : i2
    %3 = llhd.sig.extract %out from %c-2_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %4 = comb.extract %in from 2 : (i4) -> i1
    %5 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %3, %4 after %5 : !hw.inout<i1>
    %c1_i2 = hw.constant 1 : i2
    %6 = llhd.sig.extract %out from %c1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %7 = comb.extract %in from 1 : (i4) -> i1
    %8 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %6, %7 after %8 : !hw.inout<i1>
    %c0_i2 = hw.constant 0 : i2
    %9 = llhd.sig.extract %out from %c0_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %10 = comb.extract %in from 0 : (i4) -> i1
    %11 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %9, %10 after %11 : !hw.inout<i1>
    %12 = llhd.prb %out : !hw.inout<i4>
    hw.output %12 : i4
  }
  hw.module @reverse(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %c-1_i2 = hw.constant -1 : i2
    %0 = llhd.sig.extract %out from %c-1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %1 = comb.extract %in from 0 : (i4) -> i1
    %2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %0, %1 after %2 : !hw.inout<i1>
    %c-2_i2 = hw.constant -2 : i2
    %3 = llhd.sig.extract %out from %c-2_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %4 = comb.extract %in from 1 : (i4) -> i1
    %5 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %3, %4 after %5 : !hw.inout<i1>
    %c1_i2 = hw.constant 1 : i2
    %6 = llhd.sig.extract %out from %c1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %7 = comb.extract %in from 2 : (i4) -> i1
    %8 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %6, %7 after %8 : !hw.inout<i1>
    %c0_i2 = hw.constant 0 : i2
    %9 = llhd.sig.extract %out from %c0_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %10 = comb.extract %in from 3 : (i4) -> i1
    %11 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %9, %10 after %11 : !hw.inout<i1>
    %12 = llhd.prb %out : !hw.inout<i4>
    hw.output %12 : i4
  }
  hw.module @mix_bit(in %in : i4, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %c-1_i2 = hw.constant -1 : i2
    %0 = llhd.sig.extract %out from %c-1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %1 = comb.extract %in from 0 : (i4) -> i1
    %2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %0, %1 after %2 : !hw.inout<i1>
    %c-2_i2 = hw.constant -2 : i2
    %3 = llhd.sig.extract %out from %c-2_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %4 = comb.extract %in from 3 : (i4) -> i1
    %5 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %3, %4 after %5 : !hw.inout<i1>
    %c1_i2 = hw.constant 1 : i2
    %6 = llhd.sig.extract %out from %c1_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %7 = comb.extract %in from 2 : (i4) -> i1
    %8 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %6, %7 after %8 : !hw.inout<i1>
    %c0_i2 = hw.constant 0 : i2
    %9 = llhd.sig.extract %out from %c0_i2 : (!hw.inout<i4>) -> !hw.inout<i1>
    %10 = comb.extract %in from 1 : (i4) -> i1
    %11 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %9, %10 after %11 : !hw.inout<i1>
    %12 = llhd.prb %out : !hw.inout<i4>
    hw.output %12 : i4
  }
  hw.module @mix_bit2(in %in : i8, out out : i8) {
    %c0_i8 = hw.constant 0 : i8
    %out = llhd.sig %c0_i8 : i8
    %c-1_i3 = hw.constant -1 : i3
    %0 = llhd.sig.extract %out from %c-1_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %1 = comb.extract %in from 7 : (i8) -> i1
    %2 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %0, %1 after %2 : !hw.inout<i1>
    %c-2_i3 = hw.constant -2 : i3
    %3 = llhd.sig.extract %out from %c-2_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %4 = comb.extract %in from 6 : (i8) -> i1
    %5 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %3, %4 after %5 : !hw.inout<i1>
    %c-3_i3 = hw.constant -3 : i3
    %6 = llhd.sig.extract %out from %c-3_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %7 = comb.extract %in from 4 : (i8) -> i1
    %8 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %6, %7 after %8 : !hw.inout<i1>
    %c-4_i3 = hw.constant -4 : i3
    %9 = llhd.sig.extract %out from %c-4_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %10 = comb.extract %in from 5 : (i8) -> i1
    %11 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %9, %10 after %11 : !hw.inout<i1>
    %c3_i3 = hw.constant 3 : i3
    %12 = llhd.sig.extract %out from %c3_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %13 = comb.extract %in from 0 : (i8) -> i1
    %14 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %12, %13 after %14 : !hw.inout<i1>
    %c2_i3 = hw.constant 2 : i3
    %15 = llhd.sig.extract %out from %c2_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %16 = comb.extract %in from 3 : (i8) -> i1
    %17 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %15, %16 after %17 : !hw.inout<i1>
    %c1_i3 = hw.constant 1 : i3
    %18 = llhd.sig.extract %out from %c1_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %19 = comb.extract %in from 2 : (i8) -> i1
    %20 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %18, %19 after %20 : !hw.inout<i1>
    %c0_i3 = hw.constant 0 : i3
    %21 = llhd.sig.extract %out from %c0_i3 : (!hw.inout<i8>) -> !hw.inout<i1>
    %22 = comb.extract %in from 1 : (i8) -> i1
    %23 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %21, %22 after %23 : !hw.inout<i1>
    %24 = llhd.prb %out : !hw.inout<i8>
    hw.output %24 : i8
  }
  hw.module @test_mux(in %a : i4, in %b : i4, in %sel : i1, out result : i4) {
    %c0_i4 = hw.constant 0 : i4
    %result = llhd.sig %c0_i4 : i4
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.and %0, %sel : i1
    %2 = comb.extract %b from 3 : (i4) -> i1
    %true = hw.constant true
    %3 = comb.xor %sel, %true : i1
    %4 = comb.and %2, %3 : i1
    %5 = comb.or %1, %4 : i1
    %6 = comb.and %a, %7 : i4
    %7 = comb.concat %sel, %sel, %sel, %sel : i1, i1, i1, i1
    %c-1_i4 = hw.constant -1 : i4
    %8 = comb.xor %9, %c-1_i4 : i4
    %9 = comb.concat %sel, %sel, %sel, %sel : i1, i1, i1, i1
    %10 = comb.and %b, %8 : i4
    %11 = comb.or %6, %10 : i4
    %12 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %result, %11 after %12 : !hw.inout<i4>
    %13 = comb.extract %a from 2 : (i4) -> i1
    %14 = comb.and %13, %sel : i1
    %15 = comb.extract %b from 2 : (i4) -> i1
    %16 = comb.and %15, %3 : i1
    %17 = comb.or %14, %16 : i1
    %18 = comb.extract %a from 1 : (i4) -> i1
    %19 = comb.and %18, %sel : i1
    %20 = comb.extract %b from 1 : (i4) -> i1
    %21 = comb.and %20, %3 : i1
    %22 = comb.or %19, %21 : i1
    %23 = comb.extract %a from 0 : (i4) -> i1
    %24 = comb.and %23, %sel : i1
    %25 = comb.extract %b from 0 : (i4) -> i1
    %26 = comb.and %25, %3 : i1
    %27 = comb.or %24, %26 : i1
    %28 = llhd.prb %result : !hw.inout<i4>
    hw.output %28 : i4
  }
  hw.module @test_and_enable(in %a : i4, in %enable : i1, out o : i4) {
    %c0_i4 = hw.constant 0 : i4
    %o = llhd.sig %c0_i4 : i4
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.and %0, %enable : i1
    %2 = comb.and %a, %3 : i4
    %3 = comb.concat %enable, %enable, %enable, %enable : i1, i1, i1, i1
    %4 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %o, %2 after %4 : !hw.inout<i4>
    %5 = comb.extract %a from 2 : (i4) -> i1
    %6 = comb.and %5, %enable : i1
    %7 = comb.extract %a from 1 : (i4) -> i1
    %8 = comb.and %7, %enable : i1
    %9 = comb.extract %a from 0 : (i4) -> i1
    %10 = comb.and %9, %enable : i1
    %11 = llhd.prb %o : !hw.inout<i4>
    hw.output %11 : i4
  }
  hw.module @test_multiple_patterns(in %a : i4, in %b : i4, in %c : i4, out out_xor : i4, out out_and : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out_xor = llhd.sig %c0_i4 : i4
    %c0_i4_0 = hw.constant 0 : i4
    %out_and = llhd.sig %c0_i4_0 : i4
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.extract %b from 3 : (i4) -> i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.xor %a, %b : i4
    %4 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out_xor, %3 after %4 : !hw.inout<i4>
    %5 = comb.extract %a from 2 : (i4) -> i1
    %6 = comb.extract %b from 2 : (i4) -> i1
    %7 = comb.xor %5, %6 : i1
    %8 = comb.extract %a from 1 : (i4) -> i1
    %9 = comb.extract %b from 1 : (i4) -> i1
    %10 = comb.xor %8, %9 : i1
    %11 = comb.extract %a from 0 : (i4) -> i1
    %12 = comb.extract %b from 0 : (i4) -> i1
    %13 = comb.xor %11, %12 : i1
    %14 = comb.extract %c from 3 : (i4) -> i1
    %15 = comb.and %0, %14 : i1
    %16 = comb.and %a, %c : i4
    %17 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out_and, %16 after %17 : !hw.inout<i4>
    %18 = comb.extract %c from 2 : (i4) -> i1
    %19 = comb.and %5, %18 : i1
    %20 = comb.extract %c from 1 : (i4) -> i1
    %21 = comb.and %8, %20 : i1
    %22 = comb.extract %c from 0 : (i4) -> i1
    %23 = comb.and %11, %22 : i1
    %24 = llhd.prb %out_xor : !hw.inout<i4>
    %25 = llhd.prb %out_and : !hw.inout<i4>
    hw.output %24, %25 : i4, i4
  }
  hw.module @test_add(in %a : i4, in %b : i4, out o : i4) {
    %c0_i4 = hw.constant 0 : i4
    %o = llhd.sig %c0_i4 : i4
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.extract %b from 3 : (i4) -> i1
    %2 = comb.add %0, %1 : i1
    %3 = comb.add %a, %b : i4
    %4 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %o, %3 after %4 : !hw.inout<i4>
    %5 = comb.extract %a from 2 : (i4) -> i1
    %6 = comb.extract %b from 2 : (i4) -> i1
    %7 = comb.add %5, %6 : i1
    %8 = comb.extract %a from 1 : (i4) -> i1
    %9 = comb.extract %b from 1 : (i4) -> i1
    %10 = comb.add %8, %9 : i1
    %11 = comb.extract %a from 0 : (i4) -> i1
    %12 = comb.extract %b from 0 : (i4) -> i1
    %13 = comb.add %11, %12 : i1
    %14 = llhd.prb %o : !hw.inout<i4>
    hw.output %14 : i4
  }
  hw.module @CustomLogic(in %a : i8, in %b : i8, out out : i8) {
    %c0_i8 = hw.constant 0 : i8
    %out = llhd.sig %c0_i8 : i8
    %0 = comb.extract %a from 0 : (i8) -> i1
    %1 = comb.extract %b from 0 : (i8) -> i1
    %2 = comb.and %0, %1 : i1
    %true = hw.constant true
    %3 = comb.xor %0, %true : i1
    %4 = comb.or %2, %3 : i1
    %5 = comb.extract %a from 1 : (i8) -> i1
    %6 = comb.extract %b from 1 : (i8) -> i1
    %7 = comb.and %5, %6 : i1
    %true_0 = hw.constant true
    %8 = comb.xor %5, %true_0 : i1
    %9 = comb.or %7, %8 : i1
    %10 = comb.extract %a from 2 : (i8) -> i1
    %11 = comb.extract %b from 2 : (i8) -> i1
    %12 = comb.and %10, %11 : i1
    %true_1 = hw.constant true
    %13 = comb.xor %10, %true_1 : i1
    %14 = comb.or %12, %13 : i1
    %15 = comb.extract %a from 3 : (i8) -> i1
    %16 = comb.extract %b from 3 : (i8) -> i1
    %17 = comb.and %15, %16 : i1
    %true_2 = hw.constant true
    %18 = comb.xor %15, %true_2 : i1
    %19 = comb.or %17, %18 : i1
    %20 = comb.extract %a from 4 : (i8) -> i1
    %21 = comb.extract %b from 4 : (i8) -> i1
    %22 = comb.and %20, %21 : i1
    %true_3 = hw.constant true
    %23 = comb.xor %20, %true_3 : i1
    %24 = comb.or %22, %23 : i1
    %25 = comb.extract %a from 5 : (i8) -> i1
    %26 = comb.extract %b from 5 : (i8) -> i1
    %27 = comb.and %25, %26 : i1
    %true_4 = hw.constant true
    %28 = comb.xor %25, %true_4 : i1
    %29 = comb.or %27, %28 : i1
    %30 = comb.extract %a from 6 : (i8) -> i1
    %31 = comb.extract %b from 6 : (i8) -> i1
    %32 = comb.and %30, %31 : i1
    %true_5 = hw.constant true
    %33 = comb.xor %30, %true_5 : i1
    %34 = comb.or %32, %33 : i1
    %35 = comb.extract %a from 7 : (i8) -> i1
    %36 = comb.extract %b from 7 : (i8) -> i1
    %37 = comb.and %35, %36 : i1
    %true_6 = hw.constant true
    %38 = comb.xor %35, %true_6 : i1
    %39 = comb.or %37, %38 : i1
    %40 = comb.and %a, %b : i8
    %c-1_i8 = hw.constant -1 : i8
    %41 = comb.xor %a, %c-1_i8 : i8
    %42 = comb.or %40, %41 : i8
    %43 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out, %42 after %43 : !hw.inout<i8>
    %44 = llhd.prb %out : !hw.inout<i8>
    hw.output %44 : i8
  }
  hw.module @GatedXOR(in %a : i4, in %b : i4, in %enable : i1, out out : i4) {
    %c0_i4 = hw.constant 0 : i4
    %out = llhd.sig %c0_i4 : i4
    %0 = comb.extract %a from 0 : (i4) -> i1
    %1 = comb.extract %b from 0 : (i4) -> i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.and %2, %enable : i1
    %4 = comb.extract %a from 1 : (i4) -> i1
    %5 = comb.extract %b from 1 : (i4) -> i1
    %6 = comb.xor %4, %5 : i1
    %7 = comb.and %6, %enable : i1
    %8 = comb.extract %a from 2 : (i4) -> i1
    %9 = comb.extract %b from 2 : (i4) -> i1
    %10 = comb.xor %8, %9 : i1
    %11 = comb.and %10, %enable : i1
    %12 = comb.extract %a from 3 : (i4) -> i1
    %13 = comb.extract %b from 3 : (i4) -> i1
    %14 = comb.xor %12, %13 : i1
    %15 = comb.and %14, %enable : i1
    %16 = comb.xor %a, %b : i4
    %17 = comb.and %16, %18 : i4
    %18 = comb.concat %enable, %enable, %enable, %enable : i1, i1, i1, i1
    %19 = llhd.constant_time <0ns, 0d, 1e>
    llhd.drv %out, %17 after %19 : !hw.inout<i4>
    %20 = llhd.prb %out : !hw.inout<i4>
    hw.output %20 : i4
  }
}

