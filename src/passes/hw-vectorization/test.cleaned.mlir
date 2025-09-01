module {
  hw.module @linear(in %in : i4, out out : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %12 : i3, i1
    %1 = comb.concat %c0_i2, %11, %false : i2, i1, i1
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c-5_i4 : i4
    %4 = comb.concat %false, %10, %c0_i2 : i1, i1, i2
    %5 = comb.or %4, %3 : i4
    %6 = comb.and %5, %c7_i4 : i4
    %7 = comb.concat %9, %c0_i3 : i1, i3
    %8 = comb.or %7, %6 : i4
    %9 = comb.extract %in from 3 : (i4) -> i1
    %10 = comb.extract %in from 2 : (i4) -> i1
    %11 = comb.extract %in from 1 : (i4) -> i1
    %12 = comb.extract %in from 0 : (i4) -> i1
    hw.output %8 : i4
  }
  hw.module @reverse(in %in : i4, out out : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %12 : i3, i1
    %1 = comb.concat %c0_i2, %11, %false : i2, i1, i1
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c-5_i4 : i4
    %4 = comb.concat %false, %10, %c0_i2 : i1, i1, i2
    %5 = comb.or %4, %3 : i4
    %6 = comb.and %5, %c7_i4 : i4
    %7 = comb.concat %9, %c0_i3 : i1, i3
    %8 = comb.or %7, %6 : i4
    %9 = comb.extract %in from 0 : (i4) -> i1
    %10 = comb.extract %in from 1 : (i4) -> i1
    %11 = comb.extract %in from 2 : (i4) -> i1
    %12 = comb.extract %in from 3 : (i4) -> i1
    hw.output %8 : i4
  }
  hw.module @mix_bit(in %in : i4, out out : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %12 : i3, i1
    %1 = comb.concat %c0_i2, %11, %false : i2, i1, i1
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c-5_i4 : i4
    %4 = comb.concat %false, %10, %c0_i2 : i1, i1, i2
    %5 = comb.or %4, %3 : i4
    %6 = comb.and %5, %c7_i4 : i4
    %7 = comb.concat %9, %c0_i3 : i1, i3
    %8 = comb.or %7, %6 : i4
    %9 = comb.extract %in from 0 : (i4) -> i1
    %10 = comb.extract %in from 3 : (i4) -> i1
    %11 = comb.extract %in from 2 : (i4) -> i1
    %12 = comb.extract %in from 1 : (i4) -> i1
    hw.output %8 : i4
  }
  hw.module @mix_bit2(in %in : i8, out out : i8) {
    %c0_i4 = hw.constant 0 : i4
    %c0_i3 = hw.constant 0 : i3
    %c0_i5 = hw.constant 0 : i5
    %c0_i2 = hw.constant 0 : i2
    %c0_i6 = hw.constant 0 : i6
    %false = hw.constant false
    %c127_i8 = hw.constant 127 : i8
    %c-65_i8 = hw.constant -65 : i8
    %c-33_i8 = hw.constant -33 : i8
    %c-17_i8 = hw.constant -17 : i8
    %c-9_i8 = hw.constant -9 : i8
    %c-5_i8 = hw.constant -5 : i8
    %c0_i7 = hw.constant 0 : i7
    %0 = comb.concat %c0_i7, %28 : i7, i1
    %1 = comb.concat %c0_i6, %27, %false : i6, i1, i1
    %2 = comb.or %1, %0 : i8
    %3 = comb.and %2, %c-5_i8 : i8
    %4 = comb.concat %c0_i5, %26, %c0_i2 : i5, i1, i2
    %5 = comb.or %4, %3 : i8
    %6 = comb.and %5, %c-9_i8 : i8
    %7 = comb.concat %c0_i4, %25, %c0_i3 : i4, i1, i3
    %8 = comb.or %7, %6 : i8
    %9 = comb.and %8, %c-17_i8 : i8
    %10 = comb.concat %c0_i3, %24, %c0_i4 : i3, i1, i4
    %11 = comb.or %10, %9 : i8
    %12 = comb.and %11, %c-33_i8 : i8
    %13 = comb.concat %c0_i2, %23, %c0_i5 : i2, i1, i5
    %14 = comb.or %13, %12 : i8
    %15 = comb.and %14, %c-65_i8 : i8
    %16 = comb.concat %false, %22, %c0_i6 : i1, i1, i6
    %17 = comb.or %16, %15 : i8
    %18 = comb.and %17, %c127_i8 : i8
    %19 = comb.concat %21, %c0_i7 : i1, i7
    %20 = comb.or %19, %18 : i8
    %21 = comb.extract %in from 7 : (i8) -> i1
    %22 = comb.extract %in from 6 : (i8) -> i1
    %23 = comb.extract %in from 4 : (i8) -> i1
    %24 = comb.extract %in from 5 : (i8) -> i1
    %25 = comb.extract %in from 0 : (i8) -> i1
    %26 = comb.extract %in from 3 : (i8) -> i1
    %27 = comb.extract %in from 2 : (i8) -> i1
    %28 = comb.extract %in from 1 : (i8) -> i1
    hw.output %20 : i8
  }
  hw.module @test_mux(in %a : i4, in %b : i4, in %sel : i1, out result : i4) {
    %c-1_i4 = hw.constant -1 : i4
    %0 = comb.and %a, %1 : i4
    %1 = comb.replicate %sel : (i1) -> i4
    %2 = comb.xor %3, %c-1_i4 : i4
    %3 = comb.replicate %sel : (i1) -> i4
    %4 = comb.and %b, %2 : i4
    %5 = comb.or %0, %4 : i4
    hw.output %5 : i4
  }
  hw.module @test_and_enable(in %a : i4, in %enable : i1, out o : i4) {
    %0 = comb.and %a, %1 : i4
    %1 = comb.replicate %enable : (i1) -> i4
    hw.output %0 : i4
  }
  hw.module @test_multiple_patterns(in %a : i4, in %b : i4, in %c : i4, out out_xor : i4, out out_and : i4) {
    %0 = comb.xor %a, %b : i4
    %1 = comb.and %a, %c : i4
    hw.output %0, %1 : i4, i4
  }
  hw.module @test_add(in %a : i4, in %b : i4, out o : i4) {
    %0 = comb.add %a, %b : i4
    hw.output %0 : i4
  }
  hw.module @CustomLogic(in %a : i8, in %b : i8, out out : i8) {
    %c-1_i8 = hw.constant -1 : i8
    %0 = comb.and %a, %b : i8
    %1 = comb.xor %a, %c-1_i8 : i8
    %2 = comb.or %0, %1 : i8
    hw.output %2 : i8
  }
  hw.module @GatedXOR(in %a : i4, in %b : i4, in %enable : i1, out out : i4) {
    %0 = comb.xor %a, %b : i4
    %1 = comb.and %0, %2 : i4
    %2 = comb.replicate %enable : (i1) -> i4
    hw.output %1 : i4
  }
}

