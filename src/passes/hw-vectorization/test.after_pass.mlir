module {
  hw.module @simple_vectorization(in %in : i4, out out : i4) {
    hw.output %in : i4
  }
  hw.module @reverse_endianess_vectorization(in %in : i4, out out : i4) {
    %0 = comb.reverse %in : i4
    hw.output %0 : i4
  }
  hw.module @linear_and_reverse(in %in : i8, in %in2 : i4, out out : i8, out out2 : i4) {
    %0 = comb.reverse %in2 : i4
    hw.output %in, %0 : i8, i4
  }
  hw.module @bit_mixing_vectorization(in %in2 : i4, in %in : i8, out out2 : i4, out out : i8) {
    %0 = comb.extract %in2 from 0 : (i4) -> i1
    %1 = comb.extract %in2 from 1 : (i4) -> i3
    %2 = comb.concat %0, %1 : i1, i3
    %3 = comb.extract %in from 6 : (i8) -> i2
    %4 = comb.extract %in from 4 : (i8) -> i1
    %5 = comb.extract %in from 5 : (i8) -> i1
    %6 = comb.extract %in from 0 : (i8) -> i1
    %7 = comb.extract %in from 1 : (i8) -> i3
    %8 = comb.concat %3, %4, %5, %6, %7 : i2, i1, i1, i1, i3
    hw.output %2, %8 : i4, i8
  }
  hw.module @test_mux(in %a : i4, in %b : i4, in %sel : i1, out result : i4) {
    %true = hw.constant true
    %0 = comb.replicate %sel : (i1) -> i4
    %1 = comb.and %a, %0 : i4
    %2 = comb.replicate %sel : (i1) -> i4
    %3 = comb.replicate %true : (i1) -> i4
    %4 = comb.xor %2, %3 : i4
    %5 = comb.and %b, %4 : i4
    %6 = comb.or %1, %5 : i4
    hw.output %6 : i4
  }
  hw.module @test_and_enable(in %a : i4, in %enable : i1, out o : i4) {
    %0 = comb.replicate %enable : (i1) -> i4
    %1 = comb.and %a, %0 : i4
    hw.output %1 : i4
  }
  hw.module @test_multiple_patterns(in %a : i4, in %b : i4, in %c : i4, out out_xor : i4, out out_and : i4) {
    %0 = comb.xor %a, %b : i4
    %1 = comb.and %a, %c : i4
    hw.output %0, %1 : i4, i4
  }
  hw.module @test_multiple_patterns_reverse(in %a : i4, in %b : i4, out out_xor : i4) {
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.extract %b from 0 : (i4) -> i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.extract %a from 2 : (i4) -> i1
    %4 = comb.extract %b from 1 : (i4) -> i1
    %5 = comb.xor %3, %4 : i1
    %6 = comb.extract %a from 1 : (i4) -> i1
    %7 = comb.extract %b from 2 : (i4) -> i1
    %8 = comb.xor %6, %7 : i1
    %9 = comb.extract %a from 0 : (i4) -> i1
    %10 = comb.extract %b from 3 : (i4) -> i1
    %11 = comb.xor %9, %10 : i1
    %12 = comb.concat %2, %5, %8, %11 : i1, i1, i1, i1
    hw.output %12 : i4
  }
  hw.module @CustomLogic(in %a : i8, in %b : i8, out out : i8) {
    %true = hw.constant true
    %0 = comb.and %a, %b : i8
    %1 = comb.replicate %true : (i1) -> i8
    %2 = comb.xor %a, %1 : i8
    %3 = comb.or %0, %2 : i8
    hw.output %3 : i8
  }
  hw.module @GatedXOR(in %a : i4, in %b : i4, in %enable : i1, out out : i4) {
    %0 = comb.xor %a, %b : i4
    %1 = comb.replicate %enable : (i1) -> i4
    %2 = comb.and %0, %1 : i4
    hw.output %2 : i4
  }
  hw.module @with_logic_gate(in %in : i4, out out : i4) {
    %0 = comb.extract %in from 1 : (i4) -> i1
    %1 = comb.extract %in from 0 : (i4) -> i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.extract %in from 1 : (i4) -> i3
    %4 = comb.concat %3, %2 : i3, i1
    hw.output %4 : i4
  }
  hw.module @bit_duplicate(in %in : i4, out out : i4) {
    %0 = comb.extract %in from 2 : (i4) -> i2
    %1 = comb.extract %in from 0 : (i4) -> i1
    %2 = comb.extract %in from 0 : (i4) -> i1
    %3 = comb.concat %0, %1, %2 : i2, i1, i1
    hw.output %3 : i4
  }
  hw.module @ShuffledXOR(in %a : i4, in %b : i4, out out : i4) {
    %0 = comb.xor %a, %b {sv.namehint = "temp"} : i4
    %1 = comb.extract %0 from 0 : (i4) -> i1
    %2 = comb.extract %0 from 2 : (i4) -> i1
    %3 = comb.extract %0 from 3 : (i4) -> i1
    %4 = comb.extract %0 from 1 : (i4) -> i1
    %5 = comb.concat %1, %2, %3, %4 : i1, i1, i1, i1
    hw.output %5 : i4
  }
  hw.module @VectorizedEnable(in %a : i4, in %enable : i4, out o : i4) {
    %0 = comb.and %a, %enable : i4
    hw.output %0 : i4
  }
  hw.module @mixed_sources(in %in1 : i4, in %in2 : i4, out out : i8) {
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.concat %c0_i4, %in2 : i4, i4
    %1 = comb.concat %in1, %c0_i4 : i4, i4
    %2 = comb.or %1, %0 : i8
    hw.output %2 : i8
  }
  hw.module @InconsistentLogic(in %a : i4, in %b : i4, out out : i4) {
    %true = hw.constant true
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.extract %b from 3 : (i4) -> i1
    %2 = comb.and %0, %1 : i1
    %3 = comb.extract %a from 2 : (i4) -> i1
    %4 = comb.extract %b from 2 : (i4) -> i1
    %5 = comb.or %3, %4 : i1
    %6 = comb.extract %a from 1 : (i4) -> i1
    %7 = comb.extract %b from 1 : (i4) -> i1
    %8 = comb.xor %6, %7 : i1
    %9 = comb.extract %a from 0 : (i4) -> i1
    %10 = comb.xor %9, %true : i1
    %11 = comb.concat %2, %5, %8, %10 : i1, i1, i1, i1
    hw.output %11 : i4
  }
  hw.module @ShiftAndXOR(in %a : i4, in %b : i4, out out : i4) {
    %0 = comb.extract %a from 3 : (i4) -> i1
    %1 = comb.extract %b from 2 : (i4) -> i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.extract %a from 2 : (i4) -> i1
    %4 = comb.extract %b from 1 : (i4) -> i1
    %5 = comb.xor %3, %4 : i1
    %6 = comb.extract %a from 1 : (i4) -> i1
    %7 = comb.extract %b from 0 : (i4) -> i1
    %8 = comb.xor %6, %7 : i1
    %9 = comb.extract %a from 0 : (i4) -> i1
    %10 = comb.concat %2, %5, %8, %9 : i1, i1, i1, i1
    hw.output %10 : i4
  }
  hw.module @intermodule_vectorization(in %in : i4, out out : i4) {
    hw.output %in : i4
  }
  hw.module private @mybuf(in %i : i1, out o : i1) {
    hw.output %i : i1
  }
  hw.module private @bit_swapper(in %i0 : i1, in %i1 : i1, out o0 : i1, out o1 : i1) {
    hw.output %i1, %i0 : i1, i1
  }
  hw.module @top_byte_swap(in %i : i8, out o : i8) {
    %0 = comb.extract %i from 0 : (i8) -> i4
    %1 = comb.extract %i from 4 : (i8) -> i4
    %2 = comb.concat %0, %1 : i4, i4
    hw.output %2 : i8
  }
  hw.module @CarryChainAdder(in %a : i4, in %b : i4, out sum : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %0 = comb.concat %17, %false : i1, i1
    %1 = comb.concat %false, %9 : i1, i1
    %2 = comb.or %0, %1 : i2
    %3 = comb.concat %25, %c0_i2 : i1, i2
    %4 = comb.concat %false, %2 : i1, i2
    %5 = comb.or %3, %4 : i3
    %6 = comb.extract %a from 0 : (i4) -> i1
    %7 = comb.extract %b from 0 : (i4) -> i1
    %8 = comb.xor %6, %7 : i1
    %9 = comb.and %6, %7 : i1
    %10 = comb.extract %a from 1 : (i4) -> i1
    %11 = comb.extract %b from 1 : (i4) -> i1
    %12 = comb.extract %5 from 0 : (i3) -> i1
    %13 = comb.xor %10, %11, %12 : i1
    %14 = comb.and %10, %11 : i1
    %15 = comb.and %10, %12 : i1
    %16 = comb.and %11, %12 : i1
    %17 = comb.or %14, %15, %16 : i1
    %18 = comb.extract %a from 2 : (i4) -> i1
    %19 = comb.extract %b from 2 : (i4) -> i1
    %20 = comb.extract %5 from 1 : (i3) -> i1
    %21 = comb.xor %18, %19, %20 : i1
    %22 = comb.and %18, %19 : i1
    %23 = comb.and %18, %20 : i1
    %24 = comb.and %19, %20 : i1
    %25 = comb.or %22, %23, %24 : i1
    %26 = comb.extract %a from 3 : (i4) -> i1
    %27 = comb.extract %b from 3 : (i4) -> i1
    %28 = comb.extract %5 from 2 : (i3) -> i1
    %29 = comb.xor %26, %27, %28 : i1
    %30 = comb.concat %29, %21, %13, %8 : i1, i1, i1, i1
    hw.output %30 : i4
  }
  hw.module @test_add(in %a : i4, in %b : i4, out o : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %20 : i3, i1
    %1 = comb.concat %c0_i2, %17, %false : i2, i1, i1
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c-5_i4 : i4
    %4 = comb.concat %false, %14, %c0_i2 : i1, i1, i2
    %5 = comb.or %4, %3 : i4
    %6 = comb.and %5, %c7_i4 : i4
    %7 = comb.concat %11, %c0_i3 : i1, i3
    %8 = comb.or %7, %6 : i4
    %9 = comb.extract %a from 3 : (i4) -> i1
    %10 = comb.extract %b from 3 : (i4) -> i1
    %11 = comb.add %9, %10 : i1
    %12 = comb.extract %a from 2 : (i4) -> i1
    %13 = comb.extract %b from 2 : (i4) -> i1
    %14 = comb.add %12, %13 : i1
    %15 = comb.extract %a from 1 : (i4) -> i1
    %16 = comb.extract %b from 1 : (i4) -> i1
    %17 = comb.add %15, %16 : i1
    %18 = comb.extract %a from 0 : (i4) -> i1
    %19 = comb.extract %b from 0 : (i4) -> i1
    %20 = comb.add %18, %19 : i1
    hw.output %8 : i4
  }
  hw.module @VectorizedSubtraction(in %a : i8, in %b : i8, out o : i8) {
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
    %0 = comb.concat %c0_i7, %44 : i7, i1
    %1 = comb.concat %c0_i6, %41, %false : i6, i1, i1
    %2 = comb.or %1, %0 : i8
    %3 = comb.and %2, %c-5_i8 : i8
    %4 = comb.concat %c0_i5, %38, %c0_i2 : i5, i1, i2
    %5 = comb.or %4, %3 : i8
    %6 = comb.and %5, %c-9_i8 : i8
    %7 = comb.concat %c0_i4, %35, %c0_i3 : i4, i1, i3
    %8 = comb.or %7, %6 : i8
    %9 = comb.and %8, %c-17_i8 : i8
    %10 = comb.concat %c0_i3, %32, %c0_i4 : i3, i1, i4
    %11 = comb.or %10, %9 : i8
    %12 = comb.and %11, %c-33_i8 : i8
    %13 = comb.concat %c0_i2, %29, %c0_i5 : i2, i1, i5
    %14 = comb.or %13, %12 : i8
    %15 = comb.and %14, %c-65_i8 : i8
    %16 = comb.concat %false, %26, %c0_i6 : i1, i1, i6
    %17 = comb.or %16, %15 : i8
    %18 = comb.and %17, %c127_i8 : i8
    %19 = comb.concat %23, %c0_i7 : i1, i7
    %20 = comb.or %19, %18 : i8
    %21 = comb.extract %a from 7 : (i8) -> i1
    %22 = comb.extract %b from 7 : (i8) -> i1
    %23 = comb.sub %21, %22 : i1
    %24 = comb.extract %a from 6 : (i8) -> i1
    %25 = comb.extract %b from 6 : (i8) -> i1
    %26 = comb.sub %24, %25 : i1
    %27 = comb.extract %a from 5 : (i8) -> i1
    %28 = comb.extract %b from 5 : (i8) -> i1
    %29 = comb.sub %27, %28 : i1
    %30 = comb.extract %a from 4 : (i8) -> i1
    %31 = comb.extract %b from 4 : (i8) -> i1
    %32 = comb.sub %30, %31 : i1
    %33 = comb.extract %a from 3 : (i8) -> i1
    %34 = comb.extract %b from 3 : (i8) -> i1
    %35 = comb.sub %33, %34 : i1
    %36 = comb.extract %a from 2 : (i8) -> i1
    %37 = comb.extract %b from 2 : (i8) -> i1
    %38 = comb.sub %36, %37 : i1
    %39 = comb.extract %a from 1 : (i8) -> i1
    %40 = comb.extract %b from 1 : (i8) -> i1
    %41 = comb.sub %39, %40 : i1
    %42 = comb.extract %a from 0 : (i8) -> i1
    %43 = comb.extract %b from 0 : (i8) -> i1
    %44 = comb.sub %42, %43 : i1
    hw.output %20 : i8
  }
  hw.module @raw_example(in %in : i4, out out : i4) {
    %0 = comb.extract %5 from 3 : (i4) -> i1
    %1 = comb.extract %in from 2 : (i4) -> i1
    %2 = comb.and %0, %1 : i1
    %3 = comb.extract %in from 3 : (i4) -> i1
    %4 = comb.extract %in from 0 : (i4) -> i2
    %5 = comb.concat %3, %2, %4 : i1, i1, i2
    hw.output %5 : i4
  }
  hw.module @cross_dependency(in %in : i2, out out : i2) {
    %0 = comb.extract %in from 0 : (i2) -> i1
    %1 = comb.extract %6 from 1 : (i2) -> i1
    %2 = comb.xor %0, %1 : i1
    %3 = comb.extract %in from 1 : (i2) -> i1
    %4 = comb.extract %6 from 0 : (i2) -> i1
    %5 = comb.xor %3, %4 : i1
    %6 = comb.concat %5, %2 : i1, i1
    hw.output %6 : i2
  }
}

