module {
  hw.module @simple_vectorization(in %in : i4, out out : i4) {
    hw.output %in : i4
  }
  hw.module @reverse_endianess_vectorization(in %in : i4, out out : i4) {
    %0 = comb.reverse %in : i4
    hw.output %0 : i4
  }
  hw.module @bit_mixing_vectorization(in %in2 : i4, in %in : i8, out out2 : i4, out out : i8) {
    %0 = comb.extract %in2 from 1 : (i4) -> i3
    %1 = comb.extract %in2 from 0 : (i4) -> i1
    %2 = comb.concat %1, %0 : i1, i3
    %3 = comb.extract %in from 1 : (i8) -> i3
    %4 = comb.extract %in from 0 : (i8) -> i1
    %5 = comb.extract %in from 5 : (i8) -> i1
    %6 = comb.extract %in from 4 : (i8) -> i1
    %7 = comb.extract %in from 6 : (i8) -> i2
    %8 = comb.concat %7, %6, %5, %4, %3 : i2, i1, i1, i1, i3
    hw.output %2, %8 : i4, i8
  }
  hw.module @linear_and_reverse(in %in : i8, in %in2 : i4, out out : i8, out out2 : i4) {
    %0 = comb.reverse %in2 : i4
    hw.output %in, %0 : i8, i4
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
  hw.module @test_add(in %a : i4, in %b : i4, out o : i4) {
    %0 = comb.add %a, %b : i4
    hw.output %0 : i4
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
  hw.module @bit_drop(in %in : i4, out out : i4) {
    %c0_i3 = hw.constant 0 : i3
    %c7_i4 = hw.constant 7 : i4
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %0 = comb.concat %c0_i2, %8, %false : i2, i1, i1
    %1 = comb.concat %false, %7, %c0_i2 : i1, i1, i2
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c7_i4 : i4
    %4 = comb.concat %6, %c0_i3 : i1, i3
    %5 = comb.or %4, %3 : i4
    %6 = comb.extract %in from 3 : (i4) -> i1
    %7 = comb.extract %in from 2 : (i4) -> i1
    %8 = comb.extract %in from 1 : (i4) -> i1
    hw.output %5 : i4
  }
  hw.module @bit_duplicate(in %in : i4, out out : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %11 : i3, i1
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
    %11 = comb.extract %in from 0 : (i4) -> i1
    hw.output %8 : i4
  }
  hw.module @mixed_sources(in %in1 : i4, in %in2 : i4, out out : i8) {
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.concat %c0_i4, %in2 : i4, i4
    %1 = comb.concat %in1, %c0_i4 : i4, i4
    %2 = comb.or %1, %0 : i8
    hw.output %2 : i8
  }
  hw.module @with_logic_gate(in %in : i4, out out : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %13 : i3, i1
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
    %13 = comb.xor %11, %12 : i1
    hw.output %8 : i4
  }
}

