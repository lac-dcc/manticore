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
  hw.module @mix_bit2(in %in : i8, out out : i8) {
    %0 = comb.extract %in from 1 : (i8) -> i3
    %1 = comb.extract %in from 0 : (i8) -> i1
    %2 = comb.extract %in from 5 : (i8) -> i1
    %3 = comb.extract %in from 4 : (i8) -> i1
    %4 = comb.extract %in from 6 : (i8) -> i2
    %5 = comb.concat %4, %3, %2, %1, %0 : i2, i1, i1, i1, i3
    hw.output %5 : i8
  }
  hw.module @linear_and_reverse(in %in : i8, in %in2 : i4, out out : i8, out out2 : i4) {
    %0 = comb.reverse %in2 : i4
    hw.output %in, %0 : i8, i4
  }
  hw.module @bit_drop(in %in : i4, out out : i4) {
    hw.output %in : i4
  }
  hw.module @bit_duplicate(in %in : i4, out out : i4) {
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %11, %false : i1, i1
    %1 = comb.concat %false, %11 : i1, i1
    %2 = comb.or %0, %1 : i2
    %3 = comb.concat %10, %c0_i2 : i1, i2
    %4 = comb.concat %false, %2 : i1, i2
    %5 = comb.or %3, %4 : i3
    %6 = comb.concat %false, %5 : i1, i3
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
    hw.output %in : i4
  }
}

