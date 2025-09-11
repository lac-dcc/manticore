module {
  hw.module @simple_vectorization(in %in : i4, out out : i4) {
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
  hw.module @reverse_endianess_vectorization(in %in : i4, out out : i4) {
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
  hw.module @bit_mixing_vectorization(in %in2 : i4, in %in : i8, out out2 : i4, out out : i8) {
    %c0_i4 = hw.constant 0 : i4
    %c0_i5 = hw.constant 0 : i5
    %c0_i6 = hw.constant 0 : i6
    %c0_i7 = hw.constant 0 : i7
    %c0_i2 = hw.constant 0 : i2
    %false = hw.constant false
    %c127_i8 = hw.constant 127 : i8
    %c-65_i8 = hw.constant -65 : i8
    %c-33_i8 = hw.constant -33 : i8
    %c-17_i8 = hw.constant -17 : i8
    %c-9_i8 = hw.constant -9 : i8
    %c-5_i8 = hw.constant -5 : i8
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c0_i3 = hw.constant 0 : i3
    %0 = comb.concat %c0_i3, %33 : i3, i1
    %1 = comb.concat %c0_i2, %31, %false : i2, i1, i1
    %2 = comb.or %1, %0 : i4
    %3 = comb.and %2, %c-5_i4 : i4
    %4 = comb.concat %false, %32, %c0_i2 : i1, i1, i2
    %5 = comb.or %4, %3 : i4
    %6 = comb.and %5, %c7_i4 : i4
    %7 = comb.concat %30, %c0_i3 : i1, i3
    %8 = comb.or %7, %6 : i4
    %9 = comb.concat %c0_i7, %41 : i7, i1
    %10 = comb.concat %c0_i6, %40, %false : i6, i1, i1
    %11 = comb.or %10, %9 : i8
    %12 = comb.and %11, %c-5_i8 : i8
    %13 = comb.concat %c0_i5, %39, %c0_i2 : i5, i1, i2
    %14 = comb.or %13, %12 : i8
    %15 = comb.and %14, %c-9_i8 : i8
    %16 = comb.concat %c0_i4, %38, %c0_i3 : i4, i1, i3
    %17 = comb.or %16, %15 : i8
    %18 = comb.and %17, %c-17_i8 : i8
    %19 = comb.concat %c0_i3, %37, %c0_i4 : i3, i1, i4
    %20 = comb.or %19, %18 : i8
    %21 = comb.and %20, %c-33_i8 : i8
    %22 = comb.concat %c0_i2, %36, %c0_i5 : i2, i1, i5
    %23 = comb.or %22, %21 : i8
    %24 = comb.and %23, %c-65_i8 : i8
    %25 = comb.concat %false, %35, %c0_i6 : i1, i1, i6
    %26 = comb.or %25, %24 : i8
    %27 = comb.and %26, %c127_i8 : i8
    %28 = comb.concat %34, %c0_i7 : i1, i7
    %29 = comb.or %28, %27 : i8
    %30 = comb.extract %in2 from 0 : (i4) -> i1
    %31 = comb.extract %in2 from 2 : (i4) -> i1
    %32 = comb.extract %in2 from 3 : (i4) -> i1
    %33 = comb.extract %in2 from 1 : (i4) -> i1
    %34 = comb.extract %in from 7 : (i8) -> i1
    %35 = comb.extract %in from 6 : (i8) -> i1
    %36 = comb.extract %in from 4 : (i8) -> i1
    %37 = comb.extract %in from 5 : (i8) -> i1
    %38 = comb.extract %in from 0 : (i8) -> i1
    %39 = comb.extract %in from 3 : (i8) -> i1
    %40 = comb.extract %in from 2 : (i8) -> i1
    %41 = comb.extract %in from 1 : (i8) -> i1
    hw.output %8, %29 : i4, i8
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
  hw.module @linear_and_reverse(in %in : i8, in %in2 : i4, out out : i8, out out2 : i4) {
    %c0_i4 = hw.constant 0 : i4
    %c0_i3 = hw.constant 0 : i3
    %c0_i5 = hw.constant 0 : i5
    %c0_i2 = hw.constant 0 : i2
    %c0_i6 = hw.constant 0 : i6
    %false = hw.constant false
    %c7_i4 = hw.constant 7 : i4
    %c-5_i4 = hw.constant -5 : i4
    %c127_i8 = hw.constant 127 : i8
    %c-65_i8 = hw.constant -65 : i8
    %c-33_i8 = hw.constant -33 : i8
    %c-17_i8 = hw.constant -17 : i8
    %c-9_i8 = hw.constant -9 : i8
    %c-5_i8 = hw.constant -5 : i8
    %c0_i7 = hw.constant 0 : i7
    %0 = comb.concat %c0_i7, %37 : i7, i1
    %1 = comb.concat %c0_i6, %36, %false : i6, i1, i1
    %2 = comb.or %1, %0 : i8
    %3 = comb.and %2, %c-5_i8 : i8
    %4 = comb.concat %c0_i5, %35, %c0_i2 : i5, i1, i2
    %5 = comb.or %4, %3 : i8
    %6 = comb.and %5, %c-9_i8 : i8
    %7 = comb.concat %c0_i4, %34, %c0_i3 : i4, i1, i3
    %8 = comb.or %7, %6 : i8
    %9 = comb.and %8, %c-17_i8 : i8
    %10 = comb.concat %c0_i3, %33, %c0_i4 : i3, i1, i4
    %11 = comb.or %10, %9 : i8
    %12 = comb.and %11, %c-33_i8 : i8
    %13 = comb.concat %c0_i2, %32, %c0_i5 : i2, i1, i5
    %14 = comb.or %13, %12 : i8
    %15 = comb.and %14, %c-65_i8 : i8
    %16 = comb.concat %false, %31, %c0_i6 : i1, i1, i6
    %17 = comb.or %16, %15 : i8
    %18 = comb.and %17, %c127_i8 : i8
    %19 = comb.concat %30, %c0_i7 : i1, i7
    %20 = comb.or %19, %18 : i8
    %21 = comb.concat %c0_i3, %41 : i3, i1
    %22 = comb.concat %c0_i2, %40, %false : i2, i1, i1
    %23 = comb.or %22, %21 : i4
    %24 = comb.and %23, %c-5_i4 : i4
    %25 = comb.concat %false, %39, %c0_i2 : i1, i1, i2
    %26 = comb.or %25, %24 : i4
    %27 = comb.and %26, %c7_i4 : i4
    %28 = comb.concat %38, %c0_i3 : i1, i3
    %29 = comb.or %28, %27 : i4
    %30 = comb.extract %in from 7 : (i8) -> i1
    %31 = comb.extract %in from 6 : (i8) -> i1
    %32 = comb.extract %in from 5 : (i8) -> i1
    %33 = comb.extract %in from 4 : (i8) -> i1
    %34 = comb.extract %in from 3 : (i8) -> i1
    %35 = comb.extract %in from 2 : (i8) -> i1
    %36 = comb.extract %in from 1 : (i8) -> i1
    %37 = comb.extract %in from 0 : (i8) -> i1
    %38 = comb.extract %in2 from 0 : (i4) -> i1
    %39 = comb.extract %in2 from 1 : (i4) -> i1
    %40 = comb.extract %in2 from 2 : (i4) -> i1
    %41 = comb.extract %in2 from 3 : (i4) -> i1
    hw.output %20, %29 : i8, i4
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
