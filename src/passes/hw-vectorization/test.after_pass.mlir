module {
  hw.module @bsg_mesh_stitch_width_p130_x_max_p2_y_max_p2(in %outs_i : i2080, in %hor_i : i520, in %ver_i : i520, out ins_o : i2080, out hor_o : i520, out ver_o : i520) {
    %0 = comb.extract %ver_i from 390 : (i520) -> i130
    %1 = comb.extract %outs_i from 910 : (i2080) -> i130
    %2 = comb.extract %hor_i from 390 : (i520) -> i130
    %3 = comb.extract %outs_i from 1170 : (i2080) -> i130
    %4 = comb.extract %ver_i from 260 : (i520) -> i130
    %5 = comb.extract %outs_i from 390 : (i2080) -> i130
    %6 = comb.extract %outs_i from 1560 : (i2080) -> i130
    %7 = comb.extract %hor_i from 130 : (i520) -> i130
    %8 = comb.extract %outs_i from 1820 : (i2080) -> i130
    %9 = comb.extract %ver_i from 130 : (i520) -> i130
    %10 = comb.extract %hor_i from 260 : (i520) -> i130
    %11 = comb.extract %outs_i from 130 : (i2080) -> i130
    %12 = comb.extract %outs_i from 1300 : (i2080) -> i130
    %13 = comb.extract %ver_i from 0 : (i520) -> i130
    %14 = comb.extract %outs_i from 520 : (i2080) -> i130
    %15 = comb.extract %hor_i from 0 : (i520) -> i130
    %16 = comb.concat %0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15 : i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130
    %17 = comb.extract %outs_i from 1690 : (i2080) -> i130
    %18 = comb.extract %outs_i from 650 : (i2080) -> i130
    %19 = comb.extract %outs_i from 1040 : (i2080) -> i130
    %20 = comb.extract %outs_i from 0 : (i2080) -> i130
    %21 = comb.concat %17, %18, %19, %20 : i130, i130, i130, i130
    %22 = comb.extract %outs_i from 1950 : (i2080) -> i130
    %23 = comb.extract %outs_i from 1430 : (i2080) -> i130
    %24 = comb.extract %outs_i from 780 : (i2080) -> i130
    %25 = comb.extract %outs_i from 260 : (i2080) -> i130
    %26 = comb.concat %22, %23, %24, %25 : i130, i130, i130, i130
    hw.output %16, %21, %26 : i2080, i520, i520
  }
}

