module {
  hw.module @bsg_mesh_stitch_width_p130_x_max_p2_y_max_p2(in %outs_i : i2080, in %hor_i : i520, in %ver_i : i520, out ins_o : i2080, out hor_o : i520, out ver_o : i520) {
    %0 = comb.extract %hor_i from 0 : (i520) -> i130
    %1 = comb.reverse %0 : i130
    %2 = comb.extract %outs_i from 520 : (i2080) -> i130
    %3 = comb.reverse %2 : i130
    %4 = comb.extract %ver_i from 0 : (i520) -> i130
    %5 = comb.reverse %4 : i130
    %6 = comb.extract %outs_i from 1300 : (i2080) -> i130
    %7 = comb.reverse %6 : i130
    %8 = comb.extract %outs_i from 130 : (i2080) -> i130
    %9 = comb.reverse %8 : i130
    %10 = comb.extract %hor_i from 260 : (i520) -> i130
    %11 = comb.reverse %10 : i130
    %12 = comb.extract %ver_i from 130 : (i520) -> i130
    %13 = comb.reverse %12 : i130
    %14 = comb.extract %outs_i from 1820 : (i2080) -> i130
    %15 = comb.reverse %14 : i130
    %16 = comb.extract %hor_i from 130 : (i520) -> i130
    %17 = comb.reverse %16 : i130
    %18 = comb.extract %outs_i from 1560 : (i2080) -> i130
    %19 = comb.reverse %18 : i130
    %20 = comb.extract %outs_i from 390 : (i2080) -> i130
    %21 = comb.reverse %20 : i130
    %22 = comb.extract %ver_i from 260 : (i520) -> i130
    %23 = comb.reverse %22 : i130
    %24 = comb.extract %outs_i from 1170 : (i2080) -> i130
    %25 = comb.reverse %24 : i130
    %26 = comb.extract %hor_i from 390 : (i520) -> i130
    %27 = comb.reverse %26 : i130
    %28 = comb.extract %outs_i from 910 : (i2080) -> i130
    %29 = comb.reverse %28 : i130
    %30 = comb.extract %ver_i from 390 : (i520) -> i130
    %31 = comb.reverse %30 : i130
    %32 = comb.concat %1, %3, %5, %7, %9, %11, %13, %15, %17, %19, %21, %23, %25, %27, %29, %31 : i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130, i130
    %33 = comb.extract %outs_i from 0 : (i2080) -> i130
    %34 = comb.reverse %33 : i130
    %35 = comb.extract %outs_i from 1040 : (i2080) -> i130
    %36 = comb.reverse %35 : i130
    %37 = comb.extract %outs_i from 650 : (i2080) -> i130
    %38 = comb.reverse %37 : i130
    %39 = comb.extract %outs_i from 1690 : (i2080) -> i130
    %40 = comb.reverse %39 : i130
    %41 = comb.concat %34, %36, %38, %40 : i130, i130, i130, i130
    %42 = comb.extract %outs_i from 260 : (i2080) -> i130
    %43 = comb.reverse %42 : i130
    %44 = comb.extract %outs_i from 780 : (i2080) -> i130
    %45 = comb.reverse %44 : i130
    %46 = comb.extract %outs_i from 1430 : (i2080) -> i130
    %47 = comb.reverse %46 : i130
    %48 = comb.extract %outs_i from 1950 : (i2080) -> i130
    %49 = comb.reverse %48 : i130
    %50 = comb.concat %43, %45, %47, %49 : i130, i130, i130, i130
    hw.output %32, %41, %50 : i2080, i520, i520
  }
}

