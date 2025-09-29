module {
  hw.module @bsg_mesh_stitch_width_p130_x_max_p2_y_max_p1(in %outs_i : i1040, in %hor_i : i260, in %ver_i : i520, out ins_o : i1040, out hor_o : i260, out ver_o : i520) {
    %0 = comb.extract %ver_i from 390 : (i520) -> i130
    %1 = comb.extract %ver_i from 130 : (i520) -> i130
    %2 = comb.extract %hor_i from 130 : (i260) -> i130
    %3 = comb.extract %outs_i from 130 : (i1040) -> i130
    %4 = comb.extract %ver_i from 260 : (i520) -> i130
    %5 = comb.extract %ver_i from 0 : (i520) -> i130
    %6 = comb.extract %outs_i from 520 : (i1040) -> i130
    %7 = comb.extract %hor_i from 0 : (i260) -> i130
    %8 = comb.concat %0, %1, %2, %3, %4, %5, %6, %7 : i130, i130, i130, i130, i130, i130, i130, i130
    %9 = comb.extract %outs_i from 0 : (i1040) -> i130
    %10 = comb.extract %outs_i from 650 : (i1040) -> i130
    %11 = comb.concat %10, %9 : i130, i130
    %12 = comb.extract %outs_i from 260 : (i1040) -> i130
    %13 = comb.extract %outs_i from 780 : (i1040) -> i130
    %14 = comb.extract %outs_i from 390 : (i1040) -> i130
    %15 = comb.extract %outs_i from 910 : (i1040) -> i130
    %16 = comb.concat %15, %14, %13, %12 : i130, i130, i130, i130
    hw.output %8, %11, %16 : i1040, i260, i520
  }
}

