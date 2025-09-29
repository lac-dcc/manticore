#!/bin/bash

function run_vectorization {
  file=$1
  output_dir_path=$2
  file_base_name=$(basename $1)

  circt-verilog --ir-hw $file -o design.hw.mlir

  circt-opt design.hw.mlir \
    --pass-pipeline="builtin.module(simple-vec)" \
    --load-pass-plugin=./../passes/hw-vectorization/build/VectorizePass.so \
    -o design.after_pass.mlir


  circt-opt \
    --llhd-desequentialize \
    --llhd-hoist-signals \
    --llhd-sig2reg \
    --llhd-mem2reg \
    --llhd-process-lowering \
    --cse \
    --canonicalize \
    --hw-cleanup    \
    design.after_pass.mlir -o design.cleaned.mlir


  if [[ ! -z "${output_dir_path-}" ]]; then
    cp $file "$output_dir_path/non-vectorized"
    firtool design.cleaned.mlir --verilog -o "$output_dir_path/vectorized/$file_base_name"
  fi

}

function vectorize_design_collection {
  design_collection_path=$1
  output_dir_path=$2

  mkdir "$output_dir_path/non-vectorized"
  mkdir "$output_dir_path/vectorized"
  
  touch artificial-vectorization.csv
  echo ",design,pass-execution-time" > artificial-vectorization.csv

  shopt -s nullglob
  for design in $design_collection_path/*.sv $design_collection_path/*.v; do
    start=$(date +%s%3N)                   # epoch em ms
    run_vectorization "$design" "$output_dir_path"
    end=$(date +%s%3N)
    vec_time_ms=$(( end - start ))

    echo "$(basename $design),$vec_time_ms"
    echo "$(basename $design),$vec_time_ms" >> artificial-vectorization.csv
  done
}

vectorize_design_collection $1 $2
