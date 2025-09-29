#!/bin/bash

source ./bash-scripts/utilities/run-vectorization.sh

function count_design_instructions {
  shopt -s nullglob
  run_vectorization $1  ""

  before_pass_line_count=$(grep -Ev '^[[:space:]]*$|^[[:space:]]*[{}][[:space:]]*$|^[[:space:]]*//' design.hw.mlir | wc -l)
  after_pass_line_count=$(grep -Ev '^[[:space:]]*$|^[[:space:]]*[{}][[:space:]]*$|^[[:space:]]*//' design.cleaned.mlir | wc -l)

  echo "$before_pass_line_count,$after_pass_line_count"
}

function count_instructions {
  designs_dir=$1

  touch instruction_count.csv

  echo ",design,inst-count-vectorized,inst-count-non-vectorized" > instruction_count.csv

  shopt -s nullglob

  i=0
  for design in $designs_dir/*.sv $designs_dir/*.v; do
    echo $design
    line="$i,$(basename $design),$(echo $(count_design_instructions $design))" 
    echo $line >> instruction_count.csv
    ((i++))
  done
}

count_instructions $1






