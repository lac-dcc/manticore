check_sec -compile_context spec

analyze -sv ../../../designs/non-vectorized/22124_Verilog_codes_CodeConverters_bintogrey_bintogrey.v

elaborate

check_sec -compile_context imp

analyze -sv extra.sv

elaborate

clock -none

reset -none

check_sec -setup

check_sec -generate

check_return {check_sec -prove} proven
# check_return {check_sec -prove} proven

# exit 0
