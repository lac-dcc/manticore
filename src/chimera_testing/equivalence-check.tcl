check_sec -compile_context spec
analyze -sv ../../designs//real-vectorized/48233_verilog-ethernet_rtl_xgmii_deinterleave.v


elaborate        

check_sec -compile_context imp
analyze -sv ../../designs//non-vectorized/48233_verilog-ethernet_rtl_xgmii_deinterleave.v
elaborate

clock -none
reset -none
check_sec -setup
check_sec -generate
    
set result [check_sec -prove]
puts "PROOF_RES: $result"

exit 0
