
check_sec -compile_context spec
analyze -sv first.v


elaborate        

check_sec -compile_context imp
analyze -sv second.v
elaborate

clock -none
reset -none
check_sec -setup
check_sec -generate
    
set result [check_sec -prove]
puts "PROOF_RES: $result"

exit 0
