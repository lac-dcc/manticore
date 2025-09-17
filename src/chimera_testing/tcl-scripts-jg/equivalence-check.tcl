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
    

if {[catch {check_return {check_sec -prove} proven} err]} {
    exit 1
} else {
    exit 0
}


