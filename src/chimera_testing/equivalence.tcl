
check_sec -compile_context spec
analyze -sv /home/ullas/manticore/manticore/designs/non-vectorized/11111_OpenABC_leaf_level_verilog_gf12_bp_quad_bsg_concentrate_static_1b.v
elaborate        

check_sec -compile_context imp
analyze -sv /home/ullas/manticore/manticore/designs/real-vectorized/11111_OpenABC_leaf_level_verilog_gf12_bp_quad_bsg_concentrate_static_1b.v
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