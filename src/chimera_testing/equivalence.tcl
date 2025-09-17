
check_sec -compile_context spec
analyze -sv /home/ullas/manticore/manticore/designs/real-vectorized/11083_OpenABC_leaf_level_verilog_gf12_bp_quad_bp_me_cord_to_id_05.v
elaborate        

check_sec -compile_context imp
analyze -sv /home/ullas/manticore/manticore/designs/real-vectorized/11111_OpenABC_leaf_level_verilog_gf12_bp_quad_bsg_concentrate_static_1b.v
elaborate

clock -none
reset -none
check_sec -setup
check_sec -generate
    
check_sec -prove

