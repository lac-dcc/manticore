proc print_time {label us} {

    puts [format "%s: %f" $label $us]
}

proc get_mem_kb {} {
    set f [open "/proc/self/status" r]
    set data [read $f]
    close $f
    foreach line [split $data "\n"] {
        if {[string match "VmRSS:*" $line]} {
            return [lindex $line 1]
        }
    }
    return -1
}

set design_file_to_analyze $env(DESIGN_FILE)

set before [get_mem_kb]

# Analyze
set analyze_us [lindex [time {analyze -sv ${design_file_to_analyze}} 1] 0]
print_time "TIME_ANALYZE" $analyze_us

# Elaborate
set elaborate_us [lindex [time {elaborate} 1] 0]
print_time "TIME_ELABORATE" $elaborate_us

# Memory
set after [get_mem_kb]
set delta [expr {$after - $before}]
puts "MEMORY_CONSUMED: ${delta}"

exit 0