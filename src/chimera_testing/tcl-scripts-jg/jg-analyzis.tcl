proc print_time {label us} {
    puts [format "%s: %f u" $label [expr {$us}]]
}

proc get_mem_kb {} {
    set f [open "/proc/self/status" r]
    set data [read $f]
    close $f
    foreach line [split $data "\n"] {
        if {[string match "VmRSS:*" $line]} {
            # linha típica: "VmRSS:   512000 kB"
            return [lindex $line 1]
        }
    }
    return -1
}

# analyze
set before [get_mem_kb]
set analyze_us [lindex [time {analyze -sv file.v } 1] 0]
print_time "TIME_ANALYZE" $analyze_us

# elaborate
set elaborate_us [lindex [time {elaborate} 1] 0]
print_time "TIME_ELABORATE" $elaborate_us


set after [get_mem_kb]
set delta [expr {$after - $before}]
puts "MEMORY_CONSUMED: ${delta}"


exit 0

