# Mede cada etapa separadamente usando 'time' (sem 'clock')

# helper p/ imprimir em segundos
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
set analyze_us [lindex [time {analyze -sv final.sv} 1] 0]
# print_time "analyze" $analyze_us

# elaborate
set elaborate_us [lindex [time {elaborate} 1] 0]
# print_time "elaborate" $elaborate_us


set after [get_mem_kb]
set delta [expr {$after - $before}]
puts "Analyze memory: before=${before}kB after=${after}kB delta=${delta}kB"

# total
# print_time "total" [expr {$analyze_us + $elaborate_us}]

exit 0

