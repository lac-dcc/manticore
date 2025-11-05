
rm -rf jgproject

tcl_file="$1"

output=$(jg -no_gui -tcl $tcl_file -allow_unsupported_OS 2>&1)

analyze_time=$(echo "$output" | grep -oP 'TIME_ANALYZE:\s*\K[0-9]+(\.[0-9]*)?')
elaborate_time=$(echo "$output" | grep -oP 'TIME_ELABORATE:\s*\K[0-9]+(\.[0-9]*)?')
memory_consumed=$(echo "$output" | grep -oP 'MEMORY_CONSUMED:\s*\K[0-9]+(\.[0-9]*)?')

analyze_time=${analyze_time:-0}
elaborate_time=${elaborate_time:-0}
memory_consumed=${memory_consumed:-0}

total_time=$(echo "$analyze_time + $elaborate_time" | bc)

echo $total_time
echo $memory_consumed