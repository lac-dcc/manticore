rm -rf jgproject

tcl_file="$1"

mem=$(jg -no_gui -tcl $tcl_file -allow_unsupported_OS) 

analyze_time=$(echo "$mem" | grep -oP 'TIME_ANALYZE:\s*\K[0-9]+(\.[0-9]*)?')
elaborate_time=$(echo "$mem" | grep -oP 'TIME_ELABORATE:\s*\K[0-9]+(\.[0-9]*)?')
memory_consumed=$(echo "$mem" | grep -oP 'MEMORY_CONSUMED:\s*\K[0-9]+(\.[0-9]*)?')

echo $analyze_time 
echo $elaborate_time 
echo $memory_consumed 



