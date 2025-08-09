rm -rf jgproject


start=$(date +%s%N)
mem=$(/usr/bin/time -f "%M" jg -no_gui -allow_unsupported_OS -tcl jasper-commands.tcl 2>&1 >/dev/null | tail -n 1) 
end=$(date +%s%N)
nano_seconds=$((end-start))

echo "$nano_seconds $mem"




