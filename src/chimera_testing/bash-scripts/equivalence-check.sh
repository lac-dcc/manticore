rm -rf jgproject


tcl_file="$1"

res=$(jg -no_gui -allow_unsupported_OS -tcl $tcl_file)


res=$(echo $res | tail -n1 | rev | cut -c1-2 | rev)
res=${res::-1}
echo $res



