rm -rf jgproject


tcl_file="$1"

res=$(jg -no_gui -allow_unsupported_OS -tcl $tcl_file)


# olhar se a string contem proven e olhar se a string contem no property
res=$(echo $res | tail -n1 | rev | cut -c1-2 | rev)
res=${res::-1}
echo $res



