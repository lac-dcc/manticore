rm -rf jgproject

res=$(jg -no_gui -allow_unsupported_OS -tcl equivalence.tcl)


res=$(echo $res | tail -n1 | rev | cut -c1-2 | rev)
res=${res::-1}
echo $res



