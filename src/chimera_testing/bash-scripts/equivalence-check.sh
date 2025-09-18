rm -rf jgproject

tcl_file="$1"

res=$(jg -no_gui -allow_unsupported_OS -tcl $tcl_file)

if echo $res | grep -q "PROOF_RES: proven"; then
  proven="True"
else
  proven="False"
fi

echo $proven



