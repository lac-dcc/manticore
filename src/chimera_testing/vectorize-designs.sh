
initial_number=27191
other_number=10000

for design in /home/ullas/manticore/chimera/database/*.v; do

  echo $design
  number=$(basename "$design" | cut -d'_' -f1) 

  if [[ $number -lt $other_number ]]; then
    timeout 1m ./run-vectorization.sh $design
    rm  *.mlir
  fi
done







