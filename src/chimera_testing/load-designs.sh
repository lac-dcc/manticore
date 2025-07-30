chimera_path="/home/ullas/manticore/chimera"
list="vectorizable-designs.txt"

while IFS= read -r vectorizable_design; do
  mv "$chimera_path/database/$vectorizable_design" "/home/ullas/manticore/manticore/scripts/vectorizable-designs/default"
done < "$list" 

