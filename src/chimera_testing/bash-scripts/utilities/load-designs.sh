
for design in /home/ullas/manticore/manticore/designs/vectorized/*.v; do

  echo $design
  
  name=$(basename $design)

  cp /home/ullas/manticore/chimera/database/$name ../../designs/non-vectorized

done

