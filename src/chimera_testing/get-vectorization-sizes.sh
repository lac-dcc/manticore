
touch vectorization-sizes.txt

for design in ../../designs/real-vectorized/*.v; do

  name=$(basename $design)

  output="$(timeout 1m ./run-vectorization.sh  "/home/ullas/manticore/manticore/designs/non-vectorized/$name" 2>&1)"
  
  echo $output

  size=$(echo "${output#*VEC_SIZE:}")
  size="${size%%[[:space:]]*}"

  echo $size >> vectorization-sizes.txt
done

