
filename=$1

output="$(timeout 1m ./run-vectorization.sh  "/home/ullas/manticore/manticore/designs/non-vectorized/$filename" 2>&1)"
size=$(echo "${output#*VEC_SIZE:}")
size="${size%%[[:space:]]*}"

echo $size
