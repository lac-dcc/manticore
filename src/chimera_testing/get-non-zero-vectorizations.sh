

for design in ../../designs/non-vectorized/*.v; do
  echo $design

  number=$(basename "$design" | cut -d'_' -f1) 



  output="$(timeout 1m ./run-vectorization.sh "$design" 2>&1)"

  fix="VEC_${output#*VEC_}"

  l1=$(echo "${fix#*VEC_COUNT_LINEAR:}")
  l1="${l1%%[[:space:]]*}"
  l2=$(echo "${fix#*VEC_COUNT_REVERSE:}")
  l2="${l2%%[[:space:]]*}"
  l3=$(echo "${fix#*VEC_COUNT_MIX:}")
  l3="${l3%%[[:space:]]*}"
  l4=$(echo "${fix#*VEC_COUNT_STRUCTURAL:}")
  l4="${l4%%[[:space:]]*}"

  echo $l1 $l2 $l3 $l4

  if [[ "$l1" -ne 0 || "$l2" -ne 0 || "$l3" -ne 0 || "$l4" -ne 0 ]]; then
    echo "SAVE"
    name=$(basename $design)

    cp ../../designs/vectorized/$name ../../designs/real-vectorized/
  fi



done

