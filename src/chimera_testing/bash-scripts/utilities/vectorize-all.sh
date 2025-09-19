for design in /home/ullas/manticore/manticore/designs/real-vectorized/*.v; do
  echo $design

  ./bash-scripts/utilities/run-vectorization.sh $design
done







