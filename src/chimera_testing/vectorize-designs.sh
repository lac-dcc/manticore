for design in /home/ullas/manticore/manticore/scripts/vectorizable-designs/default/*.v; do
  echo $design
  ./run-vectorization.sh $design
  rm -rf *.mlir
done


