for design in /home/ullas/manticore/manticore/scripts/vectorizable-designs/default/*.v; do
  ./run-vectorization.sh $design
  rm -rf *.mlir
done


