for design in /home/ullas/manticore/manticore/scripts/vectorizable-designs/default/*.v; do
  echo $design
  ./run-vectorization.sh $design 2>&1 | grep "Detected" >> logs.txt
  rm -rf *.mlir
done
