for design in /home/ullas/manticore/manticore/scripts/vectorizable-designs/default/*.v; do
  if [ ! -f "/home/ullas/manticore/manticore/scripts/vectorizable-designs/vectorized/$(basename $design)" ]; then
    rm "$design"
  fi
done
