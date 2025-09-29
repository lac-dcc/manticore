
rm ../../designs/real-non-vectorized/*.v


lista="files-vectorization.txt"
while IFS= read -r path; do
  cp ../../designs/non-vectorized/$(basename $path) ../../designs/real-non-vectorized/

done < "$lista"
