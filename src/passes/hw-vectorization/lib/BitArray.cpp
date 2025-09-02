#include "../include/BitArray.h"

bit::bit(mlir::Value source, mlir::Value value, int index): source(source), value(value), index(index)  {}

bit::bit(): source(mlir::Value()), value(mlir::Value()), index(0) {}

bit::bit(const bit& other): source(other.source), value(other.value), index(other.index) { }

bit& bit::operator=(const bit& other) {
  if(this == &other) return *this;

  source = other.source;
  value = other.value;
  index = other.index;

  return *this;
}

bool bit::operator==(const bit& other) const {
    return source == other.source and value == other.value and index == other.index; 
}





