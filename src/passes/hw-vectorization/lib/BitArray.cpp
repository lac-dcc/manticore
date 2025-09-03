#include "../include/BitArray.h"

bit::bit(mlir::Value source, int index): source(source), index(index)  {}

bit::bit(): source(mlir::Value()), index(0) {}

bit::bit(const bit& other): source(other.source), index(other.index) { }

bit& bit::operator=(const bit& other) {
  if(this == &other) return *this;

  source = other.source;
  index = other.index;

  return *this;
}

bool bit::operator==(const bit& other) const {
    return source == other.source and index == other.index; 
}


bit_array::bit_array(llvm::DenseSet<bit>& bits, mlir::Value value): bits(bits), value(value) {}

bit_array::bit_array(const bit_array& other): bits(other.bits), value(other.value) {}

bit_array::bit_array(): bits(llvm::DenseSet<bit>()), value(mlir::Value()) {}

void bit_array::debug() {
  llvm::errs() << value << " -> (";

  for(auto b : bits) {
    llvm::errs() << b.index << ",";
  }

  llvm::errs() << ")\n";
}

