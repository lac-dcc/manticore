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


bit_array::bit_array(llvm::DenseSet<bit>& bits): bits(bits) {}

bit_array::bit_array(const bit_array& other): bits(other.bits) {}

bit_array::bit_array(): bits(llvm::DenseSet<bit>()) {}

void bit_array::debug() {
  llvm::errs() << "(";
  for(auto b : bits) {
    llvm::errs() << b.index << " | " << b.source << " " << ",";
  }
  llvm::errs() << ")";
  llvm::errs() << "\n";
}

bit_array bit_array::unite(const bit_array& a, const bit_array& b) {
  llvm::DenseSet<bit> bits_merged(a.bits);

  for(auto& bit : b.bits) bits_merged.insert(bit);

  return bit_array(bits_merged);
}

bool bit_array::all_bit_have_same_source() {
  llvm::DenseSet<mlir::Value> sources; 
  for(auto& bit : bits) {
    if(!sources.contains(bit.source)) sources.insert(bit.source);
    if(sources.size() >= 2) return false;
  }

  return true;
}

bool bit_array::is_contiguous(int size) {
  if(!all_bit_have_same_source()) return false;

  int count = 0; 
  for(auto& bit : bits) {
    if(bit.index >= size) return false;
    count++;
  }

  return count == size;
}

