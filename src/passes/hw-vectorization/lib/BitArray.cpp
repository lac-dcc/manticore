#include "../include/BitArray.h"
#include <vector>

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

bool bit::left_adjacent(const bit& other) {
  return source == other.source and index == other.index + 1;
}

bool bit::right_adjacent(const bit& other) {
  return source == other.source and index == other.index - 1;
}

assignment_group::assignment_group(): source(mlir::Value()), start(0), end(0), reversed(false) { }
assignment_group::assignment_group(mlir::Value value, int start, int end, bool reversed): source(value), start(start), end(end), reversed(reversed) { }


bit_array::bit_array(llvm::DenseMap<int,bit>& bits): bits(bits) {}

bit_array::bit_array(const bit_array& other): bits(other.bits) {}


bit_array::bit_array(): bits(llvm::DenseMap<int,bit>()) {}

void bit_array::debug() {
  llvm::errs() << "(";
  for(auto p : bits) {
    llvm::errs() << p.first << " | " << p.second.source << " | " << p.second.index << " " << ",";
  }
  llvm::errs() << ")";
  llvm::errs() << "\n";
}

bit_array bit_array::unite(const bit_array& a, const bit_array& b) {
    llvm::DenseMap<int, bit> bits_merged(a.bits);
    for (const auto& pair : b.bits) {
        bits_merged.insert(pair);
    }
    return bit_array(bits_merged);
}

bool bit_array::all_bits_have_same_source() const { 
    llvm::DenseSet<mlir::Value> sources; 
    for(const auto& [_, bit] : bits) {
        if(!sources.contains(bit.source)) sources.insert(bit.source);
        if(sources.size() >= 2) return false;
    }

    return true;
}

bool bit_array::is_linear(int size, mlir::Value sourceInput) {
    if (bits.size() != size) return false;

    for(const auto& [index, bit] : bits) {

        if (bit.source != sourceInput || bit.index != index) {
            return false;
        }
    }

    return true;
}

bool bit_array::is_reverse_and_linear(int size, mlir::Value sourceInput) {
    if (bits.size() != size) return false;

    for (const auto& [index, bit] : bits) {

        if (bit.source != sourceInput || (size - 1) - index != bit.index) {
            return false;
        }
    }

    return true;
}

bit bit_array::get_bit(int n) {
  return bits[n];
}

mlir::Value bit_array::getSingleSourceValue() const {

    if (!all_bits_have_same_source() || bits.empty()) {
        return nullptr;
    }
    return bits.begin()->second.source;
}

size_t bit_array::size() const {
    return bits.size();
}