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

bool bit::left_adjacent(const bit& other) {
  return source == other.source and index == other.index + 1;
}

bool bit::right_adjacent(const bit& other) {
  return source == other.source and index == other.index - 1;
}

bool bit::adjacent(const bit& other) {
  return left_adjacent(other) or right_adjacent(other);
}

assignment_group::assignment_group(): source(mlir::Value()), start(0), end(0), reverse(false) { }
assignment_group::assignment_group(mlir::Value source, int start, int end): source(source) {
  this->start = std::min(start, end);
  this->end = std::max(start, end);

  reverse = start < end;
}

int assignment_group::size() {
  return std::max(start, end) - std::min(start, end) + 1;
}

void assignment_group::debug() {
  llvm::errs() << "(";
  for(int i = std::min(start, end); i <= std::max(start, end); i++) {
    llvm::errs() << i << ",";
  }
  llvm::errs() << ")";
}


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
  llvm::DenseMap<int,bit> bits_merged(a.bits);

  for(auto& pair : b.bits) bits_merged.insert(pair);

  return bit_array(bits_merged);
}

bool bit_array::all_bits_have_same_source() {
  llvm::DenseSet<mlir::Value> sources; 
  for(auto& [_, bit] : bits) {
    if(!sources.contains(bit.source)) sources.insert(bit.source);
    if(sources.size() >= 2) return false;
  }

  return true;
}

bool bit_array::is_linear(int size) {
  if(!all_bits_have_same_source()) return false;

  // refatorar aqui
  int count = 0; 
  for(auto& [index, bit] : bits) {
    if(index != bit.index) return false;
    count++;
  }

  return count == size;
}

bool bit_array::is_reverse_and_linear(int size) {
  if(!all_bits_have_same_source()) return false;

  // refatorar aqui
  int count = 0; 
  for(auto& [index, bit] : bits) {
    if((size - 1) - index != bit.index) return false;
    count++;
  }

  return count == size;
}

bit bit_array::get_bit(int n) {
  return bits[n];
}

std::vector<assignment_group> bit_array::get_assignment_groups(int size) {

    for(auto& [index, bit] : bits) {
      auto barg = mlir::cast<mlir::BlockArgument>(bit.source);
      int input_index = barg.getArgNumber();

      // llvm::errs() << index << " - " << input_index << "\n";
    }

  std::vector<assignment_group> assignments;

  int i = 0;  
  int start = 0, end = 0;
  // std::set<int> collected_indexes;

  while(i < size - 1) {

    // TODO: passar para std::contains depois (via c++ 20)
    if(bits[i].adjacent(bits[i + 1])) {
      end = i + 1;
    }

    else {
      assignments.push_back(assignment_group(bits[start].source, bits[start].index, bits[end].index));
      start = i + 1;
      end = i + 1;
    }

    i++;
  }

  if(bits[i - 1].adjacent(bits[i])) {
    assignments.push_back(assignment_group(bits[start].source, bits[start].index, bits[end].index));
  }
  else {
    assignments.push_back(assignment_group(bits[i].source, bits[i].index, bits[i].index));
  }



  for(auto& a : assignments) {
    auto barg = mlir::cast<mlir::BlockArgument>(a.source);
    int input_index = barg.getArgNumber();
    llvm::errs() << input_index << " ";
  }
  llvm::errs() << "\n";

  return assignments;
}

