#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include "mlir/IR/Value.h"
#include <llvm/ADT/DenseMap.h>
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include <set>
#include <vector>

struct bit {
  mlir::Value source;
  int index;

  bit(mlir::Value source, int index);
  bit();
  bit(const bit& other);

  bit& operator=(const bit& other); 
  bool operator==(const bit& other) const;

  bool left_adjacent(const bit& other);
  bool right_adjacent(const bit& other);

};

const int INDEX_SENTINEL_VALUE = -1;
const int TOMBSTONE_SENTINEL_VALUE = -1;

namespace llvm {

inline hash_code bit_hash_code(const bit& b) {
  return llvm::hash_combine(b.source, b.index);
}

template<>
struct DenseMapInfo<bit> {
  static inline bit getEmptyKey() {
    return bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), INDEX_SENTINEL_VALUE); 
  }
  static inline bit getTombstoneKey() {
    return bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), TOMBSTONE_SENTINEL_VALUE); 
  }

  static unsigned getHashValue(const bit &A) {
    return static_cast<unsigned>(bit_hash_code(A));  
  }

  static bool isEqual(const bit& A, const bit& B) {
    return A == B;
  }
};
}

struct assignment_group {
  mlir::Value source;
  int start;
  int end;
  bool reversed;

  assignment_group(mlir::Value source, int start, int end, bool reversed);
  assignment_group();
};

struct bit_array {
  llvm::DenseMap<int,bit> bits;

  bit_array(llvm::DenseMap<int,bit>& bits);
  bit_array(const bit_array& other);
  bit_array();
  static bit_array unite(const bit_array& a, const bit_array& b);


  bit get_bit(int index);

  bool all_bits_have_same_source() const;
  bool is_linear(int size, mlir::Value sourceInput);
  bool is_reverse_and_linear(int size, mlir::Value sourceInput);

  mlir::Value getSingleSourceValue() const;
  size_t size() const;

  void debug();
};

  std::vector<std::vector<int>> get_assignment_groups(int size);

#endif
