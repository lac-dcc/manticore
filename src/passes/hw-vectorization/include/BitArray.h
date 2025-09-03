#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include "mlir/IR/Value.h"
#include <llvm/ADT/DenseMap.h>
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/Support/raw_ostream.h"
#include <set>

struct bit {
  mlir::Value source;
  int index;

  bit(mlir::Value source, int index);
  bit();
  bit(const bit& other);

  bit& operator=(const bit& other); 
  bool operator==(const bit& other) const;

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

struct bit_array {
  llvm::DenseSet<bit> bits;

  bit_array(llvm::DenseSet<bit>& bits);
  bit_array(const bit_array& other);
  bit_array();
  static bit_array unite(const bit_array& a, const bit_array& b);

  bool all_bit_have_same_source();

  bool is_contiguous(int size);


  void debug();
};


#endif
