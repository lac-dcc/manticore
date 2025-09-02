#ifndef BIT_ARRAY_H
#define BIT_ARRAY_H

#include "mlir/IR/Value.h"
#include <llvm/ADT/DenseMap.h>
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/Hashing.h"

struct bit {
  mlir::Value source;
  mlir::Value value;
  int index;

  bit(mlir::Value source, mlir::Value value, int index);
  bit();
  bit(const bit& other);

  bit& operator=(const bit& other); 
  bool operator==(const bit& other) const;

};

const int INDEX_SENTINEL_VALUE = -1;
const int TOMBSTONE_SENTINEL_VALUE = -1;

namespace llvm {
template<>
struct DenseMapInfo<bit> {
  static inline bit getEmptyKey() {
    return bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), INDEX_SENTINEL_VALUE); 
  }
  static inline bit getTombstoneKey() {
    return bit(llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), llvm::DenseMapInfo<mlir::Value>::getEmptyKey(), TOMBSTONE_SENTINEL_VALUE); 
  }

  static unsigned getHashValue(const bit &A) {
    return llvm::DenseMapInfo<mlir::Value>::getHashValue(A.value);  
  }
  static bool isEqual(const bit& A, const bit& B) {
    return A == B;
  }
};
}

struct bit_array {
  llvm::DenseSet<bit> bits;
  mlir::Value value;
};


#endif
