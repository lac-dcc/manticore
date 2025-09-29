#ifndef MANTICORE_VECTORIZATION_UTILS_H
#define MANTICORE_VECTORIZATION_UTILS_H

#include "mlir/IR/Value.h"
#include "llvm/Support/raw_ostream.h"
#include <map>
#include <string>

struct ValueComparator {
  bool operator()(mlir::Value lhs, mlir::Value rhs) const {
    return lhs.getAsOpaquePointer() < rhs.getAsOpaquePointer();
  }
};

struct VectorizationStatistics {
    std::map<std::string, int> counts;
    void increment(const std::string& pattern, int num = 1) { counts[pattern]+=num; }
    void printReport() {
        llvm::errs() << "VEC_COUNT_LINEAR:" << (counts.count("LINEAR") ? counts["LINEAR"] : 0) << "\n";
        llvm::errs() << "VEC_COUNT_REVERSE:" << (counts.count("REVERSE") ? counts["REVERSE"] : 0) << "\n";
        llvm::errs() << "VEC_COUNT_MIX:" << (counts.count("MIX") ? counts["MIX"] : 0) << "\n";
        llvm::errs() << "VEC_COUNT_STRUCTURAL:" << (counts.count("STRUCTURAL") ? counts["STRUCTURAL"] : 0) << "\n";
        llvm::errs() << "VEC_COUNT_PARTIAL:" << (counts.count("PARTIAL") ? counts["PARTIAL"] : 0) << "\n";
        llvm::errs() << "COUNT_INLINE:" << (counts.count("INLINE") ? counts["INLINE"] : 0) << "\n";
    }
    void reset() { counts.clear(); }
};
#endif // MANTICORE_VECTORIZATION_UTILS_H