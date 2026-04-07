#ifndef CANONICALIZER_HPP
#define CANONICALIZER_HPP

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"
#include "circt/Dialect/HW/HWOps.h"
#include <memory>

/**
 * @brief An alias for a stack/list of MLIR Values, commonly used for 
 * worklists or topological orderings.
 */
using ValueStack = llvm::SmallVector<mlir::Value>;

/**
 * @brief A utility class to normalize, sort, and deduplicate hardware operations.
 *
 * This class applies a series of module-level transformations (flattening, sorting, 
 * and reduction) to target operations within a CIRCT hardware module. It is designed 
 * to expose structural equivalences by bringing operations into a canonical form 
 * before extracting or deduplicating them.
 */
class Canonicalizer {

public:
    /**
     * @brief Constructs a new Canonicalizer.
     * * @param targetOps A set of MLIR operation names (e.g., "hw.add", "hw.mul") 
     * that this canonicalizer should process. Operations not in 
     * this set are ignored.
     */
    Canonicalizer(llvm::DenseSet<llvm::StringRef> targetOps);

    /**
     * @brief The main entry point that runs the full canonicalization pipeline.
     * * Executes the internal passes (flatten, sort, reduce) over the provided 
     * top-level MLIR module.
     * * @param topModule The top-level `builtin.module` containing the HW modules 
     * to be processed.
     */
    void canonicalize(mlir::ModuleOp topModule);

private:
    /// @brief The specific operations this class is configured to optimize.
    llvm::DenseSet<llvm::StringRef> targetOps;

    /**
     * @brief Checks if an operation is mathematically or structurally associative.
     * * @param op The MLIR operation to check.
     * @return true if the operation is associative, false otherwise.
     */
    bool is_associative(mlir::Operation* op);

    /**
     * @brief Checks if an operation is mathematically or structurally commutative.
     * * @param op The MLIR operation to check.
     * @return true if the operation is commutative, false otherwise.
     */
    bool is_commutative(mlir::Operation* op);

    /**
     * @brief Flattens chains of associative operations within the module.
     * * Transforms nested associative operations (e.g., `(a + b) + c`) into a 
     * single multi-operand operation (e.g., `+(a, b, c)`) to expose broader 
     * equivalence checking opportunities.
     * * @param module The hardware module to flatten.
     */
    void flatten(circt::hw::HWModuleOp module);

    /**
     * @brief Sorts the operands of commutative operations.
     * * Establishes a deterministic operand order (e.g., sorting by SSA value ID 
     * or name) so that structurally identical operations have the exact same 
     * operand layout in the IR.
     * * @param module The hardware module to sort.
     */
    void sort(circt::hw::HWModuleOp module);

     /**
     * @brief Identifies and reduces redundant algebraic identities.
     * * Scans the hardware module for operations that can be simplified or 
     * eliminated entirely based on mathematical properties (e.g., identity 
     * elements, zero elements, or self-inverses). This prepares the IR by 
     * removing trivial logic before deeper equivalence checking.
     * * @param module The hardware module to reduce.
     */
    void reduce(circt::hw::HWModuleOp module);

    /**
     * @brief Computes a topological ordering of values within the module.
     * * Useful for ensuring that operations are processed or extracted in an 
     * order that respects their data dependencies (def-use chains).
     * * @param module The hardware module to analyze.
     * @return A unique pointer to a ValueStack containing the ordered values.
     */
    std::unique_ptr<ValueStack> get_top_ord(circt::hw::HWModuleOp module);
};

#endif


