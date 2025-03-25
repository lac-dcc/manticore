find_package(MLIR REQUIRED CONFIG)
find_package(CIRCT REQUIRED CONFIG)

# Use the same C++ standard as MLIR
set(CMAKE_CXX_STANDARD 17 CACHE STRING "")

# Disable RTTI if MLIR/LLVM is built without it
if(NOT LLVM_ENABLE_RTTI)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fno-rtti")
endif()

# Ensure MLIR and CIRCT include directories are available
include_directories(SYSTEM ${MLIR_INCLUDE_DIRS} ${CIRCT_INCLUDE_DIRS})

# Create a shared plugin
add_library(MoorePrettyPrinterPass SHARED MoorePrettyPrinter.cpp)

# Ensure a single `target_link_libraries` call
target_link_libraries(MoorePrettyPrinterPass
  PRIVATE
    MLIRPass
    MLIRSupport
    MLIRIR
    MLIRTransformUtils
    MLIRAnalysis
    LLVMSupport
    LLVMCore
    LLVMOption
    CIRCTTransforms
    CIRCTSupport
    CIRCTHW
  # Handle Darwin/macOS compatibility
  "$<$<PLATFORM_ID:Darwin>:-undefined dynamic_lookup>"
)

set_target_properties(MoorePrettyPrinterPass PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED YES
)

# Set output directory for the shared library
set_target_properties(MoorePrettyPrinterPass PROPERTIES
  CXX_VISIBILITY_PRESET "default"
  VISIBILITY_INLINES_HIDDEN ON
)