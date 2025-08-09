LLVM_PREFIX ?= /usr/local
MLIR_PREFIX ?= /usr/local
CIRCT_PREFIX ?= /home/ullas/manticore/circt/include
CIRCT_BUILD_INCLUDE ?= /home/ullas/manticore/circt/build/include


LLVM_CONFIG := $(LLVM_PREFIX)/bin/llvm-config
CLANG       := clang++

CXXFLAGS    += -fPIC -shared -std=c++17 -O2 -fvisibility=default \
               -I$(MLIR_PREFIX)/include \
               -I$(LLVM_PREFIX)/include \
               -I$(MLIR_PREFIX)/tools/mlir/include \
               -I$(LLVM_PREFIX)/tools/mlir/include \
							 -I$(CIRCT_PREFIX) \
							 -I$(CIRCT_BUILD_INCLUDE) \
               `$(LLVM_CONFIG) --cxxflags`

LDFLAGS     += `$(LLVM_CONFIG) --ldflags --libs core Support` \
               -Wl,-rpath,$(LLVM_PREFIX)/lib -L$(LLVM_PREFIX)/lib \
               -lMLIRIR -lMLIRPass -lMLIRSupport -lMLIRParser -lMLIRTransforms

TARGET = VectorizePass.so
SRC    = VectorizePass.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CLANG) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

clean:
	rm -f $(TARGET)
