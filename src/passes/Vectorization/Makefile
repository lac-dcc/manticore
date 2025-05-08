LLVM_PREFIX ?= /usr/local
MLIR_PREFIX ?= /usr/local

LLVM_CONFIG := $(LLVM_PREFIX)/bin/llvm-config
CLANG       := clang++

CXXFLAGS    += -fPIC -shared -std=c++17 -O2 -fvisibility=default \
               -I$(MLIR_PREFIX)/include \
               -I$(LLVM_PREFIX)/include \
               -I$(MLIR_PREFIX)/tools/mlir/include \
               -I$(LLVM_PREFIX)/tools/mlir/include \
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
