LLVM_PREFIX ?= /usr/local
MLIR_PREFIX ?= /usr/local
CIRCT_PREFIX ?= /home/ullas/manticore/circt


LLVM_CONFIG := $(LLVM_PREFIX)/bin/llvm-config
CLANG       := clang++

CXXFLAGS    += -fPIC -shared -std=c++17 -O2 -fvisibility=default \
               -I./include \
               -I$(MLIR_PREFIX)/include \
               -I$(LLVM_PREFIX)/include \
               -I$(MLIR_PREFIX)/tools/mlir/include \
               -I$(LLVM_PREFIX)/tools/mlir/include \
               -I$(CIRCT_PREFIX)/include \
               -I$(CIRCT_PREFIX)/build/include \
               `$(LLVM_CONFIG) --cxxflags`

LDFLAGS     += `$(LLVM_CONFIG) --ldflags --libs core Support` \
               -Wl,-rpath,$(LLVM_PREFIX)/lib -L$(LLVM_PREFIX)/lib \
               -L$(CIRCT_PREFIX)/build/lib \
               -lMLIRIR -lMLIRPass -lMLIRSupport -lMLIRParser -lMLIRTransforms \
               -lCIRCTComb -lCIRCTMoore -lCIRCTSV -lCIRCTHW

SOURCES = lib/VectorizePass.cpp \
          lib/AssignmentBasedVectorization.cpp \
          lib/StructuralPatternVectorization.cpp

OBJECTS = $(SOURCES:.cpp=.o)

TARGET = VectorizePass.so

all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "Ligando para criar o $@..."
	$(CLANG) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	@echo "Compilando $<..."
	$(CLANG) $(CXXFLAGS) -c $< -o $@

clean:
	@echo "Limpando arquivos gerados..."
	rm -f $(TARGET) $(OBJECTS)
