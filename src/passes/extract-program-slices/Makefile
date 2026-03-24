CIRCT_PREFIX ?= /home/mariaguimaraes/projects/circt
LLVM_SRC_DIR := $(CIRCT_PREFIX)/llvm/llvm
MLIR_SRC_DIR := $(CIRCT_PREFIX)/llvm/mlir
LLVM_BUILD_DIR := $(CIRCT_PREFIX)/llvm/build

LLVM_CONFIG := $(LLVM_BUILD_DIR)/bin/llvm-config
CLANG       := clang++

CXXFLAGS    += -fPIC -shared -std=c++17 -O2 -fvisibility=default \
               -I./include \
               -I$(MLIR_SRC_DIR)/include \
               -I$(LLVM_SRC_DIR)/include \
               -I$(LLVM_BUILD_DIR)/include \
               -I$(LLVM_BUILD_DIR)/tools/mlir/include \
               -I$(CIRCT_PREFIX)/include \
               -I$(CIRCT_PREFIX)/build/include \
               `$(LLVM_CONFIG) --cxxflags`

LDFLAGS     += `$(LLVM_CONFIG) --ldflags --libs core Support` \
               -Wl,-rpath,$(LLVM_BUILD_DIR)/lib -L$(LLVM_BUILD_DIR)/lib \
               -L$(CIRCT_PREFIX)/build/lib \
               -lMLIRIR -lMLIRPass -lMLIRAnalysis -lMLIRSupport -lMLIRParser -lMLIRTransforms \
               -lCIRCTComb -lCIRCTMoore -lCIRCTSV -lCIRCTHW

SOURCES = lib/SliceExtractor.cpp

OBJDIR = build

OBJECTS = $(patsubst lib/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

TARGET = $(OBJDIR)/SliceExtractorPass.so

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CLANG) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

$(OBJDIR)/%.o: lib/%.cpp
	@mkdir -p $(OBJDIR)
	$(CLANG) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJDIR)

