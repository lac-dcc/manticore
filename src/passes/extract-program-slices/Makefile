LLVM_PREFIX ?= /home/rafael-paniago/circt/llvm/build
MLIR_PREFIX ?= /home/rafael-paniago/circt/llvm/build
CIRCT_PREFIX ?= /home/rafael-paniago/circt

LLVM_CONFIG := $(LLVM_PREFIX)/bin/llvm-config
CLANG        := clang++

CXXFLAGS     += -fPIC -shared -std=c++17 -O2 -fvisibility=default \
                -I./include \
                -I$(MLIR_PREFIX)/include \
                -I$(LLVM_PREFIX)/include \
                -I$(CIRCT_PREFIX)/llvm/mlir/include \
                -I$(CIRCT_PREFIX)/llvm/build/tools/mlir/include \
                -I$(CIRCT_PREFIX)/include \
                -I$(CIRCT_PREFIX)/build/include \
                `$(LLVM_CONFIG) --cxxflags`

LDFLAGS      += `$(LLVM_CONFIG) --ldflags --libs core Support` \
                -Wl,-rpath,$(LLVM_PREFIX)/lib -L$(LLVM_PREFIX)/lib \
                -L$(CIRCT_PREFIX)/build/lib \
                -lMLIRIR -lMLIRPass -lMLIRAnalysis -lMLIRSupport -lMLIRParser -lMLIRTransforms \
                -lCIRCTComb -lCIRCTMoore -lCIRCTSV -lCIRCTHW -lCIRCTLLHD

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

