# make           -> ../bgtrain (native trainer), ../bgcore.wasm + ../bgcore_wasm.js (browser module)
# make check     -> builds, then runs check_cpp.js
ROOT := ..
CXX_NATIVE ?= g++
NATIVE_FLAGS := -std=c++20 -O3 -march=native -fno-trapping-math -pthread -Wall
WASM_FLAGS := --target=wasm32 -std=c++20 -O3 -ffreestanding -nostdlib -fno-exceptions -fno-rtti -msimd128 -mbulk-memory \
	-Wl,--no-entry -Wl,--export=__wasm_call_ctors -Wl,-z,stack-size=1048576

all: $(ROOT)/bgtrain $(ROOT)/bgcore_wasm.js

$(ROOT)/bgtrain: train.cpp core.h
	$(CXX_NATIVE) $(NATIVE_FLAGS) -o $@ train.cpp

$(ROOT)/bgcore.wasm: wasm.cpp core.h
	clang++ $(WASM_FLAGS) -o $@ wasm.cpp

# the browser loads the module from a script (works from file://, where fetch does not)
$(ROOT)/bgcore_wasm.js: $(ROOT)/bgcore.wasm
	printf 'window.BG_WASM_B64="%s";\n' "$$(base64 -w0 $<)" > $@

check: all
	cd $(ROOT) && node check_cpp.js

.PHONY: all check
