.PHONY: lib, pybind, clean, all

all: lib


lib:
	@mkdir -p build
	@cd build; cmake ..
	@cd build; $(MAKE)

clean:
	rm -rf build python/needle/_ffi/main*
