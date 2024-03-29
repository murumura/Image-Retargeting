SOURCES := $(wildcard ./src/*.cpp  ./src/numerical/*.cpp  ./src/image/*.cpp ./src/image/*.cu ./src/utils/*.cpp ./src/geometry/*.cpp)
INCLUDES := $(wildcard *.h  ./src/numerical/*.h  ./src/image/*.h ./src/utils/*.h ./src/geometry/*.h)
TESTS := $(wildcard ./test/*.cpp ) 
DOCKER_DIR:= ./docker

export BUILDTYPE ?= default
buildtype := $(shell echo "$(BUILDTYPE)" | tr "[A-Z]" "[a-z]")
export BUILDDIR ?= build/default/$(buildtype)

CUDA_DEF := OFF
ifeq ($(CUDA),1)
CUDA_DEF = ON
endif

ifeq ($(shell uname -s), Darwin)
  export JOBS ?= $(shell sysctl -n hw.ncpu)
else ifeq ($(shell uname -s), Linux)
  export JOBS ?= $(shell grep --count processor /proc/cpuinfo)
else
  $(error Cannot determine host platform)
endif

.PHONY: all
all: $(BUILDDIR)/Makefile
	cmake --build "$(BUILDDIR)" -j $(JOBS)

.PHONY: patch_based_resizing
run: $(BUILDDIR)/Makefile
	cmake --build "$(BUILDDIR)" -j $(JOBS) -- patch_based_resizing

.PHONY: test
test: $(BUILDDIR)/Makefile
	cmake --build "$(BUILDDIR)" -j $(JOBS) -- resizing_test

.PHONY: run-test
run-test: test
	"$(BUILDDIR)/test/resizing_test" yellow

.PHONY: format
format:
	@echo "Format: " $(INCLUDES) $(SOURCES) $(TESTS)
	clang-format -i  $(SOURCES) $(INCLUDES) $(TESTS)

.PHONY: clean
clean:
	-@rm -rvf $(BUILDDIR) *.png

.PRECIOUS: $(BUILDDIR)/Makefile
$(BUILDDIR)/Makefile:
	mkdir -p $(BUILDDIR)
	cmake -H. -B$(BUILDDIR) -DCMAKE_BUILD_TYPE=$(BUILDTYPE) -DUSE_CUDA=$(CUDA_DEF)

.PHONY: docker-run
docker-run:
	-@sh $(DOCKER_DIR)/docker_run.sh
