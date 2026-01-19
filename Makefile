# Compiler settings
CC = gcc
MPICC = mpicc

ARCH ?= native

# Directories
SRCDIR = src
INCDIR = include
BUILDDIR = build

# Include paths
INCLUDES = -I$(INCDIR)

# Base flags
BASE_CFLAGS = -std=c99 -Wall -Wextra -Wpedantic -Wshadow -Wuninitialized -W
# Disable specific pedantic warnings that are problematic for this code
PEDANTIC_OVERRIDES = -Wno-pointer-sign
MATH_LIBS = -lm

# Optimization flags
OPT_CFLAGS = -Ofast -flto -fopenmp -march=$(ARCH)

# Alternative optimization levels for benchmarking
OPT_O1_CFLAGS = -O1 -flto -fopenmp -march=$(ARCH)
OPT_O0_CFLAGS = -O0 -fopenmp -march=$(ARCH)
OPT_NOARCH_CFLAGS = -Ofast -flto -fopenmp

# Debug flags
DEBUG_CFLAGS = -g -O2 -DDEBUG -fopenmp

# Release flags
RELEASE_CFLAGS = $(OPT_CFLAGS)

# Verbose flags (compile-time verbosity control)
VERBOSE_1_CFLAGS = $(OPT_CFLAGS) -g -DVERBOSE_LEVEL=1
VERBOSE_2_CFLAGS = $(OPT_CFLAGS) -g -DVERBOSE_LEVEL=2

# Create build directory
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Default target
all: serial_omp parallel

# Serial OMP version
serial_omp: $(BUILDDIR)/serial_omp

$(BUILDDIR)/serial_omp: $(SRCDIR)/stencil_serial_omp.c $(INCDIR)/stencil_template_serial.h | $(BUILDDIR)
	$(CC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(RELEASE_CFLAGS)

# Parallel version (original)
parallel: $(BUILDDIR)/parallel

$(BUILDDIR)/parallel: $(SRCDIR)/stencil_parallel.c $(INCDIR)/stencil.h | $(BUILDDIR)
	$(MPICC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(RELEASE_CFLAGS)

# Debug builds
debug: serial_omp_debug parallel_debug

serial_omp_debug: $(BUILDDIR)/serial_omp_debug

$(BUILDDIR)/serial_omp_debug: $(SRCDIR)/stencil_serial_omp.c $(INCDIR)/stencil_template_serial.h | $(BUILDDIR)
	$(CC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(DEBUG_CFLAGS) $(INCLUDES) -o $@ $< $(MATH_LIBS)

parallel_debug: $(BUILDDIR)/parallel_debug

$(BUILDDIR)/parallel_debug: $(SRCDIR)/stencil_parallel.c $(INCDIR)/stencil.h | $(BUILDDIR)
	$(MPICC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(DEBUG_CFLAGS) $(INCLUDES) -o $@ $< $(MATH_LIBS)

# Benchmark builds - O1 optimization
bench_o1: serial_omp_o1 parallel_o1

serial_omp_o1: $(BUILDDIR)/serial_omp_o1

$(BUILDDIR)/serial_omp_o1: $(SRCDIR)/stencil_serial_omp.c $(INCDIR)/stencil_template_serial.h | $(BUILDDIR)
	$(CC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(OPT_O1_CFLAGS)

parallel_o1: $(BUILDDIR)/parallel_o1

$(BUILDDIR)/parallel_o1: $(SRCDIR)/stencil_parallel.c $(INCDIR)/stencil.h | $(BUILDDIR)
	$(MPICC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(OPT_O1_CFLAGS)

# Benchmark builds - O0 (no optimization)
bench_o0: serial_omp_o0 parallel_o0

serial_omp_o0: $(BUILDDIR)/serial_omp_o0

$(BUILDDIR)/serial_omp_o0: $(SRCDIR)/stencil_serial_omp.c $(INCDIR)/stencil_template_serial.h | $(BUILDDIR)
	$(CC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(OPT_O0_CFLAGS)

parallel_o0: $(BUILDDIR)/parallel_o0

$(BUILDDIR)/parallel_o0: $(SRCDIR)/stencil_parallel.c $(INCDIR)/stencil.h | $(BUILDDIR)
	$(MPICC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(OPT_O0_CFLAGS)

# Benchmark builds - no architecture-specific optimizations
bench_noarch: serial_omp_noarch parallel_noarch

serial_omp_noarch: $(BUILDDIR)/serial_omp_noarch

$(BUILDDIR)/serial_omp_noarch: $(SRCDIR)/stencil_serial_omp.c $(INCDIR)/stencil_template_serial.h | $(BUILDDIR)
	$(CC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(OPT_NOARCH_CFLAGS)

parallel_noarch: $(BUILDDIR)/parallel_noarch

$(BUILDDIR)/parallel_noarch: $(SRCDIR)/stencil_parallel.c $(INCDIR)/stencil.h | $(BUILDDIR)
	$(MPICC) $(BASE_CFLAGS) $(PEDANTIC_OVERRIDES) $(INCLUDES) -o $@ $< $(MATH_LIBS) $(OPT_NOARCH_CFLAGS)

# Build all benchmark variants
bench_all: bench_o1 bench_o0 bench_noarch

# Clean targets
clean:
	rm -rf $(BUILDDIR)

clean_all: clean
	rm -f *.o *.out core.* plane_*.bin

# Test targets
test_serial_omp: $(BUILDDIR)/serial_omp
	$(BUILDDIR)/serial_omp -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0

test_parallel: $(BUILDDIR)/parallel
	mpirun -np 4 $(BUILDDIR)/parallel -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0

test_energy: $(BUILDDIR)/serial_omp $(BUILDDIR)/parallel
	@echo "Testing energy conservation with 5000x5000 grid..."
	@echo "Serial OMP version:"
	$(BUILDDIR)/serial_omp -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0
	@echo ""
	@echo "Parallel version (4 processes):"
	mpirun -np 4 $(BUILDDIR)/parallel -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0

# Benchmark test targets (use BUILD_VARIANT env var to identify in CSV)
test_bench_o1: $(BUILDDIR)/serial_omp_o1 $(BUILDDIR)/parallel_o1
	@echo "Testing O1 optimization build..."
	@echo "Serial OMP (O1):"
	BUILD_VARIANT=o1 $(BUILDDIR)/serial_omp_o1 -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0
	@echo ""
	@echo "Parallel (O1, 4 processes):"
	BUILD_VARIANT=o1 mpirun -x BUILD_VARIANT -np 4 $(BUILDDIR)/parallel_o1 -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0

test_bench_o0: $(BUILDDIR)/serial_omp_o0 $(BUILDDIR)/parallel_o0
	@echo "Testing O0 (no optimization) build..."
	@echo "Serial OMP (O0):"
	BUILD_VARIANT=o0 $(BUILDDIR)/serial_omp_o0 -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0
	@echo ""
	@echo "Parallel (O0, 4 processes):"
	BUILD_VARIANT=o0 mpirun -x BUILD_VARIANT -np 4 $(BUILDDIR)/parallel_o0 -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0

test_bench_noarch: $(BUILDDIR)/serial_omp_noarch $(BUILDDIR)/parallel_noarch
	@echo "Testing no-arch (generic x86-64) build..."
	@echo "Serial OMP (noarch):"
	BUILD_VARIANT=noarch $(BUILDDIR)/serial_omp_noarch -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0
	@echo ""
	@echo "Parallel (noarch, 4 processes):"
	BUILD_VARIANT=noarch mpirun -x BUILD_VARIANT -np 4 $(BUILDDIR)/parallel_noarch -x 5000 -y 5000 -e 1 -E 1 -n 100 -p 0

test_bench_all: test_bench_o1 test_bench_o0 test_bench_noarch

# Show current compiler flags
show_flags:
	@echo "Base flags: $(BASE_CFLAGS)"
	@echo "Release flags: $(RELEASE_CFLAGS)"
	@echo "Debug flags: $(DEBUG_CFLAGS)"
	@echo "Benchmark O1 flags: $(OPT_O1_CFLAGS)"
	@echo "Benchmark O0 flags: $(OPT_O0_CFLAGS)"
	@echo "Benchmark noarch flags: $(OPT_NOARCH_CFLAGS)"
	@echo "Includes: $(INCLUDES)"
	@echo "Math libs: $(MATH_LIBS)"
	@echo "Build dir: $(BUILDDIR)"
	@echo "Architecture: $(ARCH)"

# Help target
help:
	@echo "Available targets:"
	@echo ""
	@echo "Standard builds:"
	@echo "  all                        - Build all versions (default)"
	@echo "  serial_omp                 - Build serial OMP version"
	@echo "  serial_hypercommented      - Build serial hypercommented version"
	@echo "  parallel_hypercommented    - Build parallel hypercommented version"
	@echo "  serial_omp_hypercommented  - Build serial OMP hypercommented version"
	@echo "  parallel                   - Build parallel version"
	@echo "  debug                      - Build debug versions"
	@echo ""
	@echo "Benchmark builds (for optimization comparison):"
	@echo "  bench_o1                   - Build with -O1 optimization"
	@echo "  bench_o0                   - Build with -O0 (no optimization)"
	@echo "  bench_noarch               - Build without -march=native (generic x86-64)"
	@echo "  bench_all                  - Build all benchmark variants"
	@echo ""
	@echo "Test targets:"
	@echo "  test_serial_omp            - Run serial OMP test with 5000x5000 grid"
	@echo "  test_parallel              - Run parallel test with 4 processes"
	@echo "  test_energy                - Test energy conservation on both versions"
	@echo "  test_bench_o1              - Test O1 optimization build"
	@echo "  test_bench_o0              - Test O0 (no optimization) build"
	@echo "  test_bench_noarch          - Test no-arch build"
	@echo "  test_bench_all             - Test all benchmark variants"
	@echo ""
	@echo "Utility targets:"
	@echo "  clean                      - Remove build directory"
	@echo "  clean_all                  - Remove all generated files"
	@echo "  show_flags                 - Show current compiler flags"
	@echo "  help                       - Show this help message"
	@echo ""
	@echo "Benchmark builds use BUILD_VARIANT env variable for CSV identification."
	@echo "Example: BUILD_VARIANT=o1 ./build/parallel_o1 [args]"

.PHONY: all debug clean clean_all test_serial_omp test_parallel test_energy show_flags help \
        serial_omp serial_hypercommented parallel_hypercommented serial_omp_hypercommented parallel \
        serial_omp_debug parallel_debug \
        bench_o1 bench_o0 bench_noarch bench_all \
        serial_omp_o1 parallel_o1 serial_omp_o0 parallel_o0 serial_omp_noarch parallel_noarch \
        test_bench_o1 test_bench_o0 test_bench_noarch test_bench_all