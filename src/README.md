# Source Files Directory

This directory contains the source code implementations for the stencil computation project.

## Main Implementations

### `stencil_parallel.c`
Main parallel implementation using hybrid MPI+OpenMP parallelism. This is the production-ready version that includes:
- Full MPI domain decomposition with halo exchange
- OpenMP thread-level parallelism within each MPI task
- Communication-computation overlap optimization
- Detailed timing metrics (computation, communication, initialization)
- Support for both periodic and non-periodic boundary conditions
- Energy injection at specified source locations
- Comprehensive error handling and argument parsing

**Build Target**: `parallel` (via Makefile)

### `stencil_serial_omp.c`
Serial implementation with OpenMP multithreading on a single node. This version:
- Uses OpenMP to parallelize the stencil computation loops
- Provides timing information for performance analysis
- Supports periodic and non-periodic boundary conditions
- Includes energy injection functionality
- Used for OpenMP-only scaling analysis

**Build Target**: `serial_omp` (via Makefile)

## Template Files

### `stencil_template_parallel.c`
Template/skeleton for the parallel MPI+OpenMP implementation. Contains the basic structure but may be missing some optimizations or features. Useful as a starting point for understanding the parallel implementation structure.

**Build Target**: `template_parallel` (via Makefile)

### `stencil_template_serial.c`
Template/skeleton for the serial implementation. Provides a basic structure for single-node stencil computation without MPI.

**Build Target**: `template_serial` (via Makefile)

## Compilation

All source files are compiled using the main `Makefile` in the project root. The compilation settings can be customized via:
- Optimization flags (`OPT_CFLAGS`, `OPT_O1_CFLAGS`, etc.)
- Architecture-specific flags (`ARCH` variable)
- Debug/release configurations

See the main `README.md` and `Makefile` for detailed build instructions.

## Key Features

All implementations share common features:
- 5-point stencil computation (heat diffusion equation)
- Energy source injection at configurable locations
- Periodic and non-periodic boundary condition support
- Configurable iteration count and grid dimensions
- Output capabilities for visualization and analysis

## Performance Analysis

The implementations are designed to support comprehensive performance analysis:
- **Strong Scaling**: Fixed problem size, increasing resources (nodes/threads)
- **Weak Scaling**: Problem size scales proportionally with resources
- **OpenMP Scaling**: Thread-level parallelism analysis
- **Execution Time Breakdown**: Communication vs. computation time analysis

Results are output in CSV format for automated analysis and plotting.
