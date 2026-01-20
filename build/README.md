# Build Directory

This directory contains compiled executable binaries for the stencil computation project.

## Executables

### `parallel`
Hybrid MPI+OpenMP parallel implementation executable. Built from `src/stencil_parallel.c`.

**Usage:**
```bash
mpirun -np <num_tasks> ./build/parallel [options]
```

**Features:**
- Distributed memory parallelism via MPI
- Shared memory parallelism via OpenMP
- Communication-computation overlap optimization
- Detailed timing breakdown (computation, communication, initialization)
- Support for periodic and non-periodic boundary conditions

**Requirements:**
- MPI runtime environment
- OpenMP-enabled compiler
- Linked against MPI and OpenMP libraries

### `serial_omp`
Serial OpenMP implementation executable. Built from `src/stencil_serial_omp.c`.

**Usage:**
```bash
./build/serial_omp [options]
```

**Features:**
- Single-node execution
- OpenMP multithreading
- Timing measurements for performance analysis
- Support for periodic and non-periodic boundary conditions

**Requirements:**
- OpenMP-enabled compiler
- OpenMP runtime library

## Compilation

Executables are built using the main `Makefile` in the project root:

```bash
make parallel      # Build MPI+OpenMP version
make serial_omp    # Build OpenMP-only version
make all           # Build both
```

See the main `README.md` and `Makefile` for compilation options, optimization flags, and architecture-specific settings.

## File Information

**Note**: The executables in this directory are compiled binaries designed to run on the **Leonardo DCGP** supercomputer (CINECA). They are ELF 64-bit executables for GNU/Linux systems with x86-64 architecture, optimized for the Leonardo compute nodes.

**Target Platform:**
- **Supercomputer**: Leonardo DCGP (CINECA)
- **Partition**: `dcgp_usr_prod`
- **Node Architecture**: 2× Intel Xeon Platinum 8480+ (Sapphire Rapids)
- **Cores per Node**: 112 (56 per socket)
- **MPI**: OpenMPI 4.1.6
- **Compiler**: GCC 12.2.0 (with OpenMP support)

**Binary Details:**
- Format: ELF 64-bit LSB executable
- Architecture: x86-64
- Debug info: Included (not stripped)
- Dynamic linking: Yes (requires MPI and OpenMP runtime libraries)

## Testing

When executed with the `TEST_TYPE` environment variable set, both executables automatically:
- Measure performance metrics
- Append results to CSV files in the `data/` directory
- Format output according to test type (strong, weak, omp)

See `scripts/` directory for automated test scripts that configure environment variables and submit jobs.
