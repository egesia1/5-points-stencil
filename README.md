# Parallel 5-Point Stencil for Heat Diffusion Simulation

This project contains the material for the final assignment of the *High Performance and Cloud Computing* course for the academic year 2024-25. This course is offered by UniTS (Università degli Studi di Trieste), Trieste, Italy.

**Student**: SM3800083 - Luca Ricatti
**Platform**: Leonardo DCGP Supercomputer (CINECA)

---

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Structure](#structure)
- [Usage](#usage)
- [Results](#results)
- [Key Documents](#key-documents)
- [Contributions](#contributions)
- [References](#references)

---

## Introduction

This project presents a high-performance parallel implementation of a **5-point stencil algorithm** used to solve the 2D heat equation. The implementation is designed for modern HPC systems and leverages a **hybrid parallel programming model** combining MPI and OpenMP.

The core of this project is the numerical simulation of the heat equation, a fundamental problem in scientific computing. The continuous problem is discretized onto a 2D grid, and a finite-difference method is applied. The value of each grid point at a given timestep is calculated based on its previous value and the values of its four direct neighbors (North, South, East, West). This computational pattern, known as a **5-point stencil**, is inherently local, making it an excellent candidate for parallelization.

The primary challenge in parallelizing this algorithm lies in managing the dependencies at the borders of the decomposed domains, which requires efficient communication between processes. Our implementation addresses this through non-blocking MPI communication and strategic computation-communication overlap.

### Key Achievements

- ✅ **Scalability**: Achieved **17.46× speedup** with **109.1% parallel efficiency** at 16 nodes (1792 cores)
- ✅ **Performance**: Reduced time-to-solution from ~900s (serial) to **0.72s** (16 nodes) for a 16384×16384 grid
- ✅ **Comprehensive Testing**: Conducted **399 experimental runs** across multiple configurations and optimization levels
- ✅ **Optimization**: Implemented NUMA-aware memory allocation, communication-computation overlap, and strategic OpenMP deployment

---

## Features

*   **Hybrid MPI + OpenMP Parallelism:** The code uses a hybrid model to maximize performance. **MPI** is used for domain decomposition and communication between nodes, while **OpenMP** is used to parallelize the computational loops within each MPI process, taking full advantage of shared-memory architectures on modern multi-core nodes. Three distinct configurations were tested: 8×14 (balanced), 2×56 (few tasks), and 16×7 (many tasks).

*   **Computation-Communication Overlap:** To hide network latency, the implementation uses **non-blocking MPI calls** (`MPI_Isend`/`Irecv`). The stencil update is strategically split into interior and border computations, allowing the computation of inner grid points to proceed *while* the halo (ghost cell) data is being exchanged in the background. This critical optimization keeps communication overhead below 10% of total execution time.

*   **NUMA-Aware Optimizations:** On Leonardo's dual-socket architecture, memory placement is critical. The implementation uses **first-touch allocation** with OpenMP parallel initialization, ensuring memory pages are allocated on the local NUMA node for each thread. Combined with `OMP_PLACES=cores` and `OMP_PROC_BIND=close`, this provides approximately 2.5× bandwidth improvement.

*   **Detailed Performance Instrumentation:** The code is instrumented with high-precision timers to measure the time spent in each distinct phase of the algorithm: computation (inner vs. border), communication (MPI wait time), and initialization. This allows for a deep and accurate analysis of performance bottlenecks.

*   **Comprehensive Scalability Analysis:** The project includes a full suite of performance tests conducted on the **Leonardo DCGP** supercomputer, including:
    *   On-node OpenMP scaling from 1 to 112 threads to find the optimal thread count.
    *   Multi-node Strong Scaling (1-16 nodes) to measure speedup on a fixed-size problem.
    *   Multi-node Weak Scaling (1-16 nodes) to evaluate performance on a growing problem size.
    *   Compiler optimization impact analysis comparing 5 build variants.

*   **Parallel Buffer Operations:** Halo packing/unpacking operations are parallelized using OpenMP, providing a 3-4× speedup in the communication preparation phase.

---

## Structure

The project is organized as follows:

```
HPCC/
├── src/                          # Source code
│   ├── stencil_serial_omp.c      # Serial version with OpenMP
│   └── stencil_parallel.c        # Parallel version (MPI+OpenMP)
├── include/                      # Header files
│   └── stencil.h                 # Shared definitions
├── scripts/                      # Test and execution scripts
│   ├── README.md                 # Scripts documentation
│   ├── parametric_*.sh           # Main scalability test scripts
│   └── go_dcgp.sbatch           # SLURM batch script template
├── data/                         # Experimental results (CSV files)
│   ├── SM3800083_strong_parallel_results.csv
│   ├── SM3800083_weak_parallel_results.csv
│   └── SM3800083_omp_serial_results.csv
├── REPORT.md                     # Complete project report (MAIN DOCUMENT)
├── Makefile                      # Build configuration
└── README.md                     # This file
```

---

## Usage

### Compilation

After accessing the Leonardo supercomputer (follow the instructions at: https://docs.hpc.cineca.it/general/access.html), ensure to load the correct modules:

```bash
module purge
module load openmpi/4.1.6--gcc--12.2.0
```

A `Makefile` is provided for easy compilation:

```bash
# Compile both versions
make all

# Compile only serial version
make serial_omp

# Compile only parallel version
make parallel

# Compile debug versions
make debug

# Clean build files
make clean

# Show all available targets
make help
```

### Build Variants

Different optimization levels are available for performance comparison:

```bash
# Standard build (Ofast + march=native)
make all

# Benchmark builds
make bench_o1      # -O1 optimization
make bench_o0      # No optimization
make bench_noarch  # Without -march=native
make bench_all     # All variants
```

### Running Tests

#### Basic Execution

```bash
# Serial version
./build/serial_omp -x 16384 -y 16384 -n 500 -e 1

# Parallel version (4 MPI tasks)
mpirun -np 4 ./build/parallel -x 16384 -y 16384 -n 500 -e 1
```

#### Program Arguments

| Argument | Description | Default |
|:---------|:------------|:--------|
| `-x N` | X dimension of the grid | 1000/10000 |
| `-y N` | Y dimension of the grid | 1000/10000 |
| `-n N` | Number of iterations | 99/1000 |
| `-e N` | Number of heat sources | 1/4 |
| `-E F` | Energy per source | 1.0 |
| `-p N` | Periodic boundaries (0/1) | 0 |
| `-o N` | Output energy at each step (0/1) | 0 |
| `-v N` | Verbose output (parallel only) | 0 |

#### Scalability Tests

The performance analyses are run using SLURM batch scripts located in the `scripts/` directory:

1. **OpenMP Scaling** (1 node, 1-112 threads):
   ```bash
   bash scripts/parametric_omp_scaling.sh  # 27 jobs
   ```

2. **Strong Scaling** (1-16 nodes, fixed problem size):
   ```bash
   bash scripts/parametric_strong_scaling.sh  # 45 jobs
   ```

3. **Weak Scaling** (1-16 nodes, proportional problem size):
   ```bash
   bash scripts/parametric_weak_scaling.sh  # 45 jobs
   ```

4. **Run All Tests**:
   ```bash
   bash scripts/run_all_parametric_tests.sh  # 117 jobs total
   ```

The output of these scripts will be saved as CSV files in the `data/` directory. Results are automatically appended to the corresponding CSV files when the `TEST_TYPE` environment variable is set.

### Environment Variables

#### OpenMP Configuration (Critical for Performance)

```bash
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
```

These settings are automatically configured in the SLURM batch scripts.

#### Test Configuration

```bash
export TEST_TYPE=strong    # or 'weak' or 'omp'
export BUILD_VARIANT=ofast_omp_improved
```

---

## Results

The performance analysis yielded several key insights:

### 1. The Application is Memory-Bound

The OpenMP scaling tests revealed that performance is limited by memory bandwidth, not CPU speed. Efficiency drops significantly after 16-32 threads as the memory bus becomes saturated. This is characteristic of stencil computations, which have low arithmetic intensity.

### 2. Hybrid Configuration is Critical

A detailed tuning process tested three distinct configurations:
- **8×14 (standard)**: 8 MPI tasks × 14 OpenMP threads — balanced approach, best efficiency (103.1% at 16 nodes)
- **2×56 (few_tasks)**: 2 MPI tasks × 56 OpenMP threads — fewer MPI ranks, more threads per task
- **16×7 (many_tasks)**: 16 MPI tasks × 7 OpenMP threads — fastest absolute performance (0.72s at 16 nodes)

The **16×7 configuration** achieved the best absolute performance due to superior memory bandwidth saturation, while the **8×14 configuration** offers the best balance between performance and communication efficiency.

### 3. Excellent Strong and Weak Scaling

**Strong Scaling** (8×14 configuration, 16384×16384 grid):
- Achieved **16.50× speedup** with **103.1% parallel efficiency** at 16 nodes (1792 cores)
- Maintained >95% efficiency across all scales
- Time-to-solution reduced from 22.86s (1 node) to 1.39s (16 nodes)

**Weak Scaling** (8×14 configuration):
- Demonstrated near-perfect weak scaling with **>98% efficiency** up to 16 nodes
- Runtime remained extremely stable (11.45s to 11.65s) as problem size scaled proportionally
- Confirms the implementation efficiently handles growing problem sizes

### 4. Compiler Optimization Impact

Comparative tests with different optimization levels showed:
- **2.8-3.0× speedup** from compiler optimizations compared to unoptimized code (`-O0`)
- The `-march=native` flag provides minimal additional benefit, indicating generic `-Ofast` optimizations are already highly effective
- Link Time Optimization (LTO) enables cross-module optimizations

### 5. Communication Overhead Successfully Hidden

The computation-communication overlap strategy successfully hid the majority of MPI latency:
- **Computation**: ~90% of total time
- **Communication**: <10% of total time

The interior/border split strategy ensures the CPU remains busy performing useful work while network hardware handles data transfer.

The full analysis, detailed tables, and comprehensive results can be found in the [REPORT.md](REPORT.md).

---

## Key Documents

### 📄 [REPORT.md](REPORT.md) — **Main Project Report**
Complete technical report including:
- Problem statement and methodology
- Implementation details and optimizations
- Comprehensive experimental results
- Strong/weak scaling analysis
- Compiler optimization impact study
- Performance analysis and conclusions

### 📁 [scripts/README.md](scripts/README.md) — **Scripts Documentation**
Detailed documentation of all test scripts:
- Parametric scaling tests (strong, weak, OpenMP)
- Configuration details and usage
- Output file descriptions

---

## Platform Details

- **Supercomputer**: Leonardo DCGP (CINECA)
- **Partition**: `dcgp_usr_prod`
- **Node Architecture**: 2× Intel Xeon Platinum 8480+ (Sapphire Rapids)
- **Cores per Node**: 112 (56 per socket)
- **Memory**: 512 GB DDR5 per node
- **Interconnect**: NVIDIA Mellanox HDR100 Infiniband
- **MPI**: OpenMPI 4.1.6
- **Compiler**: GCC 12.2.0

---

## Contributions

Luca Ricatti <luca.ricatti@studenti.units.it>

---

## References

1. **HPC Course**: https://github.com/Foundations-of-HPC/High-Performance-Computing-2024
2. **Leonardo CINECA**: https://docs.hpc.cineca.it/hpc/leonardo.html

---

## Repository Information

**Tracked Files**: Source code, headers, CSV results, Makefile, documentation  
**Ignored**: Executables, build artifacts, binary output files, logs
