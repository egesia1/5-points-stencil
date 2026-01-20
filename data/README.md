# Data Directory

This directory contains CSV files with performance measurement results from scalability tests.

## Data Files

### `SM3800083_strong_parallel_results.csv`
Strong scaling test results. Contains performance metrics for fixed problem size (16384×16384 grid) with increasing number of nodes.

**Columns:**
- `TestType`: Always "strong"
- `BuildVariant`: Compilation variant (e.g., "ofast")
- `Nodes`: Number of compute nodes
- `TotalTasks`: Total MPI tasks
- `TasksPerNode`: MPI tasks per node
- `ThreadsPerTask`: OpenMP threads per MPI task
- `EnergySources`: Number of energy injection sources
- `XDim`, `YDim`: Grid dimensions (global)
- `Iterations`: Number of stencil iterations
- `TotalTime`: Total execution time (seconds)
- `ComputationTime`: Pure computation time (seconds)
- `InternalCompTime`: Time spent computing interior points
- `BorderCompTime`: Time spent computing border points
- `CommunicationTime`: MPI communication time (seconds)
- `InitTime`: Initialization time (seconds)

**Used for**: Strong scaling analysis (speedup and efficiency plots)

### `SM3800083_weak_parallel_results_corrected.csv`
Weak scaling test results (CORRECTED version). Contains performance metrics where problem size scales proportionally with the number of nodes.

**Columns:**
- Same as strong scaling, plus:
- `Periodic`: Boundary condition type (0=non-periodic, 1=periodic)
- `JobID`: SLURM job identifier

**Grid Scaling:**
- 1 node: 16384×16384
- 2 nodes: 23170×23170 (scaled by √2)
- 4 nodes: 32768×32768 (scaled by 2)
- 8 nodes: 46340×46340 (scaled by 2√2)
- 16 nodes: 65536×65536 (scaled by 4)

**Used for**: Weak scaling analysis (runtime and efficiency plots, execution time breakdown)

### `SM3800083_omp_serial_results.csv`
OpenMP scaling test results. Contains performance metrics for single-node scaling with increasing thread counts.

**Columns:**
- `TestType`: Always "omp"
- `BuildVariant`: Compilation variant
- `Threads`: Number of OpenMP threads (1, 2, 4, 8, 16, 32, 56, 84, 112)
- `EnergySources`: Number of energy injection sources
- `XDim`, `YDim`: Grid dimensions (16384×16384)
- `Iterations`: Number of stencil iterations
- `TotalTime`: Total execution time (seconds)
- `ComputationTime`: Pure computation time (seconds)

**Used for**: OpenMP speedup and efficiency analysis

## Data Generation

Results are automatically generated when running parametric test scripts with the `TEST_TYPE` environment variable set:

- **Strong Scaling**: `scripts/parametric_strong_scaling.sh`
- **Weak Scaling**: `scripts/parametric_weak_scaling.sh`
- **OpenMP Scaling**: `scripts/parametric_omp_scaling.sh`

The executables (`build/parallel` and `build/serial_omp`) append results to the appropriate CSV file when `TEST_TYPE` is set.

## Data Usage

These CSV files are used by:
- `scripts/plots/generate_report_figures.ipynb` - Comprehensive analysis and figure generation
- Individual plotting scripts in `scripts/plots/` for specific visualizations
- Analysis in `REPORT.md` and presentation materials

All timing values are in seconds, measured using `MPI_Wtime()` (for parallel) or `omp_get_wtime()` (for serial OpenMP).
