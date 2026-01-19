# Quick Test Scripts

Scripts for running quick local tests during development and debugging.

## Available Scripts

### `quick_test_all.sh`
**Master script** to run all quick tests.

Runs in sequence:
- `quick_test_omp.sh` - OpenMP test
- `quick_test_parallel.sh` - Parallel test (if MPI available)

```bash
bash other/quick_tests/quick_test_all.sh
```

### `quick_test.sh`
Basic quick test with minimal parameters.

```bash
bash other/quick_tests/quick_test.sh
```

### `quick_test_omp.sh`
Quick OpenMP scaling test.

- Grid: 1000×1000
- Iterations: 20
- Threads: variable (1, 2, 4, 8)

```bash
bash other/quick_tests/quick_test_omp.sh
```

**Output**: `data/SM3800083_omp_serial_results.csv`

### `quick_test_parallel.sh`
Quick parallel scaling test.

- Grid: 1000×1000
- Iterations: 20
- MPI tasks: variable (1, 2, 4, 8)

```bash
bash other/quick_tests/quick_test_parallel.sh
```

**Output**: `data/SM3800083_strong_parallel_results.csv`

### `quick_config_test.sh`
Quick test for a specific configuration.

**Parameters**:
1. Tasks per node
2. CPUs per task
3. Number of energy sources

**Examples**:
```bash
# Standard configuration: 8 tasks × 14 cpus, 1 energy source
bash other/quick_tests/quick_config_test.sh 8 14 1

# Few tasks configuration: 2 tasks × 56 cpus, 8 energy sources
bash other/quick_tests/quick_config_test.sh 2 56 8

# Many tasks configuration: 16 tasks × 7 cpus, 2 energy sources
bash other/quick_tests/quick_config_test.sh 16 7 2
```

## Features

- **Speed**: Small grids (1000×1000) and few iterations (20)
- **Local**: Run on local machine (not on cluster)
- **Debug**: Useful to verify code works before launching long jobs
- **Development**: Ideal during development for quick tests

## Requirements

- Compiled code (`make all`)
- OpenMP available (for quick_test_omp.sh)
- MPI available (for quick_test_parallel.sh, optional)

## Output

Results are saved in standard CSV files:
- `data/SM3800083_omp_serial_results.csv`
- `data/SM3800083_strong_parallel_results.csv`

**Note**: Quick test results may overwrite complete test results if they use the same `BUILD_VARIANT`.
