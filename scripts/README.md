# Scripts for Results Production

This directory contains the scripts used to generate the report result for the 5-point stencil scalability study on Leonardo DCGP.

## Main Scripts

### Parametric Tests

#### `parametric_strong_scaling.sh`
**Strong Scaling Tests** - Tests strong scalability with fixed problem size and increasing resources.

- **Configurations**: 3 (standard 8×14, few_tasks 2×56, many_tasks 16×7)
- **Nodes**: 5 levels (1, 2, 4, 8, 16)
- **Energy Sources**: 3 levels (1, 2, 8 sources)
- **Total jobs**: 45

```bash
bash scripts/parametric_strong_scaling.sh
```

**Output**: `data/SM3800083_strong_parallel_results.csv`

#### `parametric_weak_scaling.sh`
**Weak Scaling Tests (CORRECTED)** - Tests weak scalability with proportional problem size and resources.

- **Configurations**: 3 (standard 8×14, few_tasks 2×56, many_tasks 16×7)
- **Nodes**: 5 levels (1, 2, 4, 8, 16)
- **Energy Sources**: 3 levels (1, 2, 8 sources)
- **Grid Scaling**: Problem size scales proportionally with nodes (16384×16384 → 65536×65536)
- **Total jobs**: 45

**Important**: This is the CORRECTED version that properly scales the grid size with the number of nodes to maintain constant work per core. The grid dimensions scale as $\sqrt{N}$ where $N$ is the number of nodes.

```bash
bash scripts/parametric_weak_scaling.sh
```

**Output**: `data/SM3800083_weak_parallel_results.csv` (NOTE: For corrected results, use `SM3800083_weak_parallel_results_corrected.csv`)

#### `parametric_omp_scaling.sh`
**OpenMP Scaling Tests** - Tests OpenMP scalability on a single node.

- **Thread counts**: 9 (1, 2, 4, 8, 16, 32, 56, 84, 112)
- **Energy Sources**: 3 levels (1, 2, 8 sources)
- **Total jobs**: 27

```bash
bash scripts/parametric_omp_scaling.sh
```

**Output**: `data/SM3800083_omp_serial_results.csv`

#### `parametric_omp_scaling_no_energy.sh`
**OpenMP Scaling without Energy** - OpenMP tests without energy injection (pure stencil computation).

- **Thread counts**: 9 (1, 2, 4, 8, 16, 32, 56, 84, 112)
- **Energy Sources**: 0 (no injection)
- **Total jobs**: 9

**Note**: This is a specialized script for testing pure stencil performance without energy injection overhead. The main `parametric_omp_scaling.sh` includes energy sources and is used in the master test suite.

```bash
bash scripts/parametric_omp_scaling_no_energy.sh
```

### Master Script

#### `run_all_parametric_tests.sh`
**Master Script** - Runs all parametric tests in sequence.

- **Total jobs**: 117 (45 strong + 45 weak + 27 omp)

```bash
bash scripts/run_all_parametric_tests.sh
```

### Batch Script

#### `go_dcgp.sbatch`
**SLURM Batch Script** - Common template for all jobs on Leonardo DCGP.

- Loads MPI/OpenMP modules
- Configures OpenMP environment variables
- Executes executables (serial_omp or parallel)
- Handles output and timing

Called automatically by parametric scripts.

## Common Parameters

All tests use:
- **Grid Size**: 
  - **Strong Scaling**: 16384 × 16384 (fixed)
  - **Weak Scaling**: Scales proportionally with nodes (16384×16384 → 65536×65536)
  - **OpenMP Scaling**: 16384 × 16384 (fixed)
- **Iterations**: 500
- **Boundary Conditions**: 
  - **Strong/OpenMP**: Non-periodic (default)
  - **Weak Scaling**: Both periodic and non-periodic (tested separately)
- **Build Variant**: `ofast_omp_improved` (or `ofast` for strong scaling)
- **Partition**: `dcgp_usr_prod`
- **Account**: `****************`

## Output

Results are saved in CSV format in the `data/` directory:
- `SM3800083_strong_parallel_results.csv` - Strong scaling results
- `SM3800083_weak_parallel_results_corrected.csv` - Weak scaling results (CORRECTED version with proportional grid scaling)
- `SM3800083_omp_serial_results.csv` - OpenMP scaling results

**Note**: The weak scaling corrected CSV is the authoritative source for weak scaling analysis used in the report.

## Tested Configurations

| Configuration | Tasks/Node | Threads/Task | Total Cores/Node | Description |
|:--------------|:-----------|:-------------|:------------------|:------------|
| **standard**  | 8          | 14           | 112               | Balanced (optimal configuration) |
| **few_tasks** | 2          | 56           | 112               | Few MPI tasks, many threads |
| **many_tasks**| 16         | 7            | 112               | Many MPI tasks, few threads |

## Typical Workflow

1. **Compile the code**:
   ```bash
   make parallel serial_omp
   ```

2. **Run all tests**:
   ```bash
   bash scripts/run_all_parametric_tests.sh
   ```

3. **Or run individual test suites**:
   ```bash
   bash scripts/parametric_strong_scaling.sh
   bash scripts/parametric_weak_scaling.sh
   bash scripts/parametric_omp_scaling.sh
   ```

4. **Monitor jobs**:
   ```bash
   squeue -u $USER
   ```

5. **Check results**:
   ```bash
   ls -lh data/SM3800083_*_results*.csv
   ```

6. **Generate report figures** (using Jupyter notebook):
   ```bash
   jupyter notebook scripts/plots/generate_report_figures.ipynb
   ```
   Or use individual plotting scripts:
   ```bash
   python3 scripts/plots/plot_*.py
   ```


