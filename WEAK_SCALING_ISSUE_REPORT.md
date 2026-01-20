# Weak Scaling Test Issue Report

## Executive Summary

The weak scaling tests were incorrectly implemented, resulting in a **strong scaling test** rather than a true weak scaling analysis. The grid size remained constant (16384×16384) across all node counts instead of scaling proportionally with the number of nodes.

---

## Problem Identification

### Expected Behavior (Weak Scaling)
In a proper weak scaling test:
- **Problem size should increase proportionally with computational resources**
- **Work per core should remain constant**
- **Runtime should remain approximately constant** (ideal weak scaling efficiency = 100%)
- **Grid size should scale**: If 1 node uses 16384×16384, then:
  - 2 nodes should use ~23170×23170 (√2 × 16384)
  - 4 nodes should use 32768×32768 (2 × 16384)
  - 8 nodes should use ~46341×46341 (2√2 × 16384)
  - 16 nodes should use 65536×65536 (4 × 16384)

### Actual Behavior (What We Got)
- **Grid size remained constant**: 16384×16384 for all node counts (1, 2, 4, 8, 16)
- **Runtime decreased** as nodes increased (strong scaling behavior)
- **Efficiency values >100%** (indicating super-linear strong scaling, not weak scaling)

---

## Root Cause Analysis

### Script Analysis: `scripts/parametric_weak_scaling.sh`

**Location**: `/home/luca/DSAI/HPCC/scripts/parametric_weak_scaling.sh`

**Problematic Code** (lines 10-35):

```bash
GRID_SIZE=16384 # Same as strong scaling: 2^14
N_STEPS=500

# ...

for nodes in 1 2 4 8 16; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${energy_src}"
    # ...
done
```

**Issue**: The `GRID_SIZE` variable is set to a constant value (16384) and never scaled based on the number of nodes. The comment even explicitly states "Same as strong scaling", indicating this was a copy-paste error from the strong scaling script.

### Correct Implementation Should Be:

```bash
BASE_GRID_SIZE=16384  # Grid size for 1 node
N_STEPS=500

for nodes in 1 2 4 8 16; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    # Scale grid size proportionally to nodes
    # For 2D grid, scale each dimension by √nodes to maintain constant work per core
    GRID_SIZE=$(echo "scale=0; $BASE_GRID_SIZE * sqrt($nodes)" | bc)
    # Or simpler: scale linearly (each dimension by nodes^0.5)
    # For integer grid sizes, we can use: GRID_SIZE=$((BASE_GRID_SIZE * nodes))
    # But for proper weak scaling: GRID_SIZE = BASE_GRID_SIZE * sqrt(nodes)
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${energy_src}"
    # ...
done
```

**Alternative approach** (if we want to scale by total cores instead of nodes):
```bash
BASE_GRID_SIZE=16384
CORES_PER_NODE=112

for nodes in 1 2 4 8 16; do
    TOTAL_CORES=$((nodes * CORES_PER_NODE))
    # Scale grid to maintain constant work per core
    GRID_SIZE=$(echo "scale=0; $BASE_GRID_SIZE * sqrt($TOTAL_CORES / $CORES_PER_NODE)" | bc)
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${energy_src}"
    # ...
done
```

---

## Data Analysis

### Data Source
**File**: `data/SM3800083_weak_parallel_results.csv`

### Verification of Constant Grid Size

All entries in the weak scaling CSV show:
- **XDim**: 16384 (constant)
- **YDim**: 16384 (constant)
- **Grid Size**: 16384×16384 for all node counts (1, 2, 4, 8, 16)

**Sample Data Verification** (using `ofast_omp_improved`, 1 energy source):

| Configuration | Nodes | XDim | YDim | Total Time (s) | Behavior |
|--------------|-------|------|------|----------------|----------|
| 16×7 | 1 | 16384 | 16384 | 13.31 | Baseline |
| 16×7 | 2 | 16384 | 16384 | 6.50 | **Runtime decreases** (strong scaling) |
| 16×7 | 4 | 16384 | 16384 | 3.16 | **Runtime decreases** (strong scaling) |
| 16×7 | 8 | 16384 | 16384 | 1.57 | **Runtime decreases** (strong scaling) |
| 16×7 | 16 | 16384 | 16384 | 0.81 | **Runtime decreases** (strong scaling) |

**Expected for Weak Scaling**:
| Configuration | Nodes | XDim | YDim | Total Time (s) | Behavior |
|--------------|-------|------|------|----------------|----------|
| 16×7 | 1 | 16384 | 16384 | ~13.31 | Baseline |
| 16×7 | 2 | ~23170 | ~23170 | ~13.31 | **Runtime constant** (weak scaling) |
| 16×7 | 4 | 32768 | 32768 | ~13.31 | **Runtime constant** (weak scaling) |
| 16×7 | 8 | ~46341 | ~46341 | ~13.31 | **Runtime constant** (weak scaling) |
| 16×7 | 16 | 65536 | 65536 | ~13.31 | **Runtime constant** (weak scaling) |

### Efficiency Calculation Issue

The current efficiency calculation in `plot_weak_scaling.py`:
```python
# For weak scaling, efficiency = (baseline_runtime / runtime) * 100%
config_grouped['Efficiency'] = (baseline_runtime / config_grouped['TotalTime']) * 100.0
```

This formula is correct for weak scaling, but since the grid size is constant, we're actually measuring strong scaling efficiency, which explains why values exceed 100%.

---

## Impact Assessment

### What We Actually Measured
- **Strong scaling performance** with fixed problem size
- **Super-linear scaling** (efficiency >100%) due to improved cache locality
- **Time-to-solution reduction** as resources increase

### What We Should Have Measured
- **Weak scaling performance** with proportional problem size
- **Constant runtime** (efficiency ~100%) as resources and problem size scale together
- **Scalability of communication patterns** under increasing problem size

### Consequences
1. **No true weak scaling data available**: We cannot assess how the code scales when problem size increases proportionally with resources
2. **Misleading efficiency values**: Values >100% suggest super-linear weak scaling, but they actually reflect strong scaling behavior
3. **Incomplete scalability analysis**: We only have strong scaling data, missing the weak scaling perspective

---

## Corrected Implementation

**Note**: A corrected version of the script has been created at `scripts/parametric_weak_scaling_CORRECTED.sh` for reference.

### Required Changes to `scripts/parametric_weak_scaling.sh`

```bash
#!/bin/bash

# Parametric Weak Scaling Script
# Tests different task/cpu configurations and energy sources
# PROBLEM SIZE SCALES PROPORTIONALLY WITH NODES

echo "=========================================="
echo "Parametric Weak Scaling Tests"
echo "=========================================="

BASE_GRID_SIZE=16384  # Grid size for 1 node (2^14)
N_STEPS=500

# Array of configurations: "tasks_per_node:cpus_per_task:description"
CONFIGS=(
    "8:14:standard"
    "2:56:few_tasks"
    "16:7:many_tasks"
)

# Array of energy sources to test
ENERGY_SOURCES=(1 2 8)

for config in "${CONFIGS[@]}"; do
    IFS=':' read -r TASKS_PER_NODE CPUS_PER_TASK CONFIG_DESC <<< "$config"
    
    echo ""
    echo "Configuration: ${CONFIG_DESC} (${TASKS_PER_NODE} tasks x ${CPUS_PER_TASK} cpus)"
    echo "----------------------------------------"
    
    for energy_src in "${ENERGY_SOURCES[@]}"; do
        echo "  Testing with ${energy_src} energy source(s)..."
        
        for nodes in 1 2 4 8 16; do
            TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
            
            # Calculate grid size for weak scaling
            # Scale each dimension by sqrt(nodes) to maintain constant work per core
            # For 2D grid: area scales with nodes, so each dimension scales with sqrt(nodes)
            GRID_SIZE=$(python3 -c "import math; print(int(${BASE_GRID_SIZE} * math.sqrt(${nodes})))")
            
            PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${energy_src}"
            JOB_NAME="weak_${CONFIG_DESC}_${nodes}n_${TOTAL_TASKS}t_e${energy_src}"
            
            echo "    Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks, grid ${GRID_SIZE}×${GRID_SIZE})"
            
            sbatch --nodes=${nodes} \
                   --ntasks-per-node=${TASKS_PER_NODE} \
                   --cpus-per-task=${CPUS_PER_TASK} \
                   --time=01:00:00 \
                   --job-name=${JOB_NAME} \
                   --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="weak",BUILD_VARIANT="ofast_omp_improved" \
                   scripts/go_dcgp.sbatch
        done
    done
done

echo ""
echo "=========================================="
echo "All parametric weak scaling jobs submitted"
echo "Total jobs: $((${#CONFIGS[@]} * ${#ENERGY_SOURCES[@]} * 5))"
echo "=========================================="
```

### Grid Size Scaling Table (Corrected)

| Nodes | Scaling Factor | Grid Size (X×Y) | Total Points | Work per Core |
|-------|---------------|-----------------|--------------|---------------|
| 1 | 1.0 | 16384×16384 | 2.68×10⁸ | Baseline |
| 2 | √2 ≈ 1.414 | 23170×23170 | 5.37×10⁸ | Constant |
| 4 | 2.0 | 32768×32768 | 1.07×10⁹ | Constant |
| 8 | 2√2 ≈ 2.828 | 46341×46341 | 2.15×10⁹ | Constant |
| 16 | 4.0 | 65536×65536 | 4.29×10⁹ | Constant |

**Note**: For a 2D grid, to maintain constant work per core, the total number of grid points should scale linearly with the number of cores. Since area = width × height, if we want area ∝ cores, then each dimension should scale as √cores.

### Data Comparison: Actual vs Expected

**16×7 Configuration (ofast_omp_improved, 1 energy source)**:

| Nodes | Actual Grid | Expected Grid | Actual Runtime | Expected Runtime | Issue |
|-------|-------------|--------------|----------------|------------------|-------|
| 1 | 16384×16384 | 16384×16384 | 13.31s | 13.31s | ✅ Correct |
| 2 | 16384×16384 | 23170×23170 | 6.50s | ~13.31s | ❌ Grid too small, runtime decreases |
| 4 | 16384×16384 | 32768×32768 | 3.16s | ~13.31s | ❌ Grid too small, runtime decreases |
| 8 | 16384×16384 | 46340×46340 | 1.57s | ~13.31s | ❌ Grid too small, runtime decreases |
| 16 | 16384×16384 | 65536×65536 | 0.81s | ~13.31s | ❌ Grid too small, runtime decreases |

**2×56 Configuration**:

| Nodes | Actual Grid | Expected Grid | Actual Runtime | Expected Runtime | Issue |
|-------|-------------|--------------|----------------|------------------|-------|
| 1 | 16384×16384 | 16384×16384 | 100.31s | 100.31s | ✅ Correct |
| 2 | 16384×16384 | 23170×23170 | 46.10s | ~100.31s | ❌ Grid too small, runtime decreases |
| 4 | 16384×16384 | 32768×32768 | 23.42s | ~100.31s | ❌ Grid too small, runtime decreases |
| 8 | 16384×16384 | 46340×46340 | 12.12s | ~100.31s | ❌ Grid too small, runtime decreases |
| 16 | 16384×16384 | 65536×65536 | 6.58s | ~100.31s | ❌ Grid too small, runtime decreases |

**8×14 Configuration**:

| Nodes | Actual Grid | Expected Grid | Actual Runtime | Expected Runtime | Issue |
|-------|-------------|--------------|----------------|------------------|-------|
| 1 | 16384×16384 | 16384×16384 | 24.61s | 24.61s | ✅ Correct |
| 2 | 16384×16384 | 23170×23170 | 12.18s | ~24.61s | ❌ Grid too small, runtime decreases |
| 4 | 16384×16384 | 32768×32768 | 5.80s | ~24.61s | ❌ Grid too small, runtime decreases |
| 8 | 16384×16384 | 46340×46340 | 3.11s | ~24.61s | ❌ Grid too small, runtime decreases |
| 16 | 16384×16384 | 65536×65536 | 1.65s | ~24.61s | ❌ Grid too small, runtime decreases |

**Key Observation**: The actual runtime decreases proportionally with the number of nodes, which is characteristic of **strong scaling** (fixed problem size, more resources). For true weak scaling, the runtime should remain approximately constant while the problem size increases.

---

## Scripts and Code Involved

### 1. Test Generation Script
**File**: `scripts/parametric_weak_scaling.sh`
- **Line 10**: Sets constant `GRID_SIZE=16384`
- **Line 35**: Uses constant grid size for all nodes
- **Issue**: No scaling logic implemented

### 2. Data Processing Script
**File**: `scripts/plot_weak_scaling.py`
- **Function**: `load_and_process_data()` - Loads data from CSV
- **Function**: `plot_weak_scaling()` - Generates plots
- **Status**: Script is correct; it processes whatever data is in the CSV
- **Note**: The efficiency calculation assumes weak scaling, but data shows strong scaling behavior

### 3. Data File
**File**: `data/SM3800083_weak_parallel_results.csv`
- Contains all test results with constant grid size (16384×16384)
- **149 lines** of data
- All entries show `XDim=16384, YDim=16384` regardless of node count

---

## Recommendations

### Immediate Actions
1. **Acknowledge the issue** in the report and presentation
2. **Clarify terminology**: Refer to current "weak scaling" results as "strong scaling with fixed problem size"
3. **Update documentation**: Note that weak scaling tests need to be re-run with corrected script

### Future Actions
1. **Correct the script**: Implement proper grid size scaling in `parametric_weak_scaling.sh`
2. **Re-run tests**: Execute corrected weak scaling tests to obtain true weak scaling data
3. **Update analysis**: Regenerate plots and analysis with corrected weak scaling data
4. **Validate results**: Ensure runtime remains approximately constant with scaled problem size

### Expected Results After Correction
- **Runtime**: Should remain approximately constant (~11-13s for 8×14 config)
- **Efficiency**: Should be close to 100% (ideal weak scaling)
- **Grid Size**: Should increase proportionally with nodes
- **Communication Overhead**: May increase slightly due to larger halo exchanges

---

## Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **Test Type** | ❌ Incorrect | Strong scaling (fixed problem size) instead of weak scaling |
| **Grid Size** | ❌ Constant | 16384×16384 for all nodes (should scale) |
| **Runtime Behavior** | ✅ As expected for strong scaling | Decreases with more nodes |
| **Efficiency Values** | ⚠️ Misleading | >100% indicates strong scaling, not weak scaling |
| **Script Implementation** | ❌ Bug | `parametric_weak_scaling.sh` uses constant grid size |
| **Data Collection** | ✅ Correct | Data accurately reflects what was tested |
| **Analysis Scripts** | ✅ Correct | `plot_weak_scaling.py` correctly processes available data |

---

## Conclusion

The weak scaling tests were incorrectly implemented due to a bug in `scripts/parametric_weak_scaling.sh` where the grid size was set to a constant value instead of scaling proportionally with the number of nodes. This resulted in collecting strong scaling data rather than weak scaling data. 

**The current "weak scaling" results are actually strong scaling results** and should be interpreted as such. To obtain true weak scaling data, the script must be corrected and the tests re-executed.

---

**Report Generated**: Based on analysis of:
- `scripts/parametric_weak_scaling.sh` (lines 1-56)
- `data/SM3800083_weak_parallel_results.csv` (149 entries)
- `scripts/plot_weak_scaling.py` (data processing logic)
- Comparison with `scripts/parametric_strong_scaling.sh` (correct implementation)
