#!/bin/bash

# Parametric Weak Scaling Script (CORRECTED VERSION)
# Tests different task/cpu configurations and energy sources
# PROBLEM SIZE SCALES PROPORTIONALLY WITH NODES

echo "=========================================="
echo "Parametric Weak Scaling Tests (CORRECTED)"
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
echo ""
echo "Grid Size Scaling:"
echo "  1 node:  ${BASE_GRID_SIZE}×${BASE_GRID_SIZE}"
echo "  2 nodes: $(python3 -c "import math; print(int(${BASE_GRID_SIZE} * math.sqrt(2)))")×$(python3 -c "import math; print(int(${BASE_GRID_SIZE} * math.sqrt(2)))")"
echo "  4 nodes: $((BASE_GRID_SIZE * 2))×$((BASE_GRID_SIZE * 2))"
echo "  8 nodes: $(python3 -c "import math; print(int(${BASE_GRID_SIZE} * math.sqrt(8)))")×$(python3 -c "import math; print(int(${BASE_GRID_SIZE} * math.sqrt(8)))")"
echo "  16 nodes: $((BASE_GRID_SIZE * 4))×$((BASE_GRID_SIZE * 4))"
echo "=========================================="
