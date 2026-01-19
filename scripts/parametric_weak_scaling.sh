#!/bin/bash

# Parametric Weak Scaling Script
# Tests different task/cpu configurations and energy sources

echo "=========================================="
echo "Parametric Weak Scaling Tests"
echo "=========================================="

GRID_SIZE=16384 # Same as strong scaling: 2^14
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
            PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${energy_src}"
            JOB_NAME="weak_${CONFIG_DESC}_${nodes}n_${TOTAL_TASKS}t_e${energy_src}"
            
            echo "    Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks)"
            
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
