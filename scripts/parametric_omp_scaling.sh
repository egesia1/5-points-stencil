#!/bin/bash

# Parametric OpenMP Scaling Script
# Tests different thread counts and energy sources

echo "=========================================="
echo "Parametric OpenMP Scaling Tests"
echo "=========================================="

NODES=1
TASKS_PER_NODE=1
GRID_SIZE=16384 # 2^14 x 2^14
N_STEPS=500

# Array of thread counts (NUMA-aware: 14, 28, 42, 56, 70, 84, 98, 112)
THREAD_COUNTS=(1 2 4 8 16 32 56 84 112)

# Array of energy sources to test
ENERGY_SOURCES=(1 2 8)

for energy_src in "${ENERGY_SOURCES[@]}"; do
    echo ""
    echo "Testing with ${energy_src} energy source(s)"
    echo "----------------------------------------"
    
    for threads in "${THREAD_COUNTS[@]}"; do
        PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${energy_src}"
        JOB_NAME="omp_${threads}t_e${energy_src}"
        
        echo "  Submitting: ${JOB_NAME} (${threads} threads)"
        
    sbatch --nodes=${NODES} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${threads} \
           --time=00:30:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="omp",BUILD_VARIANT="ofast_omp_improved" \
           scripts/go_dcgp.sbatch
    done
done

echo ""
echo "=========================================="
echo "All parametric OMP scaling jobs submitted"
echo "Total jobs: $((${#ENERGY_SOURCES[@]} * ${#THREAD_COUNTS[@]}))"
echo "=========================================="
