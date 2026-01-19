#!/bin/bash

# Parametric OpenMP Scaling Script
# Tests different thread counts WITHOUT energy sources

echo "=========================================="
echo "Parametric OpenMP Scaling Tests (No Energy Sources)"
echo "=========================================="

NODES=1
TASKS_PER_NODE=1
GRID_SIZE=16384 # 2^14 x 2^14
N_STEPS=500

# Array of thread counts (NUMA-aware: 14, 28, 42, 56, 70, 84, 98, 112)
THREAD_COUNTS=(1 2 4 8 16 32 56 84 112)

echo "Testing OpenMP scaling without energy sources"
echo "----------------------------------------"

for threads in "${THREAD_COUNTS[@]}"; do
    # NO -e parameter - pure stencil computation without energy injection
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS}"
    JOB_NAME="omp_${threads}t_noenergy"
    
    echo "  Submitting: ${JOB_NAME} (${threads} threads)"
    
    sbatch --nodes=${NODES} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${threads} \
           --time=00:30:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="omp" \
           scripts/go_dcgp.sbatch
done

echo ""
echo "=========================================="
echo "All OMP scaling jobs submitted (no energy sources)"
echo "Total jobs: ${#THREAD_COUNTS[@]}"
echo "=========================================="
