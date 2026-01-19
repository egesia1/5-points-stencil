#!/bin/bash

# Quick test with O1 build variant (moderate optimization)
# Tests single configuration to compare with ofast baseline

echo "=========================================="
echo "Quick O1 Benchmark Test"
echo "=========================================="
echo ""
echo "Build variant: o1 (-O1 moderate optimization)"
echo "Configuration: standard (8×14)"
echo "Energy sources: 1"
echo "Nodes: 1, 2, 4"
echo ""

# Configuration
TASKS_PER_NODE=8
CPUS_PER_TASK=14
GRID_SIZE=16384
N_STEPS=500
ENERGY=1

# Test Strong Scaling (3 nodes only)
echo "=== Strong Scaling (O1) ==="
for nodes in 1 2 4; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY}"
    JOB_NAME="strong_o1_${nodes}n"
    
    echo "  Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks)"
    
    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=01:30:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="strong",BUILD_VARIANT="o1",EXEC="build/parallel_o1" \
           scripts/go_dcgp.sbatch
done

# Test Weak Scaling (3 nodes only)
echo ""
echo "=== Weak Scaling (O1) ==="
for nodes in 1 2 4; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY}"
    JOB_NAME="weak_o1_${nodes}n"
    
    echo "  Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks)"
    
    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=01:30:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="weak",BUILD_VARIANT="o1",EXEC="build/parallel_o1" \
           scripts/go_dcgp.sbatch
done

# Test OMP Scaling (3 thread counts only)
echo ""
echo "=== OMP Scaling (O1) ==="
NODES=1
TASKS_PER_NODE=1

for threads in 14 56 112; do
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY}"
    JOB_NAME="omp_o1_${threads}t"
    
    echo "  Submitting: ${JOB_NAME} (${threads} threads)"
    
    sbatch --nodes=${NODES} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${threads} \
           --time=01:30:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="omp",BUILD_VARIANT="o1",EXEC="build/serial_omp_o1" \
           scripts/go_dcgp.sbatch
done

echo ""
echo "=========================================="
echo "O1 benchmark jobs submitted"
echo "Total jobs: 9 (3 strong + 3 weak + 3 omp)"
echo "=========================================="
echo ""
echo "These will append to existing CSV with BuildVariant=o1"
echo "Time allocated: 1.5 hours per job"
