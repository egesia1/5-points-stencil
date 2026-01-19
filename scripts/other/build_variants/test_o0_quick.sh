#!/bin/bash

# Quick test with O0 build variant (no optimization)
# Tests single configuration to compare with ofast baseline

echo "=========================================="
echo "Quick O0 Benchmark Test"
echo "=========================================="
echo ""
echo "Build variant: o0 (-O0 no optimization)"
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
echo "=== Strong Scaling (O0) ==="
for nodes in 1 2 4; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY}"
    JOB_NAME="strong_o0_${nodes}n"
    
    echo "  Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks)"
    
    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=02:00:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="strong",BUILD_VARIANT="o0",EXEC="build/parallel_o0" \
           scripts/go_dcgp.sbatch
done

# Test Weak Scaling (3 nodes only)
echo ""
echo "=== Weak Scaling (O0) ==="
for nodes in 1 2 4; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY}"
    JOB_NAME="weak_o0_${nodes}n"
    
    echo "  Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks)"
    
    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=02:00:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="weak",BUILD_VARIANT="o0",EXEC="build/parallel_o0" \
           scripts/go_dcgp.sbatch
done

# Test OMP Scaling (3 thread counts only)
echo ""
echo "=== OMP Scaling (O0) ==="
NODES=1
TASKS_PER_NODE=1

for threads in 14 56 112; do
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY}"
    JOB_NAME="omp_o0_${threads}t"
    
    echo "  Submitting: ${JOB_NAME} (${threads} threads)"
    
    sbatch --nodes=${NODES} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${threads} \
           --time=02:00:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="omp",BUILD_VARIANT="o0",EXEC="build/serial_omp_o0" \
           scripts/go_dcgp.sbatch
done

echo ""
echo "=========================================="
echo "O0 benchmark jobs submitted"
echo "Total jobs: 9 (3 strong + 3 weak + 3 omp)"
echo "=========================================="
echo ""
echo "These will append to existing CSV with BuildVariant=o0"
echo "Time allocated: 2 hours per job (O0 is slower)"
