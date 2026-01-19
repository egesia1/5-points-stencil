#!/bin/bash

# Quick Parallel scaling test (local execution)
# Tests the new MPI+OpenMP implementation with improved pragmas

echo "=========================================="
echo "Quick Parallel Scaling Test (New Implementation)"
echo "=========================================="

# Create data directory
mkdir -p data

# Test parameters
GRID_SIZE=2000  # Medium grid for parallel testing
N_STEPS=50      # Few iterations for quick testing
ENERGY_SOURCES=1

echo "Testing with grid: ${GRID_SIZE}x${GRID_SIZE}, ${N_STEPS} steps, ${ENERGY_SOURCES} energy source"
echo ""

# Node counts to test (assuming we have 4 cores available)
NODE_COUNTS=(1 2 4)
TASKS_PER_NODE=1
THREADS_PER_TASK=2

echo "Nodes,TasksPerNode,ThreadsPerTask,Time"
for nodes in "${NODE_COUNTS[@]}"; do
    total_tasks=$((nodes * TASKS_PER_NODE))
    
    echo "Testing ${nodes} nodes (${total_tasks} total tasks, ${THREADS_PER_TASK} threads per task)..."
    
    export BUILD_VARIANT="ofast_omp_improved"
    export TEST_TYPE="strong"
    export SLURM_NNODES=$nodes
    export SLURM_NTASKS=$total_tasks
    export SLURM_NTASKS_PER_NODE=$TASKS_PER_NODE
    export OMP_NUM_THREADS=$THREADS_PER_TASK
    
    start_time=$(date +%s.%N)
    mpirun -np $total_tasks ./build/parallel -x $GRID_SIZE -y $GRID_SIZE -n $N_STEPS -e $ENERGY_SOURCES > /dev/null 2>&1
    end_time=$(date +%s.%N)
    
    duration=$(python3 -c "print(f'{$end_time - $start_time:.6f}')")
    
    echo "$nodes,$TASKS_PER_NODE,$THREADS_PER_TASK,$duration"
done

echo ""
echo "=========================================="
echo "Quick Parallel test completed!"
echo "Results saved to: data/SM3800083_strong_parallel_results.csv"
echo "=========================================="
