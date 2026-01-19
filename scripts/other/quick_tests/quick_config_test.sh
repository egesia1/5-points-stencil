#!/bin/bash

# Quick test script for testing a specific configuration
# Usage: ./quick_config_test.sh [tasks_per_node] [cpus_per_task] [energy_sources]

TASKS_PER_NODE=${1:-8}
CPUS_PER_TASK=${2:-14}
ENERGY_SOURCES=${3:-1}

GRID_SIZE=16384
N_STEPS=500

echo "=========================================="
echo "Quick Configuration Test"
echo "=========================================="
echo "Tasks per node: ${TASKS_PER_NODE}"
echo "CPUs per task:  ${CPUS_PER_TASK}"
echo "Energy sources: ${ENERGY_SOURCES}"
echo "Grid size:      ${GRID_SIZE}x${GRID_SIZE}"
echo "Iterations:     ${N_STEPS}"
echo ""

# Test on 1 and 2 nodes only for quick results
for nodes in 1 2; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY_SOURCES}"
    JOB_NAME="quick_${TASKS_PER_NODE}x${CPUS_PER_TASK}_${nodes}n_e${ENERGY_SOURCES}"
    
    echo "Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks)"
    
    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=00:15:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="strong" \
           scripts/go_dcgp.sbatch
done

echo ""
echo "Quick test jobs submitted."
echo "=========================================="
