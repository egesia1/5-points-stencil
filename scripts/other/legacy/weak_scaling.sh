#!/bin/bash

echo "Submitting Weak Scaling jobs..."

TASKS_PER_NODE=8
CPUS_PER_TASK=14
GRID_SIZE=16384 # Same as strong scaling: 2^14
N_STEPS=500

for nodes in 1 2 4 8 16; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS}"

    JOB_NAME="weak_scale_${nodes}n_${TOTAL_TASKS}t"
    echo "Submitting job: ${JOB_NAME} with ${nodes} nodes, ${TOTAL_TASKS} tasks, grid ${GRID_SIZE}x${GRID_SIZE}..."

    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=01:00:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="weak" \
           scripts/go_dcgp.sbatch
done

echo "All Weak Scaling jobs submitted."
