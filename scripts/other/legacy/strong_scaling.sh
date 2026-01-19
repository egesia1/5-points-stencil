#!/bin/bash

echo "Submitting Strong Scaling jobs..."

TASKS_PER_NODE=8
CPUS_PER_TASK=14
GRID_SIZE=16384 #2^14
N_STEPS=500
PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS}"

for nodes in 1 2 4 8 16; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    JOB_NAME="strong_scale_${nodes}n_${TOTAL_TASKS}t"
    echo "Submitting job: ${JOB_NAME} with ${nodes} nodes and ${TOTAL_TASKS} tasks..."

    sbatch --nodes=${nodes} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${CPUS_PER_TASK} \
           --time=03:00:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="strong" \
           scripts/go_dcgp.sbatch
done

echo "All Strong Scaling jobs submitted."
