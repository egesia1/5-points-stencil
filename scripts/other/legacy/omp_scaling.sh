#!/bin/bash

echo "Submitting OpenMP scaling tests"

NODES=1
TASKS_PER_NODE=1
GRID_SIZE="-x 16384 -y 16384" #2^14 x 2^14
N_STEPS="-n 500"

# 14, 28, 42, 56, 70, 84, 98 are numa limits
for threads in 1 2 4 8 16 32 56 84 112; do
    JOB_NAME="omp_scale_${threads}t"
    echo "Submitting job: ${JOB_NAME} with ${threads} threads..."

    export SLURM_CPUS_PER_TASK=${threads}

    sbatch --nodes=${NODES} \
           --ntasks-per-node=${TASKS_PER_NODE} \
           --cpus-per-task=${threads} \
           --time=00:30:00 \
           --job-name=${JOB_NAME} \
           --export=ALL,PROGRAM_ARGS="${GRID_SIZE} ${N_STEPS}",TEST_TYPE="omp",BUILD_VARIANT="ofast_omp_improved" \
           scripts/go_dcgp.sbatch
done

echo "All OpenMP jobs submitted"
