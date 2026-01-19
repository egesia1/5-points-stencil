#!/bin/bash

echo "Submitting Quick Test job..."

# Single node, single task, small grid for quick verification
sbatch --nodes=1 \
       --ntasks-per-node=1 \
       --cpus-per-task=4 \
       --time=00:01:00 \
       --job-name="quick_test" \
       --export=ALL,PROGRAM_ARGS="-x 1024 -y 1024 -n 10",TEST_TYPE="omp" \
       scripts/go_dcgp.sbatch

echo "Quick test job submitted."
