#!/bin/bash

# Relaunch of 5 failed jobs with increased time (2 hours)

echo "=========================================="
echo "Relaunching 5 Failed Jobs"
echo "=========================================="
echo ""
echo "Time increased to 02:00:00 (2 hours)"
echo ""

# Job 1: weak_many_tasks_16n_256t_e2
echo "1. Weak many_tasks 16 nodes, energy 2..."
sbatch --nodes=16 \
       --ntasks-per-node=16 \
       --cpus-per-task=7 \
       --time=02:00:00 \
       --job-name=weak_many_16n_e2_retry \
       --export=ALL,PROGRAM_ARGS="-x 16384 -y 16384 -n 500 -e 2",TEST_TYPE="weak" \
       scripts/go_dcgp.sbatch

# Job 2: weak_few_tasks_8n_16t_e1
echo "2. Weak few_tasks 8 nodes, energy 1..."
sbatch --nodes=8 \
       --ntasks-per-node=2 \
       --cpus-per-task=56 \
       --time=02:00:00 \
       --job-name=weak_few_8n_e1_retry \
       --export=ALL,PROGRAM_ARGS="-x 16384 -y 16384 -n 500 -e 1",TEST_TYPE="weak" \
       scripts/go_dcgp.sbatch

# Job 3: weak_few_tasks_8n_16t_e2
echo "3. Weak few_tasks 8 nodes, energy 2..."
sbatch --nodes=8 \
       --ntasks-per-node=2 \
       --cpus-per-task=56 \
       --time=02:00:00 \
       --job-name=weak_few_8n_e2_retry \
       --export=ALL,PROGRAM_ARGS="-x 16384 -y 16384 -n 500 -e 2",TEST_TYPE="weak" \
       scripts/go_dcgp.sbatch

# Job 4: weak_few_tasks_8n_16t_e8
echo "4. Weak few_tasks 8 nodes, energy 8..."
sbatch --nodes=8 \
       --ntasks-per-node=2 \
       --cpus-per-task=56 \
       --time=02:00:00 \
       --job-name=weak_few_8n_e8_retry \
       --export=ALL,PROGRAM_ARGS="-x 16384 -y 16384 -n 500 -e 8",TEST_TYPE="weak" \
       scripts/go_dcgp.sbatch

# Job 5: strong_many_tasks_8n_128t_e8
echo "5. Strong many_tasks 8 nodes, energy 8..."
sbatch --nodes=8 \
       --ntasks-per-node=16 \
       --cpus-per-task=7 \
       --time=02:00:00 \
       --job-name=strong_many_8n_e8_retry \
       --export=ALL,PROGRAM_ARGS="-x 16384 -y 16384 -n 500 -e 8",TEST_TYPE="strong" \
       scripts/go_dcgp.sbatch

echo ""
echo "=========================================="
echo "5 jobs relaunched with 2 hour time limit"
echo "=========================================="
echo ""
echo "These will complete the missing rows in CSV files."
echo ""
echo "Monitor with: squeue -u \$USER"
echo "Check CSV progress: wc -l data/*.csv"
