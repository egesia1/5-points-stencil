#!/bin/bash

#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH --job-name=weak_scaling_quick
#SBATCH --exclusive
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=7
#SBATCH -t 01:00:00
#SBATCH --mem=0

# Test Parameters - FIXED CONFIGURATION
# Configurazione: 16×7 (best performing from strong scaling analysis)
# Build variant: ofast_omp_improved
# Energy sources: 1
# - 16 MPI tasks per node
# - 7 OpenMP threads per task

EXEC="../build/parallel"
N_STEPS=500
NODES_LIST="1 2 4 8 16"
ENERGY_SOURCE=1  # Single energy source

# Output management
RESULTS_DIR="../results/weak_scalability"
CONFIG_NAME="16x7"
BUILD_VARIANT="ofast_omp_improved"
OUTPUT_FILE="${RESULTS_DIR}/weak_scalability_results_${CONFIG_NAME}_${BUILD_VARIANT}.txt"
mkdir -p ${RESULTS_DIR}

# Environment Setup and Job Info
module purge
module load openmpi/4.1.6--gcc--12.2.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close
export TEST_TYPE="weak"
export BUILD_VARIANT="ofast_omp_improved"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Configuration: ${SLURM_NTASKS_PER_NODE} tasks/node, ${SLURM_CPUS_PER_TASK} threads/task"
echo "Nodes to test: ${NODES_LIST}"
echo "Output file: ${OUTPUT_FILE}"
echo "===================================================="

# Main Loop
for nodes in ${NODES_LIST}; do
    TOTAL_TASKS=$((nodes * SLURM_NTASKS_PER_NODE))

    # Set grid size for weak scaling (QUADRATO, potenze di 2 quando possibile)
    # Area scales with nodes: area = base_area * nodes
    # For 2D: each dimension scales as sqrt(nodes)
    case ${nodes} in
        1)  GRID_SIZE_X=16384; GRID_SIZE_Y=16384 ;;  # 2^14
        2)  GRID_SIZE_X=23170; GRID_SIZE_Y=23170 ;;  # 16384 * sqrt(2)
        4)  GRID_SIZE_X=32768; GRID_SIZE_Y=32768 ;;  # 2^15
        8)  GRID_SIZE_X=46340; GRID_SIZE_Y=46340 ;;  # 16384 * sqrt(8)
        16) GRID_SIZE_X=65536; GRID_SIZE_Y=65536 ;;  # 2^16
    esac

    PROGRAM_ARGS="-x ${GRID_SIZE_X} -y ${GRID_SIZE_Y} -n ${N_STEPS} -e ${ENERGY_SOURCE} -t 1"

    echo "--- RUNNING: ${nodes} nodes, ${TOTAL_TASKS} MPI tasks, Grid: ${GRID_SIZE_X}×${GRID_SIZE_Y} ---"

    srun -N ${nodes} -n ${TOTAL_TASKS} -x TEST_TYPE -x BUILD_VARIANT ${EXEC} ${PROGRAM_ARGS} >> ${OUTPUT_FILE}

    echo "--- COMPLETED ---"
done

echo "===================================================="
echo "All weak scaling tests are complete."
echo "Results saved to: ${OUTPUT_FILE}"
echo "===================================================="
