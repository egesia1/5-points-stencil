#!/bin/bash

# =======================================================
#           Common Parameters Among All Tests
# =======================================================

#SBATCH --partition dcgp_usr_prod
#SBATCH -A uTS25_Tornator_0
#SBATCH -t 00:30:00
#SBATCH --job-name=stencil_job
#SBATCH --exclusive
#SBATCH --mem=0
  
# =======================================================
#      Parameters Dynamically Set by outer scripts
# =======================================================

# --nodes
# --cpus-per-task
# --ntasks-per-node

# =======================================================
#           Executable and Modules
# =======================================================

# Use EXEC if passed from outside, otherwise default to build/parallel
EXEC=${EXEC:-build/parallel}

# Unload any existing gcc module first to avoid conflicts
module purge
module load openmpi/4.1.6--gcc--12.2.0

# =======================================================
#               Environment Variables
# =======================================================

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

# =======================================================
#               Print Job Information
# =======================================================

echo "----------------------------------------------------"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Running on ${SLURM_NNODES} nodes"
echo "Total MPI tasks: ${SLURM_NTASKS}"
echo "Tasks per node: ${SLURM_NTASKS_PER_NODE}"
echo "CPUs per task (OMP_NUM_THREADS): ${SLURM_CPUS_PER_TASK}"
echo "Program arguments: ${PROGRAM_ARGS}"

# =======================================================
#                     Run the Program
# =======================================================

if [ "$TEST_TYPE" = "omp" ]; then
    # Use serial_omp for OpenMP-only tests
    EXEC=build/serial_omp
    ${EXEC} ${PROGRAM_ARGS}
else
    # Use parallel for MPI tests
    # -x propagates environment variables to all MPI ranks
    mpirun -x TEST_TYPE -x BUILD_VARIANT -np ${SLURM_NTASKS} ${EXEC} ${PROGRAM_ARGS}
fi

# =======================================================
#                     Print Results
# =======================================================

echo ""
echo "Job completed successfully"
echo "----------------------------------------------------"
