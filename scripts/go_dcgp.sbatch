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

# Unload any existing gcc module first to avoid conflicts
module purge
module load openmpi/4.1.6--gcc--12.2.0

# =======================================================
#      Select Executable Based on BUILD_VARIANT
# =======================================================

# If EXEC is explicitly set from outside, use it
# Otherwise, select based on BUILD_VARIANT and TEST_TYPE
if [ -z "$EXEC" ]; then
    if [ "$TEST_TYPE" = "omp" ]; then
        # OpenMP-only tests: use serial_omp variants
        case "$BUILD_VARIANT" in
            o0)
                EXEC=build/serial_omp_o0
                ;;
            o1)
                EXEC=build/serial_omp_o1
                ;;
            noarch)
                EXEC=build/serial_omp_noarch
                ;;
            ofast|ofast_omp_improved)
                # ofast and ofast_omp_improved use the same executable
                # (difference is in the code, not compilation flags)
                EXEC=build/serial_omp
                ;;
            *)
                # Default to standard build
                EXEC=build/serial_omp
                ;;
        esac
    else
        # MPI tests: use parallel variants
        case "$BUILD_VARIANT" in
            o0)
                EXEC=build/parallel_o0
                ;;
            o1)
                EXEC=build/parallel_o1
                ;;
            noarch)
                EXEC=build/parallel_noarch
                ;;
            ofast|ofast_omp_improved)
                # ofast and ofast_omp_improved use the same executable
                # (difference is in the code, not compilation flags)
                EXEC=build/parallel
                ;;
            *)
                # Default to standard build
                EXEC=build/parallel
                ;;
        esac
    fi
fi

# Verify executable exists
if [ ! -f "$EXEC" ]; then
    echo "ERROR: Executable not found: $EXEC"
    echo "Build variant: ${BUILD_VARIANT:-default}"
    echo "Test type: ${TEST_TYPE:-unknown}"
    echo ""
    echo "Please compile the required variant first:"
    if [ "$TEST_TYPE" = "omp" ]; then
        case "$BUILD_VARIANT" in
            o0) echo "  make serial_omp_o0" ;;
            o1) echo "  make serial_omp_o1" ;;
            noarch) echo "  make serial_omp_noarch" ;;
            *) echo "  make serial_omp" ;;
        esac
    else
        case "$BUILD_VARIANT" in
            o0) echo "  make parallel_o0" ;;
            o1) echo "  make parallel_o1" ;;
            noarch) echo "  make parallel_noarch" ;;
            *) echo "  make parallel" ;;
        esac
    fi
    exit 1
fi

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
echo "Build variant: ${BUILD_VARIANT:-default}"
echo "Executable: ${EXEC}"
echo "Program arguments: ${PROGRAM_ARGS}"

# =======================================================
#                     Run the Program
# =======================================================

if [ "$TEST_TYPE" = "omp" ]; then
    # OpenMP-only tests
    ${EXEC} ${PROGRAM_ARGS}
else
    # MPI tests
    # -x propagates environment variables to all MPI ranks
    mpirun -x TEST_TYPE -x BUILD_VARIANT -np ${SLURM_NTASKS} ${EXEC} ${PROGRAM_ARGS}
fi

# =======================================================
#                     Print Results
# =======================================================

echo ""
echo "Job completed successfully"
echo "----------------------------------------------------"
