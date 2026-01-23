#!/bin/bash

# Quick Weak Scaling Script
# Invia job separati per ogni configurazione (nodes × periodic)
# Configurazione: 16×7, ofast_omp_improved, 1 energy source

echo "=========================================="
echo "Quick Weak Scaling Tests (Separate Jobs)"
echo "=========================================="

BASE_GRID_SIZE=16384  # Grid size for 1 node (2^14)
N_STEPS=500
ENERGY_SOURCE=1
TASKS_PER_NODE=16
CPUS_PER_TASK=7
BUILD_VARIANT="ofast_omp_improved"

NODES_LIST=(1 2 4 8 16)
PERIODIC_LIST=(0 1)

echo "Configuration: ${TASKS_PER_NODE}×${CPUS_PER_TASK} (16×7)"
echo "Build variant: ${BUILD_VARIANT}"
echo "Energy sources: ${ENERGY_SOURCE}"
echo "Periodic boundaries: 0 and 1"
echo "=========================================="

TOTAL_JOBS=0

for nodes in "${NODES_LIST[@]}"; do
    TOTAL_TASKS=$((nodes * TASKS_PER_NODE))
    
    # Calculate grid size for weak scaling
    # Scale each dimension by sqrt(nodes) to maintain constant work per core
    case ${nodes} in
        1)  GRID_SIZE=16384 ;;  # 2^14
        2)  GRID_SIZE=23170 ;;  # 16384 * sqrt(2)
        4)  GRID_SIZE=32768 ;;  # 2^15
        8)  GRID_SIZE=46340 ;;  # 16384 * sqrt(8)
        16) GRID_SIZE=65536 ;;  # 2^16
    esac
    
    for periodic in "${PERIODIC_LIST[@]}"; do
        PARAMS="-x ${GRID_SIZE} -y ${GRID_SIZE} -n ${N_STEPS} -e ${ENERGY_SOURCE} -p ${periodic} -t 1"
        JOB_NAME="weak_16x7_${nodes}n_p${periodic}"
        
        echo "Submitting: ${JOB_NAME} (${nodes} nodes, ${TOTAL_TASKS} tasks, grid ${GRID_SIZE}×${GRID_SIZE}, periodic=${periodic})"
        
        sbatch --partition=dcgp_usr_prod \
               --account=uTS25_Tornator_0 \
               --job-name=${JOB_NAME} \
               --exclusive \
               --nodes=${nodes} \
               --ntasks-per-node=${TASKS_PER_NODE} \
               --cpus-per-task=${CPUS_PER_TASK} \
               --time=01:00:00 \
               --mem=0 \
               --export=ALL,PROGRAM_ARGS="${PARAMS}",TEST_TYPE="weak",BUILD_VARIANT="${BUILD_VARIANT}" \
               scripts/go_dcgp.sbatch
        
        TOTAL_JOBS=$((TOTAL_JOBS + 1))
    done
done

echo ""
echo "=========================================="
echo "All weak scaling jobs submitted"
echo "Total jobs: ${TOTAL_JOBS} (${#NODES_LIST[@]} nodes × ${#PERIODIC_LIST[@]} periodic)"
echo ""
echo "Grid Size Scaling:"
echo "  1 node:  ${BASE_GRID_SIZE}×${BASE_GRID_SIZE}"
echo "  2 nodes: 23170×23170"
echo "  4 nodes: 32768×32768"
echo "  8 nodes: 46340×46340"
echo "  16 nodes: 65536×65536"
echo "=========================================="
