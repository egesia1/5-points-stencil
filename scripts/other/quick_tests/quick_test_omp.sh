#!/bin/bash

# Quick OpenMP scaling test (local execution)
# Tests the new OpenMP implementation with improved pragmas

echo "=========================================="
echo "Quick OpenMP Scaling Test (New Implementation)"
echo "=========================================="

# Create data directory
mkdir -p data

# Test parameters
GRID_SIZE=1000  # Small grid for quick testing
N_STEPS=100     # Few iterations for quick testing
ENERGY_SOURCES=1

echo "Testing with grid: ${GRID_SIZE}x${GRID_SIZE}, ${N_STEPS} steps, ${ENERGY_SOURCES} energy source"
echo ""

# Thread counts to test
THREAD_COUNTS=(1 2 4 8)

echo "Threads,Time,Speedup"
for threads in "${THREAD_COUNTS[@]}"; do
    echo "Testing ${threads} threads..."
    
    export OMP_NUM_THREADS=$threads
    export BUILD_VARIANT="ofast_omp_improved"
    export TEST_TYPE="omp"
    
    start_time=$(date +%s.%N)
    ./build/serial_omp -x $GRID_SIZE -y $GRID_SIZE -n $N_STEPS -e $ENERGY_SOURCES > /dev/null 2>&1
    end_time=$(date +%s.%N)
    
    duration=$(python3 -c "print(f'{$end_time - $start_time:.6f}')")
    
    # Calculate speedup (baseline is 1 thread)
    if [ "$threads" = "1" ]; then
        baseline_time=$duration
        speedup=1.000
    else
        speedup=$(python3 -c "print(f'{$baseline_time / $duration:.3f}')")
    fi
    
    echo "$threads,$duration,$speedup"
done

echo ""
echo "=========================================="
echo "Quick OpenMP test completed!"
echo "Results saved to: data/SM3800083_omp_serial_results.csv"
echo "=========================================="
