#!/bin/bash

# Master script for quick testing of new OpenMP implementation
# Runs both serial and parallel tests locally

echo "=========================================="
echo "Quick Test Suite - New OpenMP Implementation"
echo "=========================================="
echo ""
echo "This will run quick tests to verify the new OpenMP pragmas work correctly."
echo "Tests will take a few minutes and run locally."
echo ""

# Make sure we're compiled
echo "Ensuring code is compiled..."
make all > /dev/null 2>&1

if [ ! -f "./build/serial_omp" ] || [ ! -f "./build/parallel" ]; then
    echo "Error: Build failed or executables not found!"
    echo "Run 'make all' to compile."
    exit 1
fi

echo "✅ Code compiled successfully"
echo ""

# Run OpenMP test
echo "=== OpenMP Scaling Test ==="
bash scripts/other/quick_tests/quick_test_omp.sh
echo ""

# Run Parallel test (if MPI is available)
if command -v mpirun &> /dev/null; then
    echo "=== Parallel Scaling Test ==="
    bash scripts/other/quick_tests/quick_test_parallel.sh
    echo ""
else
    echo "⚠️  MPI not available, skipping parallel test"
    echo ""
fi

echo "=========================================="
echo "Quick test suite completed!"
echo "=========================================="
echo ""
echo "Check results in:"
echo "- data/SM3800083_omp_serial_results.csv (OpenMP results)"
echo "- data/SM3800083_strong_parallel_results.csv (Parallel results)"
echo ""
echo "Look for BUILD_VARIANT='ofast_omp_improved' to identify new results"
