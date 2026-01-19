#!/bin/bash

# Master script to run all parametric scaling tests
# This will submit a large number of jobs - use carefully!

echo "=========================================="
echo "Master Parametric Scaling Test Suite"
echo "=========================================="
echo ""
echo "This will submit ALL parametric tests:"
echo "- Strong scaling: 3 configs x 3 energy levels x 5 node counts = 45 jobs"
echo "- Weak scaling:   3 configs x 3 energy levels x 5 node counts = 45 jobs"
echo "- OMP scaling:    3 energy levels x 9 thread counts = 27 jobs"
echo "TOTAL: 117 jobs"
echo ""
read -p "Are you sure you want to continue? (yes/no): " confirm

if [ "$confirm" != "yes" ]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting test suite submission..."
echo ""

# Run strong scaling tests
echo "=== Strong Scaling ==="
bash scripts/parametric_strong_scaling.sh
echo ""

# Run weak scaling tests
echo "=== Weak Scaling ==="
bash scripts/parametric_weak_scaling.sh
echo ""

# Run OMP scaling tests
echo "=== OpenMP Scaling ==="
bash scripts/parametric_omp_scaling.sh
echo ""

echo "=========================================="
echo "All parametric tests submitted!"
echo "=========================================="
echo ""
echo "Monitor jobs with: squeue -u \$USER"
echo "Check output with: ls -lh slurm-*.out"
echo "View CSV results in: data/"
