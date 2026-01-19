#!/bin/bash
#SBATCH --partition boost_usr_prod
#SBATCH -A uTS25_Tornator
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --job-name=boost_summary
#SBATCH --output=boost_summary.txt

echo "Bost DCGP Architecture Summary"
echo "=================================="
echo "Date: $(date)"
echo ""
lscpu | grep -E "CPU\(s\)|Thread|Core|Socket|NUMA|Model name"
numactl --hardware | grep -E "available|node.*cpus"
free -h | head -2

# Add these lines to your script
echo "=== GPU Information ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not found - no NVIDIA GPUs available"
echo ""

echo "=== Available GPU devices ==="
ls /dev/nvidia* 2>/dev/null || echo "No NVIDIA device files found"
echo ""

echo "=== CUDA Runtime ==="
nvcc --version 2>/dev/null || echo "nvcc not found - CUDA not installed"
echo ""

echo "=== SLURM GPU Support ==="
sinfo -o "%P %G" | grep gpu || echo "No GPU partitions available"
echo ""

echo "=== Available SLURM features ==="
sinfo -o "%P %f" | head -10
echo ""


echo "=== Available modules ==="
module avail 2>&1 | grep -i cuda || echo "No CUDA modules found"
module avail 2>&1 | grep -i gpu || echo "No GPU modules found"
echo ""
