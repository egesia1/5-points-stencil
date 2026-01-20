#!/bin/bash
# Simple script to generate 2D and 3D visualizations with plasma

echo "🚀 Generating Visualizations - Viridis Colormap"
echo "================================================"

# Compile if necessary
if [ ! -f "build/local/stencil_serial_omp" ]; then
    echo "📦 Compiling..."
    make -f Makefile.local serial_omp
fi

# Run simulation with binary output
echo "🔄 Running simulation..."
OMP_NUM_THREADS=4 ./build/local/stencil_serial_omp -x 200 -y 200 -e 8 -E 50 -n 500 -o 1

# Create output directory
mkdir -p viz_output

# Generate 2D visualizations (GIF + PNG)
echo "🎨 Generating 2D visualizations..."
python3 scripts/other/visualize_heat_diffusion.py --output viz_output

# Generate 3D visualizations
echo "🎨 Generating 3D visualizations..."
python3 scripts/other/visualize_heat_3d.py --output viz_output

# Clean binary files
echo "🧹 Cleaning binary files..."
rm -f plane_*.bin

echo "✅ Completed!"
echo "📁 Generated files in viz_output/:"
echo "   🎬 heat_diffusion.gif - Animated GIF"
echo "   🖼️  frame_*.png - Individual frames"
echo "   📊 energy_evolution.png - Energy evolution"
echo "   🎨 3d_surface_*.png - 3D visualizations"
