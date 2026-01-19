# Visualization Scripts

Python scripts to visualize heat diffusion simulation results.

## Available Scripts

### `visualize_heat_diffusion.py`
**2D Visualization** of heat diffusion.

Creates:
- `energy_evolution.png` - Energy vs time graph
- `heat_diffusion.gif` - Complete diffusion animation
- `evolution_grid.png` - 3×3 grid of temporal snapshots
- `frame_*.png` - Individual frames

**Usage**:
```bash
# Use default directory (viz_output)
python3 other/visualization/visualize_heat_diffusion.py

# Specify custom directory
python3 other/visualization/visualize_heat_diffusion.py --output my_results
```

**Requirements**:
- Binary files `plane_XXXXX.bin` generated with `-o 1`
- Python 3 with numpy and matplotlib

### `visualize_heat_3d.py`
**3D Visualization** of heat diffusion.

Creates 3D visualizations where surface height represents energy.

**Usage**:
```bash
# Standard visualization
python3 other/visualization/visualize_heat_3d.py

# Custom output
python3 other/visualization/visualize_heat_3d.py --output my_3d_results

# Interactive mode
python3 other/visualization/visualize_heat_3d.py --interactive
```

**Requirements**:
- Binary files `plane_XXXXX.bin`
- Python 3 with numpy, matplotlib, mpl_toolkits.mplot3d

### `run_viz.sh`
**Wrapper script** to run all visualizations.

Automatically runs:
- `visualize_heat_diffusion.py`
- `visualize_heat_3d.py`

```bash
bash other/visualization/run_viz.sh
```

**Output**: `viz_output/` directory with all results.

### `test_clear_colormap.py`
**Colormap test** to verify color maps used.

```bash
python3 other/visualization/test_clear_colormap.py
```

## Typical Workflow

### 1. Generate binary data
```bash
# Run simulation with output enabled
./build/serial_omp -x 500 -y 500 -e 3 -E 10 -n 50 -o 1
```

This creates files `plane_00000.bin`, `plane_00001.bin`, etc.

### 2. Visualize
```bash
# Simple method
bash other/visualization/run_viz.sh

# Or manually
python3 other/visualization/visualize_heat_diffusion.py
python3 other/visualization/visualize_heat_3d.py
```

### 3. Clean up (optional)
```bash
# Binary files can be large
rm -f plane_*.bin
```

## Requirements

### Python Packages
```bash
pip3 install numpy matplotlib
```

### Required Data
- Binary files `plane_XXXXX.bin` in current directory
- Generated with `-o 1` flag during execution

## File Sizes

Binary files can be large:
- 500×500 × 100 iterations ≈ 100 MB
- 1000×1000 × 200 iterations ≈ 800 MB

Clean up after visualization to save space.

## Binary File Format

- **Type**: `float32`
- **Order**: Row-major (C style)
- **Size**: `grid_x × grid_y` floats

The format is auto-detected by the script, but it's better to specify correct dimensions.
