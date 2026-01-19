#!/usr/bin/env python3
"""
Test with clearer colormap to avoid perceptual confusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import glob

def read_plane_binary(filename, shape):
    """Reads a binary file plane_XXXXX.bin."""
    try:
        data = np.fromfile(filename, dtype=np.float32)
        if len(data) != shape[0] * shape[1]:
            total = len(data)
            side = int(np.sqrt(total))
            if side * side == total:
                shape = (side, side)
        return data.reshape(shape)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def find_plane_files(pattern="plane_*.bin"):
    """Finds all plane_*.bin files and sorts them."""
    files = sorted(glob.glob(pattern))
    return files

# Read data (use existing files if available)
plane_files = find_plane_files()
if not plane_files:
    print("❌ No plane_*.bin files found!")
    print("Generating test data...")
    data = np.random.rand(200, 200) * 50
    data[90:110, 90:110] += 100  # Central source
else:
    shape = None
    for filename in plane_files:
        try:
            test_data = np.fromfile(filename, dtype=np.float32)
            total = len(test_data)
            side = int(np.sqrt(total))
            if side * side == total:
                shape = (side, side)
                break
        except:
            continue
    
    if shape is None:
        print("❌ Unable to determine shape")
        exit(1)
    
    data = read_plane_binary(plane_files[len(plane_files)//2], shape)

# Set white background
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

# Colormaps to test (clearer and more intuitive)
colormaps = {
    'viridis': 'Viridis (green → yellow)',
    'inferno': 'Inferno (black → yellow)',
    'magma': 'Magma (black → yellow)',
    'turbo': 'Turbo (blue → green → yellow → red)',
    'hot': 'Hot (black → red → yellow)',
    'copper': 'Copper (black → copper)',
    'bone': 'Bone (black → white)',
    'pink': 'Pink (white → pink)'
}

# Create comparison
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
fig.patch.set_facecolor('white')

for idx, (cmap_name, cmap_label) in enumerate(colormaps.items()):
    row = idx // 4
    col = idx % 4
    ax = axes[row, col]
    
    # Plot with colormap
    im = ax.imshow(data, cmap=cmap_name, interpolation='nearest')
    ax.set_title(f'{cmap_label}\n({cmap_name})', fontsize=12, pad=10, fontweight='bold')
    ax.axis('off')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Add perceptual evaluation
    if cmap_name == 'viridis':
        cbar.ax.text(1.15, 0.5, f'GREEN = low\nYELLOW = high\n✅ CLEAR!', 
                    transform=cbar.ax.transAxes, fontsize=8, 
                    verticalalignment='center', color='green', weight='bold')
    elif cmap_name == 'turbo':
        cbar.ax.text(1.15, 0.5, f'BLUE = low\nYELLOW = high\n✅ CLEAR!', 
                    transform=cbar.ax.transAxes, fontsize=8, 
                    verticalalignment='center', color='blue', weight='bold')
    elif cmap_name == 'hot':
        cbar.ax.text(1.15, 0.5, f'BLACK = low\nYELLOW = high\n⚠️ BLACK confusing', 
                    transform=cbar.ax.transAxes, fontsize=8, 
                    verticalalignment='center', color='red', weight='bold')
    elif cmap_name == 'plasma':
        cbar.ax.text(1.15, 0.5, f'PURPLE = low\nYELLOW = high\n❌ CONFUSING!', 
                    transform=cbar.ax.transAxes, fontsize=8, 
                    verticalalignment='center', color='purple', weight='bold')

plt.suptitle('Clear Colormap Test - Avoid Perceptual Confusion\nWhite = Zero Energy, Color = Increasing Energy', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('test_clear_colormaps.png', dpi=200, bbox_inches='tight')
plt.close()

print("✅ Saved: test_clear_colormaps.png")
print("\n🎯 RECOMMENDATIONS:")
print("   🥇 VIRIDIS: Green → Yellow (very clear)")
print("   🥈 TURBO: Blue → Green → Yellow → Red (colorful)")
print("   🥉 INFERNO: Black → Yellow (dramatic)")
print("   ❌ PLASMA: Purple → Yellow (confusing with white background)")
print("\n💡 With white background, VIRIDIS is the most intuitive!")
