#!/usr/bin/env python3
"""
3D visualization of heat diffusion.

The height of the 3D surface represents the energy at each point of the grid.
Much more intuitive than the 2D map!

Usage:
    python scripts/other/visualize_heat_3d.py [--output OUTPUT_DIR] [--interactive]
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import glob
import os
import argparse
from pathlib import Path

def read_plane_binary(filename, shape):
    """Reads a binary file plane_XXXXX.bin."""
    try:
        data = np.fromfile(filename, dtype=np.float32)
        if len(data) != shape[0] * shape[1]:
            total = len(data)
            side = int(np.sqrt(total))
            if side * side == total:
                shape = (side, side)
            else:
                for w in range(100, 10000):
                    if total % w == 0:
                        h = total // w
                        shape = (h, w)
                        break
        return data.reshape(shape)
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None

def infer_grid_shape(filename):
    """Infers the grid shape from the file."""
    data = np.fromfile(filename, dtype=np.float32)
    total = len(data)
    
    side = int(np.sqrt(total))
    if side * side == total:
        return (side, side)
    
    for w in [100, 200, 500, 1000, 2000, 5000]:
        if total % w == 0:
            h = total // w
            if abs(h - w) < w * 0.5:
                return (h, w)
    
    for w in range(100, int(np.sqrt(total)) + 1):
        if total % w == 0:
            h = total // w
            return (h, w)
    
    return None

def find_plane_files(pattern="plane_*.bin"):
    """Finds all plane_*.bin files and sorts them."""
    files = sorted(glob.glob(pattern))
    return files

def create_3d_surface(data, ax, title="", vmin=None, vmax=None, elev=30, azim=45):
    """
    Creates a 3D surface plot of the data.
    
    Args:
        data: 2D array with energy values
        ax: matplotlib Axes
        title: Plot title
        vmin, vmax: Value range for colormap
        elev, azim: Viewing angles
    """
    ax.clear()
    
    # Set white background for 3D axes
    ax.set_facecolor('white')
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.3)
    ax.yaxis.pane.set_alpha(0.3)
    ax.zaxis.pane.set_alpha(0.3)
    ax.grid(True, alpha=0.3, color='gray')
    
    # Create mesh grid
    rows, cols = data.shape
    X = np.arange(0, cols, 1)
    Y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(X, Y)
    
    # Plot 3D surface
    surf = ax.plot_surface(X, Y, data, cmap=cm.viridis, 
                           vmin=vmin, vmax=vmax,
                           linewidth=0, antialiased=True,
                           alpha=0.9, shade=True)
    
    # Style
    ax.set_xlabel('X', fontsize=10, labelpad=8, color='black')
    ax.set_ylabel('Y', fontsize=10, labelpad=8, color='black')
    ax.set_zlabel('Energy', fontsize=10, labelpad=8, color='black')
    ax.set_title(title, fontsize=12, pad=10, color='black')
    ax.view_init(elev=elev, azim=azim)
    
    # Z limits for consistency
    if vmin is not None and vmax is not None:
        ax.set_zlim(vmin, vmax)
    
    return surf

def create_3d_visualization(output_dir="viz_output_3d", interactive=False):
    """
    Creates 3D visualizations of heat diffusion.
    
    Args:
        output_dir: Directory to save results
        interactive: If True, also creates interactive visualization (Plotly)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style with white background
    plt.style.use('default')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['savefig.facecolor'] = 'white'
    
    # Find files
    plane_files = find_plane_files()
    
    if not plane_files:
        print("❌ No plane_*.bin files found!")
        print("Run first: ./build/local/stencil_serial_omp -x 500 -y 500 -e 3 -n 50 -o 1")
        return
    
    print(f"✓ Found {len(plane_files)} files")
    
    # Infer shape
    shape = infer_grid_shape(plane_files[0])
    if shape is None:
        print("❌ Unable to determine grid shape")
        return
    
    print(f"✓ Grid shape: {shape[0]} x {shape[1]}")
    
    # Read all data
    frames = []
    energies = []
    
    print("📖 Reading data...", end=" ", flush=True)
    for filename in plane_files:
        data = read_plane_binary(filename, shape)
        if data is not None:
            frames.append(data)
            energies.append(np.sum(data))
    print(f"✓ {len(frames)} frames")
    
    if not frames:
        print("❌ No valid data found")
        return
    
    # Global range for consistency
    vmin = min(np.min(f) for f in frames)
    vmax = max(np.max(f) for f in frames)
    
    # Downsample if grid is too large (for performance)
    if shape[0] > 300 or shape[1] > 300:
        print(f"⚠️  Large grid ({shape[0]}×{shape[1]}), downsampling for performance...")
        step = max(shape[0] // 200, shape[1] // 200)
        frames = [f[::step, ::step] for f in frames]
        shape = frames[0].shape
        print(f"   New size: {shape[0]}×{shape[1]}")
    
    # 1. SINGLE SURFACE PLOTS (various angles)
    print("🎨 Creating surface plots...", end=" ", flush=True)
    
    # Select key frames
    key_frames = [0, len(frames)//4, len(frames)//2, 3*len(frames)//4, -1]
    
    fig = plt.figure(figsize=(20, 12))
    fig.patch.set_facecolor('white')
    
    for idx, frame_idx in enumerate(key_frames):
        for view_idx, (elev, azim) in enumerate([(30, 45), (20, 120)]):
            ax = fig.add_subplot(len(key_frames), 2, idx*2 + view_idx + 1, projection='3d')
            create_3d_surface(frames[frame_idx], ax, 
                            f'Step {frame_idx if frame_idx >= 0 else len(frames)-1} - Vista {view_idx+1}',
                            vmin, vmax, elev, azim)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/surface_plots_multi_view.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("✓")
    
    # 2. ROTATING 3D ANIMATION
    print("🎬 Creating 3D animation...", end=" ", flush=True)
    
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111, projection='3d')
    
    def animate_3d(frame_idx):
        # Rotate view during animation
        azim = 45 + (frame_idx * 2) % 360  # Full rotation
        surf = create_3d_surface(frames[frame_idx], ax,
                                f'Heat Diffusion - Step {frame_idx}/{len(frames)-1}',
                                vmin, vmax, elev=25, azim=azim)
        return surf,
    
    # Downsample frames for smoother animation
    anim_frames = min(len(frames), 120)  # Max 120 frames for GIF
    frame_step = max(1, len(frames) // anim_frames)
    
    anim = animation.FuncAnimation(fig, animate_3d, 
                                   frames=range(0, len(frames), frame_step),
                                   interval=100, blit=False, repeat=True)
    
    # Save GIF
    gif_path = f'{output_dir}/heat_diffusion_3d.gif'
    anim.save(gif_path, writer='pillow', fps=10, dpi=80)
    print(f"✓ ({gif_path})")
    
    # Save MP4 if possible
    try:
        mp4_path = f'{output_dir}/heat_diffusion_3d.mp4'
        anim.save(mp4_path, writer='ffmpeg', fps=15, dpi=100, bitrate=3000)
        print(f"   ✓ Video saved: {mp4_path}")
    except Exception as e:
        print(f"   ⚠️  MP4 not available (ffmpeg?)")
    
    plt.close()
    
    # 3. 2D vs 3D COMPARISON
    print("📊 Creating 2D vs 3D comparison...", end=" ", flush=True)
    
    fig = plt.figure(figsize=(18, 8))
    fig.patch.set_facecolor('white')
    
    for idx, frame_idx in enumerate([0, len(frames)//2, -1]):
        # 2D view (heatmap)
        ax1 = fig.add_subplot(2, 3, idx + 1)
        im = ax1.imshow(frames[frame_idx], cmap='viridis', interpolation='nearest')
        ax1.set_title(f'2D Heatmap - Step {frame_idx if frame_idx >= 0 else len(frames)-1}')
        ax1.axis('off')
        plt.colorbar(im, ax=ax1, fraction=0.046)
        
        # 3D view
        ax2 = fig.add_subplot(2, 3, idx + 4, projection='3d')
        create_3d_surface(frames[frame_idx], ax2,
                         f'3D Surface - Step {frame_idx if frame_idx >= 0 else len(frames)-1}',
                         vmin, vmax, elev=35, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/comparison_2d_vs_3d.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("✓")
    
    # 4. WIREFRAME + CONTOUR
    print("📐 Creating wireframe and contour plots...", end=" ", flush=True)
    
    fig = plt.figure(figsize=(18, 6))
    fig.patch.set_facecolor('white')
    
    for idx, frame_idx in enumerate([0, len(frames)//2, -1]):
        data = frames[frame_idx]
        rows, cols = data.shape
        X = np.arange(0, cols, 1)
        Y = np.arange(0, rows, 1)
        X, Y = np.meshgrid(X, Y)
        
        # Wireframe
        ax1 = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # Set white background
        ax1.set_facecolor('white')
        ax1.xaxis.pane.fill = False
        ax1.yaxis.pane.fill = False
        ax1.zaxis.pane.fill = False
        ax1.xaxis.pane.set_edgecolor('gray')
        ax1.yaxis.pane.set_edgecolor('gray')
        ax1.zaxis.pane.set_edgecolor('gray')
        ax1.grid(True, alpha=0.3, color='gray')
        
        ax1.plot_wireframe(X, Y, data, cmap=cm.viridis, linewidth=0.5, alpha=0.7)
        ax1.contour(X, Y, data, zdir='z', offset=vmin, cmap=cm.viridis, levels=10, alpha=0.5)
        ax1.set_xlabel('X', color='black')
        ax1.set_ylabel('Y', color='black')
        ax1.set_zlabel('Energy', color='black')
        ax1.set_title(f'Wireframe + Contour - Step {frame_idx if frame_idx >= 0 else len(frames)-1}', color='black')
        ax1.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/wireframe_contour.png', dpi=120, bbox_inches='tight')
    plt.close()
    print("✓")
    
    # 5. ENERGY PLOT (like before but more detailed)
    print("📈 Creating energy plot...", end=" ", flush=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Total energy
    ax1.plot(energies, 'b-', linewidth=2, label='Total Energy')
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Total Energy', fontsize=12)
    ax1.set_title('Total Energy Evolution in the System', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Percentage variation
    if len(energies) > 1:
        energy_change = [(energies[i] - energies[0]) / energies[0] * 100 
                         for i in range(len(energies))]
        ax2.plot(energy_change, 'r-', linewidth=2, label='Variation %')
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax2.set_xlabel('Iteration', fontsize=12)
        ax2.set_ylabel('Energy Variation (%)', fontsize=12)
        ax2.set_title('Percentage Energy Variation', fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/energy_evolution_detailed.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓")
    
    # 6. INTERACTIVE VISUALIZATION (Plotly) if requested
    if interactive:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            
            print("🌐 Creating interactive visualization (Plotly)...", end=" ", flush=True)
            
            # Select subset of frames to avoid overloading
            interactive_frames = min(len(frames), 50)
            frame_step = max(1, len(frames) // interactive_frames)
            
            # Create figure with slider
            data_frames = []
            for i in range(0, len(frames), frame_step):
                data = frames[i]
                rows, cols = data.shape
                x = np.arange(0, cols, 1)
                y = np.arange(0, rows, 1)
                
                data_frames.append(go.Frame(
                    data=[go.Surface(x=x, y=y, z=data, colorscale='Hot')],
                    name=str(i)
                ))
            
            # Initial frame
            data0 = frames[0]
            rows, cols = data0.shape
            x = np.arange(0, cols, 1)
            y = np.arange(0, rows, 1)
            
            fig = go.Figure(
                data=[go.Surface(x=x, y=y, z=data0, colorscale='Hot')],
                frames=data_frames
            )
            
            # Add slider and buttons
            fig.update_layout(
                title='Heat Diffusion - Interactive 3D Visualization',
                scene=dict(
                    xaxis_title='X',
                    yaxis_title='Y',
                    zaxis_title='Energy',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.3))
                ),
                updatemenus=[dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play', method='animate',
                             args=[None, dict(frame=dict(duration=100, redraw=True),
                                            fromcurrent=True)]),
                        dict(label='Pause', method='animate',
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                              mode='immediate')])
                    ]
                )],
                sliders=[dict(
                    steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                    mode='immediate')],
                               method='animate',
                               label=f'Step {f.name}')
                          for f in data_frames],
                    active=0,
                    y=0,
                    len=0.9,
                    x=0.1
                )]
            )
            
            html_path = f'{output_dir}/heat_diffusion_interactive.html'
            fig.write_html(html_path)
            print(f"✓ ({html_path})")
            print(f"   Open in browser for interactive visualization!")
            
        except ImportError:
            print("⚠️  Plotly not installed (optional)")
            print("   Install with: pip install plotly")
    
    # SUMMARY
    print("\n" + "="*70)
    print("✅ 3D VISUALIZATION COMPLETED!")
    print("="*70)
    print(f"\n📁 Files saved in: {output_dir}/\n")
    print("  🎨 surface_plots_multi_view.png  - Surface plots from various angles")
    print("  🎬 heat_diffusion_3d.gif         - Rotating 3D animation")
    print("  📊 comparison_2d_vs_3d.png       - 2D vs 3D comparison")
    print("  📐 wireframe_contour.png         - Wireframe + contour")
    print("  📈 energy_evolution_detailed.png - Detailed energy plot")
    if interactive:
        print("  🌐 heat_diffusion_interactive.html - Interactive visualization (open in browser)")
    print("\n" + "="*70)
    print("\n💡 3D visualization shows energy as SURFACE HEIGHT!")
    print("   - High peaks = Concentrated energy")
    print("   - Low valleys = Low energy")
    print("   - See diffusion in a much more intuitive way!\n")

def main():
    parser = argparse.ArgumentParser(
        description='Visualize heat diffusion in 3D',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Standard 3D visualization
  python scripts/other/visualize_heat_3d.py
  
  # With custom output
  python scripts/other/visualize_heat_3d.py --output my_3d_results
  
  # With interactive Plotly visualization
  python scripts/other/visualize_heat_3d.py --interactive
        """
    )
    parser.add_argument('--output', '-o', default='viz_output_3d',
                        help='Output directory for files (default: viz_output_3d)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Also create interactive visualization with Plotly')
    
    args = parser.parse_args()
    
    create_3d_visualization(args.output, args.interactive)

if __name__ == "__main__":
    main()
