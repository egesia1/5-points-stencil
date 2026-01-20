#!/usr/bin/env python3
"""
Script to generate OpenMP Scaling Speedup plot from CSV data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path):
    """Load CSV data and calculate speedup for each energy source configuration."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for ofast_omp_improved build variant (the optimized version)
    df = df[df['BuildVariant'] == 'ofast_omp_improved'].copy()
    
    # Define thread counts we want to show (in order)
    desired_threads = [1, 2, 4, 8, 16, 32, 56, 84, 112]
    
    # Define energy sources to plot
    energy_sources = [1, 2, 8]
    
    results = {}
    
    for energy_src in energy_sources:
        # Filter data for this energy source
        energy_df = df[df['EnergySources'] == energy_src].copy()
        
        # Group by threads and take mean runtime (to handle duplicate runs)
        energy_grouped = energy_df.groupby('Threads')['TotalTime'].mean().reset_index()
        energy_grouped = energy_grouped.sort_values('Threads')
        
        # Get baseline runtime (1 thread)
        baseline = energy_grouped[energy_grouped['Threads'] == 1]
        if len(baseline) == 0:
            print(f"Warning: No baseline data found for {energy_src} energy source(s)")
            continue
        
        baseline_runtime = baseline['TotalTime'].iloc[0]
        
        # Calculate speedup: speedup = baseline_runtime / runtime
        energy_grouped['Speedup'] = baseline_runtime / energy_grouped['TotalTime']
        
        # Filter to only include desired thread counts
        energy_grouped = energy_grouped[energy_grouped['Threads'].isin(desired_threads)]
        energy_grouped = energy_grouped.sort_values('Threads')
        
        # Store results
        results[energy_src] = {
            'threads': energy_grouped['Threads'].values,
            'speedup': energy_grouped['Speedup'].values,
            'runtime': energy_grouped['TotalTime'].values,
            'baseline_runtime': baseline_runtime
        }
    
    return results

def plot_omp_speedup(results, output_path):
    """Generate OpenMP scaling speedup plot with 3 subplots (one per energy source)."""
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Energy sources in order
    energy_sources = [1, 2, 8]
    
    # Define thread counts for ideal scaling line
    ideal_threads = np.array([1, 2, 4, 8, 16, 32, 56, 84, 112])
    
    for idx, energy_src in enumerate(energy_sources):
        ax = axes[idx]
        ax.set_facecolor('white')  # Set subplot background to white
        
        if energy_src not in results:
            ax.text(0.5, 0.5, f'No data for {energy_src} energy source(s)', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        data = results[energy_src]
        
        # Plot ideal scaling line
        ideal_speedup = ideal_threads
        ax.plot(ideal_threads, ideal_speedup, 'r--', linewidth=1.5, alpha=0.5, 
               label='Ideal Scaling', zorder=1)
        
        # Plot measured speedup
        ax.plot(data['threads'], data['speedup'], 
               marker='o', 
               linewidth=2, 
               markersize=6,
               color='#1f77b4',
               label='Measured Speedup',
               zorder=2)
        
        # Add annotations for specific thread counts (32, 56, 84, 112)
        annotate_threads = [32, 56, 84, 112]
        for thread_count in annotate_threads:
            # Find index of this thread count in the data
            thread_indices = np.where(data['threads'] == thread_count)[0]
            if len(thread_indices) > 0:
                idx = thread_indices[0]
                x_pos = data['threads'][idx]
                y_pos = data['speedup'][idx]
                # Annotate with speedup value higher above the point, in black
                ax.annotate(f'{y_pos:.2f}×',
                           xy=(x_pos, y_pos),
                           xytext=(x_pos, y_pos + y_pos * 0.15),  # 15% above the point (increased from 5%)
                           ha='center',
                           va='bottom',
                           fontsize=9,
                           fontweight='bold',
                           color='black')
        
        # Customize plot
        ax.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
        ax.set_title(f'{energy_src} Energy Source(s)', fontsize=12, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 115)
        ax.set_ylim(bottom=0)
        
        # Set x-axis ticks only on desired values
        desired_threads = [1, 2, 4, 8, 16, 32, 56, 84, 112]
        ax.set_xticks(desired_threads)
        ax.set_xticklabels([str(t) for t in desired_threads], rotation=45, ha='right')
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nOpenMP Scaling Speedup Summary:")
    print("-" * 60)
    for energy_src in energy_sources:
        if energy_src in results:
            data = results[energy_src]
            print(f"\n{energy_src} Energy Source(s):")
            for i, (t, s) in enumerate(zip(data['threads'], data['speedup'])):
                print(f"  {t} threads: {s:.2f}× (runtime: {data['runtime'][i]:.2f}s)")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_omp_serial_results.csv')
    output_path = os.path.join(project_root, 'figures', 'omp_speedup.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path)
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_omp_speedup(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
