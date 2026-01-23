#!/usr/bin/env python3
"""
Script to generate Strong Scaling Speedup plot from CSV data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path):
    """Load CSV data and calculate speedup for each configuration."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for ofast build variant and 1 energy source (baseline data)
    df = df[(df['BuildVariant'] == 'ofast') & (df['EnergySources'] == 1)]
    
    # Define configurations
    configs = {
        '16×7': {'TasksPerNode': 16, 'ThreadsPerTask': 7},
        '2×56': {'TasksPerNode': 2, 'ThreadsPerTask': 56},
        '8×14': {'TasksPerNode': 8, 'ThreadsPerTask': 14}
    }
    
    results = {}
    
    for config_name, config_params in configs.items():
        # Filter data for this configuration
        config_df = df[
            (df['TasksPerNode'] == config_params['TasksPerNode']) &
            (df['ThreadsPerTask'] == config_params['ThreadsPerTask'])
        ].copy()
        
        # Group by nodes and take mean runtime (to handle duplicate runs)
        # Alternative: use .min() for best performance, or .max() for worst case
        config_grouped = config_df.groupby('Nodes')['TotalTime'].mean().reset_index()
        config_grouped = config_grouped.sort_values('Nodes')
        
        # Get baseline runtime (1 node)
        baseline = config_grouped[config_grouped['Nodes'] == 1]
        if len(baseline) == 0:
            print(f"Warning: No baseline data found for {config_name}")
            continue
        
        baseline_runtime = baseline['TotalTime'].iloc[0]
        
        # Calculate speedup: speedup = baseline_runtime / runtime
        config_grouped['Speedup'] = baseline_runtime / config_grouped['TotalTime']
        
        # Store results
        results[config_name] = {
            'nodes': config_grouped['Nodes'].values,
            'speedup': config_grouped['Speedup'].values,
            'runtime': config_grouped['TotalTime'].values,
            'baseline_runtime': baseline_runtime
        }
    
    return results

def plot_strong_speedup(results, output_path):
    """Generate strong scaling speedup plot."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot ideal scaling line
    nodes = np.array([1, 2, 4, 8, 16])
    ideal_speedup = nodes
    ax.plot(nodes, ideal_speedup, 'k--', linewidth=1.5, alpha=0.5, label='Ideal Scaling', zorder=1)
    
    # Plot data for 16×7 configuration only (best performance)
    config_color = '#666666'  # Same color as execution_time total line
    config_marker = 'o'
    
    if '16×7' in results:
        data = results['16×7']
        ax.plot(data['nodes'], data['speedup'], 
               marker=config_marker, 
               linewidth=2, 
               markersize=8,
               color=config_color,
               label='16×7',
               zorder=2)
    
    # Customize plot
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax.set_title('Strong Scaling Speedup', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nStrong Scaling Speedup Summary:")
    print("-" * 50)
    if '16×7' in results:
        data = results['16×7']
        print(f"\n16×7 Configuration:")
        for i, (n, s) in enumerate(zip(data['nodes'], data['speedup'])):
            print(f"  {n} nodes: {s:.2f}× (runtime: {data['runtime'][i]:.2f}s)")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    output_path = os.path.join(project_root, 'figures', 'strong_speedup.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path)
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_strong_speedup(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
