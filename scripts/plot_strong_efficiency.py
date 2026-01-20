#!/usr/bin/env python3
"""
Script to generate Strong Scaling Efficiency plot from CSV data.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path):
    """Load CSV data and calculate efficiency for each configuration."""
    
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
        
        # Calculate efficiency: efficiency = (speedup / nodes) * 100%
        config_grouped['Efficiency'] = (config_grouped['Speedup'] / config_grouped['Nodes']) * 100.0
        
        # Store results
        results[config_name] = {
            'nodes': config_grouped['Nodes'].values,
            'efficiency': config_grouped['Efficiency'].values,
            'speedup': config_grouped['Speedup'].values,
            'runtime': config_grouped['TotalTime'].values,
            'baseline_runtime': baseline_runtime
        }
    
    return results

def plot_strong_efficiency(results, output_path):
    """Generate strong scaling efficiency plot."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')  # Set figure background to white
    ax.set_facecolor('white')  # Set plot area background to white
    
    # Plot ideal efficiency line (100%)
    nodes = np.array([1, 2, 4, 8, 16])
    ideal_efficiency = np.ones_like(nodes) * 100.0
    ax.axhline(y=100.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
              label='Ideal Efficiency', zorder=1)
    
    # Plot data for each configuration
    colors = {'16×7': '#1f77b4', '2×56': '#ff7f0e', '8×14': '#2ca02c'}
    markers = {'16×7': 'o', '2×56': 's', '8×14': '^'}
    
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            ax.plot(data['nodes'], data['efficiency'], 
                   marker=markers[config_name], 
                   linewidth=2, 
                   markersize=8,
                   color=colors[config_name],
                   label=f'{config_name}',
                   zorder=2)
    
    # Customize plot
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax.set_title('Strong Scaling Parallel Efficiency', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(bottom=0)
    
    # Set x-axis ticks only on desired values (same as strong_speedup)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nStrong Scaling Efficiency Summary:")
    print("-" * 50)
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            print(f"\n{config_name} Configuration:")
            for i, (n, e, s) in enumerate(zip(data['nodes'], data['efficiency'], data['speedup'])):
                print(f"  {n} nodes: {e:.1f}% (speedup: {s:.2f}×, runtime: {data['runtime'][i]:.2f}s)")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    output_path = os.path.join(project_root, 'figures', 'strong_efficiency.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path)
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_strong_efficiency(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
