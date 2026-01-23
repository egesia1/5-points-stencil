#!/usr/bin/env python3
"""
Script to generate Configuration Comparison plot for Strong Scaling.
Shows runtime comparison across the three configurations on a single plot.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path, build_variant='ofast'):
    """Load CSV data and extract runtime for each configuration."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for specified build variant and 1 energy source (baseline)
    df = df[(df['BuildVariant'] == build_variant) & (df['EnergySources'] == 1)].copy()
    
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
        
        # Store results
        results[config_name] = {
            'nodes': config_grouped['Nodes'].values,
            'runtime': config_grouped['TotalTime'].values
        }
    
    return results

def plot_config_comparison(results, output_path):
    """Generate configuration comparison plot showing runtime for all three configurations."""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')  # Set figure background to white
    ax.set_facecolor('white')  # Set plot area background to white
    
    # Colors and markers for each configuration
    colors = {'16×7': '#1f77b4', '2×56': '#ff7f0e', '8×14': '#2ca02c'}
    markers = {'16×7': 'o', '2×56': 's', '8×14': '^'}
    
    # Plot data for each configuration
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            ax.plot(data['nodes'], data['runtime'], 
                   marker=markers[config_name], 
                   linewidth=2, 
                   markersize=8,
                   color=colors[config_name],
                   label=f'{config_name}',
                   zorder=2)
    
    # Customize plot
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Configuration Comparison (Strong Scaling)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(bottom=0)
    
    # Set x-axis ticks only on desired values (same as other strong scaling plots)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nConfiguration Comparison Summary:")
    print("-" * 50)
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            print(f"\n{config_name} Configuration:")
            for i, (n, r) in enumerate(zip(data['nodes'], data['runtime'])):
                print(f"  {n} nodes: {r:.2f}s")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    output_path = os.path.join(project_root, 'figures', 'config_comparison.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check which build variant has more data available
    # Try ofast_omp_improved first (if available), otherwise use ofast
    df = pd.read_csv(csv_path)
    df_filtered = df[(df['TestType'] == 'strong') & (df['EnergySources'] == 1)]
    ofast_omp_count = len(df_filtered[df_filtered['BuildVariant'] == 'ofast_omp_improved'])
    ofast_count = len(df_filtered[df_filtered['BuildVariant'] == 'ofast'])
    
    if ofast_omp_count >= ofast_count and ofast_omp_count > 0:
        build_variant = 'ofast_omp_improved'
        print(f"Using build variant: {build_variant} ({ofast_omp_count} data points)")
    else:
        build_variant = 'ofast'
        print(f"Using build variant: {build_variant} ({ofast_count} data points)")
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path, build_variant=build_variant)
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_config_comparison(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
