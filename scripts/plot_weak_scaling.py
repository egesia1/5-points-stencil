#!/usr/bin/env python3
"""
Script to generate Weak Scaling plots: Runtime, Efficiency, and Configuration Comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path, build_variant='ofast'):
    """Load CSV data and calculate efficiency and runtime for each configuration."""
    
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
        
        # Get baseline runtime (1 node)
        baseline = config_grouped[config_grouped['Nodes'] == 1]
        if len(baseline) == 0:
            print(f"Warning: No baseline data found for {config_name}")
            continue
        
        baseline_runtime = baseline['TotalTime'].iloc[0]
        
        # For weak scaling, efficiency = (baseline_runtime / runtime) * 100%
        # In ideal weak scaling, runtime should remain constant, so efficiency should be ~100%
        config_grouped['Efficiency'] = (baseline_runtime / config_grouped['TotalTime']) * 100.0
        
        # Store results
        results[config_name] = {
            'nodes': config_grouped['Nodes'].values,
            'runtime': config_grouped['TotalTime'].values,
            'efficiency': config_grouped['Efficiency'].values,
            'baseline_runtime': baseline_runtime
        }
    
    return results

def plot_weak_scaling(results, output_path):
    """Generate weak scaling plots with 3 subplots: Runtime, Efficiency, and Configuration Comparison."""
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Colors and markers for each configuration
    colors = {'16×7': '#1f77b4', '2×56': '#ff7f0e', '8×14': '#2ca02c'}
    markers = {'16×7': 'o', '2×56': 's', '8×14': '^'}
    
    # ========== LEFT SUBPLOT: RUNTIME ==========
    ax_runtime = axes[0]
    ax_runtime.set_facecolor('white')  # Set subplot background to white
    
    # Plot baseline runtime line (ideal weak scaling: runtime remains constant)
    # We'll plot a reference line at the average of baseline runtimes
    baseline_avg = np.mean([results[cfg]['baseline_runtime'] for cfg in results.keys()])
    nodes = np.array([1, 2, 4, 8, 16])
    ax_runtime.axhline(y=baseline_avg, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
                      label='Ideal Weak Scaling', zorder=1)
    
    # Plot runtime for each configuration
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            ax_runtime.plot(data['nodes'], data['runtime'], 
                           marker=markers[config_name], 
                           linewidth=2, 
                           markersize=8,
                           color=colors[config_name],
                           label=f'{config_name}',
                           zorder=2)
    
    # Customize runtime plot
    ax_runtime.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_runtime.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax_runtime.set_title('Weak Scaling Runtime', fontsize=14, fontweight='bold')
    ax_runtime.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_runtime.grid(True, alpha=0.3, linestyle='--')
    ax_runtime.set_xlim(0.5, 16.5)
    ax_runtime.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax_runtime.set_xticks([1, 2, 4, 8, 16])
    ax_runtime.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # ========== MIDDLE SUBPLOT: EFFICIENCY ==========
    ax_efficiency = axes[1]
    ax_efficiency.set_facecolor('white')  # Set subplot background to white
    
    # Plot ideal efficiency line (100%)
    ax_efficiency.axhline(y=100.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
                         label='Ideal Efficiency', zorder=1)
    
    # Plot efficiency for each configuration
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            ax_efficiency.plot(data['nodes'], data['efficiency'], 
                              marker=markers[config_name], 
                              linewidth=2, 
                              markersize=8,
                              color=colors[config_name],
                              label=f'{config_name}',
                              zorder=2)
    
    # Customize efficiency plot
    ax_efficiency.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_efficiency.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax_efficiency.set_title('Weak Scaling Parallel Efficiency', fontsize=14, fontweight='bold')
    ax_efficiency.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_efficiency.grid(True, alpha=0.3, linestyle='--')
    ax_efficiency.set_xlim(0.5, 16.5)
    ax_efficiency.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax_efficiency.set_xticks([1, 2, 4, 8, 16])
    ax_efficiency.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # ========== RIGHT SUBPLOT: CONFIGURATION COMPARISON (RUNTIME) ==========
    ax_comparison = axes[2]
    ax_comparison.set_facecolor('white')  # Set subplot background to white
    
    # Plot runtime for each configuration
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            ax_comparison.plot(data['nodes'], data['runtime'], 
                              marker=markers[config_name], 
                              linewidth=2, 
                              markersize=8,
                              color=colors[config_name],
                              label=f'{config_name}',
                              zorder=2)
    
    # Customize comparison plot
    ax_comparison.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_comparison.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax_comparison.set_title('Configuration Comparison', fontsize=14, fontweight='bold')
    ax_comparison.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_comparison.grid(True, alpha=0.3, linestyle='--')
    ax_comparison.set_xlim(0.5, 16.5)
    ax_comparison.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax_comparison.set_xticks([1, 2, 4, 8, 16])
    ax_comparison.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nWeak Scaling Summary:")
    print("-" * 70)
    for config_name in ['16×7', '2×56', '8×14']:
        if config_name in results:
            data = results[config_name]
            print(f"\n{config_name} Configuration:")
            for i, (n, r, e) in enumerate(zip(data['nodes'], data['runtime'], data['efficiency'])):
                print(f"  {n} nodes: Runtime={r:.2f}s, Efficiency={e:.1f}%")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_weak_parallel_results.csv')
    output_path = os.path.join(project_root, 'figures', 'weak_scaling.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Check which build variant has more data available
    # Try ofast_omp_improved first (if available), otherwise use ofast
    df = pd.read_csv(csv_path)
    df_filtered = df[(df['TestType'] == 'weak') & (df['EnergySources'] == 1)]
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
    plot_weak_scaling(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
