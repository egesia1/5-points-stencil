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
    """Load CSV data and calculate efficiency and runtime for periodic and non-periodic cases."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for specified build variant and 1 energy source (baseline)
    df = df[(df['BuildVariant'] == build_variant) & (df['EnergySources'] == 1)].copy()
    
    # Filter for 16×7 configuration
    config_df = df[
        (df['TasksPerNode'] == 16) &
        (df['ThreadsPerTask'] == 7)
    ].copy()
    
    results = {}
    
    # Process non-periodic (Periodic=0)
    periodic0_df = config_df[config_df['Periodic'] == 0].copy()
    periodic0_grouped = periodic0_df.groupby('Nodes')['TotalTime'].mean().reset_index()
    periodic0_grouped = periodic0_grouped.sort_values('Nodes')
    
    if len(periodic0_grouped) > 0:
        baseline0 = periodic0_grouped[periodic0_grouped['Nodes'] == 1]['TotalTime'].iloc[0]
        periodic0_grouped['Efficiency'] = (baseline0 / periodic0_grouped['TotalTime']) * 100.0
        
        results['Non-Periodic'] = {
            'nodes': periodic0_grouped['Nodes'].values,
            'runtime': periodic0_grouped['TotalTime'].values,
            'efficiency': periodic0_grouped['Efficiency'].values,
            'baseline_runtime': baseline0
        }
    
    # Process periodic (Periodic=1)
    periodic1_df = config_df[config_df['Periodic'] == 1].copy()
    periodic1_grouped = periodic1_df.groupby('Nodes')['TotalTime'].mean().reset_index()
    periodic1_grouped = periodic1_grouped.sort_values('Nodes')
    
    if len(periodic1_grouped) > 0:
        baseline1 = periodic1_grouped[periodic1_grouped['Nodes'] == 1]['TotalTime'].iloc[0]
        periodic1_grouped['Efficiency'] = (baseline1 / periodic1_grouped['TotalTime']) * 100.0
        
        results['Periodic'] = {
            'nodes': periodic1_grouped['Nodes'].values,
            'runtime': periodic1_grouped['TotalTime'].values,
            'efficiency': periodic1_grouped['Efficiency'].values,
            'baseline_runtime': baseline1
        }
    
    return results

def plot_weak_scaling(results, output_path):
    """Generate weak scaling plots with 2 subplots: Runtime and Efficiency."""
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Colors and markers for periodic and non-periodic
    colors = {'Non-Periodic': '#1f77b4', 'Periodic': '#ff7f0e'}
    markers = {'Non-Periodic': 'o', 'Periodic': 's'}
    
    # ========== LEFT SUBPLOT: RUNTIME ==========
    ax_runtime = axes[0]
    ax_runtime.set_facecolor('white')  # Set subplot background to white
    
    # Plot baseline runtime lines (ideal weak scaling: runtime remains constant)
    if 'Non-Periodic' in results:
        baseline0 = results['Non-Periodic']['baseline_runtime']
        ax_runtime.axhline(y=baseline0, color=colors['Non-Periodic'], linestyle='--', 
                          linewidth=1.5, alpha=0.3, label='Ideal (Non-Periodic)', zorder=1)
    
    if 'Periodic' in results:
        baseline1 = results['Periodic']['baseline_runtime']
        ax_runtime.axhline(y=baseline1, color=colors['Periodic'], linestyle='--', 
                          linewidth=1.5, alpha=0.3, label='Ideal (Periodic)', zorder=1)
    
    # Plot runtime for each boundary condition
    for condition in ['Non-Periodic', 'Periodic']:
        if condition in results:
            data = results[condition]
            ax_runtime.plot(data['nodes'], data['runtime'], 
                           marker=markers[condition], 
                           linewidth=2, 
                           markersize=8,
                           color=colors[condition],
                           label=condition,
                           zorder=2)
    
    # Customize runtime plot
    ax_runtime.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_runtime.set_ylabel('Runtime (seconds)', fontsize=12, fontweight='bold')
    ax_runtime.set_title('Weak Scaling Runtime (16×7 Configuration)', fontsize=14, fontweight='bold')
    ax_runtime.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_runtime.grid(True, alpha=0.3, linestyle='--')
    ax_runtime.set_xlim(0.5, 16.5)
    ax_runtime.set_ylim(bottom=0, top=14)
    
    # Set x-axis ticks
    ax_runtime.set_xticks([1, 2, 4, 8, 16])
    ax_runtime.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # ========== RIGHT SUBPLOT: EFFICIENCY ==========
    ax_efficiency = axes[1]
    ax_efficiency.set_facecolor('white')  # Set subplot background to white
    
    # Plot ideal efficiency line (100%)
    ax_efficiency.axhline(y=100.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
                         label='Ideal Efficiency', zorder=1)
    
    # Plot efficiency for each boundary condition
    for condition in ['Non-Periodic', 'Periodic']:
        if condition in results:
            data = results[condition]
            ax_efficiency.plot(data['nodes'], data['efficiency'], 
                              marker=markers[condition], 
                              linewidth=2, 
                              markersize=8,
                              color=colors[condition],
                              label=condition,
                              zorder=2)
    
    # Customize efficiency plot
    ax_efficiency.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_efficiency.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax_efficiency.set_title('Weak Scaling Parallel Efficiency (16×7 Configuration)', fontsize=14, fontweight='bold')
    ax_efficiency.legend(loc='best', fontsize=10, framealpha=0.9)
    ax_efficiency.grid(True, alpha=0.3, linestyle='--')
    ax_efficiency.set_xlim(0.5, 16.5)
    ax_efficiency.set_ylim(bottom=0, top=120)
    
    # Set x-axis ticks
    ax_efficiency.set_xticks([1, 2, 4, 8, 16])
    ax_efficiency.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nWeak Scaling Summary:")
    print("-" * 70)
    for condition in ['Non-Periodic', 'Periodic']:
        if condition in results:
            data = results[condition]
            print(f"\n{condition} (16×7 Configuration):")
            for i, (n, r, e) in enumerate(zip(data['nodes'], data['runtime'], data['efficiency'])):
                print(f"  {n} nodes: Runtime={r:.2f}s, Efficiency={e:.1f}%")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_weak_parallel_results_corrected.csv')
    output_path = os.path.join(project_root, 'figures', 'weak_efficiency.png')
    
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
