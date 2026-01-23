#!/usr/bin/env python3
"""
Script to generate Strong Scaling Speedup and Efficiency plots combined in one figure.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path):
    """Load CSV data and calculate speedup and efficiency for each configuration."""
    
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
            'speedup': config_grouped['Speedup'].values,
            'efficiency': config_grouped['Efficiency'].values,
            'runtime': config_grouped['TotalTime'].values,
            'baseline_runtime': baseline_runtime
        }
    
    return results

def plot_strong_speedup_efficiency(results, output_path_speedup, output_path_efficiency, output_path_combined):
    """Generate combined strong scaling speedup and efficiency plots."""
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Color and marker for 16×7 configuration only (best performance)
    config_color = '#666666'  # Same color as execution_time total line
    config_marker = 'o'
    
    # Nodes for ideal lines
    nodes = np.array([1, 2, 4, 8, 16])
    
    # ========== LEFT SUBPLOT: SPEEDUP ==========
    ax_speedup = axes[0]
    ax_speedup.set_facecolor('white')  # Set subplot background to white
    
    # Plot ideal scaling line
    ideal_speedup = nodes
    ax_speedup.plot(nodes, ideal_speedup, 'k--', linewidth=1.5, alpha=0.5, 
                   label='Ideal Scaling', zorder=1)
    
    # Plot data for 16×7 configuration only
    if '16×7' in results:
        data = results['16×7']
        ax_speedup.plot(data['nodes'], data['speedup'], 
                       marker=config_marker, 
                       linewidth=2, 
                       markersize=8,
                       color=config_color,
                       label='16×7',
                       zorder=2)
    
    # Customize speedup plot
    ax_speedup.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_speedup.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax_speedup.set_title('Strong Scaling Speedup', fontsize=14, fontweight='bold')
    ax_speedup.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_speedup.grid(True, alpha=0.3, linestyle='--')
    ax_speedup.set_xlim(0.5, 16.5)
    ax_speedup.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax_speedup.set_xticks([1, 2, 4, 8, 16])
    ax_speedup.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Add annotation for speedup at 16 nodes
    if '16×7' in results:
        data = results['16×7']
        if 16 in data['nodes']:
            idx_16 = np.where(data['nodes'] == 16)[0][0]
            speedup_16 = data['speedup'][idx_16]
            
            # Annotate the speedup value at 16 nodes
            ax_speedup.annotate(f'{speedup_16:.2f}×',
                       xy=(16, speedup_16),
                       xytext=(16, speedup_16 + ax_speedup.get_ylim()[1] * 0.05),  # Position above the point
                       ha='center',
                       va='bottom',
                       fontsize=10,
                       fontweight='bold',
                       color=config_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=config_color, linewidth=1.5))
    
    # ========== RIGHT SUBPLOT: EFFICIENCY ==========
    ax_efficiency = axes[1]
    ax_efficiency.set_facecolor('white')  # Set subplot background to white
    
    # Plot ideal efficiency line (100%) - only up to 16 nodes
    ideal_nodes = np.array([1, 16])
    ideal_efficiency = np.array([100.0, 100.0])
    ax_efficiency.plot(ideal_nodes, ideal_efficiency, 'k--', linewidth=1.5, alpha=0.5, 
                       label='Ideal Efficiency', zorder=1)
    
    # Plot data for 16×7 configuration only
    if '16×7' in results:
        data = results['16×7']
        ax_efficiency.plot(data['nodes'], data['efficiency'], 
                          marker=config_marker, 
                          linewidth=2, 
                          markersize=8,
                          color=config_color,
                          label='16×7',
                          zorder=2)
    
    # Customize efficiency plot
    ax_efficiency.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_efficiency.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax_efficiency.set_title('Strong Scaling Parallel Efficiency', fontsize=14, fontweight='bold')
    ax_efficiency.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_efficiency.grid(True, alpha=0.3, linestyle='--')
    ax_efficiency.set_xlim(0.5, 16.5)
    ax_efficiency.set_ylim(bottom=0, top=120)  # Set y-axis limit to 120%
    
    # Set x-axis ticks
    ax_efficiency.set_xticks([1, 2, 4, 8, 16])
    ax_efficiency.set_xticklabels(['1', '2', '4', '8', '16'])
    
    # Add annotation for efficiency at 16 nodes
    if '16×7' in results:
        data = results['16×7']
        if 16 in data['nodes']:
            idx_16 = np.where(data['nodes'] == 16)[0][0]
            efficiency_16 = data['efficiency'][idx_16]
            
            # Annotate the efficiency value at 16 nodes
            ax_efficiency.annotate(f'{efficiency_16:.1f}%',
                       xy=(16, efficiency_16),
                       xytext=(16, efficiency_16 + 5),  # Position above the point
                       ha='center',
                       va='bottom',
                       fontsize=10,
                       fontweight='bold',
                       color=config_color,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=config_color, linewidth=1.5))
    
    # Improve layout
    plt.tight_layout()
    
    # Save combined figure with white background
    plt.savefig(output_path_combined, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Combined plot saved to: {output_path_combined}")
    
    # Also save individual plots (optional, for compatibility)
    # Save speedup plot
    fig_speedup, ax_speedup_only = plt.subplots(figsize=(10, 6))
    fig_speedup.patch.set_facecolor('white')
    ax_speedup_only.set_facecolor('white')
    
    ideal_speedup = nodes
    ax_speedup_only.plot(nodes, ideal_speedup, 'k--', linewidth=1.5, alpha=0.5, 
                        label='Ideal Scaling', zorder=1)
    
    if '16×7' in results:
        data = results['16×7']
        ax_speedup_only.plot(data['nodes'], data['speedup'], 
                            marker=config_marker, 
                            linewidth=2, 
                            markersize=8,
                            color=config_color,
                            label='16×7',
                            zorder=2)
    
    ax_speedup_only.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_speedup_only.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax_speedup_only.set_title('Strong Scaling Speedup', fontsize=14, fontweight='bold')
    ax_speedup_only.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_speedup_only.grid(True, alpha=0.3, linestyle='--')
    ax_speedup_only.set_xlim(0.5, 16.5)
    ax_speedup_only.set_ylim(bottom=0)
    ax_speedup_only.set_xticks([1, 2, 4, 8, 16])
    ax_speedup_only.set_xticklabels(['1', '2', '4', '8', '16'])
    plt.tight_layout()
    plt.savefig(output_path_speedup, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_speedup)
    
    # Save efficiency plot
    fig_efficiency, ax_efficiency_only = plt.subplots(figsize=(10, 6))
    fig_efficiency.patch.set_facecolor('white')
    ax_efficiency_only.set_facecolor('white')
    
    # Plot ideal efficiency line (100%) - only up to 16 nodes
    ideal_nodes = np.array([1, 16])
    ideal_efficiency = np.array([100.0, 100.0])
    ax_efficiency_only.plot(ideal_nodes, ideal_efficiency, 'k--', linewidth=1.5, alpha=0.5, 
                           label='Ideal Efficiency', zorder=1)
    
    if '16×7' in results:
        data = results['16×7']
        ax_efficiency_only.plot(data['nodes'], data['efficiency'], 
                               marker=config_marker, 
                               linewidth=2, 
                               markersize=8,
                               color=config_color,
                               label='16×7',
                               zorder=2)
    
    ax_efficiency_only.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_efficiency_only.set_ylabel('Parallel Efficiency (%)', fontsize=12, fontweight='bold')
    ax_efficiency_only.set_title('Strong Scaling Parallel Efficiency', fontsize=14, fontweight='bold')
    ax_efficiency_only.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax_efficiency_only.grid(True, alpha=0.3, linestyle='--')
    ax_efficiency_only.set_xlim(0.5, 16.5)
    ax_efficiency_only.set_ylim(bottom=0)
    ax_efficiency_only.set_xticks([1, 2, 4, 8, 16])
    ax_efficiency_only.set_xticklabels(['1', '2', '4', '8', '16'])
    plt.tight_layout()
    plt.savefig(output_path_efficiency, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig_efficiency)
    
    print(f"Individual speedup plot saved to: {output_path_speedup}")
    print(f"Individual efficiency plot saved to: {output_path_efficiency}")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    output_path_speedup = os.path.join(project_root, 'figures', 'strong_speedup.png')
    output_path_efficiency = os.path.join(project_root, 'figures', 'strong_efficiency.png')
    output_path_combined = os.path.join(project_root, 'figures', 'strong_speedup_efficiency.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path_speedup), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path)
    
    # Generate plots
    print(f"\nGenerating plots...")
    plot_strong_speedup_efficiency(results, output_path_speedup, output_path_efficiency, output_path_combined)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
