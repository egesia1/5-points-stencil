#!/usr/bin/env python3
"""
Script to generate Strong Scaling Execution Time plot showing Computation Time,
Communication Time, and Total Time for three configurations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path, build_variant='ofast'):
    """Load CSV data and extract execution times for each configuration."""
    
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
        
        # Group by nodes and take mean (to handle duplicate runs)
        config_grouped = config_df.groupby('Nodes').agg({
            'TotalTime': 'mean',
            'ComputationTime': 'mean',
            'CommunicationTime': 'mean'
        }).reset_index()
        config_grouped = config_grouped.sort_values('Nodes')
        
        # Store results
        results[config_name] = {
            'nodes': config_grouped['Nodes'].values,
            'total_time': config_grouped['TotalTime'].values,
            'computation_time': config_grouped['ComputationTime'].values,
            'communication_time': config_grouped['CommunicationTime'].values
        }
    
    return results

def plot_strong_execution_time(results, output_path):
    """Generate strong scaling execution time and speedup plots with all configurations."""
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    ax_time = axes[0]
    ax_speedup = axes[1]
    ax_time.set_facecolor('white')
    ax_speedup.set_facecolor('white')
    
    # Configuration names in order
    config_names = ['2×56', '8×14', '16×7']
    
    # Colors and markers for each configuration (black to purple range)
    colors = {
        '2×56': '#1a1a1a',  # Very dark grey/black
        '8×14': '#6a4c93',  # Medium purple
        '16×7': '#3d1a5c'   # Dark purple
    }
    markers = {
        '2×56': 's',  # Square
        '8×14': '^',  # Triangle
        '16×7': 'o'   # Circle
    }
    
    # Calculate speedup for each configuration
    speedup_data = {}
    for config_name in config_names:
        if config_name not in results:
            continue
        
        data = results[config_name]
        
        # Get baseline runtime (1 node)
        if 1 in data['nodes']:
            idx_1 = np.where(data['nodes'] == 1)[0][0]
            baseline_runtime = data['total_time'][idx_1]
            
            # Calculate speedup
            speedup = baseline_runtime / data['total_time']
            speedup_data[config_name] = {
                'nodes': data['nodes'],
                'speedup': speedup
            }
    
    # ========== LEFT SUBPLOT: EXECUTION TIME ==========
    for config_name in config_names:
        if config_name not in results:
            continue
        
        data = results[config_name]
        
        # Plot total time
        ax_time.plot(data['nodes'], data['total_time'], 
               marker=markers[config_name],
               linewidth=2,
               markersize=8,
               color=colors[config_name],
               label=f'{config_name}',
               zorder=2)
    
    # Customize execution time plot
    ax_time.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_time.set_ylabel('Total Time (seconds)', fontsize=12, fontweight='bold')
    ax_time.set_title('Strong Scaling - Execution Time', fontsize=14, fontweight='bold')
    ax_time.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax_time.grid(True, alpha=0.3, linestyle='--')
    ax_time.set_xlim(0.5, 16.5)
    ax_time.set_ylim(bottom=0)
    
    # Set x-axis ticks
    nodes = [1, 2, 4, 8, 16]
    ax_time.set_xticks(nodes)
    ax_time.set_xticklabels([str(n) for n in nodes])
    
    # ========== RIGHT SUBPLOT: SPEEDUP ==========
    # Plot ideal scaling line
    ideal_nodes = np.array([1, 2, 4, 8, 16])
    ideal_speedup = ideal_nodes
    ax_speedup.plot(ideal_nodes, ideal_speedup, 'k--', linewidth=1.5, alpha=0.5, 
                   label='Ideal Scaling', zorder=1)
    
    # Plot speedup for each configuration
    for config_name in config_names:
        if config_name not in speedup_data:
            continue
        
        data = speedup_data[config_name]
        
        # Plot speedup
        ax_speedup.plot(data['nodes'], data['speedup'], 
               marker=markers[config_name],
               linewidth=2,
               markersize=8,
               color=colors[config_name],
               label=f'{config_name}',
               zorder=2)
    
    # Customize speedup plot
    ax_speedup.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax_speedup.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax_speedup.set_title('Strong Scaling - Speedup', fontsize=14, fontweight='bold')
    ax_speedup.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax_speedup.grid(True, alpha=0.3, linestyle='--')
    ax_speedup.set_xlim(0.5, 16.5)
    ax_speedup.set_ylim(bottom=0)
    
    # Set x-axis ticks
    ax_speedup.set_xticks(nodes)
    ax_speedup.set_xticklabels([str(n) for n in nodes])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nStrong Scaling Execution Time Summary:")
    print("-" * 70)
    for config_name in config_names:
        if config_name in results:
            data = results[config_name]
            print(f"\n{config_name} Configuration:")
            for i, n in enumerate(data['nodes']):
                print(f"  {n} nodes: Total={data['total_time'][i]:.2f}s, "
                      f"Computation={data['computation_time'][i]:.2f}s, "
                      f"Communication={data['communication_time'][i]:.2f}s")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    output_path = os.path.join(project_root, 'figures', 'strong_execution_time.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data (using ofast as it's the best optimization profile available)
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path, build_variant='ofast')
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_strong_execution_time(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
