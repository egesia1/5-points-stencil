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
    """Generate strong scaling execution time plot with 3 subplots (one per configuration)."""
    
    # Create figure with 3 subplots side by side
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Configuration names in order
    config_names = ['8×14', '2×56', '16×7']
    
    # Colors and markers for each time type
    colors = {
        'computation': '#ff9999',  # Light pink
        'communication': '#cc99ff',  # Light purple
        'total': '#666666'  # Dark grey/purple
    }
    markers = {
        'computation': 's',  # Square
        'communication': '^',  # Triangle
        'total': 'o'  # Circle
    }
    
    for idx, config_name in enumerate(config_names):
        ax = axes[idx]
        ax.set_facecolor('white')  # Set subplot background to white
        
        if config_name not in results:
            ax.text(0.5, 0.5, f'No data for {config_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        data = results[config_name]
        
        # Plot communication time
        ax.plot(data['nodes'], data['communication_time'], 
               marker=markers['communication'],
               linewidth=2,
               markersize=5,
               color=colors['communication'],
               label='Communication Time',
               zorder=3)
        
        # Plot total time
        ax.plot(data['nodes'], data['total_time'], 
               marker=markers['total'],
               linewidth=2,
               markersize=5,
               color=colors['total'],
               label='Total Time',
               zorder=2)
        
        # Customize plot
        ax.set_xlabel('Number of Nodes', fontsize=11, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
        ax.set_title(f'{config_name} Configuration', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(bottom=0)
        
        # Set x-axis ticks only on desired values (same as strong_speedup)
        nodes = [1, 2, 4, 8, 16]
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes])
    
    # Overall title
    fig.suptitle('Strong Scaling - Execution Time (as a function of nodes)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
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
