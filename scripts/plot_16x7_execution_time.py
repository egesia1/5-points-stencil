#!/usr/bin/env python3
"""
Script to generate Strong Scaling Execution Time plot for 16×7 configuration
showing Total Time and Communication Time.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path, build_variant='ofast'):
    """Load CSV data and extract execution times for 16×7 configuration."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for specified build variant and 1 energy source (baseline)
    df = df[(df['BuildVariant'] == build_variant) & (df['EnergySources'] == 1)].copy()
    
    # Filter for 16×7 configuration
    config_df = df[
        (df['TasksPerNode'] == 16) &
        (df['ThreadsPerTask'] == 7)
    ].copy()
    
    # Group by nodes and take mean (to handle duplicate runs)
    config_grouped = config_df.groupby('Nodes').agg({
        'TotalTime': 'mean',
        'ComputationTime': 'mean',
        'CommunicationTime': 'mean'
    }).reset_index()
    config_grouped = config_grouped.sort_values('Nodes')
    
    # Store results
    results = {
        'nodes': config_grouped['Nodes'].values,
        'total_time': config_grouped['TotalTime'].values,
        'computation_time': config_grouped['ComputationTime'].values,
        'communication_time': config_grouped['CommunicationTime'].values
    }
    
    return results

def plot_16x7_execution_time(results, output_path):
    """Generate execution time plot for 16×7 configuration."""
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Colors and markers
    colors = {
        'communication': '#cc99ff',  # Light purple
        'total': '#666666'  # Dark grey
    }
    markers = {
        'communication': '^',  # Triangle
        'total': 'o'  # Circle
    }
    
    # Plot communication time
    ax.plot(results['nodes'], results['communication_time'], 
           marker=markers['communication'],
           linewidth=2,
           markersize=8,
           color=colors['communication'],
           label='Communication Time',
           zorder=3)
    
    # Plot total time
    ax.plot(results['nodes'], results['total_time'], 
           marker=markers['total'],
           linewidth=2,
           markersize=8,
           color=colors['total'],
           label='Total Time',
           zorder=2)
    
    # Customize plot
    ax.set_xlabel('Number of Nodes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Strong Scaling - Execution Time (16×7 Configuration)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xlim(0.5, 16.5)
    ax.set_ylim(bottom=0)
    
    # Set x-axis ticks
    nodes = [1, 2, 4, 8, 16]
    ax.set_xticks(nodes)
    ax.set_xticklabels([str(n) for n in nodes])
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Print summary data
    print("\n16×7 Configuration - Execution Time Summary:")
    print("-" * 70)
    for i, n in enumerate(results['nodes']):
        print(f"  {n} nodes: Total={results['total_time'][i]:.2f}s, "
              f"Computation={results['computation_time'][i]:.2f}s, "
              f"Communication={results['communication_time'][i]:.2f}s")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    output_path = os.path.join(project_root, 'figures', 'strong_execution_time_16x7.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path, build_variant='ofast')
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_16x7_execution_time(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
