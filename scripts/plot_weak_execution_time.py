#!/usr/bin/env python3
"""
Script to generate Weak Scaling Execution Time plot showing Communication Time
and Total Time for periodic and non-periodic boundary conditions.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path, build_variant='ofast_omp_improved'):
    """Load CSV data and extract execution times for periodic and non-periodic cases."""
    
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
    periodic0_grouped = periodic0_df.groupby('Nodes').agg({
        'TotalTime': 'mean',
        'ComputationTime': 'mean',
        'CommunicationTime': 'mean'
    }).reset_index()
    periodic0_grouped = periodic0_grouped.sort_values('Nodes')
    
    if len(periodic0_grouped) > 0:
        results['Non-Periodic'] = {
            'nodes': periodic0_grouped['Nodes'].values,
            'total_time': periodic0_grouped['TotalTime'].values,
            'computation_time': periodic0_grouped['ComputationTime'].values,
            'communication_time': periodic0_grouped['CommunicationTime'].values
        }
    
    # Process periodic (Periodic=1)
    periodic1_df = config_df[config_df['Periodic'] == 1].copy()
    periodic1_grouped = periodic1_df.groupby('Nodes').agg({
        'TotalTime': 'mean',
        'ComputationTime': 'mean',
        'CommunicationTime': 'mean'
    }).reset_index()
    periodic1_grouped = periodic1_grouped.sort_values('Nodes')
    
    if len(periodic1_grouped) > 0:
        results['Periodic'] = {
            'nodes': periodic1_grouped['Nodes'].values,
            'total_time': periodic1_grouped['TotalTime'].values,
            'computation_time': periodic1_grouped['ComputationTime'].values,
            'communication_time': periodic1_grouped['CommunicationTime'].values
        }
    
    return results

def plot_weak_execution_time(results, output_path):
    """Generate weak scaling execution time plot with 2 subplots (one per boundary condition)."""
    
    # Create figure with 2 subplots side by side
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('white')  # Set figure background to white
    
    # Boundary condition names in order
    condition_names = ['Non-Periodic', 'Periodic']
    
    # Colors and markers for each time type (same as strong scaling)
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
    
    for idx, condition_name in enumerate(condition_names):
        ax = axes[idx]
        ax.set_facecolor('white')  # Set subplot background to white
        
        if condition_name not in results:
            ax.text(0.5, 0.5, f'No data for {condition_name}', 
                   ha='center', va='center', transform=ax.transAxes)
            continue
        
        data = results[condition_name]
        
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
        ax.set_title(f'{condition_name} Boundaries (16×7 Configuration)', fontsize=12, fontweight='bold')
        ax.legend(loc='center right', fontsize=9, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0.5, 16.5)
        ax.set_ylim(bottom=0, top=14)
        
        # Set x-axis ticks only on desired values
        nodes = [1, 2, 4, 8, 16]
        ax.set_xticks(nodes)
        ax.set_xticklabels([str(n) for n in nodes])
    
    # Overall title
    fig.suptitle('Weak Scaling - Execution Time (as a function of nodes)', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    # Improve layout
    plt.tight_layout()
    
    # Save figure with white background
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Also print summary data
    print("\nWeak Scaling Execution Time Summary:")
    print("-" * 70)
    for condition_name in condition_names:
        if condition_name in results:
            data = results[condition_name]
            print(f"\n{condition_name} (16×7 Configuration):")
            for i, n in enumerate(data['nodes']):
                print(f"  {n} nodes: Total={data['total_time'][i]:.2f}s, "
                      f"Computation={data['computation_time'][i]:.2f}s, "
                      f"Communication={data['communication_time'][i]:.2f}s")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_weak_parallel_results_corrected.csv')
    output_path = os.path.join(project_root, 'figures', 'weak_execution_time.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path, build_variant='ofast_omp_improved')
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_weak_execution_time(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
