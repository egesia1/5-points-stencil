#!/usr/bin/env python3
"""
Script to generate a table with Time, Speedup, and Efficiency for the three configurations.
"""

import pandas as pd
import os
import sys

def generate_table(csv_path):
    """Generate table with Time, Speedup, and Efficiency."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for ofast build variant and 1 energy source
    df = df[(df['BuildVariant'] == 'ofast') & (df['EnergySources'] == 1)].copy()
    
    # Define configurations
    configs = {
        '2×56': {'TasksPerNode': 2, 'ThreadsPerTask': 56},
        '8×14': {'TasksPerNode': 8, 'ThreadsPerTask': 14},
        '16×7': {'TasksPerNode': 16, 'ThreadsPerTask': 7}
    }
    
    # Prepare table data
    table_data = []
    
    for config_name, config_params in configs.items():
        # Filter data for this configuration
        config_df = df[
            (df['TasksPerNode'] == config_params['TasksPerNode']) &
            (df['ThreadsPerTask'] == config_params['ThreadsPerTask'])
        ].copy()
        
        # Group by nodes and take mean (to handle duplicate runs)
        config_grouped = config_df.groupby('Nodes')['TotalTime'].mean().reset_index()
        config_grouped = config_grouped.sort_values('Nodes')
        
        # Get baseline runtime (1 node)
        baseline = config_grouped[config_grouped['Nodes'] == 1]
        if len(baseline) == 0:
            continue
        
        baseline_runtime = baseline['TotalTime'].iloc[0]
        
        # Calculate speedup and efficiency for each node count
        for _, row in config_grouped.iterrows():
            nodes = int(row['Nodes'])
            runtime = row['TotalTime']
            speedup = baseline_runtime / runtime
            efficiency = (speedup / nodes) * 100.0
            
            table_data.append({
                'Configuration': config_name,
                'Nodes': nodes,
                'Time (s)': runtime,
                'Speedup': speedup,
                'Efficiency (%)': efficiency
            })
    
    # Create DataFrame
    table_df = pd.DataFrame(table_data)
    
    # Format the table nicely
    print("\n" + "="*80)
    print("STRONG SCALING RESULTS - Time, Speedup, and Efficiency")
    print("="*80)
    print()
    
    # Print for each configuration
    for config_name in ['2×56', '8×14', '16×7']:
        config_data = table_df[table_df['Configuration'] == config_name]
        if len(config_data) == 0:
            continue
        
        print(f"{config_name} Configuration:")
        print("-" * 80)
        print(f"{'Nodes':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
        print("-" * 80)
        
        for _, row in config_data.iterrows():
            print(f"{int(row['Nodes']):<8} {row['Time (s)']:<12.2f} {row['Speedup']:<12.2f} {row['Efficiency (%)']:<15.1f}")
        
        print()
    
    # Also create a combined table
    print("\n" + "="*80)
    print("COMBINED TABLE - All Configurations")
    print("="*80)
    print()
    
    # Pivot table for better comparison
    for nodes in [1, 2, 4, 8, 16]:
        print(f"\nNodes: {nodes}")
        print("-" * 80)
        print(f"{'Configuration':<15} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
        print("-" * 80)
        
        for config_name in ['2×56', '8×14', '16×7']:
            row_data = table_df[(table_df['Configuration'] == config_name) & 
                               (table_df['Nodes'] == nodes)]
            if len(row_data) > 0:
                row = row_data.iloc[0]
                print(f"{config_name:<15} {row['Time (s)']:<12.2f} {row['Speedup']:<12.2f} {row['Efficiency (%)']:<15.1f}")
    
    print("\n" + "="*80)
    
    # Save to CSV
    output_csv = os.path.join(os.path.dirname(csv_path), 'strong_scaling_table.csv')
    table_df.to_csv(output_csv, index=False)
    print(f"\nTable saved to: {output_csv}")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_path = os.path.join(project_root, 'data', 'SM3800083_strong_parallel_results.csv')
    
    print(f"Loading data from: {csv_path}")
    generate_table(csv_path)
    print("\nDone!")

if __name__ == '__main__':
    main()
