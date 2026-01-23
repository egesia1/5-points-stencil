#!/usr/bin/env python3
"""
Script to generate Build Variant Comparison plot showing compiler optimization impact.
This plot compares different build variants (o0, o1, noarch, ofast, ofast_omp_improved) 
across different thread counts to demonstrate the impact of compiler optimizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path):
    """Load CSV data and calculate runtime and speedup for each build variant."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for 1 energy source (baseline)
    df = df[df['EnergySources'] == 1].copy()
    
    # Define build variants in order of optimization level
    build_variants = ['o0', 'o1', 'noarch', 'ofast', 'ofast_omp_improved']
    
    # Define thread counts that have data for multiple build variants
    # We'll use 56 and 112 threads as they have the most complete data
    thread_counts = [56, 112]
    
    results = {}
    
    # Get baseline runtime (o0, single thread) for normalization
    baseline_o0_1thread = df[(df['BuildVariant'] == 'o0') & (df['Threads'] == 1)]['TotalTime']
    if len(baseline_o0_1thread) == 0:
        # If no o0/1 thread, use first available baseline
        baseline_o0_1thread = df[(df['Threads'] == 1)]['TotalTime']
    
    if len(baseline_o0_1thread) > 0:
        baseline_runtime = baseline_o0_1thread.mean()
    else:
        # Fallback: use minimum runtime as baseline
        baseline_runtime = df['TotalTime'].min()
    
    for thread_count in thread_counts:
        thread_df = df[df['Threads'] == thread_count].copy()
        
        if len(thread_df) == 0:
            continue
        
        # Group by build variant and take mean (to handle duplicate runs)
        grouped = thread_df.groupby('BuildVariant')['TotalTime'].mean().reset_index()
        grouped = grouped.sort_values('BuildVariant', key=lambda x: x.map({v: i for i, v in enumerate(build_variants)}))
        
        # Calculate speedup relative to o0 baseline for this thread count
        o0_runtime = grouped[grouped['BuildVariant'] == 'o0']['TotalTime']
        if len(o0_runtime) > 0:
            o0_runtime_val = o0_runtime.iloc[0]
            grouped['Speedup_vs_o0'] = o0_runtime_val / grouped['TotalTime']
        else:
            # If no o0 data, use baseline_runtime
            grouped['Speedup_vs_o0'] = baseline_runtime / grouped['TotalTime']
        
        # Store results
        results[thread_count] = {
            'build_variants': grouped['BuildVariant'].values,
            'runtime': grouped['TotalTime'].values,
            'speedup': grouped['Speedup_vs_o0'].values
        }
    
    return results, baseline_runtime

def plot_build_variant_comparison(results, baseline_runtime, output_path):
    """Generate build variant comparison plot showing runtime and speedup."""
    
    # Create figure with 2 subplots side by side (one per thread count)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('white')
    
    thread_counts = sorted(results.keys())
    colors = {
        'o0': '#d62728',           # red
        'o1': '#ff7f0e',           # orange
        'noarch': '#2ca02c',       # green
        'ofast': '#1f77b4',        # blue
        'ofast_omp_improved': '#9467bd'  # purple
    }
    
    variant_labels = {
        'o0': 'O0',
        'o1': 'O1',
        'noarch': 'noarch',
        'ofast': 'Ofast',
        'ofast_omp_improved': 'Ofast\n(OMP Improved)'
    }
    
    for idx, thread_count in enumerate(thread_counts):
        ax = axes[idx]
        ax.set_facecolor('white')
        
        if thread_count not in results:
            continue
        
        data = results[thread_count]
        x_pos = np.arange(len(data['build_variants']))
        
        # Create bars
        bars = ax.bar(x_pos, data['speedup'], 
                     color=[colors.get(bv, '#808080') for bv in data['build_variants']],
                     alpha=0.8,
                     edgecolor='black',
                     linewidth=1.5)
        
        # Add value labels on bars
        for i, (bar, speedup, runtime) in enumerate(zip(bars, data['speedup'], data['runtime'])):
            height = bar.get_height()
            # Show speedup value on top of bar
            ax.text(bar.get_x() + bar.get_width()/2., height + height * 0.02,
                   f'{speedup:.2f}×',
                   ha='center', va='bottom',
                   fontsize=10, fontweight='bold')
            # Show runtime value inside or below bar if space allows
            if height > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., height / 2,
                       f'{runtime:.1f}s',
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color='white')
        
        # Customize plot
        ax.set_xlabel('Build Variant', fontsize=12, fontweight='bold')
        if idx == 0:
            ax.set_ylabel('Speedup vs. O0', fontsize=12, fontweight='bold')
        ax.set_title(f'{thread_count} Threads', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([variant_labels.get(bv, bv) for bv in data['build_variants']], 
                          rotation=0, ha='center', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        ax.set_ylim(bottom=0)
        
        # Add reference line at 1.0× (baseline)
        ax.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.5, 
                  label='Baseline (O0)', zorder=0)
        if idx == 0:
            ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    
    # Add overall title
    fig.suptitle('Compiler Optimization Impact: Build Variant Comparison\n(Speedup relative to O0, 1 energy source)', 
                fontsize=14, fontweight='bold', y=1.02)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Print summary
    print("\nBuild Variant Comparison Summary:")
    print("-" * 70)
    for thread_count in thread_counts:
        if thread_count in results:
            data = results[thread_count]
            print(f"\n{thread_count} Threads:")
            for bv, speedup, runtime in zip(data['build_variants'], data['speedup'], data['runtime']):
                print(f"  {bv:20s}: {speedup:5.2f}× speedup, {runtime:7.2f}s runtime")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, 'data', 'SM3800083_omp_serial_results.csv')
    output_path = os.path.join(project_root, 'figures', 'build_variant_comparison.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results, baseline = load_and_process_data(csv_path)
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_build_variant_comparison(results, baseline, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
