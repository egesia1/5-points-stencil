#!/usr/bin/env python3
"""
Script to generate OpenMP Code Optimization Impact plot.
This plot compares ofast vs ofast_omp_improved for OpenMP scaling tests,
demonstrating the impact of OpenMP pragma optimizations (not compiler flags).
Both variants use identical compiler flags but differ in code-level OpenMP parallelization.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# Add parent directory to path for imports if needed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_and_process_data(csv_path):
    """Load CSV data and calculate speedup and efficiency for ofast vs ofast_omp_improved."""
    
    # Read CSV
    df = pd.read_csv(csv_path)
    
    # Filter for 1 energy source and only the two variants we care about
    df = df[
        (df['EnergySources'] == 1) & 
        (df['BuildVariant'].isin(['ofast', 'ofast_omp_improved']))
    ].copy()
    
    # Define thread counts we want to show (in order)
    desired_threads = [1, 2, 4, 8, 16, 32, 56, 84, 112]
    
    results = {}
    
    for variant in ['ofast', 'ofast_omp_improved']:
        variant_df = df[df['BuildVariant'] == variant].copy()
        
        # Group by threads and take mean (to handle duplicate runs)
        grouped = variant_df.groupby('Threads')['TotalTime'].mean().reset_index()
        grouped = grouped.sort_values('Threads')
        
        # Get baseline runtime (1 thread)
        baseline = grouped[grouped['Threads'] == 1]
        if len(baseline) == 0:
            continue
        
        baseline_runtime = baseline['TotalTime'].iloc[0]
        
        # Calculate speedup: speedup = baseline_runtime / runtime
        grouped['Speedup'] = baseline_runtime / grouped['TotalTime']
        
        # Calculate efficiency: efficiency = (speedup / threads) * 100%
        grouped['Efficiency'] = (grouped['Speedup'] / grouped['Threads']) * 100.0
        
        # Filter to only include desired thread counts
        grouped = grouped[grouped['Threads'].isin(desired_threads)]
        grouped = grouped.sort_values('Threads')
        
        # Store results
        results[variant] = {
            'threads': grouped['Threads'].values,
            'speedup': grouped['Speedup'].values,
            'efficiency': grouped['Efficiency'].values,
            'runtime': grouped['TotalTime'].values,
            'baseline_runtime': baseline_runtime
        }
    
    return results

def plot_omp_build_variant_comparison(results, output_path):
    """Generate OpenMP code optimization impact plot showing speedup, efficiency, and runtime comparison."""
    
    # Create figure with 3 subplots side by side (speedup, efficiency, and runtime)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor('white')
    
    # Define thread counts for ideal scaling line
    ideal_threads = np.array([1, 2, 4, 8, 16, 32, 56, 84, 112])
    
    # Colors for each variant
    colors = {
        'ofast': '#d62728',              # red
        'ofast_omp_improved': '#1f77b4'  # blue
    }
    
    labels = {
        'ofast': 'ofast (no pragmas)',
        'ofast_omp_improved': 'ofast_omp_improved (with pragmas)'
    }
    
    # Plot 1: Runtime (moved to first position)
    ax_runtime = axes[0]
    ax_runtime.set_facecolor('white')
    
    # Plot runtime for each variant
    for variant in ['ofast', 'ofast_omp_improved']:
        if variant not in results:
            continue
        
        data = results[variant]
        ax_runtime.plot(data['threads'], data['runtime'], 
                       marker='o', 
                       linewidth=2, 
                       markersize=6,
                       color=colors[variant],
                       label=labels[variant],
                       zorder=2)
        
        # Add annotation for 112 threads
        if 112 in data['threads']:
            idx = np.where(data['threads'] == 112)[0]
            if len(idx) > 0:
                idx = idx[0]
                x_val = data['threads'][idx]
                y_val = data['runtime'][idx]
                ax_runtime.annotate(f'{y_val:.1f}s', 
                                   xy=(x_val, y_val),
                                   xytext=(0, 10), textcoords='offset points',
                                   ha='center',
                                   fontsize=8, fontweight='bold',
                                   color='black',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            edgecolor=colors[variant], alpha=0.8))
    
    ax_runtime.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax_runtime.set_ylabel('Runtime (s)', fontsize=12, fontweight='bold')
    ax_runtime.set_title('Execution Time', fontsize=12, fontweight='bold')
    ax_runtime.legend(loc='center right', fontsize=9, framealpha=0.9)
    ax_runtime.grid(True, alpha=0.3, linestyle='--')
    ax_runtime.set_xlim(0, 115)
    ax_runtime.set_ylim(bottom=0)
    ax_runtime.set_xticks([1, 2, 4, 8, 16, 32, 56, 84, 112])
    ax_runtime.set_xticklabels(['1', '2', '4', '8', '16', '32', '56', '84', '112'], 
                              rotation=45, ha='right')
    
    # Plot 2: Speedup
    ax_speedup = axes[1]
    ax_speedup.set_facecolor('white')
    
    # Plot ideal scaling line
    ideal_speedup = ideal_threads
    ax_speedup.plot(ideal_threads, ideal_speedup, 'k--', linewidth=1.5, alpha=0.5, 
                   label='Ideal Scaling', zorder=1)
    
    # Plot measured speedup for each variant
    for variant in ['ofast', 'ofast_omp_improved']:
        if variant not in results:
            continue
        
        data = results[variant]
        ax_speedup.plot(data['threads'], data['speedup'], 
                       marker='o', 
                       linewidth=2, 
                       markersize=6,
                       color=colors[variant],
                       label=labels[variant],
                       zorder=2)
        
        # Add annotation for 112 threads
        if 112 in data['threads']:
            idx = np.where(data['threads'] == 112)[0]
            if len(idx) > 0:
                idx = idx[0]
                x_val = data['threads'][idx]
                y_val = data['speedup'][idx]
                ax_speedup.annotate(f'{y_val:.2f}×', 
                                   xy=(x_val, y_val),
                                   xytext=(0, 10), textcoords='offset points',
                                   ha='center',
                                   fontsize=8, fontweight='bold',
                                   color='black',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                            edgecolor=colors[variant], alpha=0.8))
    
    ax_speedup.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax_speedup.set_ylabel('Speedup', fontsize=12, fontweight='bold')
    ax_speedup.set_title('OpenMP Scaling Speedup', fontsize=12, fontweight='bold')
    ax_speedup.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax_speedup.grid(True, alpha=0.3, linestyle='--')
    ax_speedup.set_xlim(0, 115)
    ax_speedup.set_ylim(bottom=0)
    ax_speedup.set_xticks([1, 2, 4, 8, 16, 32, 56, 84, 112])
    ax_speedup.set_xticklabels(['1', '2', '4', '8', '16', '32', '56', '84', '112'], 
                               rotation=45, ha='right')
    
    # Plot 3: Efficiency
    ax_efficiency = axes[2]
    ax_efficiency.set_facecolor('white')
    
    # Plot ideal efficiency line (100%)
    ax_efficiency.axhline(y=100.0, color='k', linestyle='--', linewidth=1.5, alpha=0.5, 
                          label='Ideal Efficiency', zorder=1)
    
    # Plot measured efficiency for each variant
    for variant in ['ofast', 'ofast_omp_improved']:
        if variant not in results:
            continue
        
        data = results[variant]
        ax_efficiency.plot(data['threads'], data['efficiency'], 
                          marker='o', 
                          linewidth=2, 
                          markersize=6,
                          color=colors[variant],
                          label=labels[variant],
                          zorder=2)
        
        # Add annotation for 112 threads
        if 112 in data['threads']:
            idx = np.where(data['threads'] == 112)[0]
            if len(idx) > 0:
                idx = idx[0]
                x_val = data['threads'][idx]
                y_val = data['efficiency'][idx]
                ax_efficiency.annotate(f'{y_val:.1f}%', 
                                     xy=(x_val, y_val),
                                     xytext=(0, 10), textcoords='offset points',
                                     ha='center',
                                     fontsize=8, fontweight='bold',
                                     color='black',
                                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                              edgecolor=colors[variant], alpha=0.8))
    
    ax_efficiency.set_xlabel('Number of Threads', fontsize=11, fontweight='bold')
    ax_efficiency.set_ylabel('Efficiency (%)', fontsize=12, fontweight='bold')
    ax_efficiency.set_title('OpenMP Scaling Efficiency', fontsize=12, fontweight='bold')
    ax_efficiency.legend(loc='center right', fontsize=9, framealpha=0.9)
    ax_efficiency.grid(True, alpha=0.3, linestyle='--')
    ax_efficiency.set_xlim(0, 115)
    ax_efficiency.set_ylim(bottom=0)
    ax_efficiency.set_xticks([1, 2, 4, 8, 16, 32, 56, 84, 112])
    ax_efficiency.set_xticklabels(['1', '2', '4', '8', '16', '32', '56', '84', '112'], 
                                  rotation=45, ha='right')
    
    # Add overall title
    fig.suptitle('OpenMP Code Optimization Impact: ofast vs ofast_omp_improved\n(Identical compiler flags, different OpenMP pragmas, 1 energy source)', 
                fontsize=13, fontweight='bold', y=1.02)
    
    # Improve layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Plot saved to: {output_path}")
    
    # Print summary
    print("\nOpenMP Code Optimization Impact Summary:")
    print("-" * 70)
    for variant in ['ofast', 'ofast_omp_improved']:
        if variant not in results:
            continue
        
        data = results[variant]
        print(f"\n{variant}:")
        for thread, speedup, efficiency, runtime in zip(data['threads'], data['speedup'], 
                                                       data['efficiency'], data['runtime']):
            print(f"  {thread:3d} threads: {speedup:5.2f}× speedup, {efficiency:5.1f}% efficiency, {runtime:7.2f}s")

def main():
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    csv_path = os.path.join(project_root, 'data', 'SM3800083_omp_serial_results.csv')
    output_path = os.path.join(project_root, 'figures', 'omp_build_variant_comparison.png')
    
    # Create figures directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load and process data
    print(f"Loading data from: {csv_path}")
    results = load_and_process_data(csv_path)
    
    # Generate plot
    print(f"\nGenerating plot...")
    plot_omp_build_variant_comparison(results, output_path)
    
    print("\nDone!")

if __name__ == '__main__':
    main()
