import json
import glob
import matplotlib.pyplot as plt
import numpy as np

def plot_ablation_study(results_dir, output_path):
    """Create ablation study summary plot."""
    
    # Collect results
    non_shape_percentages = []
    accuracies = []
    latencies = []
    
    # Read all result files
    result_files = sorted(glob.glob(f"{results_dir}/eval_nsp_*.json"))
    
    for file in result_files:
        with open(file, 'r') as f:
            data = json.load(f)
        
        nsp = data['configuration']['non_shape_percentage']
        acc = data['performance']['accuracy']
        lat = data['performance']['avg_latency_ms']
        
        non_shape_percentages.append(nsp)
        accuracies.append(acc * 100)  # Convert to percentage
        latencies.append(lat)
    
    # Sort by non_shape_percentage
    sorted_indices = np.argsort(non_shape_percentages)
    non_shape_percentages = [non_shape_percentages[i] for i in sorted_indices]
    accuracies = [accuracies[i] for i in sorted_indices]
    latencies = [latencies[i] for i in sorted_indices]
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs Non-Shape Percentage
    ax1.plot(non_shape_percentages, accuracies, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Non-Shape Token Percentage', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Impact of Non-Shape Tokens on Accuracy', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-0.05, 1.05)
    
    # Add value annotations
    for x, y in zip(non_shape_percentages, accuracies):
        ax1.annotate(f'{y:.1f}', (x, y), textcoords="offset points", 
                    xytext=(0,5), ha='center', fontsize=9)
    
    # Plot 2: Latency vs Non-Shape Percentage
    ax2.plot(non_shape_percentages, latencies, 's-', linewidth=2, 
             markersize=8, color='orange')
    ax2.set_xlabel('Non-Shape Token Percentage', fontsize=12)
    ax2.set_ylabel('Average Latency (ms)', fontsize=12)
    ax2.set_title('Impact of Non-Shape Tokens on Latency', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-0.05, 1.05)
    
    plt.suptitle('Non-Shape Token Ablation Study', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create additional detailed plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate relative performance
    baseline_acc = accuracies[0] if accuracies else 100
    relative_acc = [(acc/baseline_acc)*100 for acc in accuracies]
    
    ax.plot(non_shape_percentages, relative_acc, 'o-', linewidth=2.5, 
            markersize=10, label='Relative Accuracy')
    ax.axhline(y=100, color='gray', linestyle='--', alpha=0.5, 
               label='Baseline (0% non-shape)')
    
    ax.set_xlabel('Fraction of Non-Shape Tokens', fontsize=14)
    ax.set_ylabel('Relative Accuracy (%)', fontsize=14)
    ax.set_title('Performance Degradation with Random Token Selection', fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    ax.set_xlim(-0.05, 1.05)
    
    # Highlight key percentages
    for x in [0.0, 0.25, 0.5, 0.75, 1.0]:
        if x in non_shape_percentages:
            idx = non_shape_percentages.index(x)
            ax.plot(x, relative_acc[idx], 'ro', markersize=12, zorder=5)
            ax.annotate(f'{relative_acc[idx]:.1f}%', (x, relative_acc[idx]), 
                       textcoords="offset points", xytext=(0,10), 
                       ha='center', fontsize=10, fontweight='bold')
    
    detail_path = output_path.replace('.png', '_detailed.png')
    plt.savefig(detail_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to {output_path} and {detail_path}")
    
    # Print summary statistics
    print("\nAblation Study Summary:")
    print("-" * 50)
    print(f"{'NSP %':<10} {'Accuracy':<15} {'Rel. Acc':<15} {'Latency (ms)':<15}")
    print("-" * 50)
    for nsp, acc, rel, lat in zip(non_shape_percentages, accuracies, relative_acc, latencies):
        print(f"{nsp*100:<10.0f} {acc:<15.2f} {rel:<15.2f} {lat:<15.2f}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, default='results/logs')
    parser.add_argument('--output_path', type=str, 
                       default='results/plots/ablation_non_shape_tokens.png')
    args = parser.parse_args()
    
    plot_ablation_study(args.results_dir, args.output_path)