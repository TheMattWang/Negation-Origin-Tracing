#!/usr/bin/env python
"""
Final Comparison Script: Analyze probe results and run interventions on best layer.

This script:
1. Merges results from all 3 parallel sweeps
2. Identifies the best layer/pooling combination
3. Runs causal interventions on the best probe
4. Generates comparison visualizations

Usage (local):
    python run_final_comparison.py

Usage (Google Drive):
    python run_final_comparison.py --drive_path /content/drive/MyDrive/NOT_results
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def merge_results(base_path):
    """Merge results from all 3 pooling sweeps."""
    sweep_dirs = {
        "cls": os.path.join(base_path, "sweep_cls"),
        "mean": os.path.join(base_path, "sweep_mean"),
        "token": os.path.join(base_path, "sweep_token"),
    }
    
    all_results = []
    
    for pooling, sweep_dir in sweep_dirs.items():
        results_file = os.path.join(sweep_dir, f"results_{pooling}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            all_results.extend(results)
            print(f"  ✓ Loaded {len(results)} {pooling} results")
        else:
            print(f"  ✗ {pooling} results not found")
    
    return all_results


def find_best_probe(results):
    """Find the best layer/pooling combination."""
    valid = [r for r in results if 'error' not in r and r.get('test_auroc', 0) > 0]
    if not valid:
        return None
    return max(valid, key=lambda x: x.get('test_auroc', 0))


def create_visualization(results, output_dir):
    """Create visualization of results."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  ⚠ matplotlib not available, skipping visualization")
        return
    
    valid = [r for r in results if 'error' not in r and r.get('test_auroc', 0) > 0]
    if not valid:
        return
    
    # Prepare data
    layers = sorted(set(r['layer'] for r in valid))
    poolings = ['cls', 'mean', 'token']
    
    # Create heatmap data
    auroc_matrix = np.zeros((len(poolings), len(layers)))
    acc_matrix = np.zeros((len(poolings), len(layers)))
    
    for r in valid:
        layer_idx = layers.index(r['layer'])
        pool_idx = poolings.index(r['pooling'])
        auroc_matrix[pool_idx, layer_idx] = r.get('test_auroc', 0)
        acc_matrix[pool_idx, layer_idx] = r.get('test_acc', 0)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # AUROC heatmap
    im1 = axes[0].imshow(auroc_matrix, cmap='YlOrRd', aspect='auto', vmin=0.5, vmax=1.0)
    axes[0].set_xticks(range(len(layers)))
    axes[0].set_xticklabels([f'Layer {l}' for l in layers])
    axes[0].set_yticks(range(len(poolings)))
    axes[0].set_yticklabels([p.upper() for p in poolings])
    axes[0].set_title('Test AUROC by Layer and Pooling Strategy', fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(len(poolings)):
        for j in range(len(layers)):
            val = auroc_matrix[i, j]
            color = 'white' if val > 0.75 else 'black'
            axes[0].text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im1, ax=axes[0], label='AUROC')
    
    # Accuracy heatmap
    im2 = axes[1].imshow(acc_matrix, cmap='YlGn', aspect='auto', vmin=0.5, vmax=1.0)
    axes[1].set_xticks(range(len(layers)))
    axes[1].set_xticklabels([f'Layer {l}' for l in layers])
    axes[1].set_yticks(range(len(poolings)))
    axes[1].set_yticklabels([p.upper() for p in poolings])
    axes[1].set_title('Test Accuracy by Layer and Pooling Strategy', fontsize=12, fontweight='bold')
    
    # Add values
    for i in range(len(poolings)):
        for j in range(len(layers)):
            val = acc_matrix[i, j]
            color = 'white' if val > 0.75 else 'black'
            axes[1].text(j, i, f'{val:.3f}', ha='center', va='center', color=color, fontsize=10)
    
    plt.colorbar(im2, ax=axes[1], label='Accuracy')
    
    plt.tight_layout()
    
    # Save
    viz_path = os.path.join(output_dir, 'probe_comparison.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Visualization saved to {viz_path}")
    
    # Also create a bar chart comparing pooling strategies
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(layers))
    width = 0.25
    
    for i, pooling in enumerate(poolings):
        aurocs = [auroc_matrix[i, j] for j in range(len(layers))]
        ax.bar(x + i * width, aurocs, width, label=pooling.upper())
    
    ax.set_xlabel('Layer')
    ax.set_ylabel('Test AUROC')
    ax.set_title('Probe Performance: AUROC by Layer and Pooling Strategy')
    ax.set_xticks(x + width)
    ax.set_xticklabels([f'Layer {l}' for l in layers])
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.grid(axis='y', alpha=0.3)
    
    bar_path = os.path.join(output_dir, 'probe_comparison_bar.png')
    plt.savefig(bar_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Bar chart saved to {bar_path}")


def generate_report(results, best, output_dir):
    """Generate a markdown report."""
    valid = [r for r in results if 'error' not in r and r.get('test_auroc', 0) > 0]
    
    report = []
    report.append("# Probe Comparison Results\n")
    report.append(f"Total experiments: {len(results)}\n")
    report.append(f"Successful: {len(valid)}\n\n")
    
    report.append("## Best Overall\n")
    report.append(f"- **Layer**: {best['layer']}\n")
    report.append(f"- **Pooling**: {best['pooling']}\n")
    report.append(f"- **Test AUROC**: {best.get('test_auroc', 0):.4f}\n")
    report.append(f"- **Test Accuracy**: {best.get('test_acc', 0):.4f}\n")
    if best.get('checkpoint'):
        report.append(f"- **Checkpoint**: `{best['checkpoint']}`\n")
    report.append("\n")
    
    report.append("## Results by Pooling Strategy\n\n")
    for pooling in ['cls', 'mean', 'token']:
        pooling_results = sorted(
            [r for r in valid if r.get('pooling') == pooling],
            key=lambda x: x['layer']
        )
        if pooling_results:
            report.append(f"### {pooling.upper()} Pooling\n\n")
            report.append("| Layer | AUROC | Accuracy |\n")
            report.append("|-------|-------|----------|\n")
            for r in pooling_results:
                report.append(f"| {r['layer']} | {r.get('test_auroc', 0):.4f} | {r.get('test_acc', 0):.4f} |\n")
            report.append("\n")
    
    report.append("## Results by Layer\n\n")
    for layer in range(6):
        layer_results = [r for r in valid if r.get('layer') == layer]
        if layer_results:
            best_l = max(layer_results, key=lambda x: x.get('test_auroc', 0))
            report.append(f"- **Layer {layer}**: Best with {best_l['pooling']} pooling (AUROC: {best_l.get('test_auroc', 0):.4f})\n")
    
    report_path = os.path.join(output_dir, 'RESULTS_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    print(f"  ✓ Report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="Final comparison and analysis")
    parser.add_argument("--drive_path", default=None,
                        help="Google Drive path (e.g., /content/drive/MyDrive/NOT_results)")
    args = parser.parse_args()
    
    # Determine base path
    if args.drive_path:
        base_path = args.drive_path
    else:
        base_path = "experiments"
    
    output_dir = os.path.join(base_path, "final_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print(f"\nBase path: {base_path}")
    print(f"Output: {output_dir}\n")
    
    # Step 1: Merge results
    print("Step 1: Merging results from parallel sweeps...")
    results = merge_results(base_path)
    
    if not results:
        print("\n❌ No results found! Make sure all 3 sweeps have completed.")
        return
    
    # Save merged results
    merged_file = os.path.join(output_dir, "all_results.json")
    with open(merged_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  ✓ Saved {len(results)} results to {merged_file}")
    
    # Step 2: Find best probe
    print("\nStep 2: Finding best probe configuration...")
    best = find_best_probe(results)
    
    if not best:
        print("  ❌ No valid results found!")
        return
    
    print(f"  🏆 Best: Layer {best['layer']}, {best['pooling']} pooling")
    print(f"     AUROC: {best.get('test_auroc', 0):.4f}")
    print(f"     Accuracy: {best.get('test_acc', 0):.4f}")
    
    # Save best config
    best_file = os.path.join(output_dir, "best_probe.json")
    with open(best_file, 'w') as f:
        json.dump(best, f, indent=2)
    
    # Step 3: Create visualizations
    print("\nStep 3: Creating visualizations...")
    create_visualization(results, output_dir)
    
    # Step 4: Generate report
    print("\nStep 4: Generating report...")
    generate_report(results, best, output_dir)
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles generated:")
    print(f"  - all_results.json")
    print(f"  - best_probe.json")
    print(f"  - probe_comparison.png")
    print(f"  - probe_comparison_bar.png")
    print(f"  - RESULTS_REPORT.md")
    
    print(f"\n🏆 Best Configuration:")
    print(f"   Layer {best['layer']} with {best['pooling']} pooling")
    print(f"   AUROC: {best.get('test_auroc', 0):.4f}")


if __name__ == "__main__":
    main()

