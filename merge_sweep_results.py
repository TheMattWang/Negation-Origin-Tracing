#!/usr/bin/env python
"""
Merge results from parallel sweep scripts (cls, mean, token).
Run this after all 3 sweeps complete.

Usage:
    python merge_sweep_results.py
    python merge_sweep_results.py --output_dir experiments/merged_results
"""

import json
import os
import argparse
from pathlib import Path


def merge_results(output_dir="experiments/merged_results"):
    """Merge results from all 3 pooling sweep directories."""
    
    # Default sweep directories
    sweep_dirs = {
        "cls": "experiments/sweep_cls",
        "mean": "experiments/sweep_mean",
        "token": "experiments/sweep_token",
    }
    
    all_results = []
    
    print("Merging sweep results...")
    print("=" * 50)
    
    for pooling, sweep_dir in sweep_dirs.items():
        results_file = os.path.join(sweep_dir, f"results_{pooling}.json")
        
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"✓ {pooling}: {len(results)} results from {results_file}")
            all_results.extend(results)
        else:
            print(f"✗ {pooling}: Not found at {results_file}")
    
    if not all_results:
        print("\nNo results found!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save merged results
    merged_file = os.path.join(output_dir, "results_summary.json")
    with open(merged_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Merged {len(all_results)} results to {merged_file}")
    
    # Print summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    # Filter valid results
    valid = [r for r in all_results if 'error' not in r and r.get('test_auroc', 0) > 0]
    
    if valid:
        # Best overall
        best = max(valid, key=lambda x: x.get('test_auroc', 0))
        print(f"\nBest overall:")
        print(f"  Layer {best['layer']}, {best['pooling']}")
        print(f"  AUROC: {best.get('test_auroc', 0):.4f}")
        print(f"  Accuracy: {best.get('test_acc', 0):.4f}")
        
        # Best per pooling
        print("\nBest per pooling strategy:")
        for pooling in ['cls', 'mean', 'token']:
            pooling_results = [r for r in valid if r.get('pooling') == pooling]
            if pooling_results:
                best_p = max(pooling_results, key=lambda x: x.get('test_auroc', 0))
                print(f"  {pooling}: Layer {best_p['layer']} -> AUROC {best_p.get('test_auroc', 0):.4f}")
        
        # Best per layer
        print("\nBest per layer:")
        for layer in range(6):
            layer_results = [r for r in valid if r.get('layer') == layer]
            if layer_results:
                best_l = max(layer_results, key=lambda x: x.get('test_auroc', 0))
                print(f"  Layer {layer}: {best_l['pooling']} -> AUROC {best_l.get('test_auroc', 0):.4f}")
    
    # Save CSV for easy analysis
    try:
        import pandas as pd
        df = pd.DataFrame(all_results)
        csv_file = os.path.join(output_dir, "results_summary.csv")
        df.to_csv(csv_file, index=False)
        print(f"\n✓ CSV saved to {csv_file}")
    except ImportError:
        print("\n(pandas not available, skipping CSV export)")
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge parallel sweep results")
    parser.add_argument("--output_dir", default="experiments/merged_results",
                        help="Output directory for merged results")
    args = parser.parse_args()
    
    merge_results(args.output_dir)

