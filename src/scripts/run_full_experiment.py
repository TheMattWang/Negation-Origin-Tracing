"""
Complete experiment pipeline: probe training → analysis → interventions → interpretation.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.engine.visualization import generate_all_visualizations
from src.engine.interpretation import generate_interpretation_report


def run_layer_search(args):
    """Run automated layer search."""
    print("\n" + "="*60)
    print("STEP 1: Automated Layer Search")
    print("="*60 + "\n")
    
    cmd = [
        sys.executable,
        "src/scripts/search_layers.py",
        "--data_dir", args.data_dir,
        "--model_name", args.model_name,
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs),
        "--probe_lr", str(args.probe_lr),
        "--output_dir", args.output_dir,
        "--layers", args.layers,
        "--pooling_strategies", args.pooling_strategies,
        "--seed", str(args.seed),
    ]
    
    if hasattr(args, 'devices'):
        cmd.extend(["--devices", str(args.devices)])
    
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_visualizations(results_path, output_dir):
    """Generate visualizations from probe results."""
    print("\n" + "="*60)
    print("STEP 2: Generating Visualizations")
    print("="*60 + "\n")
    
    viz_dir = os.path.join(output_dir, "visualizations")
    generate_all_visualizations(results_path, viz_dir)
    return True


def run_interventions(args, probe_results_path):
    """Run causal interventions on identified layers."""
    print("\n" + "="*60)
    print("STEP 3: Running Causal Interventions")
    print("="*60 + "\n")
    
    # Load probe results to identify best layers
    import pandas as pd
    import json
    
    if probe_results_path.endswith('.json'):
        with open(probe_results_path, 'r') as f:
            probe_results = json.load(f)
        df = pd.DataFrame(probe_results)
    else:
        df = pd.read_csv(probe_results_path)
    
    # Get top layers
    df_sorted = df.nlargest(3, 'test_auroc')
    top_layers = df_sorted['layer_idx'].unique().tolist()
    
    print(f"Running interventions on top layers: {top_layers}")
    
    # Find best checkpoint for each top layer
    intervention_dir = os.path.join(args.output_dir, "interventions")
    os.makedirs(intervention_dir, exist_ok=True)
    
    for layer_idx in top_layers:
        # Find best checkpoint for this layer
        layer_results = df[df['layer_idx'] == layer_idx]
        best_result = layer_results.loc[layer_results['test_auroc'].idxmax()]
        checkpoint_path = best_result.get('checkpoint_path')
        
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            print(f"⚠ Warning: Checkpoint not found for layer {layer_idx}, skipping interventions")
            continue
        
        print(f"\nRunning interventions for layer {layer_idx}...")
        
        # Check if we have negation dataset
        negation_data_path = os.path.join(args.data_dir, "test", "negation.parquet")
        if not os.path.exists(negation_data_path):
            print(f"⚠ Warning: Negation dataset not found at {negation_data_path}")
            print("  Skipping interventions (requires negation pairs)")
            break
        
        cmd = [
            sys.executable,
            "src/scripts/run_interventions.py",
            "--model_ckpt", checkpoint_path,
            "--mode", "probe",
            "--probe_layer", str(layer_idx),
            "--data_path", negation_data_path,
            "--intervention_type", "all",
            "--layers", str(layer_idx),
            "--output_dir", os.path.join(intervention_dir, f"layer_{layer_idx}"),
            "--batch_size", str(args.batch_size),
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"⚠ Error running interventions for layer {layer_idx}: {e}")
            continue
    
    return True


def run_interpretation(args, probe_results_path):
    """Generate interpretation report."""
    print("\n" + "="*60)
    print("STEP 4: Generating Interpretation Report")
    print("="*60 + "\n")
    
    # Find intervention results
    intervention_results_path = None
    intervention_dir = os.path.join(args.output_dir, "interventions")
    
    # Look for intervention results (use first found)
    for root, dirs, files in os.walk(intervention_dir):
        if "intervention_results.json" in files:
            intervention_results_path = os.path.join(root, "intervention_results.json")
            break
    
    if not intervention_results_path:
        print("⚠ Warning: No intervention results found, generating report from probes only")
    
    report_path = os.path.join(args.output_dir, "interpretation_report.json")
    
    report = generate_interpretation_report(
        probe_results_path,
        intervention_results_path,
        report_path,
        top_k=args.top_k
    )
    
    # Print summary
    print("\n" + "="*60)
    print("EXPERIMENT SUMMARY")
    print("="*60)
    print(f"\nBest Layer: {report['summary']['best_layer']}")
    print(f"Best Probe AUROC: {report['summary']['best_probe_auroc']:.4f}")
    print(f"\nTop {args.top_k} Negation Layers:")
    for i, layer in enumerate(report['negation_layers'], 1):
        print(f"  {i}. Layer {layer['layer_idx']}: "
              f"score={layer['composite_score']:.4f}, "
              f"AUROC={layer['avg_probe_score']:.4f}")
    
    if report['causality_verification']:
        causal_count = sum(1 for c in report['causality_verification'] if c['is_causal'])
        print(f"\nCausal Layers Found: {causal_count}/{len(report['causality_verification'])}")
    
    print("\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run complete experiment pipeline: probe → analysis → interventions → interpretation"
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Data directory",
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name",
    )
    
    # Training arguments
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=10,
        help="Max epochs for probe training",
    )
    parser.add_argument(
        "--probe_lr",
        type=float,
        default=1e-3,
        help="Probe learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Search arguments
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to search (comma-separated or 'all')",
    )
    parser.add_argument(
        "--pooling_strategies",
        type=str,
        default="all",
        help="Pooling strategies to test (comma-separated or 'all')",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/full_experiment",
        help="Output directory",
    )
    
    # Pipeline control
    parser.add_argument(
        "--skip_search",
        action="store_true",
        help="Skip layer search (use existing results)",
    )
    parser.add_argument(
        "--skip_interventions",
        action="store_true",
        help="Skip causal interventions",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of top layers for interpretation",
    )
    
    # Hardware
    parser.add_argument(
        "--devices",
        type=int,
        default=1,
        help="Number of devices",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("COMPLETE EXPERIMENT PIPELINE")
    print("="*60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Layers to search: {args.layers}")
    print(f"Pooling strategies: {args.pooling_strategies}")
    print("="*60 + "\n")
    
    # Step 1: Layer search
    probe_results_path = os.path.join(args.output_dir, "results_summary.json")
    
    if not args.skip_search:
        if not run_layer_search(args):
            print("✗ Layer search failed")
            return 1
    else:
        if not os.path.exists(probe_results_path):
            print(f"✗ Probe results not found at {probe_results_path}")
            return 1
        print("✓ Using existing probe results")
    
    # Step 2: Visualizations
    if not run_visualizations(probe_results_path, args.output_dir):
        print("✗ Visualization generation failed")
        return 1
    
    # Step 3: Interventions
    if not args.skip_interventions:
        if not run_interventions(args, probe_results_path):
            print("⚠ Interventions had errors, continuing...")
    else:
        print("⏭ Skipping interventions")
    
    # Step 4: Interpretation
    if not run_interpretation(args, probe_results_path):
        print("✗ Interpretation failed")
        return 1
    
    print("\n" + "="*60)
    print("EXPERIMENT COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {args.output_dir}")
    print(f"  - Probe results: {probe_results_path}")
    print(f"  - Visualizations: {os.path.join(args.output_dir, 'visualizations')}")
    print(f"  - Interventions: {os.path.join(args.output_dir, 'interventions')}")
    print(f"  - Interpretation: {os.path.join(args.output_dir, 'interpretation_report.json')}")
    print("="*60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

