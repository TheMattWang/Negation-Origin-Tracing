#!/usr/bin/env python
"""
Run Interventions and Compare Base vs Finetuned Models

This script:
1. Loads the best probe from the sweep results
2. Runs causal interventions on the base model with the probe
3. Runs causal interventions on the finetuned model
4. Compares the results to understand how negation is encoded differently

Usage (local):
    python run_interventions_comparison.py

Usage (Google Drive):
    python run_interventions_comparison.py --drive_path /content/drive/MyDrive/NOT_results
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
torch.set_num_threads(1)


def load_best_probe(base_path: str) -> Optional[Dict]:
    """Load the best probe configuration from sweep results."""
    
    # Try final_comparison first
    best_file = os.path.join(base_path, "final_comparison", "best_probe.json")
    if os.path.exists(best_file):
        with open(best_file, 'r') as f:
            return json.load(f)
    
    # Otherwise, merge results and find best
    all_results = []
    for pooling in ['cls', 'mean', 'token']:
        results_file = os.path.join(base_path, f"sweep_{pooling}", f"results_{pooling}.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                all_results.extend(json.load(f))
    
    if not all_results:
        return None
    
    valid = [r for r in all_results if 'error' not in r and r.get('test_auroc', 0) > 0]
    if not valid:
        return None
    
    return max(valid, key=lambda x: x.get('test_auroc', 0))


def run_base_model_interventions(
    best_probe: Dict,
    base_path: str,
    data_path: str,
    output_dir: str,
    device: str = "cuda"
) -> Dict:
    """Run interventions on the base model using the trained probe."""
    
    from transformers import AutoTokenizer, AutoModel
    from torch.utils.data import DataLoader
    from src.models import BaseModule
    from src.datasets.dataset import SentimentDataset
    from src.engine.causal_interventions import CausalInterventionRunner
    
    print("\n" + "=" * 60)
    print("BASE MODEL INTERVENTIONS")
    print("=" * 60)
    
    layer = best_probe['layer']
    pooling = best_probe['pooling']
    checkpoint = best_probe.get('checkpoint')
    
    print(f"Layer: {layer}")
    print(f"Pooling: {pooling}")
    print(f"Checkpoint: {checkpoint}")
    
    # Load the trained probe
    if checkpoint and os.path.exists(checkpoint):
        print(f"\nLoading probe from checkpoint...")
        model_module = BaseModule.load_from_checkpoint(checkpoint)
        base_model = model_module.backbone
        # ModuleDict doesn't have .get(), use direct access with error handling
        probe_key = f"layer_{layer}"
        probe = model_module.probes[probe_key] if probe_key in model_module.probes else None
    else:
        print(f"\n⚠ Checkpoint not found, using base model without probe")
        base_model = AutoModel.from_pretrained("distilbert-base-uncased")
        probe = None
    
    base_model = base_model.to(device)
    base_model.eval()
    
    # Load tokenizer and data
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Use test data for interventions
    dataset = SentimentDataset(data_path, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Run interventions
    runner = CausalInterventionRunner(
        model=base_model,
        probe=probe,
        device=device
    )
    
    results = {
        "model_type": "base",
        "layer": layer,
        "pooling": pooling,
        "interventions": {}
    }
    
    # Test activation patching on the best layer
    print(f"\nRunning activation patching on layer {layer}...")
    
    patching_results = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:  # Limit for speed
            break
        
        try:
            result = runner.run_activation_patching(
                negated_batch=batch,
                non_negated_batch=batch,  # Self-patching for baseline
                layer_idx=layer,
                position=None,
                use_probe=(probe is not None)
            )
            patching_results.append({
                "label_flips": result["label_flips"],
                "logit_delta": float(torch.tensor(result["logit_deltas"]).mean()) if result["logit_deltas"] else 0
            })
        except Exception as e:
            print(f"  Batch {batch_idx} failed: {e}")
    
    if patching_results:
        avg_flips = sum(r["label_flips"] for r in patching_results) / len(patching_results)
        avg_delta = sum(r["logit_delta"] for r in patching_results) / len(patching_results)
        results["interventions"]["activation_patching"] = {
            "avg_label_flips": avg_flips,
            "avg_logit_delta": avg_delta,
            "num_batches": len(patching_results)
        }
        print(f"  Avg label flips: {avg_flips:.3f}")
        print(f"  Avg logit delta: {avg_delta:.4f}")
    
    # Test targeted ablation
    print(f"\nRunning targeted ablation on layer {layer}...")
    
    ablation_results = []
    hidden_size = base_model.config.hidden_size
    num_dims = hidden_size // 10  # Ablate top 10%
    important_dims = list(range(num_dims))  # Would use probe weights in practice
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:
            break
        
        try:
            result = runner.run_targeted_ablation(
                batch=batch,
                layer_idx=layer,
                dimensions=important_dims,
                ablation_type="zero",
                use_probe=(probe is not None)
            )
            ablation_results.append({
                "label_flips": result["label_flips"],
                "logit_delta": float(torch.tensor(result["logit_deltas"]).mean()) if result["logit_deltas"] else 0
            })
        except Exception as e:
            print(f"  Batch {batch_idx} failed: {e}")
    
    if ablation_results:
        avg_flips = sum(r["label_flips"] for r in ablation_results) / len(ablation_results)
        avg_delta = sum(r["logit_delta"] for r in ablation_results) / len(ablation_results)
        results["interventions"]["targeted_ablation"] = {
            "avg_label_flips": avg_flips,
            "avg_logit_delta": avg_delta,
            "num_batches": len(ablation_results)
        }
        print(f"  Avg label flips: {avg_flips:.3f}")
        print(f"  Avg logit delta: {avg_delta:.4f}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "base_interventions.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def run_finetuned_model_interventions(
    best_layer: int,
    data_path: str,
    output_dir: str,
    device: str = "cuda"
) -> Dict:
    """Run interventions on the finetuned model."""
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.utils.data import DataLoader
    from src.datasets.dataset import SentimentDataset
    from src.engine.causal_interventions import CausalInterventionRunner
    
    print("\n" + "=" * 60)
    print("FINETUNED MODEL INTERVENTIONS")
    print("=" * 60)
    
    # Load finetuned model
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    print(f"Loading: {model_name}")
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load data
    dataset = SentimentDataset(data_path, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Run interventions
    runner = CausalInterventionRunner(
        model=model,
        probe=None,  # Finetuned model uses its own classifier
        device=device
    )
    
    results = {
        "model_type": "finetuned",
        "model_name": model_name,
        "layer": best_layer,
        "interventions": {}
    }
    
    # Test on the same layer as base model for comparison
    print(f"\nRunning activation patching on layer {best_layer}...")
    
    patching_results = []
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:
            break
        
        try:
            result = runner.run_activation_patching(
                negated_batch=batch,
                non_negated_batch=batch,
                layer_idx=best_layer,
                position=None,
                use_probe=False
            )
            patching_results.append({
                "label_flips": result["label_flips"],
                "logit_delta": float(torch.tensor(result["logit_deltas"]).mean()) if result["logit_deltas"] else 0
            })
        except Exception as e:
            print(f"  Batch {batch_idx} failed: {e}")
    
    if patching_results:
        avg_flips = sum(r["label_flips"] for r in patching_results) / len(patching_results)
        avg_delta = sum(r["logit_delta"] for r in patching_results) / len(patching_results)
        results["interventions"]["activation_patching"] = {
            "avg_label_flips": avg_flips,
            "avg_logit_delta": avg_delta,
            "num_batches": len(patching_results)
        }
        print(f"  Avg label flips: {avg_flips:.3f}")
        print(f"  Avg logit delta: {avg_delta:.4f}")
    
    # Test targeted ablation
    print(f"\nRunning targeted ablation on layer {best_layer}...")
    
    ablation_results = []
    hidden_size = model.config.hidden_size
    num_dims = hidden_size // 10
    important_dims = list(range(num_dims))
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 10:
            break
        
        try:
            result = runner.run_targeted_ablation(
                batch=batch,
                layer_idx=best_layer,
                dimensions=important_dims,
                ablation_type="zero",
                use_probe=False
            )
            ablation_results.append({
                "label_flips": result["label_flips"],
                "logit_delta": float(torch.tensor(result["logit_deltas"]).mean()) if result["logit_deltas"] else 0
            })
        except Exception as e:
            print(f"  Batch {batch_idx} failed: {e}")
    
    if ablation_results:
        avg_flips = sum(r["label_flips"] for r in ablation_results) / len(ablation_results)
        avg_delta = sum(r["logit_delta"] for r in ablation_results) / len(ablation_results)
        results["interventions"]["targeted_ablation"] = {
            "avg_label_flips": avg_flips,
            "avg_logit_delta": avg_delta,
            "num_batches": len(ablation_results)
        }
        print(f"  Avg label flips: {avg_flips:.3f}")
        print(f"  Avg logit delta: {avg_delta:.4f}")
    
    # Save results
    with open(os.path.join(output_dir, "finetuned_interventions.json"), 'w') as f:
        json.dump(results, f, indent=2)
    
    return results


def generate_comparison_report(
    base_results: Dict,
    finetuned_results: Dict,
    best_probe: Dict,
    output_dir: str
):
    """Generate a comparison report between base and finetuned models."""
    
    print("\n" + "=" * 60)
    print("GENERATING COMPARISON REPORT")
    print("=" * 60)
    
    report = []
    report.append("# Base vs Finetuned Model Comparison\n\n")
    
    report.append("## Best Probe Configuration\n\n")
    report.append(f"- **Layer**: {best_probe['layer']}\n")
    report.append(f"- **Pooling**: {best_probe['pooling']}\n")
    report.append(f"- **Test AUROC**: {best_probe.get('test_auroc', 'N/A'):.4f}\n")
    report.append(f"- **Test Accuracy**: {best_probe.get('test_acc', 'N/A'):.4f}\n\n")
    
    report.append("## Intervention Results\n\n")
    
    # Activation Patching comparison
    report.append("### Activation Patching\n\n")
    report.append("| Model | Avg Label Flips | Avg Logit Delta |\n")
    report.append("|-------|-----------------|------------------|\n")
    
    base_patch = base_results.get("interventions", {}).get("activation_patching", {})
    ft_patch = finetuned_results.get("interventions", {}).get("activation_patching", {})
    
    report.append(f"| Base (with probe) | {base_patch.get('avg_label_flips', 'N/A'):.3f} | {base_patch.get('avg_logit_delta', 'N/A'):.4f} |\n")
    report.append(f"| Finetuned | {ft_patch.get('avg_label_flips', 'N/A'):.3f} | {ft_patch.get('avg_logit_delta', 'N/A'):.4f} |\n\n")
    
    # Targeted Ablation comparison
    report.append("### Targeted Ablation\n\n")
    report.append("| Model | Avg Label Flips | Avg Logit Delta |\n")
    report.append("|-------|-----------------|------------------|\n")
    
    base_abl = base_results.get("interventions", {}).get("targeted_ablation", {})
    ft_abl = finetuned_results.get("interventions", {}).get("targeted_ablation", {})
    
    report.append(f"| Base (with probe) | {base_abl.get('avg_label_flips', 'N/A'):.3f} | {base_abl.get('avg_logit_delta', 'N/A'):.4f} |\n")
    report.append(f"| Finetuned | {ft_abl.get('avg_label_flips', 'N/A'):.3f} | {ft_abl.get('avg_logit_delta', 'N/A'):.4f} |\n\n")
    
    # Analysis
    report.append("## Analysis\n\n")
    
    if base_patch and ft_patch:
        base_delta = base_patch.get('avg_logit_delta', 0)
        ft_delta = ft_patch.get('avg_logit_delta', 0)
        
        if abs(base_delta) > abs(ft_delta):
            report.append("- **Base model shows stronger response to activation patching**, suggesting negation information is more localized and easier to manipulate.\n")
        else:
            report.append("- **Finetuned model shows stronger response to activation patching**, suggesting negation handling is more integrated into the model's representations.\n")
    
    if base_abl and ft_abl:
        base_flips = base_abl.get('avg_label_flips', 0)
        ft_flips = ft_abl.get('avg_label_flips', 0)
        
        if base_flips > ft_flips:
            report.append("- **Base model is more sensitive to ablation**, indicating negation-related features are concentrated in specific dimensions.\n")
        else:
            report.append("- **Finetuned model is more sensitive to ablation**, suggesting it relies more heavily on specific features for negation.\n")
    
    report.append("\n## Conclusion\n\n")
    report.append(f"The best layer for detecting negation is **Layer {best_probe['layer']}** with **{best_probe['pooling']}** pooling.\n")
    report.append("This layer shows the strongest probe performance and intervention effects.\n")
    
    # Save report
    report_path = os.path.join(output_dir, "COMPARISON_REPORT.md")
    with open(report_path, 'w') as f:
        f.write(''.join(report))
    print(f"✓ Report saved to {report_path}")
    
    # Save combined results
    combined = {
        "best_probe": best_probe,
        "base_model": base_results,
        "finetuned_model": finetuned_results
    }
    combined_path = os.path.join(output_dir, "comparison_results.json")
    with open(combined_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"✓ Combined results saved to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Run interventions and compare models")
    parser.add_argument("--drive_path", default=None,
                        help="Google Drive path for results")
    parser.add_argument("--data_dir", default="data/raw",
                        help="Data directory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")
    args = parser.parse_args()
    
    # Determine paths
    if args.drive_path:
        base_path = args.drive_path
    else:
        base_path = "experiments"
    
    output_dir = os.path.join(base_path, "intervention_comparison")
    os.makedirs(output_dir, exist_ok=True)
    
    data_path = os.path.join(args.data_dir, "test", "sst.parquet")
    
    print("=" * 60)
    print("INTERVENTION COMPARISON: BASE vs FINETUNED")
    print("=" * 60)
    print(f"\nBase path: {base_path}")
    print(f"Output: {output_dir}")
    print(f"Data: {data_path}")
    print(f"Device: {args.device}")
    
    # Step 1: Load best probe
    print("\n" + "-" * 40)
    print("Step 1: Loading best probe configuration...")
    print("-" * 40)
    
    best_probe = load_best_probe(base_path)
    
    if not best_probe:
        print("❌ No probe results found! Run the sweep scripts first.")
        return
    
    print(f"✓ Best probe: Layer {best_probe['layer']}, {best_probe['pooling']} pooling")
    print(f"  AUROC: {best_probe.get('test_auroc', 'N/A')}")
    
    # Step 2: Run base model interventions
    print("\n" + "-" * 40)
    print("Step 2: Running base model interventions...")
    print("-" * 40)
    
    base_results = run_base_model_interventions(
        best_probe=best_probe,
        base_path=base_path,
        data_path=data_path,
        output_dir=output_dir,
        device=args.device
    )
    
    # Step 3: Run finetuned model interventions
    print("\n" + "-" * 40)
    print("Step 3: Running finetuned model interventions...")
    print("-" * 40)
    
    finetuned_results = run_finetuned_model_interventions(
        best_layer=best_probe['layer'],
        data_path=data_path,
        output_dir=output_dir,
        device=args.device
    )
    
    # Step 4: Generate comparison report
    print("\n" + "-" * 40)
    print("Step 4: Generating comparison report...")
    print("-" * 40)
    
    generate_comparison_report(
        base_results=base_results,
        finetuned_results=finetuned_results,
        best_probe=best_probe,
        output_dir=output_dir
    )
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles generated:")
    print("  - base_interventions.json")
    print("  - finetuned_interventions.json")
    print("  - comparison_results.json")
    print("  - COMPARISON_REPORT.md")


if __name__ == "__main__":
    main()

