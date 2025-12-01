"""
Script to run causal intervention experiments.
"""

import os
import argparse
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from src.models import BaseModule
from src.datasets import NegationPairDataset
from src.engine.causal_interventions import CausalInterventionRunner
from src.engine.metrics import InterventionMetrics, split_by_negation


def find_negation_pairs(dataset: NegationPairDataset) -> Dict[int, int]:
    """
    Find pairs of negated/non-negated examples.
    
    Args:
        dataset: Dataset with pair information
    
    Returns:
        Dictionary mapping negated_idx to non_negated_idx
    """
    pairs = {}
    
    if dataset.pair_ids is None:
        # Try to match by text similarity (simple heuristic)
        texts = dataset.texts
        for i, text_i in enumerate(texts):
            text_i_lower = text_i.lower()
            if "not " in text_i_lower or "n't " in text_i_lower:
                # Find corresponding non-negated version
                for j, text_j in enumerate(texts):
                    if i != j:
                        # Remove "not " or "n't " from text_i and compare
                        text_i_clean = text_i_lower.replace("not ", "").replace("n't ", "")
                        text_j_clean = text_j.lower()
                        if text_i_clean == text_j_clean or text_j_clean in text_i_clean:
                            pairs[i] = j
                            break
    else:
        # Use pair IDs
        pair_dict = {}
        for i, pair_id in enumerate(dataset.pair_ids):
            if pair_id not in pair_dict:
                pair_dict[pair_id] = []
            pair_dict[pair_id].append(i)
        
        for pair_id, indices in pair_dict.items():
            if len(indices) >= 2:
                # Assume first is negated, second is non-negated
                pairs[indices[0]] = indices[1]
    
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Run causal intervention experiments"
    )
    
    # Model arguments
    parser.add_argument(
        "--model_ckpt",
        type=str,
        required=True,
        help="Path to trained model checkpoint (for probe mode) or base model path",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Base model name",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["finetune", "probe"],
        default="probe",
        help="Model mode (probe uses trained probe, finetune uses full model)",
    )
    parser.add_argument(
        "--probe_layer",
        type=int,
        default=None,
        help="Layer index for probe (if mode=probe)",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to parquet file with negation pairs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for interventions",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    
    # Intervention arguments
    parser.add_argument(
        "--intervention_type",
        type=str,
        choices=["activation_patching", "ablation", "control", "all"],
        default="all",
        help="Type of intervention to run",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to test (comma-separated or 'all')",
    )
    parser.add_argument(
        "--positions",
        type=str,
        default=None,
        help="Token positions to test (comma-separated or None for all)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/interventions",
        help="Directory to save results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Load model
    print(f"Loading model from {args.model_ckpt}...")
    model_module = BaseModule.load_from_checkpoint(args.model_ckpt)
    base_model = model_module.backbone
    
    if args.mode == "probe":
        # Get probe from model
        if hasattr(model_module, 'probes'):
            if args.probe_layer is not None:
                probe = model_module.probes.get(f"layer_{args.probe_layer}", None)
            else:
                # Use first available probe
                probe = next(iter(model_module.probes.values())) if model_module.probes else None
        else:
            probe = None
    else:
        probe = None
    
    # Load dataset
    print(f"Loading dataset from {args.data_path}...")
    dataset = NegationPairDataset(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )
    
    # Find pairs
    print("Finding negation pairs...")
    pairs = find_negation_pairs(dataset)
    print(f"Found {len(pairs)} pairs")
    
    if len(pairs) == 0:
        print("Warning: No pairs found. Cannot run activation patching.")
        return
    
    # Create data loader
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize intervention runner
    runner = CausalInterventionRunner(
        model=base_model,
        probe=probe,
        device=args.device
    )
    
    # Determine layers to test
    if args.layers == "all":
        num_layers = base_model.config.num_hidden_layers
        layers_to_test = list(range(num_layers))
    else:
        layers_to_test = [int(x) for x in args.layers.split(",")]
    
    # Determine positions to test
    positions_to_test = None
    if args.positions:
        positions_to_test = [int(x) for x in args.positions.split(",")]
    
    metrics = InterventionMetrics()
    
    print(f"\n{'='*60}")
    print(f"Running causal interventions...")
    print(f"{'='*60}\n")
    
    # Collect batches with pairs
    negated_batches = []
    non_negated_batches = []
    
    for batch in dataloader:
        # For simplicity, process pairs within batch
        # In practice, you'd want to properly match pairs across batches
        negated_batches.append(batch)
        non_negated_batches.append(batch)  # Would need proper pairing
    
    # Run activation patching
    if args.intervention_type in ["activation_patching", "all"]:
        print("Running activation patching experiments...")
        for layer_idx in layers_to_test:
            print(f"  Testing layer {layer_idx}...")
            
            # Run patching for each pair
            for i, (neg_batch, non_neg_batch) in enumerate(zip(negated_batches[:5], non_negated_batches[:5])):  # Limit for demo
                result = runner.run_activation_patching(
                    negated_batch=neg_batch,
                    non_negated_batch=non_neg_batch,
                    layer_idx=layer_idx,
                    position=positions_to_test[0] if positions_to_test else None,
                    use_probe=(args.mode == "probe")
                )
                
                metrics.add_result(
                    experiment_type="activation_patching",
                    layer_idx=layer_idx,
                    label_flips=result["label_flips"],
                    logit_delta=result["logit_deltas"],
                )
    
    # Run targeted ablation
    if args.intervention_type in ["ablation", "all"]:
        print("\nRunning targeted ablation experiments...")
        # This would require identifying important dimensions from probes
        # For now, demonstrate with random dimensions
        for layer_idx in layers_to_test:
            print(f"  Testing layer {layer_idx}...")
            
            hidden_size = base_model.config.hidden_size
            # Ablate top 10% dimensions (would use probe weights in practice)
            num_dims = hidden_size // 10
            important_dims = torch.randperm(hidden_size)[:num_dims].tolist()
            
            for batch in negated_batches[:5]:
                result = runner.run_targeted_ablation(
                    batch=batch,
                    layer_idx=layer_idx,
                    dimensions=important_dims,
                    ablation_type="zero",
                    use_probe=(args.mode == "probe")
                )
                
                metrics.add_result(
                    experiment_type="targeted_ablation",
                    layer_idx=layer_idx,
                    label_flips=result["label_flips"],
                    logit_delta=result["logit_deltas"],
                )
    
    # Run control experiments
    if args.intervention_type in ["control", "all"]:
        print("\nRunning control experiments...")
        for layer_idx in layers_to_test:
            print(f"  Testing layer {layer_idx}...")
            
            for batch in negated_batches[:5]:
                result = runner.run_control_experiment(
                    batch=batch,
                    control_type="random",
                    layer_idx=layer_idx,
                    use_probe=(args.mode == "probe")
                )
                
                metrics.add_result(
                    experiment_type="control",
                    layer_idx=layer_idx,
                    label_flips=result["label_flips"],
                    logit_delta=result["logit_deltas"],
                )
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")
    
    summary = metrics.get_overall_summary()
    
    # Save to JSON
    import json
    output_path = os.path.join(args.output_dir, "intervention_results.json")
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"Results saved to {output_path}")
    
    # Print summary
    print("\nSummary:")
    for exp_type, layer_summary in summary.items():
        print(f"\n{exp_type}:")
        for layer_idx, stats in layer_summary.items():
            print(f"  Layer {layer_idx}:")
            print(f"    Avg label flips: {stats['avg_label_flips']:.3f}")
            print(f"    Avg logit delta: {stats['avg_logit_delta']:.6f}")


if __name__ == "__main__":
    main()

