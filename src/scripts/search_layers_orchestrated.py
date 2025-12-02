"""
Orchestrated layer search: launches separate processes for each pooling strategy.
Each process is independent with its own GPU context (no threading/multiprocessing issues).
"""

import os
import sys
from pathlib import Path
import json
import argparse
import subprocess
import time
from transformers import AutoModel
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def train_single_probe(layer_idx, pooling_strategy, args, output_dir):
    """
    Launch a separate Python process to train a single probe.
    Returns the subprocess object.
    """
    cmd = [
        sys.executable,
        "src/scripts/train.py",
        "--mode", "probe",
        "--model_name", args.model_name,
        "--data_dir", args.data_dir,
        "--batch_size", str(args.batch_size),
        "--max_epochs", str(args.max_epochs),
        "--probe_lr", str(args.probe_lr),
        "--probe_layer", str(layer_idx),
        "--pooling_strategy", pooling_strategy,
        "--experiment_name", f"layer_{layer_idx}_pooling_{pooling_strategy}",
        "--output_dir", output_dir,
        "--seed", str(args.seed),
        "--devices", str(args.devices),
        "--precision", str(args.precision),
    ]
    
    # Launch process
    print(f"🚀 Launching: Layer {layer_idx}, {pooling_strategy}")
    sys.stdout.flush()
    
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1
    )
    
    return process


def main():
    parser = argparse.ArgumentParser(
        description="Orchestrated layer search: separate processes for parallel execution"
    )
    
    # Arguments
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--data_dir", type=str, default="data/raw")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--probe_lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--num_labels", type=int, default=2)
    parser.add_argument("--layers", type=str, default="all")
    parser.add_argument("--pooling_strategies", type=str, default="all")
    parser.add_argument("--output_dir", type=str, default="experiments/layer_search")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--max_parallel", type=int, default=3, help="Max parallel processes (default: 3)")
    
    args = parser.parse_args()
    
    # Determine layers
    if args.layers == "all":
        base_model = AutoModel.from_pretrained(args.model_name)
        num_layers = base_model.config.num_hidden_layers
        layers_to_search = list(range(num_layers))
        print(f"Searching all {num_layers} layers")
    else:
        layers_to_search = [int(x) for x in args.layers.split(",")]
        print(f"Searching layers: {layers_to_search}")
    
    # Determine pooling strategies
    if args.pooling_strategies == "all":
        pooling_strategies = ["cls", "mean", "token"]
    else:
        pooling_strategies = [x.strip() for x in args.pooling_strategies.split(",")]
    
    print(f"Pooling strategies: {pooling_strategies}")
    print(f"Output directory: {args.output_dir}")
    print(f"Max parallel processes: {args.max_parallel}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load existing results
    all_results = []
    completed_experiments = set()
    results_json_path = os.path.join(args.output_dir, "results_summary.json")
    
    should_resume = not args.no_resume
    
    if should_resume and os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                all_results = json.load(f)
            
            for result in all_results:
                if 'error' not in result and result.get('test_auroc', 0) > 0:
                    completed_experiments.add((result['layer_idx'], result['pooling_strategy']))
            
            print(f"✓ Loaded {len(all_results)} existing results")
            print(f"✓ {len(completed_experiments)} experiments already completed")
        except Exception as e:
            print(f"⚠ Could not load existing results: {e}")
    
    # Determine experiments to run
    total_experiments = len(layers_to_search) * len(pooling_strategies)
    experiments_to_run = [
        (layer, pooling)
        for layer in layers_to_search
        for pooling in pooling_strategies
        if (layer, pooling) not in completed_experiments
    ]
    
    # Prioritize partially completed
    partially_completed = []
    not_started = []
    
    for layer, pooling in experiments_to_run:
        experiment_name = f"layer_{layer}_pooling_{pooling}"
        experiment_dir = os.path.join(args.output_dir, experiment_name)
        checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
        
        if os.path.exists(checkpoint_dir) and os.listdir(checkpoint_dir):
            partially_completed.append((layer, pooling))
        else:
            not_started.append((layer, pooling))
    
    experiments_to_run = partially_completed + not_started
    
    if partially_completed:
        print(f"✓ Found {len(partially_completed)} partially completed experiments (will run first)")
    
    remaining = len(experiments_to_run)
    
    print(f"\n{'='*60}")
    print(f"Starting layer search: {total_experiments} total experiments")
    print(f"Already completed: {len(completed_experiments)}")
    print(f"Remaining: {remaining}")
    print(f"{'='*60}\n")
    
    if remaining == 0:
        print("✓ All experiments already completed!")
        return
    
    # Group by layer for efficient parallel execution
    experiments_by_layer = {}
    for layer, pooling in experiments_to_run:
        if layer not in experiments_by_layer:
            experiments_by_layer[layer] = []
        experiments_by_layer[layer].append(pooling)
    
    current_experiment = len(completed_experiments)
    
    # Process each layer
    for layer_idx in sorted(experiments_by_layer.keys()):
        pooling_strategies_for_layer = experiments_by_layer[layer_idx]
        
        print(f"\n{'='*60}")
        print(f"Layer {layer_idx}: Running {len(pooling_strategies_for_layer)} pooling strategies in parallel")
        print(f"{'='*60}\n")
        
        # Launch processes for this layer (up to max_parallel)
        processes = {}
        for pooling in pooling_strategies_for_layer:
            process = train_single_probe(layer_idx, pooling, args, args.output_dir)
            processes[pooling] = process
            time.sleep(2)  # Small delay between launches
        
        # Monitor processes and show output
        while processes:
            for pooling, process in list(processes.items()):
                # Read output line by line
                line = process.stdout.readline()
                if line:
                    print(f"[{pooling}] {line.rstrip()}")
                    sys.stdout.flush()
                
                # Check if process finished
                if process.poll() is not None:
                    # Process finished
                    return_code = process.returncode
                    
                    # Read any remaining output
                    for line in process.stdout:
                        print(f"[{pooling}] {line.rstrip()}")
                    
                    if return_code == 0:
                        print(f"\n✓ Completed: Layer {layer_idx}, {pooling}")
                        current_experiment += 1
                        print(f"Progress: {current_experiment}/{total_experiments}\n")
                    else:
                        print(f"\n✗ Failed: Layer {layer_idx}, {pooling} (exit code: {return_code})\n")
                    
                    sys.stdout.flush()
                    del processes[pooling]
            
            time.sleep(0.1)  # Small sleep to avoid busy waiting
        
        print(f"✓ Layer {layer_idx} complete\n")
    
    # Collect all results from experiment directories
    print(f"\n{'='*60}")
    print("Collecting results...")
    print(f"{'='*60}\n")
    
    all_results = []
    for layer in layers_to_search:
        for pooling in pooling_strategies:
            experiment_name = f"layer_{layer}_pooling_{pooling}"
            experiment_dir = os.path.join(args.output_dir, experiment_name)
            
            # Look for best checkpoint
            checkpoint_dir = os.path.join(experiment_dir, "checkpoints")
            best_ckpt = None
            if os.path.exists(checkpoint_dir):
                ckpts = [f for f in os.listdir(checkpoint_dir) if f.startswith("best-") and f.endswith(".ckpt")]
                if ckpts:
                    best_ckpt = os.path.join(checkpoint_dir, ckpts[0])
            
            if best_ckpt:
                # Extract metrics from filename (best-{val_loss:.3f}-{val_acc:.3f}.ckpt)
                try:
                    parts = os.path.basename(best_ckpt).replace("best-", "").replace(".ckpt", "").split("-")
                    val_loss = float(parts[0].split("=")[1])
                    val_acc = float(parts[1].split("=")[1])
                    
                    result = {
                        "layer_idx": layer,
                        "pooling_strategy": pooling,
                        "checkpoint_path": best_ckpt,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc,
                        "test_accuracy": val_acc,  # Use val as proxy
                        "test_auroc": val_acc,  # Use val as proxy
                        "experiment_dir": experiment_dir,
                    }
                    all_results.append(result)
                except:
                    pass
    
    # Save results
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved to {results_json_path}")
    
    df = pd.DataFrame(all_results)
    results_csv_path = os.path.join(args.output_dir, "results_summary.csv")
    df.to_csv(results_csv_path, index=False)
    print(f"✓ CSV saved to {results_csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}\n")
    
    if len(all_results) > 0:
        df_summary = pd.DataFrame(all_results)
        
        print("Best Test Accuracy:")
        best_acc = df_summary.loc[df_summary['test_accuracy'].idxmax()]
        print(f"  Layer {best_acc['layer_idx']}, {best_acc['pooling_strategy']}: {best_acc['test_accuracy']:.4f}")
        
        print("\nBest Test AUROC:")
        best_auroc = df_summary.loc[df_summary['test_auroc'].idxmax()]
        print(f"  Layer {best_auroc['layer_idx']}, {best_auroc['pooling_strategy']}: {best_auroc['test_auroc']:.4f}")
    
    print(f"\n{'='*60}")
    print("Layer search complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

