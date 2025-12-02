"""
Parallelized automated layer search script.
Trains probes on multiple layers/pooling strategies in parallel.
"""

import os
import sys
from pathlib import Path
import json
import argparse
import multiprocessing

# Set multiprocessing start method to 'spawn' for CUDA compatibility
# This MUST be done before importing torch/CUDA libraries
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass  # Already set

import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoModel
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models import BaseModule
from src.datasets import NOTDataModule
from src.utils.parser import get_parser


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_probe_for_layer(
    layer_idx: int,
    pooling_strategy: str,
    args,
    base_output_dir: str
):
    """
    Train a probe for a specific layer and pooling strategy.
    This function is designed to be called in parallel.
    
    Args:
        layer_idx: Layer index to probe
        pooling_strategy: Pooling strategy ('cls', 'mean', 'token')
        args: Arguments from parser
        base_output_dir: Base output directory
    
    Returns:
        Dictionary with results (metrics, checkpoint path, etc.)
    """
    # Create experiment name
    experiment_name = f"layer_{layer_idx}_pooling_{pooling_strategy}"
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"Training probe: Layer {layer_idx}, Pooling: {pooling_strategy}")
    print(f"{'='*60}")
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize model
    model = BaseModule(
        model_name=args.model_name,
        num_labels=args.num_labels,
        mode="probe",
        probe_layer=layer_idx,
        pooling_strategy=pooling_strategy,
        probe_lr=args.probe_lr,
    )
    
    # Initialize data module
    datamodule = NOTDataModule.from_args(args)
    datamodule.setup("fit")
    
    # Setup logging
    logger = TensorBoardLogger(
        save_dir=base_output_dir,
        name="",
        version=experiment_name,
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(experiment_dir, "checkpoints"),
        filename="best-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        verbose=False,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=args.patience if hasattr(args, 'patience') else 5,
        verbose=False,
    )
    callbacks.append(early_stop_callback)
    
    # Initialize trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=1,  # Each parallel worker uses 1 device
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.5,
        enable_progress_bar=True,
        enable_model_summary=False,
        deterministic=False,
    )
    
    # Train
    trainer.fit(model=model, datamodule=datamodule)
    
    # Test - check if test set has labels
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    
    # Check if test dataset has valid labels (not -1)
    has_test_labels = getattr(test_dataset, 'has_labels', True)
    
    if not has_test_labels:
        # Test set has no ground truth (e.g., SST-2 test set with -1 labels)
        print(f"  ⚠ Test set has no ground truth labels (-1), using validation metrics")
        
        # Run test set (will generate predictions but no metrics)
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
        
        # Get validation metrics to use as test metrics
        val_results = trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
        val_result = val_results[0] if val_results else {}
        
        result = {
            "layer_idx": layer_idx,
            "pooling_strategy": pooling_strategy,
            "checkpoint_path": checkpoint_callback.best_model_path,
            "test_accuracy": val_result.get("val_acc", 0.0),
            "test_auroc": val_result.get("val_auroc", 0.0),
            "test_loss": val_result.get("val_loss", float('inf')),
            "val_accuracy": val_result.get("val_acc", 0.0),
            "val_auroc": val_result.get("val_auroc", 0.0),
            "experiment_dir": experiment_dir,
            "note": "test_set_has_no_labels_using_val_metrics",
        }
    else:
        # Test set has labels - use it normally
        test_results = trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
        test_result = test_results[0] if test_results else {}
        result = {
            "layer_idx": layer_idx,
            "pooling_strategy": pooling_strategy,
            "checkpoint_path": checkpoint_callback.best_model_path,
            "test_accuracy": test_result.get("test_acc", 0.0),
            "test_auroc": test_result.get("test_auroc", 0.0),
            "test_loss": test_result.get("test_loss", float('inf')),
            "val_accuracy": test_result.get("val_acc", 0.0),
            "val_auroc": test_result.get("val_auroc", 0.0),
            "experiment_dir": experiment_dir,
        }
    
    print(f"✓ Layer {layer_idx}, {pooling_strategy}: "
          f"Test Acc={result['test_accuracy']:.4f}, "
          f"Test AUROC={result['test_auroc']:.4f}")
    
    return result


def train_probe_wrapper(args_tuple):
    """Wrapper for parallel execution."""
    layer_idx, pooling_strategy, args, base_output_dir = args_tuple
    try:
        return train_probe_for_layer(layer_idx, pooling_strategy, args, base_output_dir)
    except Exception as e:
        print(f"✗ Error training layer {layer_idx}, {pooling_strategy}: {e}")
        return {
            "layer_idx": layer_idx,
            "pooling_strategy": pooling_strategy,
            "error": str(e),
            "test_accuracy": 0.0,
            "test_auroc": 0.0,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Parallelized automated layer search: train probes on multiple layers in parallel"
    )
    
    # Get base parser arguments
    base_parser = get_parser()
    args, remaining = base_parser.parse_known_args()
    
    # Add layer search specific arguments
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layers to search (comma-separated or 'all'). Default: all",
    )
    parser.add_argument(
        "--pooling_strategies",
        type=str,
        default="all",
        help="Pooling strategies to test (comma-separated or 'all'). Default: all",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/layer_search",
        help="Output directory for results",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous checkpoint (skip completed experiments). Default: True (always resume)",
    )
    parser.add_argument(
        "--no_resume",
        action="store_true",
        help="Start fresh, ignoring any existing results",
    )
    parser.add_argument(
        "--parallel_workers",
        type=int,
        default=3,
        help="Number of parallel workers (default: 3 for 3 pooling strategies)",
    )
    parser.add_argument(
        "--parallel_mode",
        type=str,
        default="pooling",
        choices=["pooling", "layer", "all"],
        help="Parallelization mode: 'pooling' (parallel pooling per layer), 'layer' (parallel layers), 'all' (both)",
    )
    
    # Parse layer search specific args
    layer_search_args = parser.parse_args(remaining)
    
    # Merge arguments
    for key, value in vars(layer_search_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
    # Determine layers to search
    if args.layers == "all":
        # Get number of layers from model
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
    print(f"Parallel workers: {args.parallel_workers}")
    print(f"Parallel mode: {args.parallel_mode}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load existing results if any (unless --no_resume is specified)
    all_results = []
    completed_experiments = set()
    results_json_path = os.path.join(args.output_dir, "results_summary.json")
    
    # By default, always resume unless --no_resume is specified
    should_resume = not getattr(args, 'no_resume', False)
    
    if should_resume and os.path.exists(results_json_path):
        try:
            with open(results_json_path, 'r') as f:
                all_results = json.load(f)
            
            # Track completed experiments
            for result in all_results:
                if 'error' not in result and result.get('test_auroc', 0) > 0:
                    layer_idx = result['layer_idx']
                    pooling = result['pooling_strategy']
                    completed_experiments.add((layer_idx, pooling))
            
            print(f"✓ Loaded {len(all_results)} existing results")
            print(f"✓ {len(completed_experiments)} experiments already completed")
        except Exception as e:
            print(f"⚠ Could not load existing results: {e}")
            all_results = []
    elif not should_resume:
        print("⚠ Starting fresh (--no_resume specified)")
    
    # Search across layers and pooling strategies
    total_experiments = len(layers_to_search) * len(pooling_strategies)
    experiments_to_run = [
        (layer, pooling)
        for layer in layers_to_search
        for pooling in pooling_strategies
        if (layer, pooling) not in completed_experiments
    ]
    
    remaining = len(experiments_to_run)
    
    print(f"\n{'='*60}")
    print(f"Starting layer search: {total_experiments} total experiments")
    print(f"Already completed: {len(completed_experiments)}")
    print(f"Remaining: {remaining}")
    print(f"{'='*60}\n")
    
    if remaining == 0:
        print("✓ All experiments already completed!")
    else:
        # Thread-safe lock for saving results
        save_lock = threading.Lock()
        
        if args.parallel_mode == "pooling":
            # Parallelize pooling strategies for each layer
            print("Mode: Parallelizing pooling strategies per layer")
            
            for layer_idx in layers_to_search:
                # Get pooling strategies to run for this layer
                layer_experiments = [
                    (layer_idx, pooling, args, args.output_dir)
                    for pooling in pooling_strategies
                    if (layer_idx, pooling) in experiments_to_run
                ]
                
                if not layer_experiments:
                    continue
                
                print(f"\n{'='*60}")
                print(f"Layer {layer_idx}: Running {len(layer_experiments)} pooling strategies in parallel")
                print(f"{'='*60}")
                
                # Run pooling strategies in parallel for this layer
                with ProcessPoolExecutor(max_workers=min(args.parallel_workers, len(layer_experiments))) as executor:
                    futures = {executor.submit(train_probe_wrapper, exp): exp for exp in layer_experiments}
                    
                    for future in as_completed(futures):
                        result = future.result()
                        
                        with save_lock:
                            all_results.append(result)
                            # Save after each completed experiment
                            with open(results_json_path, "w") as f:
                                json.dump(all_results, f, indent=2)
        
        elif args.parallel_mode == "layer":
            # Parallelize layers (run multiple layers in parallel)
            print("Mode: Parallelizing layers")
            
            # Prepare all experiments
            all_experiments = [
                (layer, pooling, args, args.output_dir)
                for layer, pooling in experiments_to_run
            ]
            
            # Run in parallel
            with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = {executor.submit(train_probe_wrapper, exp): exp for exp in all_experiments}
                
                completed_count = len(completed_experiments)
                for future in as_completed(futures):
                    result = future.result()
                    completed_count += 1
                    
                    print(f"\n[{completed_count}/{total_experiments}] Completed")
                    
                    with save_lock:
                        all_results.append(result)
                        # Save after each completed experiment
                        with open(results_json_path, "w") as f:
                            json.dump(all_results, f, indent=2)
        
        else:  # "all" mode
            # Maximum parallelization - run everything in parallel
            print("Mode: Maximum parallelization (all experiments in parallel)")
            
            all_experiments = [
                (layer, pooling, args, args.output_dir)
                for layer, pooling in experiments_to_run
            ]
            
            with ProcessPoolExecutor(max_workers=args.parallel_workers) as executor:
                futures = {executor.submit(train_probe_wrapper, exp): exp for exp in all_experiments}
                
                completed_count = len(completed_experiments)
                for future in as_completed(futures):
                    result = future.result()
                    completed_count += 1
                    
                    print(f"\n[{completed_count}/{total_experiments}] Completed")
                    
                    with save_lock:
                        all_results.append(result)
                        with open(results_json_path, "w") as f:
                            json.dump(all_results, f, indent=2)
    
    # Save final results
    print(f"\n{'='*60}")
    print("Saving final results...")
    print(f"{'='*60}\n")
    
    # Save as JSON (one final time to ensure it's complete)
    with open(results_json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✓ Results saved to {results_json_path}")
    
    # Save as CSV for easy analysis
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
        
        # Best results by metric
        print("Best Test Accuracy:")
        best_acc = df_summary.loc[df_summary['test_accuracy'].idxmax()]
        print(f"  Layer {best_acc['layer_idx']}, {best_acc['pooling_strategy']}: "
              f"{best_acc['test_accuracy']:.4f}")
        
        print("\nBest Test AUROC:")
        best_auroc = df_summary.loc[df_summary['test_auroc'].idxmax()]
        print(f"  Layer {best_auroc['layer_idx']}, {best_auroc['pooling_strategy']}: "
              f"{best_auroc['test_auroc']:.4f}")
        
        # Average by layer
        print("\nAverage Performance by Layer:")
        layer_avg = df_summary.groupby('layer_idx').agg({
            'test_accuracy': 'mean',
            'test_auroc': 'mean'
        }).round(4)
        print(layer_avg)
        
        # Average by pooling strategy
        print("\nAverage Performance by Pooling Strategy:")
        pooling_avg = df_summary.groupby('pooling_strategy').agg({
            'test_accuracy': 'mean',
            'test_auroc': 'mean'
        }).round(4)
        print(pooling_avg)
    
    print(f"\n{'='*60}")
    print("Layer search complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

