"""
Threaded automated layer search script.
Trains all 3 pooling strategies simultaneously using threads (shared memory).
More efficient than multiprocessing - no process spawning, shared data loading.
"""

import os
import sys
from pathlib import Path
import json
import argparse
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from transformers import AutoModel
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    base_output_dir: str,
    datamodule: NOTDataModule
):
    """
    Train a probe for a specific layer and pooling strategy.
    Shares datamodule across threads (efficient memory usage).
    """
    # Create experiment name
    experiment_name = f"layer_{layer_idx}_pooling_{pooling_strategy}"
    experiment_dir = os.path.join(base_output_dir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"[Thread {threading.current_thread().name}] Training probe: Layer {layer_idx}, Pooling: {pooling_strategy}")
    print(f"{'='*60}")
    sys.stdout.flush()  # Ensure output is visible
    
    # Set seed for reproducibility
    set_seed(args.seed + hash(pooling_strategy) % 1000)  # Different seed per pooling
    
    # Initialize model
    model = BaseModule(
        model_name=args.model_name,
        num_labels=args.num_labels,
        mode="probe",
        probe_layer=layer_idx,
        pooling_strategy=pooling_strategy,
        probe_lr=args.probe_lr,
    )
    
    # Datamodule is already initialized and shared
    # Each thread uses the same data but different model
    
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
    # Note: With threading, we need to be careful about GPU usage
    # Each thread will use the same GPU but PyTorch handles this
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        devices=1,  # All threads share the same device
        precision=args.precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=50,
        val_check_interval=0.5,
        enable_progress_bar=True,  # Keep enabled - Lightning handles threading
        enable_model_summary=False,
        deterministic=False,
    )
    
    # Train
    trainer.fit(model=model, datamodule=datamodule)
    
    # Test
    datamodule.setup("test")
    test_dataset = datamodule.test_dataset
    has_test_labels = getattr(test_dataset, 'has_labels', True)
    
    if not has_test_labels:
        print(f"  [{pooling_strategy}] Test set has no ground truth, using validation metrics")
        sys.stdout.flush()
        
        trainer.test(model=model, datamodule=datamodule, ckpt_path="best", verbose=False)
        val_results = trainer.validate(model=model, datamodule=datamodule, ckpt_path="best", verbose=False)
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
        test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path="best", verbose=False)
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
    
    print(f"✓ [{pooling_strategy}] Layer {layer_idx}: "
          f"Test Acc={result['test_accuracy']:.4f}, "
          f"Test AUROC={result['test_auroc']:.4f}")
    sys.stdout.flush()
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Threaded layer search: train all pooling strategies simultaneously (shared memory)"
    )
    
    # Get base parser arguments
    base_parser = get_parser()
    args, remaining = base_parser.parse_known_args()
    
    # Add layer search specific arguments
    parser.add_argument("--layers", type=str, default="all", help="Layers to search")
    parser.add_argument("--pooling_strategies", type=str, default="all", help="Pooling strategies")
    parser.add_argument("--output_dir", type=str, default="experiments/layer_search", help="Output directory")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--no_resume", action="store_true", help="Start fresh")
    
    layer_search_args = parser.parse_args(remaining)
    for key, value in vars(layer_search_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    
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
    print(f"Mode: Threaded (shared memory, {len(pooling_strategies)} threads per layer)")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load existing results
    all_results = []
    completed_experiments = set()
    results_json_path = os.path.join(args.output_dir, "results_summary.json")
    
    should_resume = not getattr(args, 'no_resume', False)
    
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
            all_results = []
    
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
    else:
        # Initialize datamodule ONCE - shared across all threads
        print("Initializing shared datamodule...")
        datamodule = NOTDataModule.from_args(args)
        datamodule.setup("fit")
        print("✓ Datamodule initialized and loaded into memory\n")
        
        # Thread-safe lock for saving results
        save_lock = threading.Lock()
        
        # Group experiments by layer
        experiments_by_layer = {}
        for layer, pooling in experiments_to_run:
            if layer not in experiments_by_layer:
                experiments_by_layer[layer] = []
            experiments_by_layer[layer].append(pooling)
        
        current_experiment = len(completed_experiments)
        
        # Process each layer with threading
        for layer_idx in sorted(experiments_by_layer.keys()):
            pooling_strategies_for_layer = experiments_by_layer[layer_idx]
            
            print(f"\n{'='*60}")
            print(f"Layer {layer_idx}: Training {len(pooling_strategies_for_layer)} pooling strategies in parallel (threads)")
            print(f"{'='*60}")
            sys.stdout.flush()
            
            # Run all pooling strategies for this layer in parallel using threads
            with ThreadPoolExecutor(max_workers=len(pooling_strategies_for_layer)) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(
                        train_probe_for_layer,
                        layer_idx,
                        pooling,
                        args,
                        args.output_dir,
                        datamodule  # SHARED datamodule across threads
                    ): pooling
                    for pooling in pooling_strategies_for_layer
                }
                
                # Collect results as they complete
                for future in as_completed(futures):
                    pooling = futures[future]
                    current_experiment += 1
                    
                    try:
                        result = future.result()
                        
                        with save_lock:
                            all_results.append(result)
                            # Save after each completed experiment
                            with open(results_json_path, "w") as f:
                                json.dump(all_results, f, indent=2)
                        
                        print(f"[{current_experiment}/{total_experiments}] ✓ Completed: Layer {layer_idx}, {pooling}")
                        sys.stdout.flush()
                        
                    except Exception as e:
                        print(f"✗ Error training layer {layer_idx}, {pooling}: {e}")
                        sys.stdout.flush()
                        
                        result = {
                            "layer_idx": layer_idx,
                            "pooling_strategy": pooling,
                            "error": str(e),
                            "test_accuracy": 0.0,
                            "test_auroc": 0.0,
                        }
                        
                        with save_lock:
                            all_results.append(result)
                            with open(results_json_path, "w") as f:
                                json.dump(all_results, f, indent=2)
            
            print(f"✓ Layer {layer_idx} complete - all pooling strategies done")
            sys.stdout.flush()
    
    # Save final results
    print(f"\n{'='*60}")
    print("Saving final results...")
    print(f"{'='*60}\n")
    
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
        
        print("\nAverage Performance by Layer:")
        layer_avg = df_summary.groupby('layer_idx').agg({'test_accuracy': 'mean', 'test_auroc': 'mean'}).round(4)
        print(layer_avg)
        
        print("\nAverage Performance by Pooling Strategy:")
        pooling_avg = df_summary.groupby('pooling_strategy').agg({'test_accuracy': 'mean', 'test_auroc': 'mean'}).round(4)
        print(pooling_avg)
    
    print(f"\n{'='*60}")
    print("Layer search complete!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

