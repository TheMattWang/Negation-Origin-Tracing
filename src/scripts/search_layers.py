"""
Automated layer search script.
Trains probes on all layers with different pooling strategies and saves results.
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
        devices=args.devices,
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
        # Still run test set for inference, but use validation set for metrics
        print(f"  ⚠ Test set has no ground truth labels (-1), running inference anyway")
        print(f"  → Using validation set metrics as test metrics")
        
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
            "test_accuracy": val_result.get("val_acc", 0.0),  # Use val metrics as test
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
        # Extract metrics
        test_result = test_results[0] if test_results else {}
        result = {
            "layer_idx": layer_idx,
            "pooling_strategy": pooling_strategy,
            "checkpoint_path": checkpoint_callback.best_model_path,
            "test_accuracy": test_result.get("test_acc", 0.0),
            "test_auroc": test_result.get("test_auroc", 0.0),
            "test_loss": test_result.get("test_loss", float('inf')),
            "val_accuracy": test_result.get("val_acc", 0.0),  # May not be in test_results
            "val_auroc": test_result.get("val_auroc", 0.0),
            "experiment_dir": experiment_dir,
        }
    
    print(f"✓ Layer {layer_idx}, {pooling_strategy}: "
          f"Test Acc={result['test_accuracy']:.4f}, "
          f"Test AUROC={result['test_auroc']:.4f}")
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Automated layer search: train probes on all layers"
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
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Store all results
    all_results = []
    
    # Search across layers and pooling strategies
    total_experiments = len(layers_to_search) * len(pooling_strategies)
    current_experiment = 0
    
    print(f"\n{'='*60}")
    print(f"Starting layer search: {total_experiments} experiments")
    print(f"{'='*60}\n")
    
    for layer_idx in layers_to_search:
        for pooling_strategy in pooling_strategies:
            current_experiment += 1
            print(f"\n[{current_experiment}/{total_experiments}] ", end="")
            
            try:
                result = train_probe_for_layer(
                    layer_idx=layer_idx,
                    pooling_strategy=pooling_strategy,
                    args=args,
                    base_output_dir=args.output_dir,
                )
                all_results.append(result)
            except Exception as e:
                print(f"✗ Error training layer {layer_idx}, {pooling_strategy}: {e}")
                result = {
                    "layer_idx": layer_idx,
                    "pooling_strategy": pooling_strategy,
                    "error": str(e),
                    "test_accuracy": 0.0,
                    "test_auroc": 0.0,
                }
                all_results.append(result)
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}\n")
    
    # Save as JSON
    results_json_path = os.path.join(args.output_dir, "results_summary.json")
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

