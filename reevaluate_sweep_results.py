#!/usr/bin/env python
"""
Re-evaluate existing sweep checkpoints on validation set.

This script:
1. Reads existing results_*.json files from all 3 sweeps
2. For each checkpoint, runs validation evaluation
3. Updates test_acc and test_auroc with validation metrics
4. Creates backup of original results before updating

Usage (local):
    python reevaluate_sweep_results.py

Usage (Google Drive):
    python reevaluate_sweep_results.py --drive_path /content/drive/MyDrive/NOT_results
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

import torch
torch.set_num_threads(1)

import lightning as L
from lightning.pytorch.callbacks import TQDMProgressBar
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models import BaseModule
from src.datasets.dataset import SentimentDataset


def load_checkpoint_and_evaluate(
    checkpoint_path: str,
    layer: int,
    pooling: str,
    data_dir: str,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    device: str = "auto"
) -> Optional[Dict]:
    """
    Load a checkpoint and evaluate it on validation set.
    
    Args:
        checkpoint_path: Path to checkpoint file
        layer: Layer index
        pooling: Pooling strategy ('cls', 'mean', or 'token')
        data_dir: Root data directory
        model_name: Model name
        batch_size: Batch size for evaluation
        device: Device to use
    
    Returns:
        Dictionary with validation metrics, or None if evaluation failed
    """
    if not os.path.exists(checkpoint_path):
        print(f"  ⚠ Checkpoint not found: {checkpoint_path}")
        return None
    
    try:
        # Load model from checkpoint
        model = BaseModule.load_from_checkpoint(
            checkpoint_path,
            strict=False
        )
        
        # Ensure model is in eval mode
        model.eval()
        
        # Create tokenizer and validation dataset
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        val_ds = SentimentDataset(
            os.path.join(data_dir, "validation", "sst.parquet"),
            tokenizer,
            max_length=128
        )
        val_dl = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Create trainer for evaluation
        progress_cb = TQDMProgressBar(refresh_rate=10)
        trainer = L.Trainer(
            accelerator='auto',
            devices=1,
            callbacks=[progress_cb],
            enable_progress_bar=True,
            enable_model_summary=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        
        # Run validation
        val_results = trainer.validate(
            model=model,
            dataloaders=val_dl,
            verbose=False
        )
        
        if val_results and len(val_results) > 0:
            r = val_results[0]
            return {
                'val_acc': r.get('val_acc', 0),
                'val_auroc': r.get('val_auroc', 0),
            }
        else:
            print(f"  ⚠ No validation results returned")
            return None
            
    except Exception as e:
        print(f"  ✗ Error evaluating checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return None


def update_results_file(
    results_file: str,
    data_dir: str,
    model_name: str = "distilbert-base-uncased",
    batch_size: int = 16,
    create_backup: bool = True
) -> Dict:
    """
    Update a results JSON file by re-evaluating all checkpoints.
    
    Args:
        results_file: Path to results JSON file
        data_dir: Root data directory
        model_name: Model name
        batch_size: Batch size for evaluation
        create_backup: Whether to create backup before updating
    
    Returns:
        Dictionary with update statistics
    """
    if not os.path.exists(results_file):
        print(f"  ⚠ Results file not found: {results_file}")
        return {'processed': 0, 'updated': 0, 'failed': 0, 'skipped': 0}
    
    # Load existing results
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    if not results:
        print(f"  ⚠ No results found in {results_file}")
        return {'processed': 0, 'updated': 0, 'failed': 0, 'skipped': 0}
    
    # Create backup
    if create_backup:
        backup_file = results_file + '.backup'
        shutil.copy2(results_file, backup_file)
        print(f"  ✓ Backup created: {backup_file}")
    
    # Determine pooling strategy from filename
    pooling = None
    if 'cls' in results_file:
        pooling = 'cls'
    elif 'mean' in results_file:
        pooling = 'mean'
    elif 'token' in results_file:
        pooling = 'token'
    
    if not pooling:
        print(f"  ⚠ Could not determine pooling strategy from filename")
        return {'processed': 0, 'updated': 0, 'failed': 0, 'skipped': 0}
    
    stats = {'processed': 0, 'updated': 0, 'failed': 0, 'skipped': 0}
    
    # Process each result
    for i, result in enumerate(results):
        stats['processed'] += 1
        
        layer = result.get('layer')
        checkpoint = result.get('checkpoint')
        
        if not checkpoint:
            print(f"  ⚠ Result {i+1} (layer {layer}): No checkpoint path")
            stats['skipped'] += 1
            continue
        
        print(f"  [{i+1}/{len(results)}] Layer {layer}, {pooling} pooling...")
        
        # Evaluate checkpoint
        metrics = load_checkpoint_and_evaluate(
            checkpoint_path=checkpoint,
            layer=layer,
            pooling=pooling,
            data_dir=data_dir,
            model_name=model_name,
            batch_size=batch_size
        )
        
        if metrics:
            # Update result with validation metrics (stored as test_* for backward compatibility)
            old_acc = result.get('test_acc', 0)
            old_auroc = result.get('test_auroc', 0)
            
            result['test_acc'] = metrics['val_acc']
            result['test_auroc'] = metrics['val_auroc']
            
            # Also store original validation metrics for reference
            result['val_acc'] = metrics['val_acc']
            result['val_auroc'] = metrics['val_auroc']
            
            print(f"    ✓ Updated: acc {old_acc:.4f} → {metrics['val_acc']:.4f}, "
                  f"auroc {old_auroc:.4f} → {metrics['val_auroc']:.4f}")
            stats['updated'] += 1
        else:
            print(f"    ✗ Failed to evaluate")
            stats['failed'] += 1
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"  ✓ Results saved to {results_file}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Re-evaluate sweep checkpoints on validation set"
    )
    parser.add_argument(
        "--drive_path",
        type=str,
        default=None,
        help="Google Drive path (e.g., /content/drive/MyDrive/NOT_results)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Root data directory (default: data/raw)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="distilbert-base-uncased",
        help="Model name (default: distilbert-base-uncased)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for evaluation (default: 16)"
    )
    parser.add_argument(
        "--no_backup",
        action="store_true",
        help="Don't create backup files"
    )
    
    args = parser.parse_args()
    
    # Determine base path
    if args.drive_path:
        base_path = args.drive_path
    else:
        base_path = "experiments"
    
    print("=" * 70)
    print("RE-EVALUATE SWEEP RESULTS ON VALIDATION SET")
    print("=" * 70)
    print(f"\nBase path: {base_path}")
    print(f"Data directory: {args.data_dir}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}\n")
    
    # Define sweep result files
    sweep_files = {
        'CLS': os.path.join(base_path, 'sweep_cls', 'results_cls.json'),
        'MEAN': os.path.join(base_path, 'sweep_mean', 'results_mean.json'),
        'TOKEN': os.path.join(base_path, 'sweep_token', 'results_token.json'),
    }
    
    overall_stats = {'processed': 0, 'updated': 0, 'failed': 0, 'skipped': 0}
    
    # Process each sweep
    for name, results_file in sweep_files.items():
        print("-" * 70)
        print(f"Processing {name} sweep...")
        print("-" * 70)
        
        if not os.path.exists(results_file):
            print(f"  ⚠ Results file not found: {results_file}")
            print(f"  → Skipping {name} sweep\n")
            continue
        
        stats = update_results_file(
            results_file=results_file,
            data_dir=args.data_dir,
            model_name=args.model_name,
            batch_size=args.batch_size,
            create_backup=not args.no_backup
        )
        
        # Accumulate stats
        for key in overall_stats:
            overall_stats[key] += stats[key]
        
        print(f"\n{name} Summary:")
        print(f"  Processed: {stats['processed']}")
        print(f"  Updated: {stats['updated']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Skipped: {stats['skipped']}\n")
    
    # Final summary
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"Total processed: {overall_stats['processed']}")
    print(f"Total updated: {overall_stats['updated']}")
    print(f"Total failed: {overall_stats['failed']}")
    print(f"Total skipped: {overall_stats['skipped']}")
    print("\n✓ Re-evaluation complete!")
    print("\nNote: Validation metrics are stored as test_acc/test_auroc for")
    print("      backward compatibility with existing analysis scripts.")


if __name__ == "__main__":
    main()

