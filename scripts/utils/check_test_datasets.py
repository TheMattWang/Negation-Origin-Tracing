#!/usr/bin/env python
"""
Standalone data analysis script to check test dataset labels.

This script analyzes which test datasets are available and whether they have
valid labels for computing metrics like AUROC. If SST-2 is unavailable (has -1 labels),
it will download the CSD Negation dataset and recommend using it for all metrics.

Usage:
    python check_test_datasets.py
    python check_test_datasets.py --data_dir data/raw
    python check_test_datasets.py --download  # Auto-download CSD if SST-2 unavailable
"""

import os
import sys
import argparse
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def analyze_dataset(data_path: str, dataset_name: str) -> dict:
    """
    Analyze a dataset file to check labels.
    
    Args:
        data_path: Path to parquet file
        dataset_name: Name for reporting
    
    Returns:
        Dictionary with analysis results
    """
    result = {
        "name": dataset_name,
        "path": data_path,
        "exists": False,
        "has_labels": False,
        "label_info": None,
        "usable_for_auroc": False,
        "error": None,
    }
    
    if not os.path.exists(data_path):
        result["error"] = "File not found"
        return result
    
    result["exists"] = True
    
    try:
        # Load the parquet file to check labels directly
        df = pd.read_parquet(data_path)
        
        # Check if 'label' column exists (CSD uses 'label', SST-2 might use different name)
        if 'label' not in df.columns:
            # Try alternative column names
            if 'sentiment' in df.columns:
                labels = df['sentiment'].tolist()
            elif 'labels' in df.columns:
                labels = df['labels'].tolist()
            else:
                result["error"] = f"No 'label' column found. Available columns: {df.columns.tolist()}"
                return result
        else:
            labels = df['label'].tolist()
        unique_labels = sorted(set(labels))
        
        # Count label distribution
        label_counts = pd.Series(labels).value_counts().to_dict()
        
        result["label_info"] = {
            "unique_labels": unique_labels,
            "label_counts": label_counts,
            "total_samples": len(labels),
        }
        
        # Check if all labels are -1 (no ground truth)
        if unique_labels == [-1] or (-1 in unique_labels and len(unique_labels) == 1):
            result["has_labels"] = False
            result["usable_for_auroc"] = False
        elif set(unique_labels) == {0, 1} or set(unique_labels) == {0, 1, -1}:
            # Has valid binary labels
            result["has_labels"] = True
            result["usable_for_auroc"] = True
        else:
            # Unexpected label values
            result["has_labels"] = True  # Has labels, but might not be binary
            result["usable_for_auroc"] = False
            result["error"] = f"Unexpected label values: {unique_labels}"
        
    except Exception as e:
        result["error"] = str(e)
    
    return result


def download_hans_dataset(data_dir: str) -> bool:
    """
    Download HANS (Heuristic Analysis for NLI Systems) dataset.
    
    HANS tests for spurious correlations and heuristic shortcuts in NLI models.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        True if download successful, False otherwise
    """
    test_path = os.path.join(data_dir, "test", "hans.parquet")
    
    if os.path.exists(test_path):
        print(f"  HANS dataset already exists at {test_path}")
        return True
    
    print(f"  Downloading HANS dataset...")
    
    try:
        from datasets import load_dataset
        
        # Create directories if needed
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
        
        # Load HANS dataset
        print("    Loading HANS dataset from Hugging Face...")
        hans = load_dataset("hans")
        print(f"    Dataset structure: {hans}")
        
        # Download test set
        print("    Downloading test split...")
        test = hans.get("validation", hans.get("test", None))  # HANS uses 'validation' as test
        if test is None:
            # Try to get any available split
            splits = list(hans.keys())
            test = hans[splits[0]] if splits else None
        
        if test is None:
            raise ValueError("Could not find test split in HANS dataset")
        
        test.to_parquet(test_path)
        print(f"      ✓ Test set saved: {len(test)} examples")
        
        # Show sample structure
        if len(test) > 0:
            sample = test[0]
            print(f"\n    Sample structure:")
            print(f"      Fields: {list(sample.keys())}")
            if 'premise' in sample and 'hypothesis' in sample:
                print(f"      Example: {sample['premise'][:40]}... -> {sample['hypothesis'][:40]}...")
            if 'label' in sample:
                print(f"      Label: {sample['label']}")
        
        print(f"\n  ✓ HANS dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download HANS dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_csd_dataset(data_dir: str) -> bool:
    """
    Download CSD Negation dataset if it doesn't exist.
    
    CSD (Contrastive Sentiment Negation) from Hugging Face:
    - Source: ceval/contrastive-sentiment-negation
    - Structure: train (8000), validation (1000), test (1000)
    - Fields: text, label, original_text, negation_type
    
    Args:
        data_dir: Root data directory
    
    Returns:
        True if download successful, False otherwise
    """
    test_path = os.path.join(data_dir, "test", "negation.parquet")
    
    if os.path.exists(test_path):
        print(f"  CSD dataset already exists at {test_path}")
        return True
    
    print(f"  Downloading CSD Negation dataset from Hugging Face...")
    print(f"    Source: ceval/contrastive-sentiment-negation")
    
    try:
        from datasets import load_dataset
        
        # Create directories if needed
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
        
        # Load the full dataset
        print("    Loading dataset...")
        csd_neg = load_dataset("ceval/contrastive-sentiment-negation")
        print(f"    Dataset structure: {csd_neg}")
        
        # Download test set (1000 examples)
        print("    Downloading test split (1000 examples)...")
        test = csd_neg["test"]
        test.to_parquet(test_path)
        print(f"      ✓ Test set saved: {len(test)} examples")
        
        # Also download train/val for completeness
        train_path = os.path.join(data_dir, "train", "negation.parquet")
        val_path = os.path.join(data_dir, "validation", "negation.parquet")
        
        if not os.path.exists(train_path):
            print("    Downloading train split (8000 examples)...")
            train = csd_neg["train"]
            train.to_parquet(train_path)
            print(f"      ✓ Train set saved: {len(train)} examples")
        
        if not os.path.exists(val_path):
            print("    Downloading validation split (1000 examples)...")
            validation = csd_neg["validation"]
            validation.to_parquet(val_path)
            print(f"      ✓ Validation set saved: {len(validation)} examples")
        
        # Show sample structure
        if len(test) > 0:
            sample = test[0]
            print(f"\n    Sample structure:")
            print(f"      Fields: {list(sample.keys())}")
            if 'text' in sample:
                print(f"      Example text: {sample['text'][:60]}...")
            if 'label' in sample:
                print(f"      Label: {sample['label']}")
        
        print(f"\n  ✓ CSD dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download CSD dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def download_anli_dataset(data_dir: str) -> bool:
    """
    Download ANLI (Adversarial NLI) dataset.
    
    ANLI is an adversarial benchmark for NLI with challenging examples.
    
    Args:
        data_dir: Root data directory
    
    Returns:
        True if download successful, False otherwise
    """
    test_path = os.path.join(data_dir, "test", "anli.parquet")
    
    if os.path.exists(test_path):
        print(f"  ANLI dataset already exists at {test_path}")
        return True
    
    print(f"  Downloading ANLI dataset...")
    
    try:
        from datasets import load_dataset
        
        # Create directories if needed
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)
        
        # Load ANLI dataset (has multiple rounds)
        print("    Loading ANLI dataset from Hugging Face...")
        anli = load_dataset("anli")
        print(f"    Dataset structure: {anli}")
        
        # ANLI has rounds (r1, r2, r3) - combine test splits or use r1
        if "test_r1" in anli:
            test = anli["test_r1"]
        elif "test" in anli:
            test = anli["test"]
        else:
            # Try to get any available test split
            test_splits = [k for k in anli.keys() if "test" in k.lower()]
            if test_splits:
                test = anli[test_splits[0]]
            else:
                raise ValueError("Could not find test split in ANLI dataset")
        
        test.to_parquet(test_path)
        print(f"      ✓ Test set saved: {len(test)} examples")
        
        # Show sample structure
        if len(test) > 0:
            sample = test[0]
            print(f"\n    Sample structure:")
            print(f"      Fields: {list(sample.keys())}")
            if 'premise' in sample and 'hypothesis' in sample:
                print(f"      Example: {sample['premise'][:40]}... -> {sample['hypothesis'][:40]}...")
            if 'label' in sample:
                print(f"      Label: {sample['label']}")
        
        print(f"\n  ✓ ANLI dataset downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to download ANLI dataset: {e}")
        import traceback
        traceback.print_exc()
        return False


def print_analysis_report(results: list, data_dir: str, auto_download: bool = False):
    """Print a formatted analysis report."""
    print("\n" + "=" * 70)
    print("TEST DATASET VALIDATION")
    print("=" * 70)
    print()
    
    for result in results:
        print(f"{result['name']}:")
        print(f"  Path: {result['path']}")
        print(f"  Exists: {'Yes' if result['exists'] else 'No'}")
        
        if not result['exists']:
            print(f"  Status: Not found")
            print()
            continue
        
        if result['error']:
            print(f"  Error: {result['error']}")
            print()
            continue
        
        if result['label_info']:
            label_info = result['label_info']
            print(f"  Total samples: {label_info['total_samples']}")
            print(f"  Unique labels: {label_info['unique_labels']}")
            print(f"  Label distribution: {label_info['label_counts']}")
        
        print(f"  Has valid labels: {'Yes' if result['has_labels'] else 'No'}")
        
        if not result['has_labels']:
            print(f"  Reason: All labels are -1 (no ground truth)")
        
        print(f"  Usable for AUROC: {'Yes' if result['usable_for_auroc'] else 'No'}")
        print()
    
    # Summary and recommendation
    print("-" * 70)
    print("SUMMARY")
    print("-" * 70)
    
    sst_result = next((r for r in results if 'SST-2' in r['name']), None)
    csd_result = next((r for r in results if 'CSD' in r['name']), None)
    hans_result = next((r for r in results if 'HANS' in r['name']), None)
    anli_result = next((r for r in results if 'ANLI' in r['name']), None)
    
    usable_datasets = [r for r in results if r['usable_for_auroc']]
    unusable_datasets = [r for r in results if r['exists'] and not r['usable_for_auroc']]
    
    # Check if SST-2 is unavailable and download alternatives
    sst_unavailable = sst_result and (not sst_result['exists'] or not sst_result['usable_for_auroc'])
    csd_missing = csd_result and not csd_result['exists']
    hans_missing = hans_result and not hans_result['exists']
    anli_missing = anli_result and not anli_result['exists']
    
    if sst_unavailable and auto_download:
        # Try CSD first (sentiment dataset)
        if csd_missing:
            print("\n📥 SST-2 test set is unavailable. Downloading CSD Negation dataset...")
            if download_csd_dataset(data_dir):
                # Re-analyze CSD after download
                csd_result = analyze_dataset(csd_result['path'], csd_result['name'])
                results = [r for r in results if 'CSD' not in r['name']] + [csd_result]
                usable_datasets = [r for r in results if r['usable_for_auroc']]
                print("\n  Re-analyzing CSD dataset...")
                print(f"  {csd_result['name']}:")
                if csd_result['label_info']:
                    print(f"    Total samples: {csd_result['label_info']['total_samples']}")
                    print(f"    Unique labels: {csd_result['label_info']['unique_labels']}")
                    print(f"    Label distribution: {csd_result['label_info']['label_counts']}")
                print(f"    Usable for AUROC: {'Yes' if csd_result['usable_for_auroc'] else 'No'}")
        
        # Also try HANS
        if hans_missing:
            print("\n📥 Downloading HANS dataset for transfer evaluation...")
            if download_hans_dataset(data_dir):
                # Re-analyze HANS after download
                hans_result = analyze_dataset(hans_result['path'], hans_result['name'])
                results = [r for r in results if 'HANS' not in r['name']] + [hans_result]
                usable_datasets = [r for r in results if r['usable_for_auroc']]
                print("\n  Re-analyzing HANS dataset...")
                print(f"  {hans_result['name']}:")
                if hans_result['label_info']:
                    print(f"    Total samples: {hans_result['label_info']['total_samples']}")
                    print(f"    Unique labels: {hans_result['label_info']['unique_labels']}")
                    print(f"    Label distribution: {hans_result['label_info']['label_counts']}")
                print(f"    Usable for AUROC: {'Yes' if hans_result['usable_for_auroc'] else 'No'}")
        
        # Also try ANLI
        if anli_missing:
            print("\n📥 Downloading ANLI dataset for transfer evaluation...")
            if download_anli_dataset(data_dir):
                # Re-analyze ANLI after download
                anli_result = analyze_dataset(anli_result['path'], anli_result['name'])
                results = [r for r in results if 'ANLI' not in r['name']] + [anli_result]
                usable_datasets = [r for r in results if r['usable_for_auroc']]
                print("\n  Re-analyzing ANLI dataset...")
                print(f"  {anli_result['name']}:")
                if anli_result['label_info']:
                    print(f"    Total samples: {anli_result['label_info']['total_samples']}")
                    print(f"    Unique labels: {anli_result['label_info']['unique_labels']}")
                    print(f"    Label distribution: {anli_result['label_info']['label_counts']}")
                print(f"    Usable for AUROC: {'Yes' if anli_result['usable_for_auroc'] else 'No'}")
    
    if usable_datasets:
        print(f"\n✓ Found {len(usable_datasets)} usable test dataset(s):")
        for r in usable_datasets:
            print(f"  - {r['name']} ({r['path']})")
        
        # If SST-2 is unavailable, recommend alternatives
        if sst_unavailable:
            recommendations = []
            if csd_result and csd_result['usable_for_auroc']:
                recommendations.append("CSD Negation (sentiment classification)")
            if hans_result and hans_result['usable_for_auroc']:
                recommendations.append("HANS (transfer evaluation)")
            if anli_result and anli_result['usable_for_auroc']:
                recommendations.append("ANLI (transfer evaluation)")
            
            if recommendations:
                print(f"\n→ Recommendation: Use {' or '.join(recommendations)} for test metrics")
                print(f"  (SST-2 has -1 labels and cannot be used for AUROC computation)")
                if csd_result and csd_result['usable_for_auroc']:
                    print(f"  → CSD is best for sentiment classification tasks")
                if (hans_result and hans_result['usable_for_auroc']) or (anli_result and anli_result['usable_for_auroc']):
                    print(f"  → HANS/ANLI are useful for transfer evaluation and robustness testing")
        elif len(usable_datasets) == 1:
            print(f"\n→ Recommendation: Use {usable_datasets[0]['name']} for test metrics")
        else:
            print(f"\n→ Recommendation: You can use any of the above datasets for test metrics")
            print(f"  (Consider using the dataset with more samples or better label balance)")
    else:
        print("\n✗ No usable test datasets found!")
        if unusable_datasets:
            print("\n  Available datasets but with invalid labels:")
            for r in unusable_datasets:
                print(f"  - {r['name']}: {r.get('error', 'All labels are -1')}")
        
        if sst_unavailable and (csd_missing or hans_missing or anli_missing) and not auto_download:
            print(f"\n💡 Tip: Run with --download to automatically download CSD/HANS/ANLI datasets")
            print(f"   python check_test_datasets.py --download")
        
        print("\n→ Recommendation: Use validation set metrics instead of test set metrics")
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze test datasets to check label availability"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/raw",
        help="Root data directory (default: data/raw)",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Automatically download CSD dataset if SST-2 is unavailable",
    )
    args = parser.parse_args()
    
    # Define datasets to check
    datasets_to_check = [
        {
            "name": "SST-2 test set",
            "path": os.path.join(args.data_dir, "test", "sst.parquet"),
        },
        {
            "name": "CSD Negation test set",
            "path": os.path.join(args.data_dir, "test", "negation.parquet"),
        },
        {
            "name": "HANS test set",
            "path": os.path.join(args.data_dir, "test", "hans.parquet"),
        },
        {
            "name": "ANLI test set",
            "path": os.path.join(args.data_dir, "test", "anli.parquet"),
        },
    ]
    
    # Analyze each dataset
    results = []
    for dataset_info in datasets_to_check:
        result = analyze_dataset(
            dataset_info["path"],
            dataset_info["name"]
        )
        results.append(result)
    
    # Print report (may trigger download if auto_download is True)
    print_analysis_report(results, args.data_dir, auto_download=args.download)
    
    # Re-check after potential download
    if args.download:
        # Re-analyze all datasets if they were just downloaded
        csd_result = next((r for r in results if 'CSD' in r['name']), None)
        hans_result = next((r for r in results if 'HANS' in r['name']), None)
        anli_result = next((r for r in results if 'ANLI' in r['name']), None)
        
        if csd_result and not csd_result['exists']:
            csd_path = os.path.join(args.data_dir, "test", "negation.parquet")
            if os.path.exists(csd_path):
                csd_result = analyze_dataset(csd_path, "CSD Negation test set")
                results = [r for r in results if 'CSD' not in r['name']] + [csd_result]
        
        if hans_result and not hans_result['exists']:
            hans_path = os.path.join(args.data_dir, "test", "hans.parquet")
            if os.path.exists(hans_path):
                hans_result = analyze_dataset(hans_path, "HANS test set")
                results = [r for r in results if 'HANS' not in r['name']] + [hans_result]
        
        if anli_result and not anli_result['exists']:
            anli_path = os.path.join(args.data_dir, "test", "anli.parquet")
            if os.path.exists(anli_path):
                anli_result = analyze_dataset(anli_path, "ANLI test set")
                results = [r for r in results if 'ANLI' not in r['name']] + [anli_result]
    
    # Sanity checks on all datasets
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    print()
    
    for result in results:
        if not result['exists']:
            continue
        
        print(f"{result['name']}:")
        
        if result['error']:
            print(f"  ✗ Error: {result['error']}")
            print()
            continue
        
        if not result['label_info']:
            print(f"  ⚠ No label information available")
            print()
            continue
        
        label_info = result['label_info']
        unique_labels = label_info['unique_labels']
        label_counts = label_info['label_counts']
        
        # Sanity check 1: Has valid labels
        if result['has_labels']:
            print(f"  ✓ Has valid labels")
        else:
            print(f"  ✗ No valid labels (all -1)")
        
        # Sanity check 2: Binary classification labels
        if set(unique_labels) == {0, 1} or set(unique_labels) == {0, 1, -1}:
            print(f"  ✓ Binary classification labels (0/1)")
        else:
            print(f"  ⚠ Non-binary labels: {unique_labels}")
            print(f"    (May need label mapping for sentiment classification)")
        
        # Sanity check 3: Label balance
        if 0 in label_counts and 1 in label_counts:
            balance_ratio = min(label_counts[0], label_counts[1]) / max(label_counts[0], label_counts[1])
            if balance_ratio > 0.8:
                print(f"  ✓ Well-balanced labels (ratio: {balance_ratio:.2f})")
            elif balance_ratio > 0.5:
                print(f"  ⚠ Moderately imbalanced (ratio: {balance_ratio:.2f})")
            else:
                print(f"  ✗ Highly imbalanced (ratio: {balance_ratio:.2f})")
        
        # Sanity check 4: Sample size
        total = label_info['total_samples']
        if total >= 1000:
            print(f"  ✓ Sufficient samples ({total})")
        elif total >= 100:
            print(f"  ⚠ Small sample size ({total})")
        else:
            print(f"  ✗ Very small sample size ({total})")
        
        print()
    
    # Return exit code based on findings
    if any(r['usable_for_auroc'] for r in results):
        return 0
    else:
        print("⚠ Warning: No usable test datasets found!")
        return 1


if __name__ == "__main__":
    sys.exit(main())

