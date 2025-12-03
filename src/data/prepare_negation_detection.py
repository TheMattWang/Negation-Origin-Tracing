"""
Prepare Negation Detection Dataset from JinaAI Negation Dataset.

This script downloads the JinaAI negation dataset and converts it to a binary
classification format for negation detection:
- Label 0: No negation (anchor, entailment sentences)
- Label 1: Has negation (negative sentences)

The output is compatible with the existing SentimentDataset class.

Usage:
    python src/data/prepare_negation_detection.py [--output_dir data/raw]
"""

import os
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def download_jinaai_dataset(cache_dir: str = "data/raw/negation", version: str = "v2"):
    """
    Download JinaAI negation dataset from HuggingFace.
    
    Args:
        cache_dir: Directory to cache the raw triplet data
        version: Dataset version ("v1" or "v2")
    
    Returns:
        Dictionary with train/test splits as DataFrames, or None if failed
    """
    from datasets import load_dataset
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Try to load from cache first
    train_cache = os.path.join(cache_dir, "train.parquet")
    test_cache = os.path.join(cache_dir, "test.parquet")
    
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print(f"Loading cached JinaAI dataset from {cache_dir}...")
        return {
            "train": pd.read_parquet(train_cache),
            "test": pd.read_parquet(test_cache),
        }
    
    # Download from HuggingFace
    if version == "v2":
        dataset_name = "jinaai/negation-dataset-v2"
        print(f"Downloading {dataset_name} (50k train, 1k test)...")
    else:
        dataset_name = "jinaai/negation-dataset"
        print(f"Downloading {dataset_name}...")
    
    try:
        dataset = load_dataset(dataset_name)
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        result = {}
        for split_name, split_data in dataset.items():
            df = split_data.to_pandas()
            cache_path = os.path.join(cache_dir, f"{split_name}.parquet")
            df.to_parquet(cache_path)
            result[split_name] = df
            print(f"  Cached {split_name}: {len(df)} examples -> {cache_path}")
        
        return result
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        
        # Try v1 as fallback
        try:
            dataset_name = "jinaai/negation-dataset"
            print(f"Trying fallback: {dataset_name}...")
            dataset = load_dataset(dataset_name)
            
            result = {}
            for split_name, split_data in dataset.items():
                df = split_data.to_pandas()
                cache_path = os.path.join(cache_dir, f"{split_name}.parquet")
                df.to_parquet(cache_path)
                result[split_name] = df
                print(f"  Cached {split_name}: {len(df)} examples -> {cache_path}")
            
            return result
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None


def convert_triplets_to_detection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert triplet format (anchor, entailment, negative) to negation detection format.
    
    Args:
        df: DataFrame with 'anchor', 'entailment', 'negative' columns
    
    Returns:
        DataFrame with 'sentence' and 'label' columns
        - label 0: no negation (anchor, entailment)
        - label 1: has negation (negative)
    """
    examples = []
    
    for idx, row in df.iterrows():
        # Anchor sentence - no negation (label 0)
        if 'anchor' in row and pd.notna(row['anchor']):
            examples.append({
                'sentence': str(row['anchor']),
                'label': 0,
            })
        
        # Entailment sentence - no negation (label 0)
        # Skip if same as anchor to avoid duplicates
        if 'entailment' in row and pd.notna(row['entailment']):
            if row['entailment'] != row.get('anchor', ''):
                examples.append({
                    'sentence': str(row['entailment']),
                    'label': 0,
                })
        
        # Negative sentence - has negation (label 1)
        if 'negative' in row and pd.notna(row['negative']):
            examples.append({
                'sentence': str(row['negative']),
                'label': 1,
            })
    
    result_df = pd.DataFrame(examples)
    
    # Remove duplicates
    result_df = result_df.drop_duplicates(subset=['sentence'])
    
    # Shuffle
    result_df = result_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return result_df


def validate_labels(df: pd.DataFrame, split_name: str) -> bool:
    """
    Validate that the dataset has proper labels for training/evaluation.
    
    Args:
        df: DataFrame with 'label' column
        split_name: Name of the split for logging
    
    Returns:
        True if labels are valid, False otherwise
    """
    if 'label' not in df.columns:
        print(f"  ✗ {split_name}: No 'label' column found")
        return False
    
    unique_labels = sorted(df['label'].unique())
    label_counts = df['label'].value_counts().to_dict()
    
    print(f"  {split_name}:")
    print(f"    Samples: {len(df)}")
    print(f"    Unique labels: {unique_labels}")
    print(f"    Distribution: {label_counts}")
    
    # Check for valid binary labels
    if set(unique_labels) == {0, 1}:
        balance_ratio = min(label_counts[0], label_counts[1]) / max(label_counts[0], label_counts[1])
        print(f"    Balance ratio: {balance_ratio:.2f}")
        return True
    elif unique_labels == [-1] or (-1 in unique_labels and len(unique_labels) == 1):
        print(f"    ✗ All labels are -1 (no ground truth)")
        return False
    else:
        print(f"    ✗ Unexpected labels: {unique_labels}")
        return False


def prepare_negation_detection_dataset(
    output_dir: str = "data/raw",
    cache_dir: str = "data/raw/negation",
    version: str = "v2",
    test_split_ratio: float = 0.1,
    val_split_ratio: float = 0.1,
):
    """
    Prepare negation detection dataset from JinaAI triplets.
    
    Args:
        output_dir: Directory to save the processed dataset
        cache_dir: Directory to cache the raw JinaAI data
        version: JinaAI dataset version
        test_split_ratio: Ratio of data to use for test set if not available
        val_split_ratio: Ratio of data to use for validation set
    """
    print("=" * 60)
    print("Preparing Negation Detection Dataset")
    print("=" * 60)
    
    # Download/load JinaAI dataset
    raw_data = download_jinaai_dataset(cache_dir, version)
    
    if raw_data is None:
        print("\n✗ Failed to download JinaAI dataset")
        return False
    
    # Create output directories
    train_dir = os.path.join(output_dir, "train")
    val_dir = os.path.join(output_dir, "validation")
    test_dir = os.path.join(output_dir, "test")
    
    for d in [train_dir, val_dir, test_dir]:
        os.makedirs(d, exist_ok=True)
    
    # Convert each split to detection format
    print("\nConverting triplets to negation detection format...")
    
    converted_data = {}
    for split_name, df in raw_data.items():
        print(f"\n  Processing {split_name} split ({len(df)} triplets)...")
        converted_df = convert_triplets_to_detection(df)
        converted_data[split_name] = converted_df
        print(f"    -> {len(converted_df)} detection examples")
    
    # Determine which splits we have
    has_train = 'train' in converted_data
    has_test = 'test' in converted_data
    has_val = 'validation' in converted_data
    
    # If we don't have all splits, create them from available data
    if has_train and not has_test:
        print("\n  Creating test split from train data...")
        train_df = converted_data['train']
        n_test = int(len(train_df) * test_split_ratio)
        converted_data['test'] = train_df.tail(n_test).reset_index(drop=True)
        converted_data['train'] = train_df.head(len(train_df) - n_test).reset_index(drop=True)
        print(f"    Train: {len(converted_data['train'])}, Test: {len(converted_data['test'])}")
    
    if has_train and not has_val:
        print("\n  Creating validation split from train data...")
        train_df = converted_data['train']
        n_val = int(len(train_df) * val_split_ratio)
        converted_data['validation'] = train_df.tail(n_val).reset_index(drop=True)
        converted_data['train'] = train_df.head(len(train_df) - n_val).reset_index(drop=True)
        print(f"    Train: {len(converted_data['train'])}, Validation: {len(converted_data['validation'])}")
    
    # Save processed datasets
    print("\nSaving processed datasets...")
    
    output_paths = {
        'train': os.path.join(train_dir, "negation_detection.parquet"),
        'validation': os.path.join(val_dir, "negation_detection.parquet"),
        'test': os.path.join(test_dir, "negation_detection.parquet"),
    }
    
    for split_name, output_path in output_paths.items():
        if split_name in converted_data:
            df = converted_data[split_name]
            df.to_parquet(output_path)
            print(f"  Saved {split_name}: {len(df)} examples -> {output_path}")
    
    # Validate all splits
    print("\nValidating labels...")
    all_valid = True
    for split_name, output_path in output_paths.items():
        if os.path.exists(output_path):
            df = pd.read_parquet(output_path)
            if not validate_labels(df, split_name):
                all_valid = False
    
    if all_valid:
        print("\n✓ All datasets have valid labels!")
    else:
        print("\n⚠ Some datasets have invalid labels. Check output above.")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}")
    print("\nFiles created:")
    for split_name, output_path in output_paths.items():
        if os.path.exists(output_path):
            df = pd.read_parquet(output_path)
            print(f"  {output_path}")
            print(f"    {len(df)} examples, labels: {sorted(df['label'].unique())}")
    
    print("\nDataset format:")
    print("  - 'sentence': Text to classify")
    print("  - 'label': 0 = no negation, 1 = has negation")
    
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Negation Detection Dataset from JinaAI triplets"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw",
        help="Directory to save processed dataset (default: data/raw)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="data/raw/negation",
        help="Directory to cache raw JinaAI data (default: data/raw/negation)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="JinaAI dataset version (default: v2)",
    )
    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for test set if not available (default: 0.1)",
    )
    parser.add_argument(
        "--val_split_ratio",
        type=float,
        default=0.1,
        help="Ratio of data for validation set if not available (default: 0.1)",
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    if not os.path.isabs(args.output_dir):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    if not os.path.isabs(args.cache_dir):
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        args.cache_dir = os.path.join(project_root, args.cache_dir)
    
    success = prepare_negation_detection_dataset(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        version=args.version,
        test_split_ratio=args.test_split_ratio,
        val_split_ratio=args.val_split_ratio,
    )
    
    if success:
        print("\n✓ Preparation complete!")
    else:
        print("\n✗ Preparation failed. Check error messages above.")


if __name__ == "__main__":
    main()

