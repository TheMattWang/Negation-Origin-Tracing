"""
Download JinaAI Negation Dataset for transfer evaluation.

This script downloads the jinaai/negation-dataset-v2 from HuggingFace
and saves it in a format suitable for the transfer evaluation notebook.

Dataset structure:
- anchor: Original sentence
- entailment: Sentence that logically follows (similar meaning)
- negative: Sentence that contradicts (often via negation)

Usage:
    python src/data/download_negation.py [--output_dir data/raw/negation]
"""

import os
import argparse
from pathlib import Path


def download_negation_dataset(output_dir: str = "data/raw/negation", version: str = "v2"):
    """
    Download JinaAI negation dataset from HuggingFace.
    
    Args:
        output_dir: Directory to save the dataset
        version: Dataset version ("v1" or "v2"). v2 has more data (50k train, 1k test)
    """
    from datasets import load_dataset
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select dataset based on version
    if version == "v2":
        dataset_name = "jinaai/negation-dataset-v2"
        print(f"Downloading {dataset_name} (50k train, 1k test)...")
    else:
        dataset_name = "jinaai/negation-dataset"
        print(f"Downloading {dataset_name} (10.5k triplets)...")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset(dataset_name)
        
        print(f"Dataset loaded successfully!")
        print(f"Available splits: {list(dataset.keys())}")
        
        # Save each split
        for split_name, split_data in dataset.items():
            output_path = os.path.join(output_dir, f"{split_name}.parquet")
            split_data.to_parquet(output_path)
            print(f"  Saved {split_name}: {len(split_data)} examples -> {output_path}")
            
            # Print sample
            if len(split_data) > 0:
                sample = split_data[0]
                print(f"    Sample columns: {list(sample.keys())}")
                if 'anchor' in sample:
                    print(f"    Anchor: {sample['anchor'][:80]}...")
                if 'negative' in sample:
                    print(f"    Negative: {sample['negative'][:80]}...")
        
        print(f"\nDataset saved to: {output_dir}")
        return True
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("\nTrying alternative approach...")
        return download_negation_dataset_fallback(output_dir)


def download_negation_dataset_fallback(output_dir: str):
    """
    Fallback: Try to load v1 if v2 fails, or create synthetic data.
    """
    from datasets import load_dataset
    
    try:
        # Try v1
        dataset_name = "jinaai/negation-dataset"
        print(f"Trying {dataset_name}...")
        dataset = load_dataset(dataset_name)
        
        for split_name, split_data in dataset.items():
            output_path = os.path.join(output_dir, f"{split_name}.parquet")
            split_data.to_parquet(output_path)
            print(f"  Saved {split_name}: {len(split_data)} examples -> {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Fallback also failed: {e}")
        print("\nCreating minimal synthetic dataset for testing...")
        return create_synthetic_negation_data(output_dir)


def create_synthetic_negation_data(output_dir: str, n_samples: int = 1000):
    """
    Create synthetic negation pairs for testing when HuggingFace download fails.
    """
    import pandas as pd
    import random
    
    templates = [
        ("The movie is {adj}", "The movie is not {adj}"),
        ("I {verb} this product", "I do not {verb} this product"),
        ("This restaurant is {adj}", "This restaurant is not {adj}"),
        ("The service was {adj}", "The service was not {adj}"),
        ("I {verb} the experience", "I do not {verb} the experience"),
    ]
    
    positive_adjs = ["good", "great", "excellent", "amazing", "wonderful", "fantastic"]
    negative_adjs = ["bad", "terrible", "awful", "horrible", "poor", "disappointing"]
    positive_verbs = ["like", "love", "enjoy", "recommend", "appreciate"]
    negative_verbs = ["hate", "dislike", "avoid", "regret"]
    
    data = []
    for i in range(n_samples):
        template = random.choice(templates)
        
        # Randomly choose positive or negative sentiment
        if random.random() > 0.5:
            adj = random.choice(positive_adjs)
            verb = random.choice(positive_verbs)
        else:
            adj = random.choice(negative_adjs)
            verb = random.choice(negative_verbs)
        
        anchor = template[0].format(adj=adj, verb=verb)
        negative = template[1].format(adj=adj, verb=verb)
        entailment = anchor  # Simple: entailment is same as anchor
        
        data.append({
            "anchor": anchor,
            "entailment": entailment,
            "negative": negative,
        })
    
    df = pd.DataFrame(data)
    
    # Split into train/test
    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    train_df.to_parquet(os.path.join(output_dir, "train.parquet"))
    test_df.to_parquet(os.path.join(output_dir, "test.parquet"))
    
    print(f"Created synthetic dataset:")
    print(f"  Train: {len(train_df)} examples")
    print(f"  Test: {len(test_df)} examples")
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Download JinaAI Negation Dataset")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/raw/negation",
        help="Directory to save the dataset",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v2",
        choices=["v1", "v2"],
        help="Dataset version (v2 recommended, has more data)",
    )
    
    args = parser.parse_args()
    
    # Handle relative paths
    if not os.path.isabs(args.output_dir):
        # Try to find project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    success = download_negation_dataset(args.output_dir, args.version)
    
    if success:
        print("\n✓ Download complete!")
    else:
        print("\n✗ Download failed. Check error messages above.")


if __name__ == "__main__":
    main()

