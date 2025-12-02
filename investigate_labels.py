"""
Diagnostic script to investigate label issues in the dataset.
Run this to see what labels are actually in your data files.
"""

import pandas as pd
import os
from pathlib import Path

def investigate_dataset(data_path, split_name):
    """Investigate a single dataset file."""
    print(f"\n{'='*60}")
    print(f"Investigating: {split_name}")
    print(f"Path: {data_path}")
    print(f"{'='*60}")
    
    if not os.path.exists(data_path):
        print(f"  ⚠ File not found: {data_path}")
        return None
    
    try:
        df = pd.read_parquet(data_path)
        print(f"  ✓ Loaded {len(df)} samples")
        print(f"  Columns: {df.columns.tolist()}")
        
        # Check for label column
        if 'label' in df.columns:
            labels = df['label'].tolist()
            
            # Check for NaN
            nan_count = pd.Series(labels).isna().sum()
            if nan_count > 0:
                print(f"  ⚠ WARNING: Found {nan_count} NaN labels!")
            
            # Get unique labels
            unique_labels = sorted(set([l for l in labels if pd.notna(l)]))
            print(f"  Unique labels: {unique_labels}")
            
            # Get label distribution
            label_counts = pd.Series(labels).value_counts().sort_index()
            print(f"  Label distribution:")
            for label, count in label_counts.items():
                print(f"    {label}: {count} ({count/len(labels)*100:.1f}%)")
            
            # Check range
            try:
                labels_int = [int(float(l)) for l in labels if pd.notna(l)]
                min_label, max_label = min(labels_int), max(labels_int)
                print(f"  Label range: [{min_label}, {max_label}]")
                
                # Check if valid for binary classification
                if min_label < 0 or max_label > 1:
                    print(f"  ⚠ WARNING: Labels outside [0, 1] range!")
                    print(f"     This will cause CUDA assertion errors!")
                elif set(unique_labels) == {0, 1}:
                    print(f"  ✓ Labels are valid for binary classification [0, 1]")
                elif len(unique_labels) == 2:
                    print(f"  ⚠ Labels are {unique_labels} - need mapping to [0, 1]")
                else:
                    print(f"  ⚠ WARNING: Unexpected number of unique labels: {len(unique_labels)}")
                
                # Show sample labels
                print(f"  First 20 labels: {labels_int[:20]}")
                
            except (ValueError, TypeError) as e:
                print(f"  ⚠ ERROR: Could not convert labels to integers: {e}")
                print(f"  Sample labels: {labels[:10]}")
            
            # Check text column
            if 'sentence' in df.columns:
                print(f"  Sample text: {df.iloc[0]['sentence'][:80]}...")
            
            return {
                'path': data_path,
                'count': len(df),
                'unique_labels': unique_labels,
                'min': min_label if 'labels_int' in locals() else None,
                'max': max_label if 'labels_int' in locals() else None,
                'nan_count': nan_count,
            }
        else:
            print(f"  ⚠ No 'label' column found!")
            print(f"  Available columns: {df.columns.tolist()}")
            return None
            
    except Exception as e:
        print(f"  ✗ Error loading file: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Investigate all dataset files."""
    print("="*60)
    print("DATASET LABEL INVESTIGATION")
    print("="*60)
    
    data_dir = "data/raw"
    
    results = {}
    
    # Check SST-2 datasets
    for split in ['train', 'validation', 'test']:
        sst_path = os.path.join(data_dir, split, "sst.parquet")
        result = investigate_dataset(sst_path, f"SST-2 {split}")
        if result:
            results[f"sst_{split}"] = result
    
    # Check Negation datasets
    for split in ['train', 'validation', 'test']:
        neg_path = os.path.join(data_dir, split, "negation.parquet")
        result = investigate_dataset(neg_path, f"Negation {split}")
        if result:
            results[f"negation_{split}"] = result
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    issues = []
    for name, result in results.items():
        if result['min'] is not None and (result['min'] < 0 or result['max'] > 1):
            issues.append(f"{name}: labels [{result['min']}, {result['max']}] out of range")
        if result['nan_count'] > 0:
            issues.append(f"{name}: {result['nan_count']} NaN labels")
    
    if issues:
        print("\n⚠ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ All datasets appear to have valid labels [0, 1]")
    
    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()

