# Utility Scripts

This directory contains utility scripts for data analysis, result processing, and debugging.

## Data Analysis

- **`check_test_datasets.py`** - Check test dataset labels and download missing datasets
  ```bash
  python scripts/utils/check_test_datasets.py
  python scripts/utils/check_test_datasets.py --download
  ```

- **`investigate_labels.py`** - Diagnostic script to investigate label issues in datasets
  ```bash
  python scripts/utils/investigate_labels.py
  ```

## Result Processing

- **`merge_sweep_results.py`** - Merge results from parallel sweep scripts (cls, mean, token)
  ```bash
  python scripts/utils/merge_sweep_results.py
  python scripts/utils/merge_sweep_results.py --drive_path /content/drive/MyDrive/NOT_results
  ```

- **`reevaluate_sweep_results.py`** - Re-evaluate existing sweep checkpoints on validation set
  ```bash
  python scripts/utils/reevaluate_sweep_results.py
  python scripts/utils/reevaluate_sweep_results.py --drive_path /content/drive/MyDrive/NOT_results
  ```

## When to Use

- **`check_test_datasets.py`**: When setting up data or debugging label issues
- **`investigate_labels.py`**: When debugging CUDA assertion errors related to labels
- **`merge_sweep_results.py`**: After running parallel sweeps to combine results
- **`reevaluate_sweep_results.py`**: To update sweep results with validation metrics

