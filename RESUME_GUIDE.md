# Resume from Checkpoint Guide

This guide explains how to resume experiments that were interrupted (e.g., due to Colab disconnection, timeout, or crashes).

## Overview

The experiment scripts now automatically save progress after each completed experiment. If your session terminates, you can simply re-run the same command and it will pick up where it left off.

## How It Works

### Automatic Checkpointing

1. **After each layer/pooling experiment completes**, results are saved to `results_summary.json`
2. **On restart**, the script reads this file and identifies which experiments are already done
3. **Partially completed experiments** (with checkpoints but no results) are prioritized first
4. **Only remaining experiments** are executed

### Smart Prioritization

The resume logic is smart about which experiments to run first:

1. **Completed experiments** - Skipped entirely ✓
2. **Partially completed** - Run first (have checkpoints, need final results) 🔄
3. **Not started** - Run last (no checkpoints yet) ⏳

This ensures you complete in-progress work before starting new experiments.

### What Gets Saved

- `results_summary.json`: Contains all completed experiment results
- Individual checkpoints: Stored in `layer_{X}_pooling_{Y}/checkpoints/`
- Progress is saved **incrementally** - you won't lose work if interrupted

## Usage

### Method 1: Automatic Resume (Default)

Simply re-run the same command you used before. The script will automatically detect and skip completed experiments.

```bash
# Original command
python src/scripts/run_full_experiment.py \
    --data_dir data/raw \
    --output_dir experiments/my_experiment \
    --layers all \
    --pooling_strategies all

# After interruption, run the EXACT SAME command
python src/scripts/run_full_experiment.py \
    --data_dir data/raw \
    --output_dir experiments/my_experiment \
    --layers all \
    --pooling_strategies all
```

**The script will:**
- ✓ Load existing results
- ✓ Skip completed experiments
- ✓ Continue with remaining experiments

### Method 2: Using search_layers.py Directly

If you're running the layer search script directly:

```bash
# Resume automatically (default behavior)
python src/scripts/search_layers.py \
    --data_dir data/raw \
    --output_dir experiments/layer_search \
    --layers all \
    --pooling_strategies all

# Start fresh (ignore existing results)
python src/scripts/search_layers.py \
    --data_dir data/raw \
    --output_dir experiments/layer_search \
    --layers all \
    --pooling_strategies all \
    --no_resume
```

### Method 3: Using Shell Script

If using the bash script:

```bash
# Just re-run the script
bash run_full_comparison.sh
```

## Google Colab Specific Instructions

### Saving to Google Drive

Make sure your output directory is in Google Drive so results persist:

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Use Drive path for output
output_dir = '/content/drive/MyDrive/experiments/my_experiment'
```

### After Disconnection

1. **Reconnect to Colab**
2. **Remount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Re-run the exact same command** - it will resume automatically

### Example Colab Cell

```python
# After reconnecting to Colab
from google.colab import drive
drive.mount('/content/drive')

# Change to project directory
%cd /content/Negation-Origin-Tracing

# Re-run the experiment (will resume automatically)
!python src/scripts/run_full_experiment.py \
    --data_dir data/raw \
    --output_dir /content/drive/MyDrive/experiments/comparison_experiment \
    --layers all \
    --pooling_strategies all \
    --max_epochs 10 \
    --batch_size 32
```

## Checking Progress

### View Completed Experiments

```python
import json

# Load results
with open('experiments/my_experiment/results_summary.json', 'r') as f:
    results = json.load(f)

# Count completed
completed = sum(1 for r in results if 'error' not in r and r.get('test_auroc', 0) > 0)
print(f"Completed: {completed}/{len(results)}")

# Show what's done
for r in results:
    if 'error' not in r:
        print(f"Layer {r['layer_idx']}, {r['pooling_strategy']}: AUROC={r['test_auroc']:.4f}")
```

### View Progress During Run

The script will print:
```
Starting layer search: 18 total experiments
Already completed: 12
Remaining: 6
```

## What If Something Goes Wrong?

### Results File is Corrupted

If `results_summary.json` is corrupted:

1. **Backup the file**: `cp results_summary.json results_summary.json.backup`
2. **Try to fix it** or delete it
3. **Re-run with `--no_resume`** to start fresh

### Want to Re-run Specific Experiments

To re-run specific layers:

```bash
# Only run layers 3, 4, 5
python src/scripts/search_layers.py \
    --layers 3,4,5 \
    --pooling_strategies all \
    --output_dir experiments/my_experiment \
    --no_resume
```

### Want to Start Completely Fresh

```bash
# Option 1: Use --no_resume flag
python src/scripts/search_layers.py \
    --output_dir experiments/my_experiment \
    --no_resume

# Option 2: Use a new output directory
python src/scripts/search_layers.py \
    --output_dir experiments/my_experiment_v2
```

## Best Practices

1. **Always use Google Drive paths** in Colab for persistence
2. **Keep the same output_dir** when resuming
3. **Don't modify results_summary.json** manually
4. **Monitor progress** by checking the file periodically
5. **Save your command** in a notebook cell for easy re-running

## Troubleshooting

### "No results found" but I had progress

**Check:**
- Are you using the same `--output_dir`?
- Is the path correct (relative vs absolute)?
- Is Google Drive mounted?

### Script keeps re-running same experiments

**Possible causes:**
- Results file has errors in it
- Experiments failed (check for 'error' field in results)
- Using `--no_resume` flag

### Want to see what will be skipped

Add this before running:

```python
import json
with open('experiments/my_experiment/results_summary.json', 'r') as f:
    results = json.load(f)
    
completed = [(r['layer_idx'], r['pooling_strategy']) 
             for r in results 
             if 'error' not in r and r.get('test_auroc', 0) > 0]
             
print(f"Will skip: {completed}")
```

## Summary

- ✅ **Resume is automatic** - just re-run the same command
- ✅ **Progress is saved** after each experiment
- ✅ **Works with Colab** - save to Google Drive
- ✅ **No data loss** - incremental saves
- ✅ **Flexible** - can start fresh with `--no_resume`

You can now safely run long experiments without worrying about interruptions!

