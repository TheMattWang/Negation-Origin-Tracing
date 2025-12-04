# Colab Quick Start Guide

This guide shows you how to run experiments on Google Colab, including setup, execution, and troubleshooting.

## Table of Contents
1. [Quick Commands](#quick-commands)
2. [Automated Setup (Recommended)](#automated-setup-recommended)
3. [Manual Setup](#manual-setup)
4. [Deadlock-Free Execution](#deadlock-free-execution)
5. [Parallel Execution](#parallel-execution)
6. [Resume from Checkpoint](#resume-from-checkpoint)
7. [Troubleshooting](#troubleshooting)

## TL;DR - Quick Commands

### Option 1: Parallel Execution (Fastest - Recommended)

```bash
# Full layer search with parallel pooling strategies (3x faster!)
python src/scripts/search_layers_parallel.py \
    --model_name distilbert-base-uncased \
    --data_dir data/raw \
    --output_dir experiments/layer_search \
    --layers all \
    --pooling_strategies all \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --colab_safe \
    --max_epochs 10 \
    --batch_size 32
```

**Runtime**: ~1-1.5 hours on Colab T4 GPU (with `--colab_safe` flag)

### Option 2: Sequential Execution (Safest)

```bash
# Full layer search (sequential, slower but most stable)
python src/scripts/search_layers_colab.py \
    --model_name distilbert-base-uncased \
    --data_dir data/raw \
    --output_dir experiments/layer_search \
    --layers all \
    --pooling_strategies all \
    --max_epochs 10 \
    --batch_size 32
```

**Runtime**: ~2-3 hours on Colab T4 GPU

### Option 2: Quick Test (1 layer, 1 pooling)

```bash
# Test run (completes in ~3 minutes)
python src/scripts/search_layers_colab.py \
    --layers 0 \
    --pooling_strategies cls \
    --max_epochs 1 \
    --output_dir test_output
```

## Automated Setup (Recommended)

For fully automated experiments, use the notebook:

1. **Upload to Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Upload `notebooks/06_run_full_comparison_colab.ipynb`
   - Enable GPU (Runtime → Change runtime type → GPU)

2. **Run All Cells:**
   - Click Runtime → Run all
   - Click "Allow" when Drive asks for permission
   - Wait for completion (2-4 hours with GPU)

3. **Access Results:**
   - Results automatically saved to: `Google Drive > My Drive > Negation-Origin-Tracing-Results/`
   - Download from Drive web interface or sync with desktop app

### Notebook Workflow

If you prefer step-by-step notebooks instead of the automated one:

1. **`00_setup_colab.ipynb`** - Setup environment and install dependencies
2. **`01_download_data.ipynb`** - Download datasets
3. **`02_layer_search.ipynb`** - Train probes
4. **`03_visualize_results.ipynb`** - Generate plots
5. **`04_full_experiment.ipynb`** - Complete pipeline
6. **`06_run_full_comparison_colab.ipynb`** - **AUTOMATED: Full experiment with Drive sync**

## Environment Setup

### Enable GPU (Recommended)
1. Go to Runtime → Change runtime type
2. Select "GPU" as hardware accelerator
3. Click Save

### Install Dependencies
The setup notebook (`00_setup_colab.ipynb`) will:
- Install PyTorch, Lightning, Transformers, etc.
- Clone the repository
- Verify installation
- Check GPU availability

### File Structure in Colab

After setup, your Colab environment should have:
```
/content/
├── Negation-Origin-Tracing/
│   ├── src/
│   ├── notebooks/
│   ├── data/
│   └── experiments/
```

## Manual Setup

### Step 1: Setup Environment

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Clone repository
!git clone https://github.com/TheMattWang/Negation-Origin-Tracing.git
%cd Negation-Origin-Tracing

# Install dependencies
!pip install -q torch lightning transformers datasets pandas pyarrow matplotlib seaborn scikit-learn tqdm tensorboardX

# Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
```

### Step 2: Download Data

```python
# Download datasets
!python src/data/download.py

# Verify data
!ls -lh data/raw/train/
!ls -lh data/raw/test/
```

### Step 3: Run Layer Search

```python
# Set output directory (save to Drive)
import os
output_dir = '/content/drive/MyDrive/Negation-Results/layer_search'
os.makedirs(output_dir, exist_ok=True)

# Option A: Parallel execution (FASTER - 3x speedup!)
!python src/scripts/search_layers_parallel.py \
    --model_name distilbert-base-uncased \
    --data_dir data/raw \
    --output_dir {output_dir} \
    --layers all \
    --pooling_strategies all \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --colab_safe \
    --max_epochs 10 \
    --batch_size 32

# Option B: Sequential execution (SAFEST - if parallel has issues)
!python src/scripts/search_layers_colab.py \
    --model_name distilbert-base-uncased \
    --data_dir data/raw \
    --output_dir {output_dir} \
    --layers all \
    --pooling_strategies all \
    --max_epochs 10 \
    --batch_size 32
```

### Step 4: Monitor Progress

```python
# Check progress (run this in a separate cell)
import json
import os

results_file = f"{output_dir}/results_summary.json"
if os.path.exists(results_file):
    with open(results_file) as f:
        results = json.load(f)
    
    print(f"Completed: {len(results)} experiments")
    
    # Show recent results
    for r in results[-3:]:
        print(f"  Layer {r['layer_idx']}, {r['pooling_strategy']}: "
              f"AUROC={r['test_auroc']:.4f}")
else:
    print("No results yet - experiment starting...")
```

### Step 5: View Results

```python
# Load and display results
import json
import pandas as pd

with open(f"{output_dir}/results_summary.json") as f:
    results = json.load(f)

df = pd.DataFrame(results)

# Best results
print("Best by AUROC:")
best = df.loc[df['test_auroc'].idxmax()]
print(f"  Layer {best['layer_idx']}, {best['pooling_strategy']}: {best['test_auroc']:.4f}")

# Average by layer
print("\nAverage by Layer:")
print(df.groupby('layer_idx')['test_auroc'].mean().round(4))

# Average by pooling
print("\nAverage by Pooling:")
print(df.groupby('pooling_strategy')['test_auroc'].mean().round(4))
```

## Why This Works (No Deadlocks)

### The Problem
- **Before**: Parallel processes + DataLoader workers (num_workers=4) = deadlock
- **Cause**: 3 parallel processes × 4 DataLoader workers = 12 workers competing for 1 GPU

### The Fix
Set `num_workers=0` to disable DataLoader multiprocessing:
```python
datamodule = NOTDataModule(
    # ... other args ...
    num_workers=0,  # No DataLoader workers - prevents deadlocks
)
```

### Why Parallel NOW Works
With `num_workers=0`:
- ✅ **3 parallel experiments** = OK (each uses GPU directly, no nested workers)
- ✅ **Each experiment** loads data in its main process
- ✅ **No resource contention** from DataLoader workers
- ✅ **3x speedup** from running 3 pooling strategies in parallel

### Two Approaches

**Parallel (Faster - Recommended)**:
- Uses `search_layers_parallel.py` with `--colab_safe`
- Runs 3 pooling strategies in parallel per layer
- ~1-1.5 hours for full search
- Safe with `num_workers=0`

**Sequential (Safest)**:
- Uses `search_layers_colab.py`
- Runs one experiment at a time
- ~2-3 hours for full search
- Maximum stability

## Configuration Options

### Model Selection

```bash
# DistilBERT (default, faster)
--model_name distilbert-base-uncased

# BERT (slower, more parameters)
--model_name bert-base-uncased

# RoBERTa
--model_name roberta-base
```

### Layer Selection

```bash
# All layers (recommended)
--layers all

# Specific layers
--layers 0,1,2,3,4,5

# Single layer (for testing)
--layers 5
```

### Pooling Strategies

```bash
# All strategies (recommended)
--pooling_strategies all

# Specific strategies
--pooling_strategies cls,mean

# Single strategy (for testing)
--pooling_strategies cls
```

### Training Configuration

```bash
# Epochs (10 is good default)
--max_epochs 10

# Batch size (adjust based on GPU memory)
--batch_size 32  # T4 GPU (16GB)
--batch_size 16  # If OOM errors

# Precision (mixed precision for speed)
--precision 16  # Faster, less memory
--precision 32  # Default, more stable
```

## Expected Runtimes (Colab T4 GPU)

| Configuration | Parallel (3 workers) | Sequential | Notes |
|--------------|---------------------|------------|-------|
| 1 layer, 1 pooling, 1 epoch | ~3 min | ~3 min | Quick test |
| 1 layer, 3 pooling, 10 epochs | ~5 min | ~15 min | **3x speedup** |
| 6 layers, 3 pooling, 10 epochs | ~30-45 min | ~90 min | **2-3x speedup** |
| All layers (6), 3 pooling, 10 epochs | **~1-1.5 hours** | ~2-3 hours | **Parallel recommended** |

## Troubleshooting

### Out of Memory (OOM)

**Solution**: Reduce batch size
```bash
--batch_size 16  # or even 8
```

### Colab Disconnects

**Solution**: Results are auto-saved, just resume
```bash
# Automatically resumes from last checkpoint
python src/scripts/search_layers_colab.py --layers all
```

### Slow Training

**Solutions**:
1. Use mixed precision: `--precision 16`
2. Reduce epochs: `--max_epochs 5`
3. Use fewer layers: `--layers 0,1,2,3,4,5`

### Process Appears Frozen

**Check**:
```python
# GPU should show activity
!nvidia-smi

# Files should be updating
!ls -lt {output_dir}/*/checkpoints/ | head -20
```

If GPU is active and files are updating, it's working (just slow).

## Resume After Interruption

The script automatically resumes:

```bash
# Just run the same command again
python src/scripts/search_layers_colab.py \
    --layers all \
    --output_dir experiments/layer_search

# It will:
# 1. Load existing results
# 2. Skip completed experiments
# 3. Continue from where it left off
```

To start fresh (ignore previous results):
```bash
--no_resume
```

## Save Results to Google Drive

```python
# Set output to Drive
output_dir = '/content/drive/MyDrive/Negation-Results/layer_search'

!python src/scripts/search_layers_colab.py \
    --output_dir {output_dir} \
    --layers all
```

Results persist even if Colab disconnects!

## Advanced: Full Comparison Experiment

Run base vs finetuned comparison:

```bash
# Set Drive output
export DRIVE_OUTPUT=/content/drive/MyDrive/Negation-Results

# Run full comparison
bash run_full_comparison.sh
```

This will:
1. Train probes on base model (all layers)
2. Find best layer
3. Run interventions on base model
4. Run interventions on finetuned model
5. Generate comparison summary

**Runtime**: ~3-4 hours on Colab T4

## Comparison: Before vs After Fix

### Before (Deadlock Issue)

```bash
# This would hang/deadlock
python src/scripts/search_layers_parallel.py \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --num_workers 4  # Too many workers!
```

**Result**: ❌ Deadlock after starting

### After (Fixed)

```bash
# This works reliably
python src/scripts/search_layers_colab.py \
    --layers all
```

**Result**: ✅ Completes in 2-3 hours

## Key Differences from Local Execution

| Aspect | Local (Multi-GPU) | Colab (Single GPU) |
|--------|------------------|-------------------|
| Parallel mode | ✅ Recommended | ❌ Causes deadlocks |
| num_workers | 4 (faster) | 0 (required) |
| Execution | Parallel | Sequential |
| Runtime | ~1 hour | ~2-3 hours |
| Stability | High | High (with fix) |

## Summary

**Use this command on Colab**:
```bash
python src/scripts/search_layers_colab.py --layers all
```

**Key features**:
- ✅ No deadlocks
- ✅ Auto-resume
- ✅ GPU optimized
- ✅ Progress tracking
- ✅ Error handling
- ✅ Drive integration

**Expected time**: 2-3 hours for full layer search

---

**Need help?** See [`docs/COLAB_DEADLOCK_FIX.md`](docs/COLAB_DEADLOCK_FIX.md) for detailed technical explanation.

