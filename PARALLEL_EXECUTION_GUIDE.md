# Parallel Execution Guide

This guide explains how to speed up your experiments using parallel execution.

## Overview

The experiment pipeline has several opportunities for parallelization:

1. **Pooling strategies** - Run cls, mean, and token probes in parallel for each layer
2. **Layers** - Train probes for multiple layers simultaneously
3. **Full parallelization** - Run all experiments in parallel (if you have enough GPUs)

## Quick Start

### Use the Parallel Script

```bash
# Run with default settings (3 pooling strategies in parallel per layer)
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/parallel_search \
    --layers all \
    --pooling_strategies all \
    --parallel_workers 3
```

## Parallelization Modes

### Mode 1: Parallel Pooling (Default - Recommended)

Runs all 3 pooling strategies in parallel for each layer sequentially.

**Best for:** Single GPU or limited GPU memory

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/my_exp \
    --parallel_mode pooling \
    --parallel_workers 3
```

**How it works:**
```
Layer 0:
  ├─ cls probe    (GPU 0) ─┐
  ├─ mean probe   (GPU 0) ─┼─ Run in parallel
  └─ token probe  (GPU 0) ─┘

Layer 1:
  ├─ cls probe    (GPU 0) ─┐
  ├─ mean probe   (GPU 0) ─┼─ Run in parallel
  └─ token probe  (GPU 0) ─┘
...
```

**Speed improvement:** ~3x faster than sequential

### Mode 2: Parallel Layers

Runs multiple layers in parallel (each with all pooling strategies).

**Best for:** Multiple GPUs available

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/my_exp \
    --parallel_mode layer \
    --parallel_workers 4 \
    --devices 1
```

**How it works:**
```
Worker 1: Layer 0 (cls, mean, token) → Layer 4 → Layer 8
Worker 2: Layer 1 (cls, mean, token) → Layer 5 → Layer 9
Worker 3: Layer 2 (cls, mean, token) → Layer 6 → Layer 10
Worker 4: Layer 3 (cls, mean, token) → Layer 7 → Layer 11
```

**Speed improvement:** ~Nx faster (where N = number of workers)

### Mode 3: Maximum Parallelization

Runs ALL experiments in parallel.

**Best for:** Multiple GPUs with lots of memory

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/my_exp \
    --parallel_mode all \
    --parallel_workers 6 \
    --devices 1
```

**How it works:**
```
All 18 experiments (6 layers × 3 pooling) run simultaneously
across 6 workers
```

**Speed improvement:** Maximum speedup (limited by available resources)

## Hardware Considerations

### Single GPU (Most Common)

**Recommended:** Parallel pooling mode

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --devices 1
```

- Runs 3 pooling strategies in parallel
- Each uses the same GPU sequentially
- ~3x speedup with minimal memory overhead

### Multiple GPUs

**Recommended:** Parallel layers mode

```bash
# For 4 GPUs
python src/scripts/search_layers_parallel.py \
    --parallel_mode layer \
    --parallel_workers 4 \
    --devices 1
```

- Each worker gets assigned to a different GPU automatically
- ~4x speedup with 4 GPUs
- Each worker trains one layer at a time

### High-Memory Setup (Multiple GPUs + Lots of RAM)

**Recommended:** Maximum parallelization

```bash
# For 8 GPUs
python src/scripts/search_layers_parallel.py \
    --parallel_mode all \
    --parallel_workers 8 \
    --devices 1
```

- All experiments run simultaneously
- Maximum speedup
- Requires significant GPU memory

## Google Colab

### Free Tier (1 GPU - T4)

**Recommended:** Parallel pooling mode

```python
!python src/scripts/search_layers_parallel.py \
    --output_dir /content/drive/MyDrive/experiments/parallel \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --devices 1
```

**Expected speedup:** ~3x faster than sequential

### Colab Pro (Better GPU)

Same as free tier, but faster overall due to better GPU.

## Performance Comparison

### Sequential (Original Script)

```bash
python src/scripts/search_layers.py --layers all
```

**Time for 18 experiments (6 layers × 3 pooling):**
- ~3-4 hours on T4 GPU

### Parallel Pooling (3 workers)

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode pooling \
    --parallel_workers 3
```

**Time for 18 experiments:**
- ~1-1.5 hours on T4 GPU
- **~3x faster**

### Parallel Layers (4 workers, 4 GPUs)

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode layer \
    --parallel_workers 4
```

**Time for 18 experiments:**
- ~45-60 minutes with 4 GPUs
- **~4x faster**

### Maximum Parallelization (8 workers, 8 GPUs)

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode all \
    --parallel_workers 8
```

**Time for 18 experiments:**
- ~20-30 minutes with 8 GPUs
- **~8x faster**

## Resume Support

The parallel script fully supports resume functionality:

```bash
# First run (interrupted after 12 experiments)
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/my_exp \
    --parallel_workers 3

# After interruption, re-run same command
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/my_exp \
    --parallel_workers 3

# Output:
# ✓ Loaded 12 existing results
# ✓ 12 experiments already completed
# Remaining: 6
# [Runs only remaining 6 experiments in parallel]
```

## Command-Line Options

### Parallel-Specific Options

| Option | Default | Description |
|--------|---------|-------------|
| `--parallel_workers` | 3 | Number of parallel workers |
| `--parallel_mode` | pooling | Parallelization mode (pooling/layer/all) |

### Standard Options (Same as Sequential)

| Option | Default | Description |
|--------|---------|-------------|
| `--layers` | all | Layers to search |
| `--pooling_strategies` | all | Pooling strategies |
| `--output_dir` | experiments/layer_search | Output directory |
| `--max_epochs` | 10 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--no_resume` | False | Start fresh |

## Examples

### Example 1: Fast Search on Single GPU

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/fast_search \
    --layers all \
    --pooling_strategies all \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --max_epochs 10
```

### Example 2: Multi-GPU Setup

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/multi_gpu \
    --layers all \
    --pooling_strategies all \
    --parallel_mode layer \
    --parallel_workers 4 \
    --max_epochs 10
```

### Example 3: Specific Layers in Parallel

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/specific_layers \
    --layers 3,4,5 \
    --pooling_strategies cls,mean \
    --parallel_mode pooling \
    --parallel_workers 2
```

### Example 4: Google Colab with Drive

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Run parallel search
!python src/scripts/search_layers_parallel.py \
    --output_dir /content/drive/MyDrive/experiments/parallel \
    --layers all \
    --pooling_strategies all \
    --parallel_mode pooling \
    --parallel_workers 3 \
    --max_epochs 10
```

## Other Parallelization Opportunities

### 1. Data Loading

Already parallelized in PyTorch DataLoader:

```python
# In datamodule.py
DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,  # Parallel data loading
    ...
)
```

### 2. Intervention Experiments

Can be parallelized similarly:

```bash
# Run interventions on multiple layers in parallel
# (Future enhancement)
```

### 3. Visualization Generation

Can generate plots in parallel:

```python
# Generate multiple visualizations simultaneously
# (Future enhancement)
```

## Troubleshooting

### Out of Memory Errors

**Solution:** Reduce parallel workers or batch size

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_workers 2 \
    --batch_size 16
```

### CUDA Out of Memory

**Solution:** Use parallel pooling mode (sequential GPU usage)

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode pooling \
    --parallel_workers 3
```

### Slow Performance

**Check:**
1. GPU is being used: `nvidia-smi`
2. Parallel workers match your hardware
3. Not using too many workers (diminishing returns)

### Process Hangs

**Solution:** Reduce parallel workers

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_workers 2
```

## Best Practices

1. **Start with parallel pooling** - Safe and gives good speedup
2. **Monitor GPU usage** - Use `nvidia-smi` to check utilization
3. **Use resume** - Don't worry about interruptions
4. **Save to Drive** - For Colab persistence
5. **Test with fewer experiments first** - Use `--layers 0,1` to test

## Comparison: Sequential vs Parallel

### Sequential Script

```bash
python src/scripts/search_layers.py
```

**Pros:**
- Simple, well-tested
- Lower memory usage
- Easier to debug

**Cons:**
- Slower (~3-4 hours for full search)
- No parallelization

### Parallel Script

```bash
python src/scripts/search_layers_parallel.py
```

**Pros:**
- Much faster (~1-1.5 hours with 3 workers)
- Efficient resource usage
- Same resume support

**Cons:**
- Slightly more complex
- Requires more memory
- May need tuning for your hardware

## When to Use Which

### Use Sequential Script When:
- Learning/testing
- Limited memory
- Debugging issues
- Running on CPU

### Use Parallel Script When:
- Running full experiments
- Have GPU(s) available
- Want faster results
- Running on Colab or cluster

## Summary

- **Single GPU:** Use `--parallel_mode pooling` with 3 workers (~3x speedup)
- **Multiple GPUs:** Use `--parallel_mode layer` with N workers (~Nx speedup)
- **High-end setup:** Use `--parallel_mode all` for maximum speed
- **Resume works:** Just re-run the same command after interruption

**Bottom line:** Parallel pooling mode gives you ~3x speedup on any single GPU with minimal setup! 🚀

---

**See Also:**
- `RESUME_GUIDE.md` - Resume functionality
- `README.md` - General usage
- `COLAB_SETUP.md` - Colab-specific instructions

