# Parallel Execution Update

## Summary

Updated scripts to use parallel execution by default for faster experiments.

## What Changed

### 1. `run_full_comparison.sh`
Now uses `search_layers_parallel.py` instead of `search_layers.py`

**Before:**
```bash
python src/scripts/search_layers.py --layers all
```

**After:**
```bash
python src/scripts/search_layers_parallel.py \
    --layers all \
    --parallel_workers 3 \
    --parallel_mode pooling
```

**Speedup:** ~3x faster

### 2. `run_full_experiment.py`
Added `--parallel` flag to enable parallel execution

**Usage:**
```bash
# Sequential (original)
python src/scripts/run_full_experiment.py --output_dir experiments/my_exp

# Parallel (3x faster)
python src/scripts/run_full_experiment.py \
    --output_dir experiments/my_exp \
    --parallel \
    --parallel_workers 3 \
    --parallel_mode pooling
```

## New Command-Line Options

### `run_full_experiment.py`

| Option | Default | Description |
|--------|---------|-------------|
| `--parallel` | False | Enable parallel execution |
| `--parallel_workers` | 3 | Number of parallel workers |
| `--parallel_mode` | pooling | Mode: pooling/layer/all |

## Examples

### Quick Start (Parallel by default)

```bash
# run_full_comparison.sh now uses parallel by default
bash run_full_comparison.sh
```

### Full Experiment with Parallel

```bash
python src/scripts/run_full_experiment.py \
    --output_dir experiments/my_exp \
    --parallel \
    --parallel_workers 3
```

### Google Colab

```python
# Now 3x faster!
!bash run_full_comparison.sh
```

## Performance

### Before (Sequential)
- **Time:** ~3-4 hours for 18 experiments
- **Script:** `search_layers.py`

### After (Parallel)
- **Time:** ~1-1.5 hours for 18 experiments
- **Script:** `search_layers_parallel.py`
- **Speedup:** ~3x

## Backward Compatibility

Both scripts still work:
- `search_layers.py` - Sequential (slower but simpler)
- `search_layers_parallel.py` - Parallel (3x faster)

You can still use the sequential version if needed:
```bash
python src/scripts/search_layers.py --layers all
```

## Recommendation

**Use parallel execution for:**
- Full experiments
- Production runs
- When you want faster results

**Use sequential execution for:**
- Debugging
- Testing
- Limited memory situations

## See Also

- `PARALLEL_EXECUTION_GUIDE.md` - Comprehensive guide
- `PARALLELIZATION_OPPORTUNITIES.md` - Analysis of all opportunities

