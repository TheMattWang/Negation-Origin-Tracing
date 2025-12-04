# Shell Scripts Guide

This directory contains various shell scripts for running experiments. Here's what each script does and when to use it.

## Main Scripts (Recommended)

### `run_full_comparison.sh` ⭐ **START HERE**
**Purpose:** Complete end-to-end experiment pipeline  
**What it does:**
- Trains probes on base model (all layers)
- Identifies best layer
- Runs interventions on base model
- Runs interventions on finetuned model
- Generates comparison summary

**Usage:**
```bash
# Local
bash run_full_comparison.sh

# Colab (with Google Drive)
export DRIVE_OUTPUT="/content/drive/MyDrive/Negation-Results"
bash run_full_comparison.sh
```

**When to use:** For complete experiments comparing base vs finetuned models

---

## Layer Sweep Scripts

### `run_layer_sweep.sh` ⭐ **RECOMMENDED**
**Purpose:** Robust sequential layer search (deadlock-safe)  
**Features:**
- Sequential execution (no deadlocks)
- Resume support (skips completed experiments)
- Timeout detection
- Works on Colab and local

**Usage:**
```bash
bash run_layer_sweep.sh
```

**When to use:** For comprehensive layer search across all layers and pooling strategies

---

## Pooling-Specific Sweeps

These scripts run sweeps for specific pooling strategies. They're designed to run in parallel across multiple Colab runtimes or terminals.

### Sentiment Classification Task

- **`run_sweep_cls.sh`** - CLS pooling sweep (sentiment task)
- **`run_sweep_mean.sh`** - Mean pooling sweep (sentiment task)  
- **`run_sweep_token.sh`** - Token pooling sweep (sentiment task)

### Negation Detection Task

- **`run_sweep_cls_negation_detection.sh`** - CLS pooling (negation detection)
- **`run_sweep_mean_negation_detection.sh`** - Mean pooling (negation detection)
- **`run_sweep_token_negation_detection.sh`** - Token pooling (negation detection)

**Usage (for parallel execution):**
```bash
# Terminal 1
bash run_sweep_cls.sh

# Terminal 2
bash run_sweep_mean.sh

# Terminal 3
bash run_sweep_token.sh
```

**When to use:** When you want to run pooling strategies in parallel across multiple terminals/Colab runtimes

---

## Legacy/Alternative Scripts

These scripts are kept for compatibility but `run_layer_sweep.sh` is recommended:

- **`run_sequential_safe.sh`** - Deadlock-proof sequential sweep (legacy)
- **`run_sweep_simple.sh`** - Simple sequential sweep (legacy)

**Note:** These are older versions. Use `run_layer_sweep.sh` instead.

---

## Quick Reference

| Script | Purpose | Recommended? |
|--------|---------|--------------|
| `run_full_comparison.sh` | Complete experiment pipeline | ✅ Yes |
| `run_layer_sweep.sh` | Layer search (all strategies) | ✅ Yes |
| `run_sweep_*_negation_detection.sh` | Negation detection sweeps | ✅ Yes (if needed) |
| `run_sweep_*.sh` | Sentiment classification sweeps | ✅ Yes (if needed) |
| `run_sequential_safe.sh` | Legacy sequential sweep | ⚠️ Use `run_layer_sweep.sh` instead |
| `run_sweep_simple.sh` | Legacy simple sweep | ⚠️ Use `run_layer_sweep.sh` instead |

---

## Which Script Should I Use?

### For Complete Experiments
→ Use **`run_full_comparison.sh`**

### For Layer Search Only
→ Use **`run_layer_sweep.sh`**

### For Parallel Execution (Multiple Terminals)
→ Use **`run_sweep_cls.sh`**, **`run_sweep_mean.sh`**, **`run_sweep_token.sh`** in separate terminals

### For Negation Detection Task
→ Use **`run_sweep_*_negation_detection.sh`** scripts

---

## Tips

1. **Colab users:** Set `DRIVE_OUTPUT` environment variable to save results to Google Drive
2. **Resume support:** Most scripts automatically resume from checkpoints
3. **Deadlock prevention:** All scripts use `num_workers=0` to prevent deadlocks on single GPU
4. **Check logs:** Logs are saved in `experiments/*/logs/` or `$OUTPUT_DIR/logs/`

---

## Need Help?

- See `README.md` for general usage
- See `COLAB_QUICK_START.md` for Colab-specific instructions
- See `docs/RESUME_GUIDE.md` for resume functionality

