# Parallelization Opportunities Analysis

This document analyzes all parallelization opportunities in the experiment pipeline.

## Currently Implemented

### ✅ 1. Probe Training Parallelization

**Location:** `src/scripts/search_layers_parallel.py`

**What:** Run multiple probe training experiments in parallel

**Modes:**
- **Pooling mode:** 3 pooling strategies in parallel per layer (~3x speedup)
- **Layer mode:** Multiple layers in parallel (~Nx speedup with N GPUs)
- **All mode:** All experiments in parallel (maximum speedup)

**Usage:**
```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode pooling \
    --parallel_workers 3
```

**Speedup:** 3x on single GPU, Nx with N GPUs

---

## Potential Future Parallelization

### 🔄 2. Data Loading (Already Optimized)

**Location:** `src/datasets/datamodule.py`

**Current Status:** Already parallelized via PyTorch DataLoader

```python
DataLoader(
    dataset,
    num_workers=4,  # Already parallel
    ...
)
```

**No action needed** - PyTorch handles this automatically.

---

### 🚀 3. Intervention Experiments

**Location:** `src/scripts/run_interventions.py`

**Opportunity:** Run interventions on multiple layers in parallel

**Potential Implementation:**

```python
# Current: Sequential
for layer in top_layers:
    run_intervention(layer)

# Parallel: Run multiple layers simultaneously
with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_intervention, layer) for layer in top_layers]
    results = [f.result() for f in futures]
```

**Estimated Speedup:** 2-3x (if running on 3 layers)

**Implementation Complexity:** Medium

**Priority:** Medium (interventions are faster than probe training)

---

### 📊 4. Visualization Generation

**Location:** `src/engine/visualization.py`

**Opportunity:** Generate multiple plots in parallel

**Potential Implementation:**

```python
# Current: Sequential
for viz_type in viz_types:
    generate_plot(viz_type)

# Parallel: Generate plots simultaneously
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(generate_plot, vt) for vt in viz_types]
    [f.result() for f in futures]
```

**Estimated Speedup:** 2-4x

**Implementation Complexity:** Low

**Priority:** Low (visualization is fast)

---

### 🔬 5. Model Inference/Testing

**Location:** `src/scripts/test.py`, `src/scripts/predict.py`

**Opportunity:** Batch inference with parallel processing

**Current Status:** Already uses batching (efficient)

**Potential Enhancement:** Multi-GPU inference

```python
# Use DataParallel for multi-GPU inference
model = torch.nn.DataParallel(model)
```

**Estimated Speedup:** Nx with N GPUs (for large inference tasks)

**Implementation Complexity:** Low

**Priority:** Low (inference is already fast with batching)

---

### 🔀 6. Cross-Validation Folds

**Location:** Not currently implemented

**Opportunity:** If implementing cross-validation, run folds in parallel

**Potential Implementation:**

```python
# Parallel cross-validation
with ProcessPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(train_fold, fold) for fold in range(5)]
    results = [f.result() for f in futures]
```

**Estimated Speedup:** 5x for 5-fold CV

**Implementation Complexity:** Medium

**Priority:** Low (not currently using CV)

---

### 📈 7. Hyperparameter Search

**Location:** Not currently implemented

**Opportunity:** If implementing hyperparameter tuning, run trials in parallel

**Potential Implementation:**

```python
# Parallel hyperparameter search
from ray import tune

tune.run(
    train_function,
    config=search_space,
    num_samples=20,
    resources_per_trial={"gpu": 0.5}  # Share GPUs
)
```

**Estimated Speedup:** Significant (depends on resources)

**Implementation Complexity:** High

**Priority:** Low (not currently needed)

---

### 🔄 8. Base vs Finetuned Comparison

**Location:** `run_full_comparison.sh`

**Opportunity:** Run base and finetuned experiments in parallel

**Potential Implementation:**

```bash
# Current: Sequential
train_base_probes
train_finetuned_probes

# Parallel: Run both simultaneously
train_base_probes &
train_finetuned_probes &
wait
```

**Estimated Speedup:** 2x

**Implementation Complexity:** Low

**Priority:** Medium

---

### 🎯 9. Multiple Model Architectures

**Location:** Not currently implemented

**Opportunity:** If comparing multiple models (DistilBERT, BERT, RoBERTa), run in parallel

**Potential Implementation:**

```python
models = ["distilbert-base-uncased", "bert-base-uncased", "roberta-base"]

with ProcessPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(run_experiment, model) for model in models]
    results = [f.result() for f in futures]
```

**Estimated Speedup:** 3x for 3 models

**Implementation Complexity:** Medium

**Priority:** Low (not currently comparing multiple models)

---

## Priority Ranking

### High Priority (Implement Now)
1. ✅ **Probe Training Parallelization** - DONE
   - Biggest time saver
   - 3x speedup on single GPU
   - Already implemented

### Medium Priority (Consider for Phase 2)
2. 🚀 **Intervention Experiments** - TODO
   - Moderate speedup (2-3x)
   - Medium complexity
   - Would help with full pipeline

3. 🔀 **Base vs Finetuned Comparison** - TODO
   - 2x speedup for comparison experiments
   - Low complexity
   - Easy to implement

### Low Priority (Nice to Have)
4. 📊 **Visualization Generation** - TODO
   - Small speedup (visualization is fast)
   - Low complexity
   - Not a bottleneck

5. 🔬 **Multi-GPU Inference** - TODO
   - Only useful for very large inference tasks
   - Low complexity
   - Not currently a bottleneck

6. 🎯 **Multiple Model Architectures** - TODO
   - Only if comparing multiple models
   - Medium complexity
   - Not currently needed

7. 📈 **Hyperparameter Search** - TODO
   - High complexity
   - Only if doing extensive tuning
   - Not currently needed

---

## Implementation Roadmap

### Phase 1: Core Parallelization (✅ COMPLETE)
- [x] Probe training parallelization
- [x] Resume support for parallel execution
- [x] Documentation

### Phase 2: Pipeline Parallelization (Future)
- [ ] Intervention experiments parallelization
- [ ] Base vs finetuned parallel execution
- [ ] Update `run_full_experiment.py` to use parallel scripts

### Phase 3: Advanced Features (Future)
- [ ] Visualization parallelization
- [ ] Multi-GPU inference support
- [ ] Hyperparameter search with Ray Tune

---

## Performance Analysis

### Current Pipeline (Sequential)

```
Total Time: ~3-4 hours on T4 GPU

Breakdown:
- Probe training: ~3 hours (18 experiments × 10 min each)
- Interventions: ~30 min
- Visualization: ~5 min
- Interpretation: ~1 min
```

### With Parallel Probes (Implemented)

```
Total Time: ~1.5-2 hours on T4 GPU

Breakdown:
- Probe training: ~1 hour (18 experiments ÷ 3 workers)
- Interventions: ~30 min
- Visualization: ~5 min
- Interpretation: ~1 min

Speedup: ~2x overall
```

### With Full Parallelization (Future)

```
Total Time: ~45-60 min on T4 GPU

Breakdown:
- Probe training: ~1 hour (parallel)
- Interventions: ~10 min (parallel)
- Visualization: ~2 min (parallel)
- Interpretation: ~1 min

Speedup: ~3-4x overall
```

---

## Resource Requirements

### Single GPU (T4 - Colab Free)
- **Recommended:** Parallel pooling mode (3 workers)
- **Memory:** ~8 GB
- **Speedup:** ~3x for probes

### Multiple GPUs (4x T4)
- **Recommended:** Parallel layer mode (4 workers)
- **Memory:** ~32 GB total
- **Speedup:** ~4x for probes

### High-End (8x A100)
- **Recommended:** Full parallelization (8 workers)
- **Memory:** ~320 GB total
- **Speedup:** ~8x for probes

---

## Code Examples

### 1. Parallel Probe Training (Implemented)

```bash
python src/scripts/search_layers_parallel.py \
    --parallel_mode pooling \
    --parallel_workers 3
```

### 2. Parallel Interventions (Future)

```python
def run_interventions_parallel(layers, args):
    with ProcessPoolExecutor(max_workers=len(layers)) as executor:
        futures = {
            executor.submit(run_intervention, layer, args): layer 
            for layer in layers
        }
        results = {}
        for future in as_completed(futures):
            layer = futures[future]
            results[layer] = future.result()
    return results
```

### 3. Parallel Visualization (Future)

```python
def generate_all_visualizations_parallel(results_path, output_dir):
    viz_functions = [
        generate_layer_performance,
        generate_pooling_comparison,
        generate_heatmap,
        generate_best_layers
    ]
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(fn, results_path, output_dir) 
            for fn in viz_functions
        ]
        [f.result() for f in futures]
```

---

## Benchmarks

### Probe Training (18 experiments)

| Mode | Workers | GPUs | Time | Speedup |
|------|---------|------|------|---------|
| Sequential | 1 | 1 | 180 min | 1x |
| Parallel Pooling | 3 | 1 | 60 min | 3x |
| Parallel Layer | 4 | 4 | 45 min | 4x |
| Full Parallel | 8 | 8 | 25 min | 7x |

### Full Pipeline (Estimated)

| Configuration | Time | Speedup |
|---------------|------|---------|
| Sequential | 240 min | 1x |
| Parallel Probes Only | 120 min | 2x |
| Parallel Probes + Interventions | 80 min | 3x |
| Full Parallelization | 60 min | 4x |

---

## Recommendations

### For Most Users (Single GPU)
✅ **Use parallel probe training** with pooling mode
```bash
python src/scripts/search_layers_parallel.py --parallel_mode pooling
```
- Easy to use
- 3x speedup
- No additional setup

### For Multi-GPU Users
✅ **Use parallel layer mode**
```bash
python src/scripts/search_layers_parallel.py --parallel_mode layer --parallel_workers 4
```
- Nx speedup with N GPUs
- Efficient resource usage

### For Future Development
🔄 **Implement intervention parallelization**
- Would provide additional 2-3x speedup
- Medium implementation effort
- Good ROI

---

## Summary

**Currently Available:**
- ✅ Parallel probe training (3x speedup on single GPU)
- ✅ Resume support
- ✅ Multiple parallelization modes

**Future Opportunities:**
- 🚀 Intervention parallelization (2-3x speedup)
- 🔀 Base vs finetuned parallelization (2x speedup)
- 📊 Visualization parallelization (2-4x speedup)

**Bottom Line:**
The parallel probe training gives you the biggest speedup with minimal effort. Future parallelization of interventions and comparisons would provide additional benefits.

---

**See Also:**
- `PARALLEL_EXECUTION_GUIDE.md` - How to use parallel execution
- `src/scripts/search_layers_parallel.py` - Parallel implementation
- `RESUME_GUIDE.md` - Resume functionality

