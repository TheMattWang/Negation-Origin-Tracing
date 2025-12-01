# Implementation Plan: Layer Search, Analysis, and Transfer Evaluation

## Phase 1: Automated Layer Search and Analysis

### 1.1 Automated Layer Search Script
**File:** `src/scripts/search_layers.py`

**Functionality:**
- Train probes on all layers (0-5 for DistilBERT) with different pooling strategies
- Save results for each layer with metrics (accuracy, AUROC, F1)
- Organize results in structured format (JSON/CSV)
- Support parallel training or sequential with progress tracking

**Key Features:**
- Iterate through all layers automatically
- Test all pooling strategies (cls, mean, token) for each layer
- Save checkpoints and metrics for each probe
- Generate summary report comparing all layers

**Output:**
- `experiments/layer_search/results_summary.json` - All results
- Individual checkpoints for each layer: `experiments/layer_search/layer_{i}_pooling_{strategy}/`
- CSV file with layer-wise metrics

### 1.2 Analysis and Visualization Tools
**File:** `src/scripts/analyze_probes.py` or `src/engine/visualization.py`

**Functionality:**
- Load probe results from layer search
- Generate visualizations:
  - Line plots: Accuracy/AUROC by layer
  - Bar charts: Comparison across pooling strategies
  - Heatmaps: Layer × pooling strategy performance
  - Confusion matrices: Split by negated vs non-negated

**Visualizations:**
1. **Layer Performance Plot**: X=layer, Y=accuracy/AUROC, lines for each pooling strategy
2. **Pooling Strategy Comparison**: Bar chart comparing cls/mean/token across layers
3. **Negation Split Analysis**: Separate plots for negated vs non-negated examples
4. **Best Layer Identification**: Highlight peak performance layers

**Dependencies:**
- matplotlib or plotly for plotting
- pandas for data manipulation

### 1.3 Complete Experiment Pipeline
**File:** `src/scripts/run_full_experiment.py`

**Functionality:**
- End-to-end script that:
  1. Trains probes on all layers (or uses existing results)
  2. Identifies best layer(s) from probe results
  3. Runs causal interventions on identified layers
  4. Generates analysis and visualizations
  5. Produces final report

**Workflow:**
```
1. Train probes → 2. Analyze results → 3. Identify key layers → 
4. Run interventions → 5. Verify causality → 6. Generate report
```

**Output:**
- Complete experiment report with:
  - Probe performance summary
  - Identified "negation layers"
  - Intervention results
  - Causal verification
  - Visualizations

### 1.4 Results Interpretation Tools
**File:** `src/engine/interpretation.py`

**Functionality:**
- Identify "negation layers" based on:
  - High probe accuracy/AUROC
  - High label-flip rates under activation patching
  - Significant logit deltas under ablation
- Statistical significance testing
- Layer ranking and importance scoring

**Key Functions:**
- `identify_negation_layers()`: Returns ranked list of layers
- `verify_causality()`: Checks if interventions confirm probe findings
- `generate_interpretation_report()`: Creates human-readable summary

---

## Phase 2: Transfer Evaluation and Comparison

### 2.1 HANS/ANLI Dataset Support
**Files:** 
- `src/data/download.py` (update)
- `src/datasets/dataset.py` (add HANS/ANLI dataset classes)

**Functionality:**
- Download HANS and ANLI datasets
- Create dataset classes compatible with existing pipeline
- Handle different data formats (HANS has different structure)
- Save to parquet format for consistency

**HANS Dataset:**
- Heuristic Analysis for NLI Systems
- Tests for spurious correlations
- Useful for transfer testing

**ANLI Dataset:**
- Adversarial NLI
- More challenging examples
- Tests robustness

**Implementation:**
- Add download functions to `download.py`
- Create `HANSDataset` and `ANLIDataset` classes
- Ensure compatibility with `NOTDataModule`

### 2.2 Transfer Evaluation Script
**File:** `src/scripts/evaluate_transfer.py`

**Functionality:**
- Load trained models (probe and fine-tuned)
- Evaluate on multiple datasets:
  - SST-2 (training set)
  - CSD Negation (already supported)
  - HANS (new)
  - ANLI (new)
- Compare performance:
  - Probe vs fine-tuned on each dataset
  - Negated vs non-negated performance
  - Error analysis

**Key Metrics:**
- Accuracy, AUROC, F1 for each dataset
- Performance drop from training to transfer
- Negation-specific performance

**Output:**
- Transfer evaluation report (JSON/CSV)
- Per-dataset performance metrics
- Comparison tables

### 2.3 Comparison Analysis Tools
**File:** `src/engine/comparison.py`

**Functionality:**
- Compare probe vs fine-tuned models:
  - Overall performance
  - Negated vs non-negated breakdown
  - Error patterns (confusion matrices)
  - Generalization gap
- Statistical tests for significance
- Layer consistency analysis (same best layer across datasets?)

**Key Functions:**
- `compare_models()`: Probe vs fine-tuned comparison
- `analyze_generalization()`: Transfer performance analysis
- `layer_consistency_check()`: Check if best layer is consistent
- `error_analysis()`: Detailed error pattern comparison

### 2.4 Comparison Visualizations
**File:** `src/scripts/visualize_comparisons.py` or extend `src/engine/visualization.py`

**Functionality:**
- Generate comparison plots:
  1. **Model Comparison**: Probe vs fine-tuned performance across datasets
  2. **Transfer Performance**: Performance drop visualization
  3. **Negation Analysis**: Negated vs non-negated comparison
  4. **Layer Consistency**: Best layer across datasets (if consistent)
  5. **Error Patterns**: Confusion matrices side-by-side

**Visualizations:**
- Multi-dataset performance bar charts
- Transfer gap plots (training vs transfer)
- Negation-specific performance heatmaps
- Error distribution comparisons

---

## Implementation Order

### Phase 1 (Current Focus)
1. ✅ Create `search_layers.py` - Automated layer search
2. ✅ Create visualization module - Plot generation
3. ✅ Create `run_full_experiment.py` - End-to-end pipeline
4. ✅ Create `interpretation.py` - Results interpretation

### Phase 2 (After Phase 1)
1. ✅ Add HANS/ANLI to `download.py`
2. ✅ Create dataset classes for HANS/ANLI
3. ✅ Create `evaluate_transfer.py` - Transfer evaluation
4. ✅ Create `comparison.py` - Analysis tools
5. ✅ Extend visualization for comparisons

---

## File Structure

```
src/
├── scripts/
│   ├── search_layers.py          # NEW: Automated layer search
│   ├── run_full_experiment.py    # NEW: Complete pipeline
│   ├── analyze_probes.py         # NEW: Analysis script
│   ├── evaluate_transfer.py       # NEW: Transfer evaluation
│   └── visualize_comparisons.py  # NEW: Comparison plots
├── engine/
│   ├── visualization.py           # NEW: Plotting utilities
│   ├── interpretation.py          # NEW: Results interpretation
│   └── comparison.py              # NEW: Comparison analysis
└── datasets/
    └── dataset.py                 # UPDATE: Add HANS/ANLI classes
```

---

## Dependencies to Add

- `matplotlib` or `plotly` for visualization
- `seaborn` for enhanced plots (optional)
- `pandas` for data manipulation
- `scipy` for statistical tests (optional)

---

## Success Criteria

**Phase 1:**
- Can automatically search all layers and identify best ones
- Can visualize probe performance across layers
- Can run complete experiment from start to finish
- Can interpret results and identify "negation layers"

**Phase 2:**
- Can evaluate on HANS/ANLI datasets
- Can compare probe vs fine-tuned performance
- Can analyze transfer/generalization
- Can visualize all comparisons

