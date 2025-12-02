# Finding Where Negation Lives: Sparse Probing + Causal Tracing in Small Language Models

## Motivation
Despite strong benchmark scores, LLMs still miss basic semantic phenomena—counting, compositionality, and especially **negation**. These errors show up in everyday tasks like sentiment analysis, where *“not good”* is misread as *“good.”*  
The problem is even more pronounced in smaller models, which we increasingly rely on for **local, low-compute deployment**.  

To make compact LMs reliable on real hardware, we need to know **where negation is represented inside the network** and whether those features actually drive predictions.  
This project targets that gap by localizing negation in small LMs and testing causality—aiming for insights that translate into lightweight, deployable fixes.

---

## Problem
Negation is fundamental to human language—it lets us set boundaries, reject ideas, and express disagreement.  
Yet even large LLMs often fail to handle it correctly. They may correctly identify that *“good”* is positive but fail when negation is introduced, incorrectly assigning *“not good”* as positive.  

Despite knowing that this happens, **we don’t yet know why**. This project aims to uncover where and how negation is encoded in smaller, interpretable models.

---

## Idea
We test **where negation lives** in small, CPU-friendly LMs by:
1. Training **sparse linear probes** across layers to detect negation features.
2. Running **causal tests**—activation patching and targeted ablations—to verify whether these identified representations actually influence model decisions on sentiment polarity.

---

Here’s an updated version that accurately reflects your current plan (using **CSD Negation** for evaluation while noting bias considerations):

---

## Dataset

We study **polarity classification** with and without negation using:

* **SST-2**: The standard sentiment classification dataset from GLUE, used for training and baseline evaluation.
* **CSD Negation Subset**: A curated contrastive dataset from the *Contrastive Sentiment Data (CSD)* corpus, containing human-verified negated counterparts of sentiment phrases (e.g., *“good” → “not good”*, *“bad” → “not bad”*). Used primarily for **testing model robustness to negation**.

**Data splits**

| Dataset      | Train  | Dev   | Test  |
| ------------ | ------ | ----- | ----- |
| SST-2        | 10,000 | 2,000 | 2,000 |
| CSD Negation | —      | —     | 2,000 |

**Optional transfer tests**: 1,000–2,000 examples from **HANS** or **ANLI**.

We report: percentage of negated samples, average sentence length, and vocabulary size.
⚠️ *Note:* The CSD Negation subset is balanced across negation types (e.g., “not good” and “not bad”), which can introduce mild bias toward negated negatives. For naturalistic analysis, we rely on SST-2 and use CSD primarily as a controlled contrastive benchmark.

---

## Method
- **Base model:** `DistilBERT-base-uncased` (frozen).  
- **Feature extraction:** Layerwise hidden states (token-level and [CLS]).  
- **Linear probes:** Sparse probes trained per layer to predict negation-sensitive polarity.  
  - Pooling strategies: `[CLS]`, mean-pooling, and token around “not.”  

### Causal Interventions
1. **Activation patching:** Swap hidden states between negated/non-negated pairs at specific layers/positions.  
2. **Targeted ablations:** Zero out or project out probe-identified dimensions.  
3. **Controls:**  
   - Label-shuffled probes  
   - Representation-shuffled probes  
   - Random-site/dimension interventions  

**Metrics:** 
- Probe accuracy and AUROC by layer
- Label-flip rates under activation patching
- Logit deltas under targeted ablation
- Confusion matrices (split by negated vs. non-negated)
- Macro-F1 scores

Results are compared against a **BERT-base reference** (planned for Phase 2).

---

## Evaluation
We evaluate with:
- **Baselines:** Fine-tuned SST-2 and negation-augmented model.
- **Metrics:** Accuracy, Macro-F1, and confusion matrices (split by negated vs. non-negated examples).  
- **Interpretability:**  
  - Layerwise probe accuracy  
  - Label-flip rate under activation patching  
- **Comparisons:** Against `BERT-base-uncased` to contextualize performance and robustness.

---

---

## Usage

> **⚠️ Important for Colab Users**: If running on Google Colab, see [`COLAB_QUICK_START.md`](COLAB_QUICK_START.md) for deadlock-free execution instructions.

### Setup

#### Option 1: Local Setup

1. **Install dependencies:**
   ```bash
   conda env create -f environment.yml
   conda activate not
   pip install -r requirements.txt
   ```

2. **Download data:**
   ```bash
   python src/data/download.py
   ```
   This will download SST-2 and CSD Negation datasets to `data/raw/`.

#### Option 2: Google Colab (Recommended)

1. **Clone repository in Colab:**
   ```python
   !git clone https://github.com/TheMattWang/Negation-Origin-Tracing.git
   %cd Negation-Origin-Tracing
   ```

2. **Install dependencies:**
   ```python
   !pip install -q torch lightning transformers datasets pandas pyarrow matplotlib seaborn scikit-learn tqdm tensorboardX
   ```

3. **Download data:**
   ```python
   !python src/data/download.py
   ```

4. **Run layer search (parallel - faster!):**
   ```python
   # Parallel execution with 3 workers (3x faster, ~1-1.5 hours)
   !python src/scripts/search_layers_parallel.py \
       --model_name distilbert-base-uncased \
       --data_dir data/raw \
       --output_dir experiments/layer_search \
       --layers all \
       --pooling_strategies all \
       --parallel_mode pooling \
       --parallel_workers 3 \
       --colab_safe \
       --max_epochs 10
   ```
   
   **⚠️ Important**: Use `--colab_safe` flag to avoid deadlocks on single GPU. See [`COLAB_QUICK_START.md`](COLAB_QUICK_START.md) for details.

**Alternative: Use notebooks** (step-by-step):
   - `notebooks/00_setup_colab.ipynb` - Setup environment and install dependencies
   - `notebooks/01_download_data.ipynb` - Download datasets
   - `notebooks/02_layer_search.ipynb` - Automated layer search
   - `notebooks/03_visualize_results.ipynb` - Generate visualizations
   - `notebooks/04_full_experiment.ipynb` - Complete experiment pipeline

   See `COLAB_SETUP.md` for detailed notebook instructions.

### Quick Start: Complete Experiment Pipeline

Run the complete experiment pipeline (recommended):

```bash
python src/scripts/run_full_experiment.py \
    --output_dir experiments/full_experiment \
    --layers all \
    --pooling_strategies all \
    --max_epochs 10
```

This will:
1. Train probes on all layers with all pooling strategies
2. Generate visualizations
3. Run causal interventions on top layers
4. Generate interpretation report

**🔄 Resume from Checkpoint:** If your experiment is interrupted (e.g., Colab disconnect), simply re-run the same command. The script will automatically skip completed experiments and continue from where it left off. See [`RESUME_GUIDE.md`](RESUME_GUIDE.md) for details.

### Automated Layer Search

Search across all layers to find where negation is represented:

```bash
python src/scripts/search_layers.py \
    --output_dir experiments/layer_search \
    --layers all \
    --pooling_strategies all \
    --max_epochs 10
```

This trains probes on all layers and saves results for analysis.

**Resume capability:** Results are saved after each experiment. If interrupted, re-run the same command to continue.

**⚡ Faster with Parallel Execution:** Run 3 pooling strategies in parallel (~3x speedup):

```bash
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/layer_search \
    --layers all \
    --pooling_strategies all \
    --parallel_workers 3 \
    --parallel_mode pooling
```

See [`PARALLEL_EXECUTION_GUIDE.md`](PARALLEL_EXECUTION_GUIDE.md) for details on parallel execution modes and performance optimization.

### Training Individual Models

#### Fine-tune Full Model

Fine-tune DistilBERT on SST-2 for sentiment classification:

```bash
python src/scripts/train.py \
    --mode finetune \
    --experiment_name sst2_finetune \
    --max_epochs 3 \
    --batch_size 32 \
    --lr 2e-5 \
    --data_dir data/raw
```

#### Train Linear Probes

Train linear probes on frozen model representations:

**Probe a specific layer:**
```bash
python src/scripts/train.py \
    --mode probe \
    --experiment_name probe_layer5 \
    --probe_layer 5 \
    --pooling_strategy cls \
    --probe_lr 1e-3 \
    --batch_size 32 \
    --max_epochs 10
```

**Probe all layers:**
```bash
python src/scripts/train.py \
    --mode probe \
    --experiment_name probe_all_layers \
    --pooling_strategy mean \
    --probe_lr 1e-3
```

**Available pooling strategies:**
- `cls`: Use [CLS] token representation
- `mean`: Mean pooling over sequence (masked)
- `token`: Pool around "not" token position

### Visualization and Analysis

Generate visualizations from layer search results:

```bash
python src/engine/visualization.py \
    --results_path experiments/layer_search/results_summary.json \
    --output_dir experiments/layer_search/visualizations
```

This creates:
- Layer performance plots (accuracy/AUROC by layer)
- Pooling strategy comparisons
- Performance heatmaps
- Best layers visualization

### Causal Interventions

Run causal intervention experiments:

```bash
python src/scripts/run_interventions.py \
    --model_ckpt experiments/runs/probe_layer5/checkpoints/best.ckpt \
    --mode probe \
    --probe_layer 5 \
    --data_path data/raw/test/negation.parquet \
    --intervention_type all \
    --layers all \
    --output_dir experiments/interventions
```

Intervention types:
- `activation_patching`: Swap hidden states between negated/non-negated pairs
- `ablation`: Zero out or project out specific dimensions
- `control`: Random and shuffled interventions for comparison

### Results Interpretation

Generate interpretation report from probe and intervention results:

```bash
python src/engine/interpretation.py \
    --probe_results experiments/layer_search/results_summary.json \
    --intervention_results experiments/interventions/intervention_results.json \
    --output experiments/interpretation_report.json \
    --top_k 5
```

This identifies:
- Most important layers for negation (composite scoring)
- Causal verification (whether layers actually drive predictions)
- Recommendations for further analysis

### Evaluation

Test a trained model:

```bash
python src/scripts/test.py \
    --ckpt_path experiments/runs/sst2_finetune/checkpoints/best.ckpt \
    --data_dir data/raw \
    --batch_size 32
```

### Inference

Run inference on new text:

```bash
# Single text
python src/scripts/predict.py \
    --ckpt_path experiments/runs/sst2_finetune/checkpoints/best.ckpt \
    --text "This movie is not good"

# From file
python src/scripts/predict.py \
    --ckpt_path experiments/runs/sst2_finetune/checkpoints/best.ckpt \
    --text_file input.txt \
    --output_file predictions.txt
```

### Command-Line Arguments

**Common arguments:**
- `--mode`: `finetune` or `probe`
- `--experiment_name`: Name for logging/checkpoint folders
- `--model_name`: HuggingFace model (default: `distilbert-base-uncased`)
- `--data_dir`: Data directory (default: `data/raw`)
- `--batch_size`: Batch size (default: 32)
- `--max_epochs`: Training epochs (default: 3)
- `--lr`: Learning rate for finetune (default: 2e-5)
- `--seed`: Random seed (default: 42)

**Probe-specific arguments:**
- `--probe_layer`: Layer index to probe (None = all layers)
- `--pooling_strategy`: `cls`, `mean`, or `token` (default: `cls`)
- `--probe_lr`: Learning rate for probes (default: 1e-3)

**Hardware:**
- `--devices`: Number of GPUs/CPUs (default: 1)
- `--precision`: Training precision `16` or `32` (default: 32)

### Output Structure

```
experiments/
├── layer_search/
│   ├── results_summary.json          # All probe results
│   ├── results_summary.csv           # CSV format
│   ├── visualizations/               # Generated plots
│   └── layer_*/pooling_*/            # Individual probe checkpoints
├── full_experiment/
│   ├── results_summary.json
│   ├── visualizations/
│   ├── interventions/                # Intervention results
│   └── interpretation_report.json    # Final analysis
└── runs/
    └── <experiment_name>/
        ├── checkpoints/
        │   ├── best.ckpt
        │   ├── last.ckpt
        │   └── epoch=XX-val_loss=X.XXX.ckpt
        └── events.out.tfevents.*     # TensorBoard logs
```

View training metrics:
```bash
tensorboard --logdir experiments/runs
```

### Resume from Checkpoint

All experiment scripts now support automatic checkpointing and resume functionality. If your experiment is interrupted (Colab disconnect, timeout, crash), you can simply re-run the same command and it will pick up where it left off.

**How it works:**
- Results are saved after each completed experiment
- On restart, the script loads existing results and skips completed work
- No data loss from interruptions

**Example:**
```bash
# First run (completes 12 out of 18 experiments before interruption)
python src/scripts/run_full_experiment.py --output_dir experiments/my_exp

# After interruption, run the SAME command
python src/scripts/run_full_experiment.py --output_dir experiments/my_exp
# Will automatically resume and run only the remaining 6 experiments
```

**For Google Colab:**
1. Save to Google Drive: `--output_dir /content/drive/MyDrive/experiments/my_exp`
2. After disconnection: Remount Drive and re-run the same command
3. Script automatically resumes from checkpoint

See [`RESUME_GUIDE.md`](RESUME_GUIDE.md) for detailed instructions and troubleshooting.

### Key Scripts

| Script | Purpose |
|--------|---------|
| `src/scripts/run_full_experiment.py` | Complete end-to-end pipeline |
| `src/scripts/search_layers.py` | Automated layer search (sequential) |
| `src/scripts/search_layers_colab.py` | **Colab-optimized layer search (deadlock-free)** |
| `src/scripts/search_layers_parallel.py` | Parallel layer search (multi-GPU) |
| `src/scripts/train.py` | Train individual models/probes |
| `src/scripts/run_interventions.py` | Run causal interventions |
| `src/scripts/test.py` | Evaluate trained models |
| `src/scripts/predict.py` | Inference on new text |
| `src/engine/visualization.py` | Generate plots |
| `src/engine/interpretation.py` | Interpret results |

---

## Troubleshooting

### Deadlocks on Google Colab

**Problem**: Experiments hang or freeze when running on Colab.

**Solution**: Use the Colab-optimized script:
```bash
python src/scripts/search_layers_colab.py --layers all
```

This script automatically:
- Sets `num_workers=0` (prevents nested multiprocessing deadlocks)
- Runs experiments sequentially (stable on single GPU)
- Manages GPU memory efficiently
- Provides better error handling

**See**: [`COLAB_DEADLOCK_FIX.md`](COLAB_DEADLOCK_FIX.md) for technical details and [`COLAB_QUICK_START.md`](COLAB_QUICK_START.md) for usage guide.

### Out of Memory (OOM) Errors

**Solutions**:
1. Reduce batch size: `--batch_size 16` (or 8)
2. Use mixed precision: `--precision 16`
3. Clear GPU cache between runs:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

### Colab Disconnects

**Solution**: Results are auto-saved. Just re-run the same command to resume:
```bash
# Automatically resumes from last checkpoint
python src/scripts/search_layers_colab.py --layers all --output_dir /content/drive/MyDrive/results
```

---

## Project Status

### ✅ Completed (Phase 1)
- Data pipeline (SST-2, CSD Negation)
- Model training (finetune and probe modes)
- Automated layer search across all layers
- Visualization tools (plots, heatmaps, comparisons)
- Causal interventions (activation patching, ablation, controls)
- Metrics (accuracy, AUROC, F1, label-flip rates, logit deltas)
- Results interpretation and layer identification
- Complete experiment pipeline
- Jupyter notebooks for Google Colab

### 🚧 Planned (Phase 2)
- HANS/ANLI dataset support
- Transfer evaluation across datasets
- Probe vs fine-tuned model comparison
- BERT-base reference comparison
- Sparse probe implementation (L1/L2 regularization)
- Advanced visualization and comparison tools

## Summary
This project combines **probing and causal tracing** to identify and validate where negation emerges in small LMs.  
The ultimate goal: improve reliability and interpretability of lightweight models suitable for **low-compute, real-world deployment**.

## Citation

If you use this code, please cite:
```bibtex
@misc{negation-origin-tracing,
  title={Finding Where Negation Lives: Sparse Probing + Causal Tracing in Small Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/TheMattWang/Negation-Origin-Tracing}
}
```
