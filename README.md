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

**Metrics:** Probe accuracy by layer, label-flip rates, and logit deltas under intervention.  
Results are compared against a **BERT-base reference**.

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

### Setup

1. **Install dependencies:**
   ```bash
   conda env create -f environment.yml
   conda activate not
   pip install torch lightning transformers datasets pandas pyarrow
   ```

2. **Download data:**
   ```bash
   python src/data/download.py
   ```
   This will download SST-2 and CSD Negation datasets to `data/raw/`.

### Training

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
└── runs/
    └── <experiment_name>/
        ├── checkpoints/
        │   ├── best.ckpt
        │   ├── last.ckpt
        │   └── epoch=XX-val_loss=X.XXX.ckpt
        └── events.out.tfevents.*  # TensorBoard logs
```

View training metrics:
```bash
tensorboard --logdir experiments/runs
```

---

## Summary
This project combines **probing and causal tracing** to identify and validate where negation emerges in small LMs.  
The ultimate goal: improve reliability and interpretability of lightweight models suitable for **low-compute, real-world deployment**.
