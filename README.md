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

## Summary
This project combines **probing and causal tracing** to identify and validate where negation emerges in small LMs.  
The ultimate goal: improve reliability and interpretability of lightweight models suitable for **low-compute, real-world deployment**.
