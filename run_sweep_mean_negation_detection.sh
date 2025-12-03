#!/bin/bash
# =============================================================================
# Sweep: MEAN Pooling - Negation Detection Task
# =============================================================================
# Train probes to detect negation (label 0 = no negation, label 1 = has negation)
# Uses JinaAI negation dataset converted to detection format
#
# Run this in one Colab runtime while running cls/token in others
#
# For Google Drive persistence, set DRIVE_OUTPUT before running:
#   export DRIVE_OUTPUT="/content/drive/MyDrive/NOT_results"
#   ./run_sweep_mean_negation_detection.sh

set -o pipefail

MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
DATA_DIR="${DATA_DIR:-data/raw}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
SEED="${SEED:-42}"

# Use Google Drive if DRIVE_OUTPUT is set, otherwise local
if [ -n "$DRIVE_OUTPUT" ]; then
    OUTPUT_DIR="$DRIVE_OUTPUT/sweep_mean_negation_detection"
    echo "Saving to Google Drive: $OUTPUT_DIR"
else
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/sweep_mean_negation_detection}"
    echo "Saving locally: $OUTPUT_DIR"
fi

echo "========================================"
echo "Layer Sweep: MEAN Pooling - NEGATION DETECTION"
echo "========================================"
echo "Task: Binary classification (0=no negation, 1=has negation)"
echo "Layers: 0-5"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

# Activate conda if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate not 2>/dev/null || true
fi

RESULTS_FILE="$OUTPUT_DIR/results_mean_negation_detection.json"
echo "[]" > "$RESULTS_FILE"

for layer in 0 1 2 3 4 5; do
    echo ""
    echo "[Layer $layer] MEAN pooling - Negation Detection..."
    LOG="$OUTPUT_DIR/logs/layer${layer}_mean.log"
    CKPT_DIR="$OUTPUT_DIR/checkpoints/layer${layer}_mean"
    mkdir -p "$CKPT_DIR"
    
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_LAUNCH_BLOCKING=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    python -c "
import os, sys, json, warnings
warnings.filterwarnings('ignore')

# Enable tqdm in Colab/notebooks
os.environ['TQDM_DISABLE'] = '0'

import torch
torch.set_num_threads(1)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm.auto import tqdm

sys.path.insert(0, '.')
from src.models import BaseModule
from src.datasets.dataset import SentimentDataset

torch.manual_seed($SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all($SEED)

model = BaseModule(
    model_name='$MODEL_NAME',
    num_labels=2,
    mode='probe',
    probe_layer=$layer,
    pooling_strategy='mean',
    probe_lr=1e-3,
)

tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')

# Load negation detection dataset (converted from JinaAI triplets)
train_ds = SentimentDataset('$DATA_DIR/train/negation_detection.parquet', tokenizer, 128)
val_ds = SentimentDataset('$DATA_DIR/validation/negation_detection.parquet', tokenizer, 128)
test_ds = SentimentDataset('$DATA_DIR/test/negation_detection.parquet', tokenizer, 128)

train_dl = DataLoader(train_ds, batch_size=$BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_dl = DataLoader(val_ds, batch_size=$BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_dl = DataLoader(test_ds, batch_size=$BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

ckpt_cb = ModelCheckpoint(dirpath='$CKPT_DIR', filename='best-{val_acc:.3f}', monitor='val_acc', mode='max', save_top_k=1)
early_cb = EarlyStopping(monitor='val_loss', mode='min', patience=3)

# Progress bar callback for better display
progress_cb = TQDMProgressBar(refresh_rate=10)

trainer = L.Trainer(
    max_epochs=$MAX_EPOCHS,
    accelerator='auto',
    devices=1,
    callbacks=[ckpt_cb, early_cb, progress_cb],
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
    num_sanity_val_steps=0,
)

trainer.fit(model, train_dl, val_dl)

# Check if test set has valid labels
has_test_labels = getattr(test_ds, 'has_labels', True)

if has_test_labels:
    # Use test set for evaluation (negation detection has valid test labels)
    results = trainer.test(model, test_dl, ckpt_path='best', verbose=False)
    r = results[0] if results else {}
    note = 'test_set_evaluation'
else:
    # Fallback to validation metrics if test set has no labels
    print('  ⚠ Test set has no labels, using validation metrics')
    results = trainer.test(model, val_dl, ckpt_path='best', verbose=False)
    r = results[0] if results else {}
    note = 'validation_fallback_no_test_labels'

out = {
    'layer': $layer,
    'pooling': 'mean',
    'task': 'negation_detection',
    'test_acc': r.get('test_acc', 0),
    'test_auroc': r.get('test_auroc', 0),
    'checkpoint': ckpt_cb.best_model_path,
    'note': note,
}
print('RESULT:' + json.dumps(out))
" 2>&1 | tee -a "$LOG"
    
    # Note: Progress bars display in terminal, logs saved to file
    
    # Extract and save result
    RESULT=$(grep "^RESULT:" "$LOG" | tail -1 | sed 's/^RESULT://')
    if [ -n "$RESULT" ]; then
        python -c "
import json
with open('$RESULTS_FILE', 'r') as f: results = json.load(f)
results.append($RESULT)
with open('$RESULTS_FILE', 'w') as f: json.dump(results, f, indent=2)
"
        ACC=$(echo "$RESULT" | python -c "import sys,json; print(f\"{json.load(sys.stdin).get('test_acc',0):.4f}\")")
        AUROC=$(echo "$RESULT" | python -c "import sys,json; print(f\"{json.load(sys.stdin).get('test_auroc',0):.4f}\")")
        echo "  ✓ Layer $layer MEAN: acc=$ACC, auroc=$AUROC"
    else
        echo "  ✗ Layer $layer MEAN: failed"
    fi
    
    # GPU cleanup
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
    sleep 2
done

echo ""
echo "========================================"
echo "MEAN Negation Detection Sweep Complete!"
echo "========================================"
echo "Results: $RESULTS_FILE"

python -c "
import json
with open('$RESULTS_FILE') as f: results = json.load(f)
if results:
    best = max(results, key=lambda x: x.get('test_auroc', 0))
    print(f\"Best: Layer {best['layer']} -> AUROC {best.get('test_auroc', 0):.4f}, Acc {best.get('test_acc', 0):.4f}\")
    print(f\"\\nAll layers (sorted by AUROC):\")
    for r in sorted(results, key=lambda x: x.get('test_auroc', 0), reverse=True):
        print(f\"  Layer {r['layer']}: AUROC={r.get('test_auroc', 0):.4f}, Acc={r.get('test_acc', 0):.4f}\")
"

