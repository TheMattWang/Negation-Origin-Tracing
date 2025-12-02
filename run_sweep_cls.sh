#!/bin/bash
# =============================================================================
# Sweep: CLS Pooling Only (Run in its own Colab/terminal)
# =============================================================================
# Run this in one Colab runtime while running mean/token in others

set -o pipefail

MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
DATA_DIR="${DATA_DIR:-data/raw}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
SEED="${SEED:-42}"
OUTPUT_DIR="${OUTPUT_DIR:-experiments/sweep_cls}"

echo "========================================"
echo "Layer Sweep: CLS Pooling"
echo "========================================"
echo "Layers: 0-5"
echo "Output: $OUTPUT_DIR"
echo ""

mkdir -p "$OUTPUT_DIR/logs"

# Activate conda if available
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate not 2>/dev/null || true
fi

RESULTS_FILE="$OUTPUT_DIR/results_cls.json"
echo "[]" > "$RESULTS_FILE"

for layer in 0 1 2 3 4 5; do
    echo ""
    echo "[Layer $layer] CLS pooling..."
    LOG="$OUTPUT_DIR/logs/layer${layer}_cls.log"
    
    CUDA_VISIBLE_DEVICES=0 \
    CUDA_LAUNCH_BLOCKING=1 \
    OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    python -c "
import os, sys, json, warnings
warnings.filterwarnings('ignore')

import torch
torch.set_num_threads(1)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

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
    pooling_strategy='cls',
    probe_lr=1e-3,
)

tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')

train_ds = SentimentDataset('$DATA_DIR/train/sst.parquet', tokenizer, 128)
val_ds = SentimentDataset('$DATA_DIR/validation/sst.parquet', tokenizer, 128)
test_ds = SentimentDataset('$DATA_DIR/test/sst.parquet', tokenizer, 128)

train_dl = DataLoader(train_ds, batch_size=$BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
val_dl = DataLoader(val_ds, batch_size=$BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
test_dl = DataLoader(test_ds, batch_size=$BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

ckpt_dir = '$OUTPUT_DIR/checkpoints/layer${layer}_cls'
os.makedirs(ckpt_dir, exist_ok=True)

ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir, filename='best-{val_acc:.3f}', monitor='val_acc', mode='max', save_top_k=1)
early_cb = EarlyStopping(monitor='val_loss', mode='min', patience=3)

trainer = L.Trainer(
    max_epochs=$MAX_EPOCHS,
    accelerator='auto',
    devices=1,
    callbacks=[ckpt_cb, early_cb],
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,
    num_sanity_val_steps=0,
)

trainer.fit(model, train_dl, val_dl)
results = trainer.test(model, test_dl, ckpt_path='best', verbose=False)
r = results[0] if results else {}

out = {'layer': $layer, 'pooling': 'cls', 'test_acc': r.get('test_acc', 0), 'test_auroc': r.get('test_auroc', 0)}
print('RESULT:' + json.dumps(out))
" 2>&1 | tee "$LOG"
    
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
        echo "  ✓ Layer $layer CLS: acc=$ACC"
    else
        echo "  ✗ Layer $layer CLS: failed"
    fi
    
    # GPU cleanup
    python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
    sleep 2
done

echo ""
echo "========================================"
echo "CLS Sweep Complete!"
echo "========================================"
echo "Results: $RESULTS_FILE"

python -c "
import json
with open('$RESULTS_FILE') as f: results = json.load(f)
if results:
    best = max(results, key=lambda x: x.get('test_auroc', 0))
    print(f\"Best: Layer {best['layer']} -> AUROC {best.get('test_auroc', 0):.4f}\")
"

