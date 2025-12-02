#!/bin/bash
# =============================================================================
# DEADLOCK-PROOF Sequential Layer Sweep
# =============================================================================
# This script is designed to NEVER deadlock on single-GPU systems.
#
# Key fixes:
# 1. CUDA_LAUNCH_BLOCKING=1 - Forces synchronous GPU operations
# 2. OMP_NUM_THREADS=1 - Prevents OpenMP threading issues  
# 3. num_workers=0 - No multiprocessing in DataLoader
# 4. pin_memory=False - Avoids CUDA memory pinning issues
# 5. Sequential execution - Only one job at a time
# 6. Timeout detection - Catches any remaining hangs
# =============================================================================

set -o pipefail

# Configuration
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
DATA_DIR="${DATA_DIR:-data/raw}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
SEED="${SEED:-42}"
TIMEOUT="${TIMEOUT:-30m}"  # 30 min timeout per experiment

# Output directory
OUTPUT_DIR="${OUTPUT_DIR:-experiments/sweep_$(date +%Y%m%d_%H%M%S)}"

# Layers and pooling
LAYERS="${LAYERS:-0 1 2 3 4 5}"
POOLING="${POOLING:-cls mean token}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}Deadlock-Proof Sequential Layer Sweep${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR/checkpoints"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup Python
PYTHON_CMD="${PYTHON_CMD:-python}"
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    conda activate not 2>/dev/null || true
fi

echo "Model: $MODEL_NAME"
echo "Layers: $LAYERS"
echo "Pooling: $POOLING"
echo "Output: $OUTPUT_DIR"
echo "Timeout: $TIMEOUT per run"
echo ""

# Results file
RESULTS_FILE="$OUTPUT_DIR/results.json"
echo "[]" > "$RESULTS_FILE"

# Count experiments
LAYER_ARR=($LAYERS)
POOL_ARR=($POOLING)
TOTAL=$((${#LAYER_ARR[@]} * ${#POOL_ARR[@]}))
CURRENT=0
SUCCESS=0
FAILED=0

echo -e "${BLUE}Starting $TOTAL experiments...${NC}"
echo ""

for layer in $LAYERS; do
    for pool in $POOLING; do
        CURRENT=$((CURRENT + 1))
        RUN_ID="layer${layer}_${pool}"
        LOG="$OUTPUT_DIR/logs/${RUN_ID}.log"
        CKPT_DIR="$OUTPUT_DIR/checkpoints/${RUN_ID}"
        
        echo -e "${BLUE}[$CURRENT/$TOTAL]${NC} Layer $layer, Pooling: $pool"
        
        mkdir -p "$CKPT_DIR"
        
        # Run with all deadlock prevention measures
        timeout $TIMEOUT env \
            CUDA_VISIBLE_DEVICES=0 \
            CUDA_LAUNCH_BLOCKING=1 \
            OMP_NUM_THREADS=1 \
            MKL_NUM_THREADS=1 \
            TOKENIZERS_PARALLELISM=false \
            PYTHONUNBUFFERED=1 \
            $PYTHON_CMD -c "
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

# Force spawn method for multiprocessing (safer with CUDA)
import multiprocessing
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

import torch
torch.set_num_threads(1)

# Disable CUDA multiprocessing
if torch.cuda.is_available():
    torch.cuda.set_device(0)

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

sys.path.insert(0, '$SCRIPT_DIR')
from src.models import BaseModule
from src.datasets.dataset import SentimentDataset
from transformers import AutoTokenizer

# Set seeds
torch.manual_seed($SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all($SEED)

print('Initializing model...')
model = BaseModule(
    model_name='$MODEL_NAME',
    num_labels=2,
    mode='probe',
    probe_layer=$layer,
    pooling_strategy='$pool',
    probe_lr=1e-3,
)

print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_NAME')

print('Loading datasets...')
train_dataset = SentimentDataset(
    data_path='$DATA_DIR/train/sst.parquet',
    tokenizer=tokenizer,
    max_length=128,
)
val_dataset = SentimentDataset(
    data_path='$DATA_DIR/validation/sst.parquet',
    tokenizer=tokenizer,
    max_length=128,
)
test_dataset = SentimentDataset(
    data_path='$DATA_DIR/test/sst.parquet',
    tokenizer=tokenizer,
    max_length=128,
)

# Create DataLoaders with NO multiprocessing, NO pin_memory
print('Creating dataloaders (num_workers=0, pin_memory=False)...')
train_loader = DataLoader(
    train_dataset,
    batch_size=$BATCH_SIZE,
    shuffle=True,
    num_workers=0,  # NO multiprocessing
    pin_memory=False,  # NO CUDA memory pinning
    persistent_workers=False,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=$BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=$BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
)

print(f'Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}')

# Callbacks
checkpoint_cb = ModelCheckpoint(
    dirpath='$CKPT_DIR',
    filename='best-{val_acc:.3f}',
    monitor='val_acc',
    mode='max',
    save_top_k=1,
)

early_stop_cb = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=3,
)

# Trainer with minimal features
print('Initializing trainer...')
trainer = L.Trainer(
    max_epochs=$MAX_EPOCHS,
    accelerator='auto',
    devices=1,
    callbacks=[checkpoint_cb, early_stop_cb],
    enable_progress_bar=True,
    enable_model_summary=False,
    logger=False,  # Disable logging to avoid file I/O issues
    num_sanity_val_steps=0,
    deterministic=False,
    # Disable any distributed features
    strategy='auto',
)

print('Starting training...')
trainer.fit(model, train_loader, val_loader)

print('Running test...')
test_results = trainer.test(model, test_loader, ckpt_path='best', verbose=False)
test_result = test_results[0] if test_results else {}

result = {
    'layer': $layer,
    'pooling': '$pool',
    'test_acc': test_result.get('test_acc', 0.0),
    'test_auroc': test_result.get('test_auroc', 0.0),
    'checkpoint': checkpoint_cb.best_model_path,
}

print('RESULT:' + json.dumps(result))
print('Done!')
" 2>&1 | tee "$LOG"
        
        EXIT_CODE=${PIPESTATUS[0]}
        
        # Parse result
        if [ $EXIT_CODE -eq 124 ]; then
            echo -e "  ${RED}TIMEOUT${NC}"
            RESULT="{\"layer\": $layer, \"pooling\": \"$pool\", \"error\": \"timeout\"}"
            FAILED=$((FAILED + 1))
        elif [ $EXIT_CODE -ne 0 ]; then
            echo -e "  ${RED}FAILED (exit $EXIT_CODE)${NC}"
            RESULT="{\"layer\": $layer, \"pooling\": \"$pool\", \"error\": \"exit_$EXIT_CODE\"}"
            FAILED=$((FAILED + 1))
        else
            RESULT_LINE=$(grep "^RESULT:" "$LOG" | tail -1 | sed 's/^RESULT://')
            if [ -n "$RESULT_LINE" ]; then
                RESULT="$RESULT_LINE"
                ACC=$(echo "$RESULT" | python -c "import sys,json; print(f\"{json.load(sys.stdin).get('test_acc',0):.4f}\")" 2>/dev/null || echo "?")
                echo -e "  ${GREEN}SUCCESS${NC} (acc: $ACC)"
                SUCCESS=$((SUCCESS + 1))
            else
                RESULT="{\"layer\": $layer, \"pooling\": \"$pool\", \"error\": \"no_result\"}"
                FAILED=$((FAILED + 1))
            fi
        fi
        
        # Append to results
        python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
results.append($RESULT)
with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
" 2>/dev/null
        
        # Force GPU cleanup
        python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
" 2>/dev/null || true
        
        # Brief pause between runs
        sleep 3
    done
done

echo ""
echo -e "${BLUE}======================================${NC}"
echo -e "${BLUE}COMPLETE${NC}"
echo -e "${BLUE}======================================${NC}"
echo ""
echo -e "Success: ${GREEN}$SUCCESS${NC}"
echo -e "Failed:  ${RED}$FAILED${NC}"
echo -e "Total:   $TOTAL"
echo ""
echo "Results: $RESULTS_FILE"

# Show best result
python -c "
import json
with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)
valid = [r for r in results if 'error' not in r]
if valid:
    best = max(valid, key=lambda x: x.get('test_auroc', 0))
    print(f\"Best: Layer {best['layer']}, {best['pooling']} -> AUROC {best.get('test_auroc', 0):.4f}\")
" 2>/dev/null || true

echo ""
echo -e "${GREEN}Done!${NC}"

