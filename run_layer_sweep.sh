#!/bin/bash
# =============================================================================
# Robust Sequential Layer Sweep Script
# =============================================================================
# This script runs layer search experiments SEQUENTIALLY to avoid GPU deadlocks.
#
# Why sequential?
# - PyTorch Lightning + CUDA + multiprocessing = hangs, OOMs, or frozen NCCL ops
# - Even though each experiment works alone, parallelizing on a single GPU
#   guarantees contention and deadlocks
#
# Features:
# - Only one training job touches the GPU at once → no deadlocks
# - Automatic hang detection via timeout
# - Per-run logs in logs/ for debugging
# - Resume support: skips already-completed experiments
# - Safe + reproducible experiment sweeps
# =============================================================================

set -o pipefail  # Catch errors in pipes

# Configuration
MODEL_NAME="${MODEL_NAME:-distilbert-base-uncased}"
DATA_DIR="${DATA_DIR:-data/raw}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
PROBE_LR="${PROBE_LR:-1e-3}"
SEED="${SEED:-42}"
TIMEOUT_HOURS="${TIMEOUT_HOURS:-2}"  # Kill run if it hangs > 2 hours

# Output directory (support Google Drive for Colab)
if [ -n "$DRIVE_OUTPUT" ]; then
    OUTPUT_DIR="$DRIVE_OUTPUT/layer_sweep_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="${OUTPUT_DIR:-experiments/layer_sweep_$(date +%Y%m%d_%H%M%S)}"
fi

# Layers and pooling strategies to search
LAYERS="${LAYERS:-0,1,2,3,4,5}"  # Default: all 6 layers of DistilBERT
POOLING_STRATEGIES="${POOLING_STRATEGIES:-cls,mean,token}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =============================================================================
# Setup
# =============================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Sequential Layer Sweep (Deadlock-Safe)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Model: $MODEL_NAME"
echo "Layers: $LAYERS"
echo "Pooling: $POOLING_STRATEGIES"
echo "Output: $OUTPUT_DIR"
echo "Timeout: ${TIMEOUT_HOURS}h per experiment"
echo ""

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/logs"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup Python command
if command -v conda &> /dev/null; then
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    
    if conda env list | grep -q "^not "; then
        conda activate not 2>/dev/null || true
    fi
    
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python" ]; then
        PYTHON_CMD="$CONDA_PREFIX/bin/python"
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python"
fi

echo -e "${GREEN}Using Python: $PYTHON_CMD${NC}"

# Detect GPU
GPU_CHECK=$($PYTHON_CMD -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
if [ "$GPU_CHECK" = "cuda" ]; then
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME${NC}"
    ACCELERATOR="gpu"
    DEVICES=1
else
    echo -e "${YELLOW}⚠ No GPU detected - using CPU (slower)${NC}"
    ACCELERATOR="cpu"
    DEVICES=1
fi

# =============================================================================
# Results tracking
# =============================================================================

RESULTS_FILE="$OUTPUT_DIR/results_summary.json"
STATUS_FILE="$OUTPUT_DIR/sweep_status.json"

# Initialize or load results
if [ -f "$RESULTS_FILE" ]; then
    echo -e "${GREEN}✓ Found existing results, will resume${NC}"
else
    echo "[]" > "$RESULTS_FILE"
fi

# Function to check if experiment is already completed
is_completed() {
    local layer=$1
    local pooling=$2
    
    if [ ! -f "$RESULTS_FILE" ]; then
        return 1
    fi
    
    $PYTHON_CMD -c "
import json
import sys
try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
    for r in results:
        if r.get('layer_idx') == $layer and r.get('pooling_strategy') == '$pooling':
            if 'error' not in r and r.get('test_auroc', 0) > 0:
                sys.exit(0)  # Completed successfully
    sys.exit(1)  # Not completed
except:
    sys.exit(1)
" 2>/dev/null
}

# Function to append result
append_result() {
    local result_json=$1
    
    $PYTHON_CMD -c "
import json
import sys

result = json.loads('''$result_json''')

try:
    with open('$RESULTS_FILE', 'r') as f:
        results = json.load(f)
except:
    results = []

# Remove any existing result for this layer/pooling combo
results = [r for r in results 
           if not (r.get('layer_idx') == result['layer_idx'] and 
                   r.get('pooling_strategy') == result['pooling_strategy'])]
results.append(result)

with open('$RESULTS_FILE', 'w') as f:
    json.dump(results, f, indent=2)
"
}

# =============================================================================
# Main sweep loop
# =============================================================================

# Parse layers and pooling strategies
IFS=',' read -ra LAYER_ARRAY <<< "$LAYERS"
IFS=',' read -ra POOL_ARRAY <<< "$POOLING_STRATEGIES"

TOTAL_EXPERIMENTS=$((${#LAYER_ARRAY[@]} * ${#POOL_ARRAY[@]}))
COMPLETED=0
FAILED=0
SKIPPED=0
CURRENT=0

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Starting sweep: $TOTAL_EXPERIMENTS experiments${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

for layer in "${LAYER_ARRAY[@]}"; do
    for pooling in "${POOL_ARRAY[@]}"; do
        CURRENT=$((CURRENT + 1))
        RUN_ID="L${layer}_P${pooling}"
        LOG_FILE="$OUTPUT_DIR/logs/${RUN_ID}.log"
        EXPERIMENT_DIR="$OUTPUT_DIR/layer_${layer}_pooling_${pooling}"
        
        echo -e "\n${BLUE}[$CURRENT/$TOTAL_EXPERIMENTS]${NC} $RUN_ID"
        
        # Check if already completed
        if is_completed "$layer" "$pooling"; then
            echo -e "  ${GREEN}✓ Already completed, skipping${NC}"
            SKIPPED=$((SKIPPED + 1))
            continue
        fi
        
        echo "  Running... (log: $LOG_FILE)"
        
        # Create experiment directory
        mkdir -p "$EXPERIMENT_DIR/checkpoints"
        
        # Run the training with timeout
        # CUDA_VISIBLE_DEVICES=0 ensures only one GPU is used
        START_TIME=$(date +%s)
        
        timeout "${TIMEOUT_HOURS}h" bash -c "
            CUDA_VISIBLE_DEVICES=0 $PYTHON_CMD -c \"
import os
import sys
import json
import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path

# Add project root to path
sys.path.insert(0, '$SCRIPT_DIR')

from src.models import BaseModule
from src.datasets import NOTDataModule

# Set seed
torch.manual_seed($SEED)
torch.cuda.manual_seed_all($SEED)

# Initialize model
model = BaseModule(
    model_name='$MODEL_NAME',
    num_labels=2,
    mode='probe',
    probe_layer=$layer,
    pooling_strategy='$pooling',
    probe_lr=$PROBE_LR,
)

# Initialize data module
class Args:
    data_dir = '$DATA_DIR'
    batch_size = $BATCH_SIZE
    num_workers = 0  # Avoid multiprocessing issues
    model_name = '$MODEL_NAME'

datamodule = NOTDataModule.from_args(Args())
datamodule.setup('fit')

# Setup callbacks
checkpoint_callback = ModelCheckpoint(
    dirpath='$EXPERIMENT_DIR/checkpoints',
    filename='best-{val_loss:.3f}-{val_acc:.3f}',
    monitor='val_acc',
    mode='max',
    save_top_k=1,
    save_last=True,
)

early_stop = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=5,
)

logger = TensorBoardLogger(
    save_dir='$OUTPUT_DIR',
    name='',
    version='layer_${layer}_pooling_${pooling}',
)

# Initialize trainer - single GPU, no multiprocessing
trainer = L.Trainer(
    max_epochs=$MAX_EPOCHS,
    accelerator='$ACCELERATOR',
    devices=$DEVICES,
    logger=logger,
    callbacks=[checkpoint_callback, early_stop],
    log_every_n_steps=50,
    val_check_interval=0.5,
    enable_progress_bar=True,
    enable_model_summary=False,
    deterministic=False,
    num_sanity_val_steps=0,  # Skip sanity check to save time
)

# Train
trainer.fit(model=model, datamodule=datamodule)

# Test
datamodule.setup('test')
test_results = trainer.test(model=model, datamodule=datamodule, ckpt_path='best', verbose=False)
test_result = test_results[0] if test_results else {}

# Save result
result = {
    'layer_idx': $layer,
    'pooling_strategy': '$pooling',
    'checkpoint_path': checkpoint_callback.best_model_path,
    'test_accuracy': test_result.get('test_acc', 0.0),
    'test_auroc': test_result.get('test_auroc', 0.0),
    'test_loss': test_result.get('test_loss', float('inf')),
    'experiment_dir': '$EXPERIMENT_DIR',
}

print('RESULT_JSON:' + json.dumps(result))
\"
        " 2>&1 | tee "$LOG_FILE"
        
        EXIT_CODE=${PIPESTATUS[0]}
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))
        
        # Parse result from log
        if [ $EXIT_CODE -eq 124 ]; then
            echo -e "  ${RED}✗ TIMEOUT (likely deadlock after ${DURATION}s)${NC}"
            RESULT_JSON="{\"layer_idx\": $layer, \"pooling_strategy\": \"$pooling\", \"error\": \"timeout\", \"test_accuracy\": 0.0, \"test_auroc\": 0.0}"
            FAILED=$((FAILED + 1))
        elif [ $EXIT_CODE -ne 0 ]; then
            echo -e "  ${RED}✗ FAILED (exit code $EXIT_CODE after ${DURATION}s)${NC}"
            ERROR_MSG=$(tail -5 "$LOG_FILE" | tr '\n' ' ' | sed 's/"/\\"/g' | cut -c1-200)
            RESULT_JSON="{\"layer_idx\": $layer, \"pooling_strategy\": \"$pooling\", \"error\": \"$ERROR_MSG\", \"test_accuracy\": 0.0, \"test_auroc\": 0.0}"
            FAILED=$((FAILED + 1))
        else
            # Extract result JSON from log
            RESULT_LINE=$(grep "RESULT_JSON:" "$LOG_FILE" | tail -1 | sed 's/RESULT_JSON://')
            if [ -n "$RESULT_LINE" ]; then
                RESULT_JSON="$RESULT_LINE"
                TEST_AUROC=$($PYTHON_CMD -c "import json; print(json.loads('$RESULT_JSON').get('test_auroc', 0))" 2>/dev/null || echo "0")
                echo -e "  ${GREEN}✓ SUCCESS (AUROC: $TEST_AUROC, ${DURATION}s)${NC}"
                COMPLETED=$((COMPLETED + 1))
            else
                echo -e "  ${YELLOW}⚠ Completed but no result found${NC}"
                RESULT_JSON="{\"layer_idx\": $layer, \"pooling_strategy\": \"$pooling\", \"error\": \"no_result_found\", \"test_accuracy\": 0.0, \"test_auroc\": 0.0}"
                FAILED=$((FAILED + 1))
            fi
        fi
        
        # Append result to file
        append_result "$RESULT_JSON"
        
        # Save status after each experiment
        echo "{\"completed\": $COMPLETED, \"failed\": $FAILED, \"skipped\": $SKIPPED, \"total\": $TOTAL_EXPERIMENTS}" > "$STATUS_FILE"
        
        # Clean up GPU memory between runs
        $PYTHON_CMD -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null || true
        
        # Small delay between runs to let GPU cool down
        sleep 2
    done
done

# =============================================================================
# Summary
# =============================================================================

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}SWEEP COMPLETE${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Completed: ${GREEN}$COMPLETED${NC}"
echo -e "Failed:    ${RED}$FAILED${NC}"
echo -e "Skipped:   ${YELLOW}$SKIPPED${NC}"
echo -e "Total:     $TOTAL_EXPERIMENTS"
echo ""
echo "Results: $RESULTS_FILE"
echo "Logs:    $OUTPUT_DIR/logs/"
echo ""

# Print best results
if [ -f "$RESULTS_FILE" ]; then
    $PYTHON_CMD -c "
import json

with open('$RESULTS_FILE', 'r') as f:
    results = json.load(f)

# Filter out errors
valid = [r for r in results if 'error' not in r and r.get('test_auroc', 0) > 0]

if valid:
    best = max(valid, key=lambda x: x.get('test_auroc', 0))
    print(f\"Best result: Layer {best['layer_idx']}, {best['pooling_strategy']}\")
    print(f\"  Test AUROC: {best['test_auroc']:.4f}\")
    print(f\"  Test Accuracy: {best['test_accuracy']:.4f}\")
else:
    print('No valid results found')
" 2>/dev/null || true
fi

echo ""
echo -e "${GREEN}✓ Done!${NC}"

