#!/bin/bash
# Full experiment pipeline: Base model probes → Interventions on base → Interventions on finetuned
# This script compares how negation is encoded in base vs finetuned models

set -e  # Exit on error

# Configuration
BASE_MODEL="distilbert-base-uncased"
FINETUNED_MODEL="distilbert-base-uncased-finetuned-sst-2-english"
DATA_DIR="data/raw"
OUTPUT_DIR="experiments/comparison_$(date +%Y%m%d_%H%M%S)"
BATCH_SIZE=16
MAX_EPOCHS=10
PROBE_LR=1e-3

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Negation Encoding Comparison Experiment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Base model: $BASE_MODEL"
echo "Finetuned model: $FINETUNED_MODEL"
echo "Output directory: $OUTPUT_DIR"
echo ""

# Activate conda environment
CONDA_ENV_PATH="/Users/mattwang/miniconda3/envs/not"
if [ -d "$CONDA_ENV_PATH" ]; then
    echo -e "${YELLOW}Using conda environment at: $CONDA_ENV_PATH${NC}"
    export PATH="$CONDA_ENV_PATH/bin:$PATH"
    # Set Python to use the environment's Python
    PYTHON_CMD="$CONDA_ENV_PATH/bin/python"
    if [ ! -f "$PYTHON_CMD" ]; then
        echo -e "${YELLOW}Warning: Python not found at $PYTHON_CMD, using system python${NC}"
        PYTHON_CMD="python"
    fi
elif command -v conda &> /dev/null; then
    echo -e "${YELLOW}Activating conda environment 'not'...${NC}"
    # Try to source conda.sh first
    if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
        source "$(conda info --base)/etc/profile.d/conda.sh"
    fi
    # Initialize conda for bash shell
    eval "$(conda shell.bash hook)" 2>/dev/null || true
    # Activate the environment
    conda activate not 2>/dev/null || {
        echo -e "${YELLOW}Warning: Could not activate 'not' via conda command${NC}"
        echo -e "${YELLOW}Please activate manually before running: conda activate not${NC}"
        PYTHON_CMD="python"
    } || PYTHON_CMD="python"
else
    echo -e "${YELLOW}Warning: conda not found, using system python${NC}"
    echo -e "${YELLOW}If you see import errors, please activate the 'not' environment manually${NC}"
    PYTHON_CMD="python"
fi

# Default to python if PYTHON_CMD not set
PYTHON_CMD="${PYTHON_CMD:-python}"
echo -e "${YELLOW}Using Python: $PYTHON_CMD${NC}"

# Detect GPU availability
echo -e "\n${YELLOW}Checking GPU availability...${NC}"
GPU_CHECK=$($PYTHON_CMD -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')" 2>/dev/null || echo "cpu")
if [ "$GPU_CHECK" = "cuda" ]; then
    GPU_NAME=$($PYTHON_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null || echo "Unknown")
    GPU_MEM=$($PYTHON_CMD -c "import torch; print(f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}')" 2>/dev/null || echo "Unknown")
    echo -e "${GREEN}✓ GPU detected: $GPU_NAME (${GPU_MEM} GB)${NC}"
    DEVICE="cuda"
    DEVICES=1
else
    echo -e "${YELLOW}⚠ No GPU detected - will use CPU (much slower)${NC}"
    DEVICE="cpu"
    DEVICES=1
fi
echo -e "${YELLOW}Using device: $DEVICE${NC}\n"

# Create output directory
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/base_probes"
mkdir -p "$OUTPUT_DIR/base_interventions"
mkdir -p "$OUTPUT_DIR/finetuned_interventions"

# ============================================
# STEP 1: Train probes on base model
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}STEP 1: Training probes on base model${NC}"
echo -e "${GREEN}========================================${NC}\n"

$PYTHON_CMD src/scripts/search_layers.py \
    --model_name "$BASE_MODEL" \
    --mode probe \
    --data_dir "$DATA_DIR" \
    --batch_size "$BATCH_SIZE" \
    --max_epochs "$MAX_EPOCHS" \
    --probe_lr "$PROBE_LR" \
    --layers all \
    --pooling_strategies all \
    --output_dir "$OUTPUT_DIR/base_probes" \
    --seed 42 \
    --devices "$DEVICES"

# Check if probe results exist
PROBE_RESULTS="$OUTPUT_DIR/base_probes/results_summary.json"
if [ ! -f "$PROBE_RESULTS" ]; then
    echo -e "${YELLOW}Error: Probe results not found at $PROBE_RESULTS${NC}"
    exit 1
fi

echo -e "\n${GREEN}✓ Probe training complete!${NC}"

# ============================================
# STEP 2: Identify best layer from probe results
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}STEP 2: Identifying best layer${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Use Python to extract best layer from results
BEST_LAYER=$($PYTHON_CMD -c "
import json
import sys
with open('$PROBE_RESULTS', 'r') as f:
    results = json.load(f)
# Find best layer by test_auroc
best = max(results, key=lambda x: x.get('test_auroc', 0))
layer = best['layer_idx']
pooling = best['pooling_strategy']
ckpt = best.get('checkpoint_path', '')
print(f\"{layer}\")
sys.stdout.write(f\"{pooling}\")
" 2>/dev/null || echo "5")

BEST_POOLING=$($PYTHON_CMD -c "
import json
with open('$PROBE_RESULTS', 'r') as f:
    results = json.load(f)
best = max(results, key=lambda x: x.get('test_auroc', 0))
print(best['pooling_strategy'])
" 2>/dev/null || echo "cls")

BEST_CKPT=$($PYTHON_CMD -c "
import json
with open('$PROBE_RESULTS', 'r') as f:
    results = json.load(f)
best = max(results, key=lambda x: x.get('test_auroc', 0))
print(best.get('checkpoint_path', ''))
" 2>/dev/null || echo "")

echo "Best layer: $BEST_LAYER"
echo "Best pooling: $BEST_POOLING"
echo "Checkpoint: $BEST_CKPT"

if [ -z "$BEST_CKPT" ] || [ ! -f "$BEST_CKPT" ]; then
    echo -e "${YELLOW}Warning: Best checkpoint not found, will try to find it...${NC}"
    # Try to find checkpoint in the experiment directory
    BEST_CKPT=$(find "$OUTPUT_DIR/base_probes" -name "*.ckpt" -type f | head -1)
    if [ -z "$BEST_CKPT" ]; then
        echo -e "${YELLOW}Error: No checkpoint found. Skipping base model interventions.${NC}"
        SKIP_BASE_INTERVENTIONS=true
    fi
fi

# ============================================
# STEP 3: Run interventions on base model (with probes)
# ============================================
if [ -z "$SKIP_BASE_INTERVENTIONS" ]; then
    echo -e "\n${GREEN}========================================${NC}"
    echo -e "${GREEN}STEP 3: Running interventions on BASE model${NC}"
    echo -e "${GREEN}========================================${NC}\n"

    NEGATION_DATA="$DATA_DIR/test/negation.parquet"
    if [ ! -f "$NEGATION_DATA" ]; then
        echo -e "${YELLOW}Warning: Negation dataset not found at $NEGATION_DATA${NC}"
        echo -e "${YELLOW}Skipping base model interventions${NC}"
    else
        $PYTHON_CMD src/scripts/run_interventions.py \
            --model_ckpt "$BEST_CKPT" \
            --model_name "$BASE_MODEL" \
            --mode probe \
            --probe_layer "$BEST_LAYER" \
            --data_path "$NEGATION_DATA" \
            --intervention_type activation_patching \
            --layers "$BEST_LAYER" \
            --batch_size "$BATCH_SIZE" \
            --output_dir "$OUTPUT_DIR/base_interventions" \
            --device "$DEVICE"

        echo -e "\n${GREEN}✓ Base model interventions complete!${NC}"
    fi
else
    echo -e "\n${YELLOW}Skipping base model interventions (no checkpoint found)${NC}"
fi

# ============================================
# STEP 4: Run interventions on finetuned model
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}STEP 4: Running interventions on FINETUNED model${NC}"
echo -e "${GREEN}========================================${NC}\n"

NEGATION_DATA="$DATA_DIR/test/negation.parquet"
if [ ! -f "$NEGATION_DATA" ]; then
    echo -e "${YELLOW}Warning: Negation dataset not found at $NEGATION_DATA${NC}"
    echo -e "${YELLOW}Skipping finetuned model interventions${NC}"
else
    # Run on the same layer(s) as base model for comparison
    $PYTHON_CMD src/scripts/run_interventions.py \
        --mode finetune \
        --model_name "$FINETUNED_MODEL" \
        --data_path "$NEGATION_DATA" \
        --intervention_type activation_patching \
        --layers "$BEST_LAYER" \
        --batch_size "$BATCH_SIZE" \
        --output_dir "$OUTPUT_DIR/finetuned_interventions" \
        --device cpu

    echo -e "\n${GREEN}✓ Finetuned model interventions complete!${NC}"
fi

# ============================================
# STEP 5: Generate comparison summary
# ============================================
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}STEP 5: Generating comparison summary${NC}"
echo -e "${GREEN}========================================${NC}\n"

$PYTHON_CMD -c "
import json
import os

output_dir = '$OUTPUT_DIR'
summary = {
    'experiment_config': {
        'base_model': '$BASE_MODEL',
        'finetuned_model': '$FINETUNED_MODEL',
        'best_layer': int('$BEST_LAYER'),
        'best_pooling': '$BEST_POOLING',
    },
    'probe_results': None,
    'base_interventions': None,
    'finetuned_interventions': None,
}

# Load probe results
probe_file = '$PROBE_RESULTS'
if os.path.exists(probe_file):
    with open(probe_file, 'r') as f:
        probe_results = json.load(f)
        summary['probe_results'] = {
            'total_experiments': len(probe_results),
            'best_layer': int('$BEST_LAYER'),
            'best_auroc': max(r.get('test_auroc', 0) for r in probe_results),
        }

# Load base interventions
base_int_file = '$OUTPUT_DIR/base_interventions/intervention_results.json'
if os.path.exists(base_int_file):
    with open(base_int_file, 'r') as f:
        summary['base_interventions'] = json.load(f)

# Load finetuned interventions
finetuned_int_file = '$OUTPUT_DIR/finetuned_interventions/intervention_results.json'
if os.path.exists(finetuned_int_file):
    with open(finetuned_int_file, 'r') as f:
        summary['finetuned_interventions'] = json.load(f)

# Save summary
summary_file = os.path.join(output_dir, 'comparison_summary.json')
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f'Comparison summary saved to: {summary_file}')
"

# ============================================
# Final summary
# ============================================
echo -e "\n${BLUE}========================================${NC}"
echo -e "${BLUE}EXPERIMENT COMPLETE!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "Files generated:"
echo "  - Probe results: $OUTPUT_DIR/base_probes/results_summary.json"
if [ -f "$OUTPUT_DIR/base_interventions/intervention_results.json" ]; then
    echo "  - Base interventions: $OUTPUT_DIR/base_interventions/intervention_results.json"
fi
if [ -f "$OUTPUT_DIR/finetuned_interventions/intervention_results.json" ]; then
    echo "  - Finetuned interventions: $OUTPUT_DIR/finetuned_interventions/intervention_results.json"
fi
echo "  - Comparison summary: $OUTPUT_DIR/comparison_summary.json"
echo ""
echo -e "${GREEN}✓ All done!${NC}\n"

