#!/bin/bash
# =============================================================================
# Simple Sequential Sweep Script (Deadlock-Safe)
# =============================================================================
# Runs experiments one at a time to avoid GPU contention.
# Uses timeout to detect hangs automatically.
# =============================================================================

mkdir -p logs

# Configuration - customize these
LAYERS="${LAYERS:-0 1 2 3 4 5}"
POOLING="${POOLING:-cls mean token}"
TIMEOUT="${TIMEOUT:-2h}"

echo "Starting sequential layer sweep..."
echo "Layers: $LAYERS"
echo "Pooling: $POOLING"
echo "Timeout: $TIMEOUT per run"
echo ""

for layer in $LAYERS; do
    for pool in $POOLING; do
        run_id="L${layer}_P${pool}"
        log="logs/${run_id}.log"
        
        echo "Running $run_id"
        
        # Run sequentially, kill if it hangs
        timeout $TIMEOUT bash -c \
            "CUDA_VISIBLE_DEVICES=0 python src/scripts/train.py \
                --mode probe \
                --probe_layer $layer \
                --pooling_strategy $pool \
                --experiment_name $run_id \
                --max_epochs 10 \
                --batch_size 16 \
                --num_workers 0 2>&1 | tee $log"
        
        status=$?
        
        if [ $status -eq 124 ]; then
            echo "$run_id: TIMEOUT (likely deadlock)" | tee -a "$log"
        elif [ $status -ne 0 ]; then
            echo "$run_id: FAILED (exit code $status)" | tee -a "$log"
        else
            echo "$run_id: SUCCESS" | tee -a "$log"
        fi
        
        # Clear GPU memory between runs
        python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" 2>/dev/null
        
        echo ""
    done
done

echo "Sweep complete! Check logs/ for details."

