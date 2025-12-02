# Dataset Label Investigation Summary

## Problem
CUDA assertion error: `t >= 0 && t < n_classes` failed during testing phase of probe training.

## Root Cause
The SST-2 test set contains labels of `-1` (no ground truth), which is standard for SST-2. The dataset was trying to use these `-1` labels in loss computation, causing CUDA assertion errors because PyTorch expects labels in the range `[0, num_classes-1]`.

## Investigation Results

### Dataset Label Analysis
- **SST-2 Train**: Labels `[0, 1]` ✓ Valid
- **SST-2 Validation**: Labels `[0, 1]` ✓ Valid  
- **SST-2 Test**: Labels `[-1]` ⚠ No ground truth (standard for SST-2)

### Files Investigated
- `data/raw/train/sst.parquet`: 67,349 samples, labels [0, 1]
- `data/raw/validation/sst.parquet`: 872 samples, labels [0, 1]
- `data/raw/test/sst.parquet`: 1,821 samples, labels [-1] (no ground truth)

## Solution

### Changes Made

1. **`src/datasets/dataset.py`** - `SentimentDataset`:
   - Detects `-1` labels and marks dataset as `has_labels = False`
   - Does not include `labels` key in returned items when label is `-1`
   - Improved error messages with file basename for debugging

2. **`src/models/base_model.py`** - `BaseModule`:
   - `forward()` method: Checks if labels are valid (>= 0) before computing loss
   - `test_step()` method: Handles batches without labels gracefully, skipping loss/metrics computation

### Behavior
- **Train/Validation sets**: Work normally with labels [0, 1]
- **Test sets with -1 labels**: 
  - Dataset returns items without `labels` key
  - Model skips loss computation
  - Model skips accuracy/AUROC computation
  - Predictions are still generated (logits available)

## Testing
✅ Test set with -1 labels loads correctly
✅ DataLoader handles missing labels correctly  
✅ Model forward pass works without labels
✅ No CUDA assertion errors

## Next Steps
The fix is ready. The `search_layers.py` script should now work correctly with the SST-2 test set.

