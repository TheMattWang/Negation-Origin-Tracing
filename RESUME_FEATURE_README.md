# Resume Feature - Quick Start

## What's New? 🎉

Your experiment scripts now **automatically save progress** and can **resume from where they left off** if interrupted!

## Why This Matters

**Before:**
- Colab disconnects after 2 hours
- Lost all progress
- Had to start over 😢

**After:**
- Colab disconnects after 2 hours
- Progress is saved
- Just re-run the same command
- Continues from where it stopped 🎉

## How to Use

### Step 1: Run Your Experiment (Same as Before)

```bash
python src/scripts/run_full_experiment.py \
    --output_dir experiments/my_experiment \
    --layers all \
    --pooling_strategies all
```

### Step 2: If Interrupted, Just Re-run the Same Command

```bash
# After Colab disconnect or crash, run the EXACT SAME command:
python src/scripts/run_full_experiment.py \
    --output_dir experiments/my_experiment \
    --layers all \
    --pooling_strategies all
```

**That's it!** The script will:
- ✅ Load your existing results
- ✅ Skip completed experiments
- ✅ Continue with remaining experiments

## Example Output

### First Run (Interrupted)
```
Starting layer search: 18 total experiments
Already completed: 0
Remaining: 18

[1/18] Training probe: Layer 0, Pooling: cls
✓ Layer 0, cls: Test Acc=0.8500, Test AUROC=0.9200

[2/18] Training probe: Layer 0, Pooling: mean
✓ Layer 0, mean: Test Acc=0.8300, Test AUROC=0.9000

...

[12/18] Training probe: Layer 3, Pooling: token
✓ Layer 3, token: Test Acc=0.8700, Test AUROC=0.9400

❌ [Colab disconnects]
```

### After Reconnecting (Resume)
```
✓ Loaded 12 existing results
✓ 12 experiments already completed

Starting layer search: 18 total experiments
Already completed: 12
Remaining: 6

[13/18] Training probe: Layer 4, Pooling: cls
✓ Layer 4, cls: Test Acc=0.8600, Test AUROC=0.9300

...

[18/18] Training probe: Layer 5, Pooling: token
✓ Layer 5, token: Test Acc=0.8800, Test AUROC=0.9500

✓ All experiments complete!
```

## For Google Colab Users

### Important: Save to Google Drive!

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# 2. Use Drive path for output
!python src/scripts/run_full_experiment.py \
    --output_dir /content/drive/MyDrive/experiments/my_exp \
    --layers all
```

### After Disconnection

1. **Reconnect to Colab**
2. **Remount Drive:**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Re-run the same command** - it will resume automatically!

## Check Your Progress

Want to see how many experiments are done?

```python
import json

with open('experiments/my_exp/results_summary.json', 'r') as f:
    results = json.load(f)

completed = sum(1 for r in results if 'error' not in r and r.get('test_auroc', 0) > 0)
total = len(results)

print(f"Progress: {completed}/{total} experiments completed ({completed/total*100:.1f}%)")

# Show what's done
for r in results:
    if 'error' not in r and r.get('test_auroc', 0) > 0:
        print(f"  ✓ Layer {r['layer_idx']}, {r['pooling_strategy']}: AUROC={r['test_auroc']:.4f}")
```

## What Gets Saved?

After each experiment completes:
- `results_summary.json` - All results (this is the checkpoint file)
- `layer_X_pooling_Y/checkpoints/` - Model checkpoints
- TensorBoard logs

## Common Questions

### Q: Do I need to do anything special?
**A:** No! Just re-run the same command. Resume is automatic.

### Q: What if I want to start fresh?
**A:** Use the `--no_resume` flag:
```bash
python src/scripts/search_layers.py --output_dir experiments/my_exp --no_resume
```

### Q: Will it re-run completed experiments?
**A:** No! It automatically skips them.

### Q: What if an experiment failed?
**A:** Failed experiments are not marked as completed, so they'll be re-run.

### Q: Can I resume multiple times?
**A:** Yes! You can disconnect and resume as many times as needed.

### Q: Does this work with the notebook?
**A:** Yes! The notebook `06_run_full_comparison_colab.ipynb` includes resume instructions.

## Documentation

- **Quick Start:** This file
- **Detailed Guide:** [`RESUME_GUIDE.md`](RESUME_GUIDE.md)
- **Technical Details:** [`CHANGELOG_RESUME.md`](CHANGELOG_RESUME.md)
- **Visual Diagrams:** [`RESUME_FLOW.md`](RESUME_FLOW.md)
- **Summary:** [`RESUME_SUMMARY.md`](RESUME_SUMMARY.md)

## Test It

Want to verify it works?

```bash
python test_resume.py
```

Should output:
```
✓ Resume detection test passed
✓ Incremental save test passed
✓ Resume workflow test passed
✓ All tests passed!
```

## Need Help?

1. Check [`RESUME_GUIDE.md`](RESUME_GUIDE.md) for detailed instructions
2. Verify Google Drive is mounted (for Colab)
3. Check that `results_summary.json` exists and is valid JSON
4. Use `--no_resume` to start fresh if needed

## Bottom Line

**Just re-run the same command after an interruption. The script handles the rest!** 🚀

---

**Feature Status:** ✅ Fully Implemented and Tested  
**Documentation:** ✅ Complete  
**Ready to Use:** ✅ Yes!

