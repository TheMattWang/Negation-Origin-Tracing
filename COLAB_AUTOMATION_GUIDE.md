# Automated Colab Experiment Guide

## Quick Start (Fully Automated)

### Step 1: Upload Notebook to Colab
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/06_run_full_comparison_colab.ipynb`
3. Go to Runtime → Change runtime type → Select **GPU** → Save

### Step 2: Run All Cells
Click **Runtime → Run all** or run cells one by one:
1. **Mount Drive** - Click "Allow" when prompted
2. **Clone Repository** - Downloads latest code
3. **Install Dependencies** - Installs packages
4. **Download Datasets** - Gets SST-2 data
5. **Run Experiment** - Takes 2-4 hours with GPU
6. **Verify Results** - Checks Drive for saved files

### Step 3: Access Results
Results are automatically saved to:
```
Google Drive > My Drive > Negation-Origin-Tracing-Results > comparison_YYYYMMDD_HHMMSS/
```

**Option A: Download from Drive Web**
1. Open [drive.google.com](https://drive.google.com)
2. Navigate to `My Drive/Negation-Origin-Tracing-Results/`
3. Right-click the comparison folder → Download

**Option B: Sync with Desktop App (Recommended)**
1. Install [Google Drive desktop app](https://www.google.com/drive/download/)
2. Results appear automatically at:
   ```
   ~/Google Drive/My Drive/Negation-Origin-Tracing-Results/
   ```
3. Copy to your local repo:
   ```bash
   cp -r ~/Google\ Drive/My\ Drive/Negation-Origin-Tracing-Results/comparison_* \
         ~/Documents/NLP/Negation-Origin-Tracing/experiments/
   ```

## What Gets Saved

The experiment creates a timestamped folder with:
```
comparison_YYYYMMDD_HHMMSS/
├── base_probes/
│   ├── results_summary.json          # Probe performance across all layers
│   ├── layer_0_pooling_cls/
│   │   └── checkpoints/
│   │       └── best-val_loss=*.ckpt  # Best probe checkpoint
│   └── ...
├── base_interventions/
│   └── intervention_results.json     # Intervention effects on base model
├── finetuned_interventions/
│   └── intervention_results.json     # Intervention effects on finetuned model
└── comparison_summary.json           # Overall summary
```

## How It Works

### Modified Shell Script
The `run_full_comparison.sh` script now supports a `DRIVE_OUTPUT` environment variable:

```bash
# If DRIVE_OUTPUT is set, save to Drive
if [ -n "$DRIVE_OUTPUT" ]; then
    OUTPUT_DIR="$DRIVE_OUTPUT/comparison_$(date +%Y%m%d_%H%M%S)"
else
    OUTPUT_DIR="experiments/comparison_$(date +%Y%m%d_%H%M%S)"
fi
```

### Notebook Automation
The notebook sets this variable before running the script:

```python
import os
os.environ['DRIVE_OUTPUT'] = '/content/drive/MyDrive/Negation-Origin-Tracing-Results'
!bash run_full_comparison.sh
```

## Troubleshooting

### Drive Not Mounted
**Error:** `No such file or directory: /content/drive/MyDrive/...`

**Solution:** Run the first cell again to mount Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### GPU Not Available
**Warning:** `No GPU detected - training will be much slower!`

**Solution:** 
1. Go to Runtime → Change runtime type
2. Select "GPU" as hardware accelerator
3. Click Save
4. Restart and rerun cells

### Results Not in Drive
**Issue:** Experiment finished but no results in Drive

**Solution:** Check the last cell output for the exact path, or manually copy:
```python
import shutil
import glob
comparison_dirs = glob.glob('/content/Negation-Origin-Tracing/experiments/comparison_*')
if comparison_dirs:
    latest = max(comparison_dirs, key=os.path.getmtime)
    dest = f'/content/drive/MyDrive/Negation-Origin-Tracing-Results/{os.path.basename(latest)}'
    shutil.copytree(latest, dest)
    print(f"Copied to: {dest}")
```

### Colab Disconnected
**Issue:** Browser closed or connection lost during experiment

**Solution:** 
- Results are still being saved to Drive in real-time
- Reconnect and check Drive for partial results
- Use Colab Pro for longer guaranteed runtimes

## Advanced: Manual Control

If you want more control, you can run the script manually:

```python
# Mount Drive first
from google.colab import drive
drive.mount('/content/drive')

# Clone and setup
!git clone https://github.com/TheMattWang/Negation-Origin-Tracing.git
%cd Negation-Origin-Tracing
!pip install -q torch lightning transformers datasets pandas pyarrow matplotlib seaborn scikit-learn tqdm tensorboardX

# Download data
!python src/data/download.py

# Run with custom settings
import os
os.environ['DRIVE_OUTPUT'] = '/content/drive/MyDrive/my-custom-path'
!bash run_full_comparison.sh
```

## Estimated Timings (with T4 GPU)

| Step | Time |
|------|------|
| Setup & Data Download | ~5 min |
| Probe Training (all layers) | ~1-2 hours |
| Base Model Interventions | ~15-30 min |
| Finetuned Model Interventions | ~15-30 min |
| **Total** | **~2-4 hours** |

## Next Steps After Download

Once results are on your local machine:

1. **Analyze results:**
   ```bash
   cd ~/Documents/NLP/Negation-Origin-Tracing
   jupyter notebook notebooks/05_base_vs_finetuned_comparison.ipynb
   ```

2. **Generate visualizations:**
   - Open `05_base_vs_finetuned_comparison.ipynb`
   - Update the `RESULTS_DIR` variable to point to your downloaded results
   - Run all cells

3. **Commit results (optional):**
   ```bash
   git add experiments/comparison_*
   git commit -m "Add experiment results from Colab"
   git push
   ```

## Tips

1. **Keep browser tab open** during long experiments (or use Colab Pro)
2. **Check Drive periodically** to ensure results are being saved
3. **Use the preview cell** at the end to quickly check results without downloading
4. **Download as ZIP** (last optional cell) for a backup copy
5. **Monitor GPU usage** with `!nvidia-smi` to ensure GPU is being used


