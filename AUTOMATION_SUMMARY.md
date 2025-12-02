# 🚀 Colab Automation Setup Complete!

## What I Did

### 1. Modified `run_full_comparison.sh`
- Added support for `DRIVE_OUTPUT` environment variable
- When set, results save directly to Google Drive instead of local `experiments/` folder
- Backward compatible - still works without the variable

### 2. Created `notebooks/06_run_full_comparison_colab.ipynb`
- **Fully automated notebook** for running experiments in Colab
- Handles everything from setup to result saving
- No manual intervention needed after starting

### 3. Updated Documentation
- `COLAB_SETUP.md` - Added automated workflow instructions
- `notebooks/README.md` - Documented new notebook
- `COLAB_AUTOMATION_GUIDE.md` - Comprehensive guide with troubleshooting

## How to Use (TL;DR)

1. **Upload to Colab:**
   - Go to [colab.research.google.com](https://colab.research.google.com)
   - Upload `notebooks/06_run_full_comparison_colab.ipynb`
   - Enable GPU (Runtime → Change runtime type → GPU)

2. **Run:**
   - Click Runtime → Run all
   - Click "Allow" when Drive mount asks for permission
   - Wait 2-4 hours (with GPU)

3. **Download Results:**
   - Results automatically saved to `Google Drive/My Drive/Negation-Origin-Tracing-Results/`
   - Download from Drive web interface OR
   - Install Google Drive desktop app for automatic sync

## Files Changed

```
✅ run_full_comparison.sh                    # Modified: Added DRIVE_OUTPUT support
✅ notebooks/06_run_full_comparison_colab.ipynb  # Created: Automated notebook
✅ COLAB_SETUP.md                            # Updated: Added automation section
✅ notebooks/README.md                       # Updated: Documented new notebook
✅ COLAB_AUTOMATION_GUIDE.md                 # Created: Detailed guide
✅ AUTOMATION_SUMMARY.md                     # Created: This file
```

## What the Notebook Does

**Step 1: Mount Google Drive**
- Mounts Drive at `/content/drive`
- Creates `Negation-Origin-Tracing-Results` folder

**Step 2: Clone Repository**
- Clones from GitHub (or pulls latest if exists)
- Changes to repo directory

**Step 3: Install Dependencies**
- Installs PyTorch, Lightning, Transformers, etc.
- Verifies GPU availability

**Step 4: Download Datasets**
- Downloads SST-2 data
- Verifies files exist

**Step 5: Run Full Experiment**
- Sets `DRIVE_OUTPUT` environment variable
- Runs `run_full_comparison.sh`
- Results save directly to Drive

**Step 6: Verify Results**
- Lists files in Drive
- Checks for key result files
- Shows access instructions

**Optional: Preview Results**
- Displays summary without downloading
- Shows best layer, AUROC, etc.

**Optional: Download as ZIP**
- Creates backup ZIP file
- Can download from Colab file browser

## Result Structure

```
Google Drive/My Drive/Negation-Origin-Tracing-Results/
└── comparison_20251202_143022/
    ├── base_probes/
    │   ├── results_summary.json
    │   ├── layer_0_pooling_cls/
    │   │   └── checkpoints/
    │   │       └── best-val_loss=0.615-val_acc=0.704.ckpt
    │   ├── layer_1_pooling_cls/
    │   └── ...
    ├── base_interventions/
    │   └── intervention_results.json
    ├── finetuned_interventions/
    │   └── intervention_results.json
    └── comparison_summary.json
```

## Next Steps

### On Colab:
1. Upload `notebooks/06_run_full_comparison_colab.ipynb`
2. Enable GPU
3. Run all cells
4. Wait for completion

### On Your Mac:
1. Install Google Drive desktop app (optional but recommended)
2. Results sync automatically to:
   ```
   ~/Google Drive/My Drive/Negation-Origin-Tracing-Results/
   ```
3. Copy to your repo:
   ```bash
   cp -r ~/Google\ Drive/My\ Drive/Negation-Origin-Tracing-Results/comparison_* \
         ~/Documents/NLP/Negation-Origin-Tracing/experiments/
   ```
4. Analyze with `notebooks/05_base_vs_finetuned_comparison.ipynb`

## Benefits

✅ **Fully Automated** - No manual file copying
✅ **Persistent Storage** - Results saved to Drive, not lost if Colab disconnects
✅ **Easy Access** - Download from Drive or auto-sync with desktop app
✅ **Backward Compatible** - Shell script still works locally without Drive
✅ **Verified** - Automatic result verification at the end
✅ **Preview** - Quick summary without downloading everything

## Troubleshooting

See `COLAB_AUTOMATION_GUIDE.md` for detailed troubleshooting, including:
- Drive not mounted
- GPU not available
- Results not in Drive
- Colab disconnected
- Manual recovery steps

## Questions?

Read the full guide: `COLAB_AUTOMATION_GUIDE.md`
