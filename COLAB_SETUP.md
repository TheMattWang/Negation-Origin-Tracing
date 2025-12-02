# Google Colab Setup Guide

This guide explains how to run the Negation-Origin-Tracing experiments in Google Colab.

## Quick Start

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Upload or Clone Repository**:
   - Option A: Upload the notebooks from `notebooks/` directory
   - Option B: Clone the repo in Colab:
     ```python
     !git clone https://github.com/TheMattWang/Negation-Origin-Tracing.git
     %cd Negation-Origin-Tracing
     ```

3. **Run Notebooks in Order**:
   - `00_setup_colab.ipynb` - Setup environment
   - `01_download_data.ipynb` - Download datasets
   - `02_layer_search.ipynb` - Train probes
   - `03_visualize_results.ipynb` - Generate plots
   - `04_full_experiment.ipynb` - Complete pipeline
   - `06_run_full_comparison_colab.ipynb` - **AUTOMATED: Full experiment with Drive sync**

## Environment Setup

### Enable GPU (Recommended)
1. Go to Runtime → Change runtime type
2. Select "GPU" as hardware accelerator
3. Click Save

### Install Dependencies
The setup notebook (`00_setup_colab.ipynb`) will:
- Install PyTorch, Lightning, Transformers, etc.
- Clone the repository
- Verify installation
- Check GPU availability

## File Structure in Colab

After setup, your Colab environment should have:
```
/content/
├── Negation-Origin-Tracing/
│   ├── src/
│   ├── notebooks/
│   ├── data/
│   └── experiments/
```

## Saving Results

### **RECOMMENDED: Automated Google Drive Sync**

Use the new `06_run_full_comparison_colab.ipynb` notebook which automatically:
1. Mounts Google Drive
2. Runs the full experiment
3. Saves all results to `My Drive/Negation-Origin-Tracing-Results/`
4. Verifies results were saved

**To use:**
1. Upload `06_run_full_comparison_colab.ipynb` to Colab
2. Run all cells (or run them one by one)
3. Results automatically appear in your Google Drive
4. Download from Drive to your local machine

**Local sync (optional):**
- Install [Google Drive desktop app](https://www.google.com/drive/download/)
- Results sync automatically to your Mac

### Alternative: Manual Download from Colab
- Right-click files in Colab file browser
- Select "Download"

### Alternative: Manual Drive Mount
```python
from google.colab import drive
drive.mount('/content/drive')

# Use drive path
config['output_dir'] = '/content/drive/MyDrive/experiments'
```

### Alternative: Save to GitHub
```python
# After experiments, commit and push
!git add experiments/
!git commit -m "Add experiment results"
!git push
```

## Resume from Checkpoint (NEW!)

**All experiments now support automatic resume!** If your Colab session disconnects or times out, you can simply re-run the same command and it will pick up where it left off.

### How It Works
- Results are saved after **each completed experiment**
- On restart, the script loads existing results and skips completed work
- No data loss from interruptions

### After Disconnection
1. **Reconnect to Colab** (Runtime → Reconnect)
2. **Remount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. **Re-run the exact same command** - it will resume automatically

### Example
```python
# First run (completes 12 out of 18 experiments before disconnect)
!python src/scripts/run_full_experiment.py \
    --output_dir /content/drive/MyDrive/experiments/my_exp

# After reconnecting, run the SAME command
!python src/scripts/run_full_experiment.py \
    --output_dir /content/drive/MyDrive/experiments/my_exp
# Will automatically resume and run only the remaining 6 experiments
```

### Check Progress
```python
import json

# Load results
with open('/content/drive/MyDrive/experiments/my_exp/results_summary.json', 'r') as f:
    results = json.load(f)

# Count completed
completed = sum(1 for r in results if 'error' not in r and r.get('test_auroc', 0) > 0)
print(f"Progress: {completed}/{len(results)} experiments completed")
```

See [`RESUME_GUIDE.md`](RESUME_GUIDE.md) for detailed instructions.

## Tips

1. **Long Experiments**: Use Colab Pro for longer runtimes (12+ hours)
2. **Data Persistence**: Mount Google Drive to save data between sessions
3. **Checkpoints**: Save model checkpoints to Drive to resume training
4. **Monitoring**: Use TensorBoard or print statements to monitor progress
5. **Resume Support**: Don't worry about disconnections - experiments can resume automatically!

## Troubleshooting

### Import Errors
- Make sure you've run `00_setup_colab.ipynb` first
- Check that you're in the correct directory: `os.getcwd()`

### GPU Not Available
- Check Runtime → Change runtime type → GPU is selected
- Restart runtime if needed

### Out of Memory
- Reduce batch size in config
- Use fewer layers or pooling strategies
- Clear variables: `%reset -f`

### Slow Training
- Ensure GPU is enabled
- Check GPU utilization: `!nvidia-smi`
- Reduce batch size if needed

## Next Steps After Colab

1. Download results from Colab
2. Analyze results locally
3. Generate final visualizations
4. Write up findings

