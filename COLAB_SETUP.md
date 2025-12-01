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

### Option 1: Download from Colab
- Right-click files in Colab file browser
- Select "Download"

### Option 2: Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')

# Use drive path
config['output_dir'] = '/content/drive/MyDrive/experiments'
```

### Option 3: Save to GitHub
```python
# After experiments, commit and push
!git add experiments/
!git commit -m "Add experiment results"
!git push
```

## Tips

1. **Long Experiments**: Use Colab Pro for longer runtimes (12+ hours)
2. **Data Persistence**: Mount Google Drive to save data between sessions
3. **Checkpoints**: Save model checkpoints to Drive to resume training
4. **Monitoring**: Use TensorBoard or print statements to monitor progress

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

