# Jupyter Notebooks for Google Colab

This directory contains Jupyter notebooks for running experiments in Google Colab.

## Setup Instructions

1. **Open Google Colab** and create a new notebook
2. **Run the setup notebook** (`00_setup_colab.ipynb`) first to:
   - Install dependencies
   - Clone the repository
   - Set up the environment

3. **Run notebooks in order:**
   - `00_setup_colab.ipynb` - Initial setup
   - `01_download_data.ipynb` - Download datasets
   - `02_layer_search.ipynb` - Train probes on all layers
   - `03_visualize_results.ipynb` - Generate visualizations
   - `04_full_experiment.ipynb` - Run complete pipeline
   - `05_base_vs_finetuned_comparison.ipynb` - Analysis of base vs finetuned models
   - `06_run_full_comparison_colab.ipynb` - **AUTOMATED: Full experiment with Drive sync**

## Quick Start

### Recommended: Clone from GitHub in Colab

1. **Open Google Colab**: Go to [colab.research.google.com](https://colab.research.google.com)

2. **Enable GPU**: Runtime → Change runtime type → Select "GPU" → Save

3. **Clone repository**:
   ```python
   !git clone https://github.com/TheMattWang/Negation-Origin-Tracing.git
   %cd Negation-Origin-Tracing
   ```

4. **Open and run notebooks in order**:
   - `notebooks/00_setup_colab.ipynb` - Setup (installs dependencies)
   - `notebooks/01_download_data.ipynb` - Download datasets
   - `notebooks/02_layer_search.ipynb` - Train probes on all layers
   - `notebooks/03_visualize_results.ipynb` - Generate plots
   - `notebooks/04_full_experiment.ipynb` - Complete pipeline
   - `notebooks/06_run_full_comparison_colab.ipynb` - **AUTOMATED: Full comparison with Drive sync**

### Alternative: Upload notebooks
1. Upload notebooks from `notebooks/` directory to Colab
2. Run them in sequence (still need to clone repo for code)

## Notebook Descriptions

### 00_setup_colab.ipynb
- Installs required packages
- Clones repository (if needed)
- Verifies installation
- Checks GPU availability

### 01_download_data.ipynb
- Downloads SST-2 dataset
- Downloads CSD Negation dataset (if available)
- Verifies data files

### 02_layer_search.ipynb
- Trains probes on all layers
- Tests different pooling strategies
- Saves results for analysis

### 03_visualize_results.ipynb
- Generates performance plots
- Creates heatmaps and comparisons
- Displays visualizations inline

### 04_full_experiment.ipynb
- Runs complete experiment pipeline
- Includes probe training, visualization, interventions, and interpretation
- Generates final report

### 05_base_vs_finetuned_comparison.ipynb
- Analyzes differences between base and finetuned models
- Compares probe performance and intervention effects
- Generates comparison visualizations

### 06_run_full_comparison_colab.ipynb ⭐ **RECOMMENDED FOR AUTOMATION**
- **Fully automated end-to-end experiment**
- Automatically mounts Google Drive
- Runs complete base vs finetuned comparison
- Saves all results to Drive automatically
- Includes result verification and preview
- **Best for long-running experiments (2-4 hours)**

## Tips for Colab

1. **Enable GPU**: Runtime → Change runtime type → GPU
2. **Save results**: Use `06_run_full_comparison_colab.ipynb` for automatic Drive sync
3. **Long experiments**: Use Colab Pro for longer runtimes (12+ hours)
4. **Data persistence**: Mount Google Drive to save data between sessions

## Automated Result Saving (Recommended)

The `06_run_full_comparison_colab.ipynb` notebook handles everything automatically:
- Mounts Google Drive
- Runs experiments
- Saves to `My Drive/Negation-Origin-Tracing-Results/`
- Verifies results were saved

**To access results locally:**
1. Install [Google Drive desktop app](https://www.google.com/drive/download/)
2. Results sync automatically to your computer
3. Or download manually from Drive web interface

## Manual Google Drive Mount (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')

# Use drive path for data/results
config['output_dir'] = '/content/drive/MyDrive/experiments'
```

