# Scripts Directory

This directory contains Python scripts for running experiments and analysis.

## Main Experiment Scripts

- **`run_final_comparison.py`** - Analyze probe results and run interventions on best layer
  - Merges results from parallel sweeps
  - Identifies best layer/pooling combination
  - Runs causal interventions
  - Generates visualizations

- **`run_interventions_comparison.py`** - Run interventions and compare base vs finetuned models
  - Loads best probe from sweep results
  - Runs interventions on base model (with probe)
  - Runs interventions on finetuned model
  - Generates comparison report

## Subdirectories

- **`utils/`** - Utility scripts for data analysis and result processing
- **`tests/`** - Test scripts for verifying functionality

## Usage

These scripts are typically called from:
- Shell scripts (e.g., `run_full_comparison.sh`)
- Jupyter notebooks (e.g., `notebooks/08_final_analysis.ipynb`)

For standalone usage, see individual script docstrings.

