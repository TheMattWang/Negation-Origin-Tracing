# 🚀 Quick Start: Automated Colab Experiments

## For Colab (Automated)

### 1. Upload & Setup (2 minutes)
```
1. Go to: https://colab.research.google.com
2. Upload: notebooks/06_run_full_comparison_colab.ipynb
3. Runtime → Change runtime type → GPU → Save
```

### 2. Run Experiment (2-4 hours)
```
1. Click: Runtime → Run all
2. Click: "Allow" when Drive asks for permission
3. Wait for completion (monitor progress in cells)
```

### 3. Get Results (5 minutes)
```
Results automatically saved to:
Google Drive > My Drive > Negation-Origin-Tracing-Results/

Option A: Download from Drive
  1. Go to drive.google.com
  2. Navigate to folder
  3. Right-click → Download

Option B: Auto-sync (Recommended)
  1. Install Google Drive desktop app
  2. Results appear at: ~/Google Drive/My Drive/Negation-Origin-Tracing-Results/
  3. Copy to repo:
     cp -r ~/Google\ Drive/My\ Drive/Negation-Origin-Tracing-Results/comparison_* \
           ~/Documents/NLP/Negation-Origin-Tracing/experiments/
```

## For Local Mac (Manual)

### Run Experiment
```bash
cd ~/Documents/NLP/Negation-Origin-Tracing
bash run_full_comparison.sh
```

### Results Location
```
experiments/comparison_YYYYMMDD_HHMMSS/
```

## What You Get

```
comparison_YYYYMMDD_HHMMSS/
├── base_probes/
│   └── results_summary.json          # Which layer encodes negation best?
├── base_interventions/
│   └── intervention_results.json     # How does base model use negation?
├── finetuned_interventions/
│   └── intervention_results.json     # How does finetuned model use negation?
└── comparison_summary.json           # Overall comparison
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| No GPU in Colab | Runtime → Change runtime type → GPU |
| Drive not mounted | Run first cell again |
| Results not in Drive | Check last cell for path, or see COLAB_QUICK_START.md |
| Colab disconnected | Results still saved to Drive, reconnect and check |

## Next Steps

1. **Analyze Results:**
   ```bash
   jupyter notebook notebooks/05_base_vs_finetuned_comparison.ipynb
   ```

2. **Update result path in notebook:**
   ```python
   RESULTS_DIR = "experiments/comparison_YYYYMMDD_HHMMSS"
   ```

3. **Generate visualizations and insights**

## Need More Help?

- **Colab setup and troubleshooting:** `COLAB_QUICK_START.md`
- **Colab detailed setup:** `COLAB_QUICK_START.md`
- **Resume from checkpoint:** `docs/RESUME_GUIDE.md`

---

**That's it! Upload → Run → Download → Analyze** 🎉
