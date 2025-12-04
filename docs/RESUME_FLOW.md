# Resume Flow Diagram

## How Resume Works

```
┌─────────────────────────────────────────────────────────────┐
│                    FIRST RUN                                 │
└─────────────────────────────────────────────────────────────┘

Start Experiment
    ↓
Check for results_summary.json
    ↓
Not found → Start fresh
    ↓
┌────────────────────────────────┐
│ Experiment 1: Layer 0, cls     │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
┌────────────────────────────────┐
│ Experiment 2: Layer 0, mean    │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
┌────────────────────────────────┐
│ Experiment 3: Layer 0, token   │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
    ... (more experiments)
    ↓
┌────────────────────────────────┐
│ Experiment 12: Layer 3, token  │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
❌ COLAB DISCONNECTS
    ↓
Progress saved: 12/18 experiments ✓


┌─────────────────────────────────────────────────────────────┐
│                    AFTER RECONNECTION                        │
└─────────────────────────────────────────────────────────────┘

Reconnect to Colab
    ↓
Remount Google Drive
    ↓
Re-run SAME command
    ↓
Check for results_summary.json
    ↓
Found! → Load existing results
    ↓
Identify completed experiments:
  ✓ Layer 0, cls
  ✓ Layer 0, mean
  ✓ Layer 0, token
  ✓ Layer 1, cls
  ... (12 total)
    ↓
Calculate remaining experiments:
  ⏳ Layer 4, cls
  ⏳ Layer 4, mean
  ⏳ Layer 4, token
  ... (6 remaining)
    ↓
Print status:
  "Already completed: 12"
  "Remaining: 6"
    ↓
┌────────────────────────────────┐
│ Experiment 13: Layer 4, cls    │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
┌────────────────────────────────┐
│ Experiment 14: Layer 4, mean   │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
    ... (continue with remaining)
    ↓
┌────────────────────────────────┐
│ Experiment 18: Layer 5, token  │ → Save to results_summary.json
└────────────────────────────────┘
    ↓
✅ ALL EXPERIMENTS COMPLETE
```

## File Structure During Resume

```
/content/drive/MyDrive/experiments/my_exp/
│
├── results_summary.json          ← CHECKPOINT FILE (read on resume)
│   └── Contains:
│       - All completed experiments
│       - Metrics for each
│       - Checkpoint paths
│
├── results_summary.csv           ← CSV version for analysis
│
├── layer_0_pooling_cls/          ← Completed experiment
│   ├── checkpoints/
│   │   ├── best-*.ckpt          ✓ Saved
│   │   └── last.ckpt            ✓ Saved
│   └── events.out.tfevents.*    ✓ Saved
│
├── layer_0_pooling_mean/         ← Completed experiment
│   └── ... (same structure)
│
├── ... (more completed)
│
└── layer_4_pooling_cls/          ← Will be created on resume
    └── ... (created during resume)
```

## Decision Flow

```
                    Start Script
                         ↓
                         ↓
        ┌────────────────────────────────┐
        │ Does results_summary.json      │
        │ exist in output_dir?           │
        └────────────────────────────────┘
                 ↓           ↓
            Yes  │           │  No
                 ↓           ↓
    ┌────────────────┐   ┌──────────────┐
    │ Load existing  │   │ Start fresh  │
    │ results        │   │ (empty list) │
    └────────────────┘   └──────────────┘
                 ↓           ↓
                 └─────┬─────┘
                       ↓
        ┌────────────────────────────────┐
        │ For each result in loaded:     │
        │   If no error AND auroc > 0:   │
        │     Mark (layer, pooling) as   │
        │     completed                  │
        └────────────────────────────────┘
                       ↓
        ┌────────────────────────────────┐
        │ Generate all possible          │
        │ (layer, pooling) combinations  │
        └────────────────────────────────┘
                       ↓
        ┌────────────────────────────────┐
        │ Filter out completed           │
        │ experiments                    │
        └────────────────────────────────┘
                       ↓
        ┌────────────────────────────────┐
        │ Run only remaining             │
        │ experiments                    │
        └────────────────────────────────┘
                       ↓
        ┌────────────────────────────────┐
        │ For each experiment:           │
        │   1. Train probe               │
        │   2. Save checkpoint           │
        │   3. Append to results         │
        │   4. Save results_summary.json │ ← Incremental save
        └────────────────────────────────┘
                       ↓
                    Done!
```

## Experiment Identification

Each experiment is uniquely identified by:

```python
experiment_id = (layer_idx, pooling_strategy)

# Examples:
(0, 'cls')    # Layer 0, CLS pooling
(0, 'mean')   # Layer 0, mean pooling
(1, 'cls')    # Layer 1, CLS pooling
```

An experiment is considered "completed" if:
```python
def is_completed(result):
    return (
        'error' not in result and
        result.get('test_auroc', 0) > 0
    )
```

## Incremental Save Pattern

```python
# Initialize
all_results = []  # or load from existing file

# For each experiment
for layer, pooling in experiments_to_run:
    # Run experiment
    result = train_probe(layer, pooling)
    
    # Append to list
    all_results.append(result)
    
    # Save immediately (incremental save)
    with open('results_summary.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # ✓ Progress is now saved!
    # If script crashes here, this experiment is saved
```

## Resume Example Timeline

```
Time    Event                           Saved Experiments
─────────────────────────────────────────────────────────
10:00   Start experiment                0/18
10:15   Complete Layer 0, cls           1/18 ✓
10:30   Complete Layer 0, mean          2/18 ✓
10:45   Complete Layer 0, token         3/18 ✓
11:00   Complete Layer 1, cls           4/18 ✓
11:15   Complete Layer 1, mean          5/18 ✓
...
13:00   Complete Layer 3, token         12/18 ✓
13:05   ❌ COLAB DISCONNECTS            12/18 saved
─────────────────────────────────────────────────────────
13:30   Reconnect to Colab              12/18 loaded
13:31   Remount Drive                   12/18 loaded
13:32   Re-run same command             12/18 loaded
13:33   Script detects 12 completed     
13:34   Resume with 6 remaining         
13:35   Complete Layer 4, cls           13/18 ✓
13:50   Complete Layer 4, mean          14/18 ✓
...
15:00   Complete Layer 5, token         18/18 ✓
15:01   ✅ ALL DONE
```

## Key Points

1. **Automatic**: No special flags needed (resume is default)
2. **Incremental**: Saves after EACH experiment
3. **Safe**: No data loss from interruptions
4. **Transparent**: Shows progress on restart
5. **Flexible**: Can start fresh with `--no_resume`

## Common Scenarios

### Scenario 1: Fresh Start
```
No results_summary.json
    ↓
Run all 18 experiments
    ↓
Save after each
    ↓
Complete all 18
```

### Scenario 2: Resume Once
```
12/18 completed before disconnect
    ↓
Load 12 existing results
    ↓
Run remaining 6
    ↓
Complete all 18
```

### Scenario 3: Resume Multiple Times
```
First run: 6/18 completed → disconnect
    ↓
Resume: Load 6, run 6 more (12 total) → disconnect
    ↓
Resume: Load 12, run 6 more (18 total) → complete!
```

### Scenario 4: Partial Failure
```
10/18 completed, 1 failed, 7 remaining
    ↓
Load 10 successful results
    ↓
Skip 10 completed
    ↓
Re-run 1 failed + 7 remaining = 8 experiments
```

---

**Remember:** Just re-run the same command. The script handles everything! 🎯

