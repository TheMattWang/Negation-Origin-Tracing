# Parallel Execution - Known Issues

## Issue: No Output with Parallel Script

### Symptoms
- Script runs but no output is displayed
- RAM and CPU usage increases
- Processes appear to be running but no progress shown
- Output buffering issues

### Root Cause
When using `multiprocessing.set_start_method('spawn')` (required for CUDA), child process output is not properly flushed to the parent process stdout in real-time.

### Current Status
**Temporarily reverted to sequential execution** in `run_full_comparison.sh` to ensure reliable output and monitoring.

### Workarounds

#### Option 1: Use Sequential Script (Recommended for now)
```bash
python src/scripts/search_layers.py \
    --output_dir experiments/my_exp \
    --layers all
```

**Pros:**
- Reliable output
- Easy to monitor progress
- Stable and well-tested

**Cons:**
- Slower (~3-4 hours for full search)

#### Option 2: Use Parallel Script Directly (Advanced)
```bash
# Run parallel script with output redirection
python src/scripts/search_layers_parallel.py \
    --output_dir experiments/my_exp \
    --layers all \
    --parallel_workers 3 \
    --parallel_mode pooling 2>&1 | tee output.log

# Monitor progress in another terminal
watch -n 5 "cat experiments/my_exp/results_summary.json | jq length"
```

**Pros:**
- Faster (~1-1.5 hours)

**Cons:**
- Limited output visibility
- Harder to monitor
- May appear frozen

#### Option 3: Run Experiments Sequentially but Manually in Parallel

Run different layers in different terminals:

```bash
# Terminal 1
python src/scripts/search_layers.py --layers 0,1 --output_dir exp/part1

# Terminal 2  
python src/scripts/search_layers.py --layers 2,3 --output_dir exp/part2

# Terminal 3
python src/scripts/search_layers.py --layers 4,5 --output_dir exp/part3
```

Then merge results.

### Monitoring Progress

Even without output, you can monitor progress:

```bash
# Check number of completed experiments
cat experiments/my_exp/results_summary.json | python -m json.tool | grep "layer_idx" | wc -l

# Check for checkpoint directories
ls -la experiments/my_exp/*/checkpoints/ | grep "best"

# Monitor GPU usage
watch -n 1 nvidia-smi

# Check process status
ps aux | grep python
```

### Future Fix

We're investigating solutions:
1. Use `logging` module with proper handlers for multiprocessing
2. Implement a progress monitoring subprocess
3. Use shared memory for progress updates
4. Consider alternative parallelization (threading vs multiprocessing)
5. Use `tqdm` with multiprocessing support

### Recommendation

**For production runs:** Use sequential script (`search_layers.py`)
- Reliable
- Good output
- Easy to monitor
- Resume works perfectly

**For experimentation:** Try parallel script with monitoring
- Faster
- Check progress via file system
- Use resume if interrupted

## Related Files

- `src/scripts/search_layers.py` - Sequential (reliable)
- `src/scripts/search_layers_parallel.py` - Parallel (faster but output issues)
- `run_full_comparison.sh` - Currently uses sequential

## Updates

- **2024-12-02:** Identified output buffering issue with spawn multiprocessing
- **2024-12-02:** Reverted `run_full_comparison.sh` to sequential for reliability
- **Future:** Will fix output handling in parallel script

---

**Status:** Known issue, workaround in place  
**Priority:** Medium (functionality works, just output visibility)  
**Tracking:** Will fix in next update

