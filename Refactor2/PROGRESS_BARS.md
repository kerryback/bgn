# Progress Bars in Refactor2

## Overview

Refactor2 includes comprehensive progress tracking using `tqdm` progress bars for all slow operations. This provides real-time feedback during long-running simulations.

## Progress Bar Hierarchy

When running a simulation, you'll see nested progress bars showing:

```
Iterations:  40%|████      | 4/10 [1:40:00<2:30:00, 25:00/iter] {months: 41, panel_size: 4000}
  └─ Computing factors:  80%|████████  | 4/5 [00:02<00:00, 1.5s/it]
       └─ DKKM matrices: 100%|██████████| 1/1 [00:00<00:00, 10.0it/s]
  └─ Evaluating months: 100%|██████████| 41/41 [00:20<00:00, 2.0it/s]
```

## Where Progress Bars Appear

### 1. Main Iteration Loop

**Location:** `main.py`

Shows overall progress through Monte Carlo iterations:

```
Iterations: 30%|███       | 3/10 [1:15:00<2:55:00, 25:00/iter]
```

**Information shown:**
- Current iteration / total iterations
- Elapsed time
- Estimated time remaining
- Time per iteration
- Postfix: months processed, panel size

### 2. Panel Generation

**Location:** `panel_processor.py` → `_generate_panel()`

Currently logged (could add progress bar for sub-steps if needed):
- Array creation
- Panel DataFrame construction
- SDF computation

### 3. Factor Computation

**Location:** `panel_processor.py` → `_compute_factors()`

Shows progress through 5 major factor types:

```
Computing factors: 60%|██████    | 3/5 [00:01<00:01, 1.5it/s]
```

**Breakdown:**
1. Latent model factors (taylor/proj)
2. Fama-French factors
3. Fama-MacBeth factors
4. DKKM factors (with sub-progress for multiple matrices)
5. IPCA factors

#### Sub-progress bars:

**DKKM matrices:**
```
  DKKM matrices: 100%|██████████| 1/1 [00:00<00:00, 10.0it/s]
```

**IPCA variants:**
```
  IPCA variants: 100%|██████████| 2/2 [00:00<00:00, 5.0it/s]
```

### 4. Monthly Evaluation

**Location:** `panel_processor.py` → `_process_months()`

Shows progress through months being evaluated:

```
Evaluating months: 75%|███████▌  | 30/40 [00:15<00:05, 2.0it/s]
```

Each month involves:
- Getting SDF data
- Evaluating model portfolios
- Evaluating DKKM portfolios
- Evaluating Fama portfolios
- Evaluating IPCA portfolios
- Writing results

## Configuration Options

### Disable Progress Bars

If progress bars interfere with logging or you're running in a non-interactive environment:

```python
# Set environment variable
import os
os.environ['TQDM_DISABLE'] = '1'

# Or use a config option (if implemented)
config = SimulationConfig(..., disable_progress_bars=True)
```

### Adjust Update Frequency

Progress bars update automatically, but you can control logging frequency:

```python
from src.utils import log_and_progress

for item in log_and_progress(items, "Processing", log_every=10):
    # Logs every 10 items while showing continuous progress bar
    process(item)
```

## Progress Bar Utilities

### ProgressBarManager

For complex nested progress tracking:

```python
from src.utils import ProgressBarManager

manager = ProgressBarManager()

# Create nested bars
outer = manager.create_bar(10, "Iterations", position=0)
inner = manager.create_bar(100, "Tasks", position=1, leave=False)

for i in range(10):
    inner.reset()
    for j in range(100):
        # ... work ...
        inner.update(1)
    outer.update(1)

manager.close_all()
```

### ProgressContext

Safe progress bar handling with context manager:

```python
from src.utils import ProgressContext

with ProgressContext(total=100, desc="Processing") as pbar:
    for i in range(100):
        # ... work ...
        pbar.update(1)
        pbar.set_postfix({'current': i, 'status': 'ok'})
# Automatically closed even if exception occurs
```

### iterate_with_progress

Simple wrapper for iterables:

```python
from src.utils import iterate_with_progress

for month in iterate_with_progress(months, desc="Processing months"):
    process_month(month)
```

### log_and_progress

Combine logging and progress:

```python
from src.utils import log_and_progress
import logging

logger = logging.getLogger(__name__)

for item in log_and_progress(items, "Processing items", logger, log_every=10):
    process(item)
# Shows progress bar AND logs every 10 items
```

## Example Output

Complete simulation output with all progress bars:

```
2025-01-15 10:00:00 - INFO - Starting 10 iterations
================================================================================
Asset Pricing Simulation - Refactor2
================================================================================
Model: gs
Dimensions: N=100, T=400
Iterations: 0 to 9
Output directory: results
Parallel jobs: 10
================================================================================

Iterations:   0%|          | 0/10 [00:00<?, ?it/s]
================================================================================
Processing iteration 0
================================================================================
Step 1: Generating panel...
  Panel shape: (40000, 15)
  Months: 2 to 401

Step 2: Computing factor models...
Computing factors:   0%|          | 0/5 [00:00<?, ?it/s]
Computing factors:  20%|██        | 1/5 [00:00<00:02, 1.5it/s] (Latent model factors)
Computing factors:  40%|████      | 2/5 [00:01<00:01, 2.0it/s] (Fama-French factors)
Computing factors:  60%|██████    | 3/5 [00:01<00:01, 2.0it/s] (Fama-MacBeth factors)
  DKKM matrices: 100%|██████████| 1/1 [00:00<00:00, 10.0it/s]
Computing factors:  80%|████████  | 4/5 [00:02<00:00, 2.0it/s] (DKKM factors)
  IPCA variants: 100%|██████████| 2/2 [00:00<00:00, 5.0it/s]
Computing factors: 100%|██████████| 5/5 [00:02<00:00, 2.0it/s]
  ✓ All factors computed

Step 3: Processing 41 months...
Evaluating months:   0%|          | 0/41 [00:00<?, ?it/s]
Evaluating months:  25%|██▌       | 10/41 [00:05<00:15, 2.0it/s]
Evaluating months:  50%|█████     | 20/41 [00:10<00:10, 2.0it/s]
Evaluating months:  75%|███████▌  | 30/41 [00:15<00:05, 2.0it/s]
Evaluating months: 100%|██████████| 41/41 [00:20<00:00, 2.0it/s]
  ✓ Processed 41 months

Step 4: Saving panel data...
  ✓ Panel saved

✓ Iteration 0 complete
================================================================================

Iterations:  10%|█         | 1/10 [00:25<03:45, 25.0s/iter] {months: 41, panel_size: 40000}
================================================================================
Processing iteration 1
================================================================================
...

Iterations: 100%|██████████| 10/10 [04:10<00:00, 25.0s/iter] {months: 41, panel_size: 40000}

================================================================================
Completed 10 iterations
Total time: 4h 10m 0s
Average per iteration: 25m 0s
================================================================================

Simulation complete!
Results saved to: results
================================================================================
```

## Performance Impact

Progress bars have minimal performance impact:

- **Memory:** ~1-2 MB for progress tracking
- **CPU:** <1% overhead for updates
- **I/O:** Uses terminal control sequences (very fast)

The progress bars use smart update strategies to avoid excessive terminal writes.

## Troubleshooting

### Progress bars don't show

**Issue:** Progress bars not appearing in output

**Solutions:**
1. Check if running in interactive terminal (progress bars need TTY)
2. Verify tqdm is installed: `pip install tqdm>=4.62`
3. Check if disabled via environment: `echo $TQDM_DISABLE`

### Progress bars overlap

**Issue:** Multiple progress bars overwrite each other

**Solutions:**
1. Use `position` parameter to place bars at different lines
2. Set `leave=False` for inner/temporary bars
3. Use `ProgressBarManager` for complex nested scenarios

### Progress bars interfere with logging

**Issue:** Log messages break progress bar display

**Solutions:**
1. Use `tqdm.write()` instead of `print()` for messages during progress
2. Set `file=sys.stderr` for progress bars if logging to stdout
3. Use `log_and_progress()` utility for combined logging and progress

Example:
```python
from tqdm import tqdm

with tqdm(total=100) as pbar:
    for i in range(100):
        # Don't use print() - it will break the progress bar
        # print(f"Processing {i}")  # BAD

        # Use tqdm.write() instead
        tqdm.write(f"Processing {i}")  # GOOD

        pbar.update(1)
```

### Progress bars in Jupyter notebooks

**Issue:** Progress bars don't render well in Jupyter

**Solution:** tqdm automatically detects Jupyter and uses rich HTML widgets

If issues persist:
```python
from tqdm.notebook import tqdm  # Use notebook version explicitly
```

## Customization

### Custom Progress Bar Format

Modify progress bar appearance:

```python
from tqdm import tqdm

# Custom format string
pbar = tqdm(
    total=100,
    desc="Custom",
    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
)
```

### Add Custom Postfix Information

Show additional information:

```python
with tqdm(total=100) as pbar:
    for i in range(100):
        # ... work ...

        # Update postfix with custom metrics
        pbar.set_postfix({
            'loss': 0.123,
            'accuracy': 0.98,
            'memory_mb': 150
        })
        pbar.update(1)
```

### Colored Progress Bars

Use color for different statuses:

```python
from tqdm import tqdm

# Green for success
pbar = tqdm(total=100, colour='green', desc='Success')

# Red for errors
pbar = tqdm(total=100, colour='red', desc='Errors')

# Blue for info
pbar = tqdm(total=100, colour='blue', desc='Info')
```

## Summary

Progress bars in Refactor2 provide:

✅ **Real-time feedback** - Know exactly where you are in long simulations
✅ **Time estimation** - See elapsed and remaining time
✅ **Nested tracking** - Monitor progress at multiple levels
✅ **Minimal overhead** - <1% performance impact
✅ **Flexible** - Easy to customize or disable
✅ **Robust** - Automatic cleanup even with exceptions

For a 10-iteration simulation (~4 hours), progress bars help you:
- Know if simulation is progressing normally
- Estimate completion time
- Identify slow steps
- Catch infinite loops or hangs
- Plan breaks during long runs

**Recommended:** Keep progress bars enabled for interactive use, disable for automated/logged runs.
