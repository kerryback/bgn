# Quick Start Guide - Refactor2

## Installation

```bash
cd Refactor2
pip install -r requirements.txt
```

**Important:** The simulation requires `Jstar.csv` in the Refactor2 directory. This file is already included and contains interpolation data for the BGN model's growth option valuation.

## Basic Usage

### Run with default parameters

```bash
python main.py
```

This runs 10 iterations of the GS model with N=100 firms and T=400 time periods.

### Customize parameters

```bash
python main.py \
    --model gs \
    --N 100 \
    --T 400 \
    --num-iters 10 \
    --n-jobs 10 \
    --output-dir results/experiment_1 \
    --log-level INFO
```

### Use a configuration file

Create `config.yaml`:
```yaml
model: gs
N: 100
T: 400
num_iters: 10
n_jobs: 10
output_dir: results/experiment_1
```

Then run:
```bash
python main.py --config config.yaml
```

## Understanding the Output

After running, you'll find these files in your output directory:

### Results Files

- `results_model_gs.csv` - Latent factor model metrics
- `results_dkkm_gs.csv` - DKKM method metrics
- `results_fama_gs.csv` - Fama-French/MacBeth metrics
- `results_ipca_gs.csv` - IPCA metrics

Each contains columns:
- `iter`: Iteration number
- `month`: Month number
- Method-specific identifiers (e.g., `alpha`, `nfeatures`)
- `stdev`: Portfolio standard deviation
- `mean`: Expected return
- `xret`: Realized return
- `hjd`: Hansen-Jagannathan distance

### Portfolio Weights Files (Optional)

- `port_model_gs.csv` - Model portfolio weights
- `port_dkkm_gs.csv` - DKKM portfolio weights
- `port_fama_gs.csv` - Fama portfolio weights
- `port_ipca_gs.csv` - IPCA portfolio weights

Each contains the same identifier columns plus `firm_1`, `firm_2`, ..., `firm_N` with portfolio weights.

### Panel Data File (Optional)

- `panel_gs.csv` - All panel data across iterations

Contains firm-month observations with characteristics and returns.

### Log File

- `simulation.log` - Detailed execution log

## Python API Usage

```python
from src.config import SimulationConfig
from src.panel_processor import PanelProcessor
from src.io import CSVResultsWriter
from src.utils import setup_logging

# Setup
setup_logging(level='INFO')

# Configure
config = SimulationConfig(
    model='gs',
    N=100,
    T=400,
    num_iters=10,
    n_jobs=10,
    output_dir='results'
)

# Initialize writer
writer = CSVResultsWriter(config.output_dir)
writer.initialize(N=config.N, model=config.model)

# Create processor
processor = PanelProcessor(config)

# Run iterations
for i in range(config.num_iters):
    iter_num = config.start_iter + i
    summary = processor.process_iteration(iter_num, writer)
    print(f"Iteration {iter_num}: {summary['months_processed']} months processed")

# Finalize
writer.finalize()
print(f"Results saved to {config.output_dir}")
```

## Key Differences from Original Code

### Original Code (main.py)

```python
# Everything in one big function
def run_panel(iter):
    arr_tuple = panels[model].create_arrays(N, T+burnin)
    panel = panels[model].create_panel(N, T+burnin, arr_tuple)
    sdf_loop = sdf[model].sdf_compute(N, T+burnin, arr_tuple)
    # ... 378 lines of analysis logic ...
    # CSV writes scattered throughout

for iter in range(numiters):
    run_panel(iter)
```

### Refactor2

```python
# Clean separation
config = SimulationConfig(...)
processor = PanelProcessor(config)
writer = CSVResultsWriter(config.output_dir)
writer.initialize(N=config.N, model=config.model)

for iter in range(config.num_iters):
    processor.process_iteration(iter, writer)

writer.finalize()
```

**Benefits:**
- Modular and testable
- Type-safe configuration
- Separated I/O
- Clear structure

## Key Differences from Refactor v1

### Refactor v1 (Two-Stage - BROKEN)

```bash
# Stage 1: Generate panels
python generate_panels.py --model gs --num-iters 10
# Saves: panel_gs_iter0.csv, panel_gs_iter1.csv, ...

# Stage 2: Analyze panels (PROBLEM: Missing SDF data!)
python analyze_panels.py --panel-dir panels/
```

**Problem:** Can't compute conditional covariances and risk premia without `arr_tuple` and `sdf_loop`!

### Refactor v2 (Unified - CORRECT)

```bash
# One stage: Generate + Analyze together
python main.py --model gs --num-iters 10
```

**Solution:** Keep generation and analysis together so all data is available.

## Testing

### Small Test Run

```python
from src.config import SimulationConfig
from src.panel_processor import PanelProcessor
from src.io import InMemoryResultsWriter
from src.utils import setup_logging

# Small test
setup_logging(level='DEBUG')
config = SimulationConfig(model='gs', N=10, T=50, num_iters=2, n_jobs=1)

writer = InMemoryResultsWriter()
writer.initialize(N=10, model='gs')

processor = PanelProcessor(config)

for i in range(2):
    summary = processor.process_iteration(i, writer)
    print(f"Iteration {i}: {summary}")

# Check results
results = writer.to_dataframes()
print(f"Generated {len(results)} result types")
print(f"Model results shape: {results.get('model', pd.DataFrame()).shape}")
```

## Troubleshooting

### Import Errors

If you get import errors for `panel_functions`, `sdf_compute`, etc.:

```python
# The processor automatically adds the parent directory to sys.path
# But if running from a different location, you may need to:
import sys
sys.path.insert(0, '/path/to/Code')  # Root directory with original code
```

### Memory Issues

For very large simulations (N > 500, T > 1000):
- Reduce `n_jobs` to avoid memory pressure from parallelization
- Consider processing fewer iterations at a time
- Monitor memory usage

### Performance

Typical timings (N=100, T=400):
- Panel generation: ~10 seconds
- SDF computation: ~30 seconds
- Factor estimation: ~2 minutes
- Monthly evaluation: ~20 minutes
- **Total per iteration: ~25 minutes**

For 100 iterations: ~42 hours

Use more parallel jobs (`--n-jobs 20`) to speed up factor estimation.

## Next Steps

1. **Explore the architecture**: Read [ARCHITECTURE.md](ARCHITECTURE.md)
2. **Understand the code**: Read [README.md](README.md)
3. **Customize**: Modify `src/panel_processor.py` to add new methods
4. **Extend**: Add new portfolio strategies or output formats

## Common Patterns

### Running Multiple Models

```bash
for model in bgn kp gs; do
    python main.py \
        --model $model \
        --num-iters 100 \
        --output-dir results/$model \
        --log-level INFO
done
```

### Experimenting with Parameters

```python
# Test different regularization parameters
for alpha_list in [[0], [0, 0.01, 0.1], [0, 0.001, 0.01, 0.1, 1]]:
    config = SimulationConfig(
        model='gs',
        alpha_lst=alpha_list,
        output_dir=f'results/alpha_{len(alpha_list)}'
    )
    # ... run simulation ...
```

### Saving Only Specific Results

Modify `CSVResultsWriter` to skip portfolio weights:

```python
class CompactCSVWriter(CSVResultsWriter):
    def write_monthly_results(self, results):
        # Only write metrics, not weights
        # Saves disk space
        ...
```

## Support

For questions or issues:
1. Check [README.md](README.md) for overview
2. Check [ARCHITECTURE.md](ARCHITECTURE.md) for design details
3. Look at the code comments
4. Read the original `main.py` for algorithm details

## Summary

Refactor2 provides a **clean, unified workflow** that:
- ✅ Keeps panel generation and analysis together
- ✅ Maintains all necessary data availability
- ✅ Provides type-safe configuration
- ✅ Separates concerns cleanly
- ✅ Easy to use and extend

Just run `python main.py` and you're good to go!
