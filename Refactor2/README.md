# Refactor2: Unified Panel Generation and Analysis

## Overview

This is a refactored version of the asset pricing simulation code that **keeps panel generation and analysis together**, correcting the mistake in the original Refactor where they were separated.

## The Problem with Refactor (v1)

The original Refactor separated the workflow into two stages:
1. **Stage 1 (generate_panels.py)**: Generate panel data and save to disk
2. **Stage 2 (analyze_panels.py)**: Load panel data and analyze

**The Critical Mistake**: Panel analysis requires data from panel generation that wasn't being saved:
- `arr_tuple` - Raw arrays from model simulation
- `sdf_loop` - SDF computation function that needs `arr_tuple`
- Conditional covariances, risk premia computed from SDF data
- These are needed for portfolio evaluation but aren't available from just the panel CSV

## The Solution in Refactor2

**Keep panel generation and analysis together in a single iteration.** The data flow is:

```
for each iteration:
  1. Generate panel (create_arrays → create_panel)
  2. Compute SDF data (sdf_compute with arr_tuple)
  3. Compute all factor models (still in memory)
  4. For each month:
      - Use SDF data to get cond_var, risk_premia
      - Evaluate all portfolios
      - Write results
  5. Optionally save panel for reference
```

All the data needed for analysis stays in memory from generation.

## Key Architectural Principles

1. **Single Iteration Workflow**: One function processes an entire iteration from generation through analysis
2. **Modular Strategies**: Factor models implemented as strategy classes (from Refactor v1)
3. **Separation of I/O**: Results writing decoupled from computation
4. **Type Safety**: Configuration dataclasses with validation
5. **Clean Utilities**: Logging, progress tracking, constants

## Directory Structure

```
Refactor2/
├── src/
│   ├── config.py                   # Configuration management
│   ├── panel_processor.py          # Core: unified panel generation + analysis
│   ├── strategies/                 # Portfolio strategies (adapted from Refactor)
│   │   ├── fama_strategies.py
│   │   ├── dkkm_strategies.py
│   │   ├── ipca_strategies.py
│   │   └── portfolio_evaluation.py
│   ├── io/
│   │   └── results_writer.py       # CSV and in-memory writers
│   └── utils/
│       ├── constants.py            # Named constants
│       ├── exceptions.py           # Custom exceptions
│       ├── logging_config.py       # Logging setup
│       └── progress.py             # Progress tracking
├── main.py                         # Main execution script
├── README.md                       # This file
└── setup.py                        # Setup script

```

## Usage

### Basic Usage

```python
from src.config import SimulationConfig
from src.panel_processor import PanelProcessor
from src.io import CSVResultsWriter

# Create configuration
config = SimulationConfig(
    model='gs',
    N=100,
    T=400,
    num_iters=10,
    n_jobs=10
)

# Create results writer
writer = CSVResultsWriter(config.output_dir)
writer.initialize(N=config.N, model=config.model)

# Create processor
processor = PanelProcessor(config)

# Run simulation
for i in range(config.num_iters):
    iter_num = config.start_iter + i

    # Process entire iteration: generation + analysis
    results = processor.process_iteration(iter_num, writer)

writer.finalize()
```

### From Command Line

```bash
# Run with default parameters
python main.py --model gs --num-iters 10

# Custom configuration
python main.py \
    --model gs \
    --N 200 \
    --T 600 \
    --num-iters 100 \
    --n-jobs 20 \
    --output-dir results/experiment_1

# Load from config file
python main.py --config config.yaml
```

## Comparison with Original Code

### Original Code (main.py)

```python
def run_panel(iter):
    # Generate panel
    arr_tuple = panels[model].create_arrays(N, T+burnin)
    panel = panels[model].create_panel(N, T+burnin, arr_tuple)

    # Compute SDF
    sdf_loop = sdf[model].sdf_compute(N, T+burnin, arr_tuple)

    # ... compute factors ...

    # Process each month
    def run_month(month, iter):
        sdf_ret, max_sr, rp, cond_var = sdf_loop(month-1, iter)
        # ... evaluate portfolios ...

    for month in months:
        run_month(month, iter)
```

**Issues:**
- 378-line monolithic function
- Global variables everywhere
- I/O mixed with computation
- Hard to test

### Refactor v1 (generate_panels.py + analyze_panels.py)

```python
# Stage 1: Generate
def generate_single_panel(model, N, T, burnin, iter):
    arr_tuple = PANELS[model].create_arrays(N, T + burnin)
    panel = PANELS[model].create_panel(N, T + burnin, arr_tuple)
    # Save panel to disk
    panel.to_csv(f'panel_{model}_iter{iter}.csv')
    # BUT: arr_tuple and sdf_loop are lost!

# Stage 2: Analyze (later, separate process)
def analyze_panel(panel, metadata, config, writer, iter):
    # Load panel from disk
    panel = pd.read_csv(panel_file)
    # Problem: No access to arr_tuple or sdf_loop!
    # Can't compute conditional covariances, risk premia
```

**Issues:**
- ❌ SDF data not available in Stage 2
- ❌ Can't compute conditional covariances
- ❌ Can't evaluate portfolios properly
- ❌ Need to recompute or save/load arr_tuple

### Refactor v2 (This Version - panel_processor.py)

```python
class PanelProcessor:
    def process_iteration(self, iter: int, writer: ResultsWriter):
        # 1. Generate panel
        arr_tuple = self.panels[self.model].create_arrays(N, T + burnin)
        panel = self.panels[self.model].create_panel(N, T + burnin, arr_tuple)

        # 2. Compute SDF (arr_tuple still in memory!)
        sdf_loop = self.sdf[self.model].sdf_compute(N, T + burnin, arr_tuple)

        # 3. Initialize strategies with panel
        strategies = self._initialize_strategies(panel)

        # 4. Process each month (sdf_loop available!)
        for month in months:
            # Get SDF data
            sdf_ret, max_sr, rp, cond_var = sdf_loop(month-1, iter)

            # Evaluate all strategies
            results = self._evaluate_month(
                month, iter, panel, strategies,
                rp, cond_var, sdf_ret, max_sr
            )

            # Write results
            writer.write_monthly_results(results)

        # 5. Optionally save panel
        writer.write_panel(panel, iter)
```

**Benefits:**
- ✅ All data available when needed
- ✅ Clean separation of concerns
- ✅ Modular strategies
- ✅ Type safe
- ✅ Testable
- ✅ Maintains single-iteration workflow

## Key Classes and Functions

### PanelProcessor

Main class that orchestrates panel generation and analysis:

```python
class PanelProcessor:
    def process_iteration(self, iter: int, writer: ResultsWriter) -> IterationResults:
        """Process one complete iteration: generation + analysis."""

    def _generate_panel(self, iter: int) -> Tuple[pd.DataFrame, dict]:
        """Generate panel and SDF data."""

    def _initialize_strategies(self, panel: pd.DataFrame) -> Dict[str, Any]:
        """Initialize all factor model strategies."""

    def _evaluate_month(
        self,
        month: int,
        iter: int,
        panel: pd.DataFrame,
        strategies: Dict,
        risk_premia: np.ndarray,
        cond_var: np.ndarray,
        sdf_ret: float,
        max_sr: float
    ) -> MonthlyResults:
        """Evaluate all strategies for one month."""
```

### ResultsWriter

Interface for writing results:

```python
class ResultsWriter(Protocol):
    def initialize(self, N: int, model: str) -> None: ...
    def write_monthly_results(self, results: MonthlyResults) -> None: ...
    def write_panel(self, panel: pd.DataFrame, iter: int) -> None: ...
    def finalize(self) -> None: ...
```

Implementations:
- `CSVResultsWriter`: Writes to CSV files
- `InMemoryResultsWriter`: Accumulates in memory for testing

## Benefits Over Both Previous Versions

### vs. Original Code

| Original | Refactor2 |
|----------|-----------|
| 378-line function | Modular classes (~50 lines each) |
| Global variables | Dependency injection |
| Mixed I/O | Separated writers |
| No types | Full type hints |
| No tests | Testable architecture |
| Hard to understand | Clear structure |

### vs. Refactor v1

| Refactor v1 | Refactor2 |
|-------------|-----------|
| Two-stage workflow | Single iteration |
| Missing SDF data | All data available |
| Can't evaluate portfolios | Full evaluation |
| Need disk I/O | In-memory processing |
| Slower iteration | Fast experimentation |
| Complex saving/loading | Simple workflow |

## Migration from Original Code

To migrate existing code:

1. **Keep original panel/SDF generation** - No changes needed
2. **Adapt strategy initialization** - Use classes instead of functions
3. **Use ResultsWriter** - Replace scattered CSV writes
4. **Add configuration** - Replace global variables
5. **Wrap in PanelProcessor** - Orchestrate the workflow

The actual simulation logic (panel generation, SDF computation, factor estimation) remains unchanged. We just organize it better.

## Performance

Same performance as original code since we're keeping the same workflow:
- No extra disk I/O
- Same computation
- Same parallelization
- Just better organized

## Testing

```python
# Test with in-memory writer
from src.io import InMemoryResultsWriter

config = SimulationConfig(model='gs', N=10, T=50, num_iters=2)
writer = InMemoryResultsWriter()
writer.initialize(N=10, model='gs')

processor = PanelProcessor(config)
for i in range(2):
    processor.process_iteration(i, writer)

# Retrieve results
results = writer.to_dataframes()
assert 'model' in results
assert 'dkkm' in results
```

## Future Enhancements

1. **Checkpointing**: Save intermediate state to resume interrupted runs
2. **Parallel Iterations**: Run multiple iterations in parallel
3. **Selective Analysis**: Choose which methods to run
4. **Memory Optimization**: Stream results instead of accumulating
5. **Database Backend**: Store results in database instead of CSV

## Summary

Refactor2 corrects the mistake in Refactor v1 by keeping panel generation and analysis together, while maintaining all the benefits of the refactored architecture:

- ✅ **Unified workflow**: Generation + analysis in one iteration
- ✅ **All data available**: SDF data accessible when needed
- ✅ **Clean architecture**: Modular, type-safe, testable
- ✅ **Same performance**: No overhead from reorganization
- ✅ **Easy to use**: Simple API, clear documentation

This is the correct way to refactor the original code!
