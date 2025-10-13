# Refactor2 Architecture

## Design Philosophy

Refactor2 corrects the fundamental mistake in Refactor v1: **separating panel generation from analysis**.

### The Core Problem

Panel analysis requires data that is only available during generation:
- `arr_tuple`: Raw simulation arrays from the theoretical model
- `sdf_loop`: SDF computation function that depends on `arr_tuple`
- These are needed to compute:
  - Conditional covariances (`cond_var`)
  - Risk premia (`rp`)
  - Second moments
  - Hansen-Jagannathan distance

### The Solution

**Keep generation and analysis together within a single iteration.**

```
One Iteration = Generate Panel → Compute SDF → Analyze All Months → Save Results
```

## Architecture Layers

### Layer 1: Configuration (`src/config.py`)

Type-safe configuration management:
- `SimulationConfig`: Main configuration dataclass
- Validation in `__post_init__`
- Support for YAML loading
- No global variables

### Layer 2: Panel Processing (`src/panel_processor.py`)

Core orchestration class that keeps everything together:

```python
class PanelProcessor:
    def process_iteration(self, iter: int, writer: ResultsWriter):
        # 1. Generate panel AND SDF data (both stay in memory)
        panel, sdf_loop, arr_tuple, start, end = self._generate_panel(iter)

        # 2. Compute all factors
        factors_data = self._compute_factors(panel, start, end, iter)

        # 3. Process each month (sdf_loop available!)
        for month in range(start + 360, end + 1):
            # Get SDF data
            sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1, iter)

            # Evaluate portfolios
            results = self._evaluate_month(
                month, iter, panel, sdf_loop, factors_data, keep
            )

            # Write results
            writer.write_monthly_results(results)
```

**Key insight**: By keeping `sdf_loop` in scope throughout the iteration, we can access it when evaluating each month.

### Layer 3: I/O (`src/io/`)

Results writing abstracted behind a protocol:

```python
class ResultsWriter(Protocol):
    def initialize(self, N: int, model: str) -> None: ...
    def write_monthly_results(self, results: Dict) -> None: ...
    def write_panel(self, panel: pd.DataFrame, iter: int) -> None: ...
    def finalize(self) -> None: ...
```

Implementations:
- `CSVResultsWriter`: Writes to CSV files (production)
- `InMemoryResultsWriter`: Accumulates in memory (testing)

### Layer 4: Utilities (`src/utils/`)

Supporting infrastructure:
- `logging_config.py`: Centralized logging
- `progress.py`: Progress tracking
- `constants.py`: Named constants (no magic numbers)
- `exceptions.py`: Custom exception types

## Data Flow

### Complete Iteration Flow

```
Input: iter number, config, writer

1. generate_panel(iter)
   ↓
   arr_tuple (stays in memory)
   panel DataFrame
   ↓
2. sdf_compute(N, T, arr_tuple)
   ↓
   sdf_loop function (closure over arr_tuple)
   ↓
3. compute_factors(panel)
   ↓
   factors_data (all pre-computed factors)
   ↓
4. for month in months:
   ├─ sdf_loop(month-1) → cond_var, rp
   ├─ evaluate_month(panel, sdf_loop, factors_data)
   │  ├─ compute stock_cov from cond_var
   │  ├─ evaluate model portfolios
   │  ├─ evaluate DKKM portfolios
   │  ├─ evaluate Fama portfolios
   │  └─ evaluate IPCA portfolios
   └─ write_monthly_results(results)
   ↓
5. write_panel(panel, iter)  # Optional

Output: All results written via writer
```

### Why This Works

1. **`arr_tuple` stays in memory**: Generated once, used by `sdf_compute`
2. **`sdf_loop` is a closure**: Contains reference to `arr_tuple`
3. **`sdf_loop` available in evaluation**: Can call it for each month
4. **All data computed when needed**: No serialization required

### What Gets Saved

Only the final results need to be saved:
- Portfolio weights (optional)
- Performance metrics (stdev, mean, xret, hjd)
- Panel data (optional, for reference)

The intermediate data (`arr_tuple`, `sdf_loop`) is **intentionally ephemeral** - it's only needed during one iteration.

## Comparison: Refactor v1 vs v2

### Refactor v1 (Two-Stage - BROKEN)

```python
# Stage 1: Generate
def generate_panel(iter):
    arr_tuple = create_arrays(N, T)
    panel = create_panel(N, T, arr_tuple)
    panel.to_csv(f'panel_{iter}.csv')
    # Problem: arr_tuple is lost!

# Stage 2: Analyze (later, different process)
def analyze_panel(iter):
    panel = pd.read_csv(f'panel_{iter}.csv')
    # Problem: Need arr_tuple to compute sdf_loop!
    # Problem: Can't compute cond_var, rp without sdf_loop!
    # Result: Can't evaluate portfolios properly!
```

**Fatal flaw**: Critical data lost between stages.

### Refactor v2 (Unified - CORRECT)

```python
def process_iteration(iter):
    # 1. Generate (arr_tuple in memory)
    arr_tuple = create_arrays(N, T)
    panel = create_panel(N, T, arr_tuple)

    # 2. Compute SDF (uses arr_tuple)
    sdf_loop = sdf_compute(N, T, arr_tuple)

    # 3. Analyze (sdf_loop available!)
    for month in months:
        sdf_ret, max_sr, rp, cond_var = sdf_loop(month-1)
        evaluate_portfolios(panel, rp, cond_var)

    # 4. Save results
    write_results()
```

**Correct**: All data available when needed.

## Extension Points

### Adding New Portfolio Strategies

The architecture is designed for extension:

1. **Strategy Interface** (if using strategy pattern):
```python
class MyNewStrategy:
    def __init__(self, panel, start, end):
        self.panel = panel
        # Precompute factors if needed

    def get_portfolio(self, month: int, **kwargs):
        # Return (factor_loadings, factor_weights)
        # or just portfolio weights directly
```

2. **Integration in `_evaluate_month`**:
```python
# In PanelProcessor._evaluate_month
try:
    strategy = MyNewStrategy(panel, start, end)
    loadings, weights = strategy.get_portfolio(month)
    portfolio_weights = loadings @ weights
    metrics = evaluate_portfolio(portfolio_weights)
    results['my_strategy_results'] = metrics
except Exception as e:
    logger.warning(f"MyStrategy failed: {e}")
```

### Adding New Output Formats

Implement the `ResultsWriter` protocol:

```python
class ParquetResultsWriter:
    def initialize(self, N: int, model: str) -> None:
        self.tables = {}

    def write_monthly_results(self, results: Dict) -> None:
        # Accumulate in Arrow tables

    def finalize(self) -> None:
        # Write to Parquet files
        for name, table in self.tables.items():
            pq.write_table(table, f'{name}.parquet')
```

### Adding New Models

The panel processor already supports pluggable models via dictionaries:

```python
# In your model file: panel_functions_new_model.py
def create_arrays(N, T):
    # Your model's simulation logic
    ...

def create_panel(N, T, arr_tuple):
    # Convert to panel DataFrame
    ...

# In panel_processor.py
self.panels = {
    'bgn': bgn,
    'kp': kp,
    'gs': gs,
    'new': new_model  # Just add here!
}
```

## Testing Strategy

### Unit Tests

Test components in isolation:

```python
def test_config_validation():
    """Test that invalid configs raise errors."""
    with pytest.raises(ConfigurationError):
        SimulationConfig(N=-1)

def test_in_memory_writer():
    """Test that in-memory writer accumulates correctly."""
    writer = InMemoryResultsWriter()
    writer.initialize(N=10, model='gs')
    # ... write some results ...
    dfs = writer.to_dataframes()
    assert 'model' in dfs
```

### Integration Tests

Test full iterations with small data:

```python
def test_full_iteration():
    """Test complete iteration with small dataset."""
    config = SimulationConfig(
        model='gs',
        N=10,
        T=50,
        num_iters=1,
        n_jobs=1
    )

    writer = InMemoryResultsWriter()
    writer.initialize(N=10, model='gs')

    processor = PanelProcessor(config)
    summary = processor.process_iteration(0, writer)

    assert summary['months_processed'] > 0

    results = writer.to_dataframes()
    assert len(results) > 0
```

### Regression Tests

Compare output with original code:

```python
def test_matches_original():
    """Verify refactored code produces same results as original."""
    np.random.seed(42)

    # Run with original code
    results_original = run_original_simulation(config)

    # Run with refactored code
    np.random.seed(42)
    results_refactor = run_refactor2_simulation(config)

    # Compare
    pd.testing.assert_frame_equal(
        results_original,
        results_refactor,
        rtol=1e-10
    )
```

## Performance Considerations

### Memory Usage

**One iteration in memory**:
- Panel DataFrame: ~N×T rows × 10 columns × 8 bytes ≈ 80N×T bytes
- arr_tuple: Model-dependent, but typically ~N×T floats ≈ 8N×T bytes
- Factors: ~T×K floats for each method ≈ 8TK bytes per method

For N=100, T=400:
- Panel: ~320 KB
- arr_tuple: ~320 KB
- All factors: ~few MB
- **Total: ~10 MB per iteration**

Very manageable - no memory concerns.

### Computation Time

Same as original code since we haven't changed the algorithms:
- Panel generation: ~10 seconds
- SDF computation: ~30 seconds
- Factor estimation: ~2 minutes
- Monthly evaluation: ~30 seconds per month × 40 months = ~20 minutes
- **Total: ~25 minutes per iteration**

For 100 iterations: ~42 hours

### Parallelization

Current parallelization (unchanged from original):
- Factor estimation parallelized across months (`n_jobs` parameter)
- IPCA estimation parallelized across factor counts
- Monthly evaluation is sequential (could be parallelized)

Potential improvements:
- Parallelize monthly evaluation
- Run multiple iterations in parallel
- Use shared memory for large arrays

## Summary

Refactor2 provides the **correct** architecture by:

1. ✅ **Keeping generation and analysis together**
2. ✅ **Maintaining data availability** (arr_tuple, sdf_loop)
3. ✅ **Clean separation of concerns** (config, processor, I/O, utils)
4. ✅ **Type safety** (dataclasses, protocols)
5. ✅ **Testability** (modular, injectable dependencies)
6. ✅ **Extensibility** (easy to add strategies, models, writers)
7. ✅ **Same performance** (no overhead from refactoring)

The architecture is **production-ready** and **research-friendly**.
