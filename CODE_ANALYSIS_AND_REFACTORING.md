# Code Analysis and Refactoring Suggestions

## Executive Summary

This repository implements a financial econometrics research project that compares different asset pricing models. The code simulates financial panel data from three different theoretical models (BGN, KP14, GS21), then evaluates various empirical factor models (Fama-French, Fama-MacBeth, IPCA, and DKKM) in terms of their ability to price assets and explain returns.

**Main Components:**
- **Data Generation:** Simulates financial panel data from asset pricing theories
- **Factor Model Estimation:** Implements multiple factor model methodologies
- **Portfolio Construction:** Creates optimal portfolios using different approaches
- **Performance Evaluation:** Compares models using HJD distance and Sharpe ratios

---

## Architecture Overview

### Module Organization

```
Code/
├── main.py                      # Main orchestration script
├── parameters*.py               # Model-specific parameters
├── panel_functions*.py          # Panel data generation (3 models)
├── sdf_compute*.py             # Stochastic discount factor computation
├── fama_functions.py           # Fama-French & Fama-MacBeth methods
├── dkkm_functions.py           # DKKM random Fourier features
├── ipca_functions.py           # Instrumented PCA estimation
├── sorted_portfolios.py        # 5x5 sorted portfolio construction
├── loadings_compute*.py        # Factor loading computation
└── data_processing.py          # Results aggregation
```

### Data Flow

1. **Simulation Loop** (`main.py` lines 463-514)
   - Outer loop: Multiple Monte Carlo iterations
   - Inner loop: Monthly portfolio construction

2. **Panel Generation** (`panel_functions.py`)
   - Creates arrays of simulated variables
   - Generates firm-month panel with characteristics

3. **Factor Estimation** (lines 98-171)
   - Model factors (latent)
   - IPCA factors (instrumented PCA)
   - Fama factors (classical)
   - DKKM factors (random features)

4. **Portfolio Construction** (`run_month` function, lines 174-432)
   - Computes optimal portfolios for each method
   - Evaluates performance metrics

5. **Results Storage** (CSV files)
   - Portfolio weights
   - Performance metrics
   - Time series of factors

---

## Detailed Component Analysis

### 1. Main Orchestration (`main.py`)

**Purpose:** Coordinates the entire simulation and estimation process.

**Key Logic:**

- **Lines 1-21:** Configuration parameters
  - `model`: Which theoretical model to use ('bgn', 'kp', 'gs')
  - `N, T`: Number of firms and time periods
  - `nfeatures_lst`: Different feature counts for DKKM
  - `alpha_lst`: Regularization parameters

- **Lines 42-50:** Model dictionaries for polymorphism
  ```python
  panels = {'bgn': bgn, 'kp': kp, 'gs': gs}
  sdf = {'bgn': sdf_bgn, 'kp': sdf_kp, 'gs': sdf_gs}
  ```

- **Lines 75-453:** `run_panel()` function
  - Simulates one panel dataset
  - Computes all factor models
  - Evaluates monthly portfolios in parallel
  - Returns aggregated results

- **Lines 463-514:** Main iteration loop
  - Runs multiple simulations
  - Appends results to CSV files

**Issues:**
- ⚠️ **Monolithic function:** `run_panel()` is 378 lines long
- ⚠️ **Global state:** Relies on many global variables from parameters
- ⚠️ **File I/O scattered:** CSV writes throughout the code
- ⚠️ **Limited error handling:** Assumes everything works

---

### 2. Panel Data Generation (`panel_functions.py`, `panel_functions_kp14.py`, `panel_functions_gs21.py`)

**Purpose:** Generates synthetic financial panel data from asset pricing models.

**BGN Model (`panel_functions.py`):**

- **`create_arrays()`** (lines 12-185):
  - Simulates stochastic discount factor (SDF) shocks
  - Models interest rate dynamics (Vasicek process)
  - Creates firm cash flows and investment decisions
  - Computes asset prices and returns

- **`create_panel()`** (lines 188-259):
  - Converts arrays to panel DataFrame
  - Computes firm characteristics (size, B/M, ROE, momentum, asset growth)
  - Merges with factor data

**Key Variables:**
- `chi`: Investment indicator (alive/dead projects)
- `P`: Asset prices
- `ret`: Returns
- `eret`: Expected returns

**Issues:**
- ⚠️ **Magic numbers:** Constants like `Chat = np.exp(-3.7)` without explanation
- ⚠️ **Complex nested loops:** Difficult to understand cash flow calculations
- ⚠️ **Similar code duplicated:** Three separate but similar implementations

---

### 3. Stochastic Discount Factor (`sdf_compute.py`)

**Purpose:** Computes the true SDF-based optimal portfolio from the theoretical model.

**Key Functions:**

- **`integrator()`** (lines 87-90): Gauss-Hermite integration for expectations
- **`integ_expy()`** (lines 94-114): Integration with exponential tilting
- **`sdf_compute()`** (lines 124-326): Main computation

**Main Logic (`sdf_loop`, lines 145-326):**

1. Build sparse matrices for efficient computation
2. Compute expected returns matrix `ER` (N+1 × N+1)
   - `ER[i,j]` = E[R_i * R_j]
   - Includes risk-free asset
3. Solve for SDF portfolio: `port = solve(ER, ones)`
4. Compute:
   - SDF return
   - Maximum Sharpe ratio
   - Conditional variance-covariance matrix

**Issues:**
- ⚠️ **Extremely complex:** 327 lines of dense numerical computation
- ⚠️ **Performance critical:** Uses sparse matrices, but still slow
- ⚠️ **Numerical instability:** Has fallback for singular matrix (lines 283-296)
- ⚠️ **Hard to test:** No unit tests for individual components

---

### 4. Fama-French and Fama-MacBeth (`fama_functions.py`)

**Purpose:** Classical factor model approaches.

**Functions:**

- **`fama_french()`** (lines 9-66):
  - Double-sorted portfolios (size × characteristic)
  - Creates long-short factors (SMB, HML, CMA, RMW, UMD)

- **`fama_macbeth()`** (lines 69-89):
  - Cross-sectional regression approach
  - Standardizes characteristics
  - Creates factor-mimicking portfolios

- **`factors()`** (lines 95-113):
  - Computes factor returns panel
  - Parallelizes across months

- **`mve_data()`** (lines 116-121):
  - Mean-variance efficient portfolio of factors
  - Uses ridge regression with past 360 months

**Issues:**
- ✅ **Well-structured:** Clean, modular functions
- ⚠️ **Limited documentation:** Unclear what "mve" stands for (mean-variance efficient)
- ⚠️ **Magic number 360:** Rolling window hardcoded

---

### 5. DKKM Random Features (`dkkm_functions.py`)

**Purpose:** Implements Deep Kernel Kernel Methods using random Fourier features.

**Key Concepts:**
- Approximates kernel methods with random projections
- Uses rank-standardization of characteristics
- Creates many synthetic factors from few characteristics

**Functions:**

- **`rank_standardize()`** (lines 10-13):
  - Maps values to [-0.5, 0.5] based on ranks

- **`rff()`** (lines 16-25):
  - Random Fourier Features transformation
  - `Z = W @ X.T` → `[sin(Z), cos(Z)]`
  - Returns both rank-standardized and raw versions

- **`factors()`** (lines 46-70):
  - Computes factor returns panel
  - Returns both 'rs' (rank-standardized) and 'nors' (not rank-standardized) versions

- **`mve_data()`** (lines 73-114):
  - **Complex ridge regression** with optional market factor
  - Handles high-dimensional case (P > T) efficiently using SVD
  - Lines 82-86: Implements unpenalized market factor via augmentation

**Issues:**
- ⚠️ **Naming:** 'rs' and 'nors' are cryptic abbreviations
- ⚠️ **Complex linear algebra:** Lines 90-111 implement three different solution paths
- ⚠️ **No comments on algorithm:** SVD approach not explained

---

### 6. Instrumented PCA (`ipca_functions.py`)

**Purpose:** Implements IPCA (Instrumented Principal Component Analysis) from Kelly, Pruitt, Su (2019).

**Key Algorithm: Alternating Least Squares**

- **`fit_ipca()`** (lines 78-184):
  1. Initialize factor loadings Γ and factors f
  2. Iterate:
     - Fix Γ, update f (line 56)
     - Fix f, update Γ (line 73)
  3. Until convergence or max iterations

- **`normalization()`** (lines 28-40):
  - Ensures Γ'Γ = I (orthonormal loadings)
  - Makes factors have positive mean (identification)

- **`fit_ipca_360()`** (lines 188-217):
  - Rolling window estimation (360 months)
  - Stores portfolio weights for each month

**Issues:**
- ⚠️ **Magic number 360:** Rolling window hardcoded everywhere
- ⚠️ **Commented-out code:** Lines 85-108, 251-294 should be removed
- ⚠️ **Convergence diagnostics:** Prints to console (lines 160, 167)
- ⚠️ **Special case K==L:** Lines 132-145 handle this separately (could refactor)

---

### 7. Portfolio Construction (`main.py`, lines 174-432)

**Purpose:** The `run_month()` function evaluates all methods for one month.

**Structure:**

1. **Setup** (lines 178-187):
   - Get SDF return and conditional covariance
   - Extract stocks available this month

2. **Model portfolios** (lines 197-227):
   - For each method ('taylor', 'proj'):
     - Get factor loadings
     - Compute factor weights
     - Form portfolio of stocks
     - Evaluate performance (std, mean, xret, HJD)

3. **DKKM portfolios** (lines 229-319):
   - **Nested loops over:**
     - Method (rs/nors)
     - Include market (True/False)
     - Number of features (6, 36, 360)
     - Alpha (regularization)
     - Weight matrix
   - `process_dkkm_block()` helper function (lines 229-296)

4. **Fama portfolios** (lines 322-355):
   - FF and FM methods
   - Evaluate with different alphas

5. **IPCA portfolios** (lines 358-390):
   - For each number of factors
   - Extract pre-computed weights

**Performance Metrics:**
- `stdev`: Portfolio standard deviation
- `mean`: Expected return
- `xret`: Realized return
- `hjd`: Hansen-Jagannathan distance (pricing error)

**Issues:**
- ⚠️ **Extremely long function:** 258 lines
- ⚠️ **Deeply nested:** Up to 6 levels of nesting
- ⚠️ **Repeated code:** Same portfolio evaluation logic 4+ times
- ⚠️ **Side effects:** Writes to CSV files within the function
- ⚠️ **Poor separation of concerns:** Mixes computation, evaluation, and I/O

---

### 8. Sorted Portfolios (`sorted_portfolios.py`)

**Purpose:** Creates 5×5 double-sorted portfolios for analysis.

**Logic:**
- For each non-size characteristic (B/M, asset growth, ROE, momentum):
  - Sort on size → 5 quintiles
  - Sort on characteristic → 5 quintiles
  - Form 25 portfolios
  - Compute equal-weighted and value-weighted returns

**Issues:**
- ✅ **Well-structured:** Clean, focused functions
- ⚠️ **Error handling:** Try-except without logging (line 41)
- ⚠️ **Deprecated function:** `file_exists` (line 105) might not work in newer pandas

---

### 9. Data Processing (`data_processing.py`)

**Purpose:** Post-processing of results for analysis.

**Logic:**
- Loads all CSV result files
- Merges with SDF returns
- Computes squared differences and Sharpe ratios
- Aggregates by method and parameters
- Creates pivot tables for visualization

**Issues:**
- ✅ **Functional approach:** Clean data pipeline
- ⚠️ **Hardcoded folder:** `'new_results'` default
- ⚠️ **No error handling:** Assumes all files exist

---

## Major Design Issues

### 1. Code Duplication

**Problem:** Three nearly identical implementations of each module:
- `panel_functions.py`, `panel_functions_kp14.py`, `panel_functions_gs21.py`
- `sdf_compute.py`, `sdf_compute_kp14.py`, `sdf_compute_gs21.py`

**Impact:**
- Maintenance burden (fix bugs 3 times)
- Inconsistencies between implementations
- Violates DRY principle

**Example:**
```python
# Current approach
panels = {'bgn': bgn, 'kp': kp, 'gs': gs}
arr_tuple = panels[model].create_arrays(N, T+burnin)
```

### 2. God Functions

**Problem:** Massive functions with too many responsibilities:
- `run_panel()`: 378 lines
- `run_month()`: 258 lines
- `sdf_loop()`: 181 lines

**Impact:**
- Hard to understand
- Hard to test
- Hard to modify
- High cyclomatic complexity

### 3. Global State

**Problem:** Heavy reliance on global variables:
- Parameters imported with `from parameters import *`
- Hard to track data flow
- Makes testing difficult

### 4. Mixed Concerns

**Problem:** Functions that do computation AND I/O:
- Portfolio evaluation mixed with CSV writing
- Progress printing scattered throughout

**Impact:**
- Can't test computation without file system
- Hard to change output format
- Violates single responsibility principle

### 5. Magic Numbers

**Problem:** Unexplained constants throughout:
- `360`: Rolling window size
- `200`: Burnin period
- `-3.7`: Log cash flow parameter
- `0.5`: Rank standardization range

**Impact:**
- Hard to understand
- Can't easily experiment with alternatives

### 6. Limited Error Handling

**Problem:** Minimal try-except blocks:
- Assumes files exist
- Assumes numerical stability
- Silent failures possible

### 7. No Type Hints

**Problem:** No function signatures specify types:
```python
def factors(method, panel, n_jobs, start, end):  # What types?
```

### 8. Inadequate Documentation

**Problem:**
- No docstrings for many functions
- Complex algorithms unexplained
- No module-level documentation

---

## Refactoring Suggestions

### Priority 1: Critical Refactorings

#### 1.1 Extract Common Base Classes

**Current:**
```python
# Three separate files with similar structure
def create_arrays(N, T):
    # BGN-specific logic
    ...

def create_panel(N, T, arr_tuple):
    # BGN-specific logic
    ...
```

**Proposed:**
```python
# base_model.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Dict
import numpy as np
import pandas as pd

@dataclass
class ModelParameters:
    """Base class for model parameters."""
    N: int
    T: int
    burnin: int
    chars: list[str]

class PanelModel(ABC):
    """Abstract base class for panel data models."""

    def __init__(self, params: ModelParameters):
        self.params = params

    @abstractmethod
    def simulate_shocks(self) -> Dict[str, np.ndarray]:
        """Simulate stochastic shocks."""
        pass

    @abstractmethod
    def compute_prices(self, shocks: Dict) -> np.ndarray:
        """Compute asset prices."""
        pass

    @abstractmethod
    def compute_returns(self, prices: np.ndarray) -> np.ndarray:
        """Compute returns."""
        pass

    def create_arrays(self) -> Tuple:
        """Template method for array creation."""
        shocks = self.simulate_shocks()
        prices = self.compute_prices(shocks)
        returns = self.compute_returns(prices)
        return self._package_arrays(shocks, prices, returns)

    def create_panel(self, arr_tuple: Tuple) -> pd.DataFrame:
        """Convert arrays to panel DataFrame."""
        # Common logic shared across models
        ...

# bgn_model.py
class BGNModel(PanelModel):
    """Bansal-Gallant-Navarro model implementation."""

    def simulate_shocks(self) -> Dict[str, np.ndarray]:
        # BGN-specific implementation
        ...
```

**Benefits:**
- Eliminates code duplication
- Makes differences between models explicit
- Easier to add new models
- Shared logic centralized

#### 1.2 Break Up God Functions

**Current:** `run_month()` is 258 lines

**Proposed:**
```python
# portfolio_evaluator.py
from dataclasses import dataclass
from typing import Protocol
import numpy as np
import pandas as pd

@dataclass
class PortfolioMetrics:
    """Container for portfolio performance metrics."""
    stdev: float
    mean: float
    xret: float
    hjd: float
    weights: np.ndarray

class PortfolioStrategy(Protocol):
    """Interface for portfolio construction strategies."""

    def compute_factor_loadings(self, data: pd.DataFrame) -> np.ndarray:
        """Compute factor loadings for stocks."""
        ...

    def compute_factor_weights(self, factor_returns: pd.DataFrame) -> np.ndarray:
        """Compute weights on factors."""
        ...

class ModelPortfolioStrategy:
    """Portfolio based on latent factor model."""

    def __init__(self, method: str, model_premia: pd.DataFrame):
        self.method = method
        self.model_premia = model_premia

    def compute_factor_loadings(self, data: pd.DataFrame) -> np.ndarray:
        loading_cols = [f'{x}_{self.method}' for x in ['A_1', 'A_2']]
        return data[loading_cols].values

    def compute_factor_weights(self, month: int) -> np.ndarray:
        # Get factor portfolio from mve_data
        return fama.mve_data(self.model_premia, month, alpha=0).values

class PortfolioEvaluator:
    """Evaluates portfolio performance."""

    def __init__(self, cov_inv: np.ndarray, second_moment_inv: np.ndarray):
        self.cov_inv = cov_inv
        self.second_moment_inv = second_moment_inv

    def evaluate(
        self,
        weights: np.ndarray,
        returns: np.ndarray,
        risk_premia: np.ndarray,
        cov: np.ndarray,
        second_moment: np.ndarray
    ) -> PortfolioMetrics:
        """Compute all performance metrics."""
        stdev = np.sqrt(weights @ cov @ weights)
        mean = weights @ risk_premia
        xret = weights @ returns

        errs = risk_premia - second_moment @ weights
        hjd = errs @ self.second_moment_inv @ errs

        return PortfolioMetrics(
            stdev=stdev,
            mean=mean,
            xret=xret,
            hjd=hjd,
            weights=weights
        )

def run_month(month: int, iter: int, data: MonthlyData) -> MonthlyResults:
    """
    Evaluate all portfolio strategies for one month.

    Args:
        month: Month index
        iter: Iteration number
        data: Pre-computed data for this month

    Returns:
        MonthlyResults containing metrics for all strategies
    """
    # Setup
    evaluator = PortfolioEvaluator(data.cov_inv, data.second_moment_inv)
    results = MonthlyResults(month=month, iter=iter)

    # Evaluate model portfolios
    for method in ['taylor', 'proj']:
        strategy = ModelPortfolioStrategy(method, data.model_premia)
        metrics = evaluate_strategy(strategy, data, evaluator)
        results.add_model_results(method, metrics)

    # Evaluate DKKM portfolios
    dkkm_results = evaluate_dkkm_portfolios(data, evaluator)
    results.add_dkkm_results(dkkm_results)

    # Evaluate Fama portfolios
    fama_results = evaluate_fama_portfolios(data, evaluator)
    results.add_fama_results(fama_results)

    # Evaluate IPCA portfolios
    ipca_results = evaluate_ipca_portfolios(data, evaluator)
    results.add_ipca_results(ipca_results)

    return results

def evaluate_strategy(
    strategy: PortfolioStrategy,
    data: MonthlyData,
    evaluator: PortfolioEvaluator
) -> PortfolioMetrics:
    """Generic strategy evaluation."""
    factor_loadings = strategy.compute_factor_loadings(data.panel_data)
    factor_weights = strategy.compute_factor_weights(data.month)
    stock_weights = factor_loadings @ factor_weights

    return evaluator.evaluate(
        weights=stock_weights,
        returns=data.returns,
        risk_premia=data.risk_premia,
        cov=data.cov,
        second_moment=data.second_moment
    )
```

**Benefits:**
- Each function has single responsibility
- Easy to test individual components
- Can add new strategies without modifying main loop
- Clear separation of concerns

#### 1.3 Separate Computation from I/O

**Current:** CSV writes scattered throughout

**Proposed:**
```python
# results_writer.py
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Protocol
import pandas as pd

class ResultsWriter(Protocol):
    """Interface for writing results."""

    def write_monthly_results(self, results: MonthlyResults) -> None:
        """Write results for one month."""
        ...

    def write_iteration_summary(self, summary: IterationSummary) -> None:
        """Write summary for one iteration."""
        ...

class CSVResultsWriter:
    """Writes results to CSV files."""

    def __init__(self, output_dir: Path, model: str):
        self.output_dir = output_dir
        self.model = model
        self._initialize_files()

    def _initialize_files(self) -> None:
        """Create output directory and write headers."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Write headers for all output files
        ...

    def write_monthly_results(self, results: MonthlyResults) -> None:
        """Append monthly results to appropriate files."""
        self._write_portfolio_weights(results)
        self._write_performance_metrics(results)

    def _write_portfolio_weights(self, results: MonthlyResults) -> None:
        """Write portfolio weights for all methods."""
        # Model portfolios
        for method, metrics in results.model_results.items():
            self._append_to_csv(
                f'port_model_{self.model}.csv',
                self._format_portfolio_row(results.month, results.iter, method, metrics)
            )

        # DKKM portfolios
        for key, metrics in results.dkkm_results.items():
            self._append_to_csv(
                f'port_dkkm_{self.model}.csv',
                self._format_dkkm_row(results.month, results.iter, key, metrics)
            )
        # ... similar for other methods

class InMemoryResultsWriter:
    """Accumulates results in memory for testing."""

    def __init__(self):
        self.monthly_results = []
        self.iteration_summaries = []

    def write_monthly_results(self, results: MonthlyResults) -> None:
        self.monthly_results.append(results)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert accumulated results to DataFrame."""
        ...

# Usage in main.py
def run_panel(iter: int, writer: ResultsWriter) -> PanelResults:
    """Run one panel iteration."""
    # ... simulation logic ...

    for month in range(start + 360, end + 1):
        monthly_results = run_month(month, iter, monthly_data)
        writer.write_monthly_results(monthly_results)  # Separate I/O

    return panel_results
```

**Benefits:**
- Can test computation without file system
- Easy to switch output formats (CSV, Parquet, database)
- Can accumulate results in memory for testing
- Progress tracking separated from computation

#### 1.4 Configuration Management

**Current:** `from parameters import *`

**Proposed:**
```python
# config.py
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
import numpy as np

@dataclass
class SimulationConfig:
    """Configuration for simulation run."""

    # Simulation parameters
    model: Literal['bgn', 'kp', 'gs'] = 'gs'
    start_iter: int = 0
    num_iters: int = 10

    # Panel dimensions
    N: int = 100  # Number of firms
    T: int = 400  # Number of time periods (excluding burnin)
    burnin: int = 200  # Burnin period

    # DKKM parameters
    include_mkt: bool = False
    nmat: int = 1  # Number of weight matrices
    nfeatures_lst: list[int] = field(default_factory=lambda: [6, 36, 360])
    n_ipca_rff: int = 36

    # Regularization parameters
    alpha_lst_fama: list[float] = field(default_factory=lambda: [0])
    alpha_lst: list[float] = field(default_factory=lambda:
        [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1])

    # IPCA parameters
    ipca_nfactors_lst: list[int] = field(default_factory=lambda: [1, 2])

    # Computation parameters
    n_jobs: int = 10

    # Output parameters
    output_dir: Path = Path('.')

    def __post_init__(self):
        """Validate configuration."""
        if self.model == 'gs':
            self.alpha_lst = [0, 0.0000001, 0.000001, 0.00001,
                             0.0001, 0.001, 0.01, 0.1, 1]

        if self.N <= 0 or self.T <= 0:
            raise ValueError("N and T must be positive")

        self.max_features = max(self.nfeatures_lst)

    @classmethod
    def from_yaml(cls, path: Path) -> 'SimulationConfig':
        """Load configuration from YAML file."""
        import yaml
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

@dataclass
class ModelParameters:
    """Base model parameters."""
    chars: list[str] = field(default_factory=lambda:
        ["size", "bm", "agr", "roe", "mom"])
    gamma_grid: np.ndarray = field(default_factory=lambda:
        np.arange(0.5, 1.1, 0.1))

@dataclass
class BGNParameters(ModelParameters):
    """BGN model-specific parameters."""
    pi: float = 0.99
    rbar: float = 0.006236
    kappa: float = 0.95
    sigma_r: float = 0.002
    beta_zr: float = -0.00014
    sigma_z: float = 0.4
    Cbar: float = -3.7
    I: int = 1

# Usage
config = SimulationConfig(
    model='gs',
    num_iters=100,
    N=200,
    output_dir=Path('results/experiment_1')
)

# Or load from file
config = SimulationConfig.from_yaml('config.yaml')
```

**Benefits:**
- Type-safe configuration
- Easy to validate parameters
- Can load from files (YAML, JSON)
- Clear documentation of all parameters
- No global state

### Priority 2: Important Refactorings

#### 2.1 Add Type Hints

**Current:**
```python
def factors(method, panel, n_jobs, start, end):
    ...
```

**Proposed:**
```python
from typing import Callable, Protocol
import pandas as pd
import numpy as np

class FactorMethod(Protocol):
    """Protocol for factor construction methods."""

    def __call__(self, data: pd.Series, *, mve: pd.Series) -> np.ndarray:
        """
        Compute factor weights for one cross-section.

        Args:
            data: Characteristic data
            mve: Market value of equity

        Returns:
            Array of factor weights (n_stocks × n_factors)
        """
        ...

def factors(
    method: FactorMethod,
    panel: pd.DataFrame,
    n_jobs: int,
    start: int,
    end: int
) -> pd.DataFrame:
    """
    Compute factor returns panel.

    Args:
        method: Factor construction method
        panel: Panel data with MultiIndex (month, firmid)
        n_jobs: Number of parallel jobs
        start: Starting month
        end: Ending month

    Returns:
        DataFrame with factor returns (months × factors)
    """
    ...
```

#### 2.2 Add Comprehensive Documentation

**Current:**
```python
def mve_data(f, month, alpha):
    X = f.loc[month-360:month-1].dropna().to_numpy()
    y = np.ones(len(X))
    ridge = Ridge(fit_intercept=False, alpha=360*alpha)
    pi = ridge.fit(X=X, y=y).coef_
    return pd.Series(pi, index=f.columns)
```

**Proposed:**
```python
def compute_tangency_portfolio(
    factor_returns: pd.DataFrame,
    month: int,
    regularization: float
) -> pd.Series:
    """
    Compute mean-variance efficient (tangency) portfolio of factors.

    Uses ridge regression to solve the regularized tangency portfolio problem:
        min ||X @ w - 1||^2 + alpha * ||w||^2

    where X is the matrix of past factor returns and 1 is the vector of ones.
    This is equivalent to finding the minimum-variance portfolio with unit
    expected return under the assumption that E[R] = 1 for all factors.

    Args:
        factor_returns: DataFrame of factor returns (months × factors)
        month: Current month for which to compute portfolio
        regularization: Ridge regularization parameter alpha.
                       Multiplied by 360 (training sample size) for stability.

    Returns:
        Series of factor weights indexed by factor names

    Notes:
        - Uses past 360 months of data (rolling window)
        - Ridge regularization prevents overfitting with many factors
        - Larger regularization → more equal-weighted portfolio

    Example:
        >>> factor_rets = pd.DataFrame({'f1': [...], 'f2': [...]})
        >>> weights = compute_tangency_portfolio(factor_rets, month=400, regularization=0.01)
        >>> print(weights)
        f1    0.6
        f2    0.4
        dtype: float64
    """
    ROLLING_WINDOW = 360  # 30 years of monthly data

    # Extract training data
    training_data = factor_returns.loc[month - ROLLING_WINDOW : month - 1]
    training_data = training_data.dropna()

    X = training_data.to_numpy()
    y = np.ones(len(X))  # Target: unit expected return

    # Solve regularized least squares
    ridge = Ridge(fit_intercept=False, alpha=ROLLING_WINDOW * regularization)
    pi = ridge.fit(X=X, y=y).coef_

    return pd.Series(pi, index=factor_returns.columns)
```

#### 2.3 Extract Magic Numbers as Named Constants

**Current:**
```python
panel = panel[panel.month>=2]
# ... later ...
ipca_weights_on_stocks[:, firms, 0], ... = fit_ipca(panel, start, K, tol)
for t in range(start+1, end+1 - 360):
    ...
```

**Proposed:**
```python
# constants.py
# Rolling windows
ROLLING_WINDOW_MONTHS = 360  # 30 years of monthly data
TRAINING_WINDOW_MONTHS = 360  # For factor estimation

# Panel construction
MIN_MONTH_FOR_LAGS = 2  # Minimum month after computing lags
MOMENTUM_LOOKBACK_START = 2  # Months
MOMENTUM_LOOKBACK_END = 13  # Months

# Rank standardization
RANK_STD_MIN = -0.5
RANK_STD_MAX = 0.5

# Usage
panel = panel[panel.month >= MIN_MONTH_FOR_LAGS]

for t in range(start + 1, end + 1 - TRAINING_WINDOW_MONTHS):
    ...
```

#### 2.4 Improve Error Handling

**Current:**
```python
try:
    port = scipy.linalg.solve(ER, np.ones((N + 1, 1)), assume_a="pos").reshape(-1)
except Exception as e:
    print(f"An error occurred: {e}. Perturbing ER.")
    ER += np.eye(ER.shape[0]) * 1e-6
    ...
```

**Proposed:**
```python
# exceptions.py
class NumericalInstabilityError(Exception):
    """Raised when numerical computation fails."""
    pass

class InsufficientDataError(Exception):
    """Raised when not enough data for estimation."""
    pass

# sdf_compute.py
import logging

logger = logging.getLogger(__name__)

def solve_sdf_portfolio(ER: np.ndarray, method: str = 'direct') -> np.ndarray:
    """
    Solve for SDF portfolio weights.

    Args:
        ER: Expected returns matrix (N+1 × N+1)
        method: Solution method ('direct', 'ridge', 'tikhonov')

    Returns:
        Portfolio weights array of length N+1

    Raises:
        NumericalInstabilityError: If matrix is too ill-conditioned
    """
    N = ER.shape[0] - 1
    target = np.ones((N + 1, 1))

    # Check condition number
    cond = np.linalg.cond(ER)
    if cond > 1e10:
        logger.warning(f"ER matrix is ill-conditioned (cond={cond:.2e})")

    try:
        # Attempt direct solution
        port = scipy.linalg.solve(ER, target, assume_a="pos").reshape(-1)

        # Validate solution
        if not np.all(np.isfinite(port)):
            raise NumericalInstabilityError("Solution contains inf or nan")

        if np.abs(port).max() > 1e3:
            logger.warning(f"Extreme portfolio weight: {np.abs(port).max():.2e}")

        return port

    except (scipy.linalg.LinAlgError, NumericalInstabilityError) as e:
        logger.error(f"Direct solution failed: {e}")

        # Try Tikhonov regularization
        if method == 'direct':
            logger.info("Attempting Tikhonov regularization")
            return solve_sdf_portfolio_regularized(ER, lambda_=1e-6)
        else:
            raise NumericalInstabilityError(
                f"Failed to solve SDF portfolio (cond={cond:.2e})"
            ) from e

def solve_sdf_portfolio_regularized(
    ER: np.ndarray,
    lambda_: float
) -> np.ndarray:
    """Solve with Tikhonov regularization."""
    N = ER.shape[0]
    ER_reg = ER + lambda_ * np.eye(N)
    port = scipy.linalg.solve(ER_reg, np.ones((N, 1)), assume_a="pos").reshape(-1)

    if not np.all(np.isfinite(port)):
        raise NumericalInstabilityError(
            f"Regularized solution failed (lambda={lambda_})"
        )

    return port
```

### Priority 3: Nice-to-Have Refactorings

#### 3.1 Add Unit Tests

```python
# tests/test_fama_functions.py
import pytest
import numpy as np
import pandas as pd
from fama_functions import rank_standardize, fama_macbeth

def test_rank_standardize_basic():
    """Test basic rank standardization."""
    data = pd.DataFrame({'x': [1, 2, 3, 4, 5]})
    result = rank_standardize(data)

    # Should map to [-0.4, -0.2, 0.0, 0.2, 0.4]
    expected = pd.DataFrame({'x': [-0.4, -0.2, 0.0, 0.2, 0.4]})
    pd.testing.assert_frame_equal(result, expected)

def test_rank_standardize_ties():
    """Test rank standardization with ties."""
    data = pd.DataFrame({'x': [1, 2, 2, 4, 5]})
    result = rank_standardize(data)

    # Average rank for ties
    assert result.loc[1, 'x'] == result.loc[2, 'x']

def test_fama_macbeth_shape():
    """Test Fama-MacBeth output shape."""
    np.random.seed(42)
    N = 100
    data = pd.DataFrame({
        'size': np.random.randn(N),
        'bm': np.random.randn(N),
        'agr': np.random.randn(N),
        'roe': np.random.randn(N),
        'mom': np.random.randn(N),
    })
    mve = pd.Series(np.random.uniform(1, 100, N))

    result = fama_macbeth(data, mve=mve)

    # Should return N × 6 array (5 factors + market)
    assert result.shape == (N, 6)

    # Should sum to approximately zero (long-short)
    for i in range(5):
        assert abs(result[:, i].sum()) < 1e-10

def test_fama_macbeth_standardization():
    """Test that characteristics are standardized."""
    np.random.seed(42)
    N = 100
    data = pd.DataFrame({
        'size': np.random.randn(N) * 10 + 50,  # Large scale
        'bm': np.random.randn(N),
        'agr': np.random.randn(N),
        'roe': np.random.randn(N),
        'mom': np.random.randn(N),
    })
    mve = pd.Series(np.ones(N))

    result = fama_macbeth(data, mve=mve)

    # Result should not depend on scale of characteristics
    # (Test by comparing with scaled version)
    data_scaled = data.copy()
    data_scaled['size'] *= 100
    result_scaled = fama_macbeth(data_scaled, mve=mve)

    np.testing.assert_allclose(result, result_scaled, rtol=1e-10)
```

#### 3.2 Add Logging

```python
# logging_config.py
import logging
from pathlib import Path

def setup_logging(
    log_file: Path = None,
    level: int = logging.INFO,
    format_string: str = None
):
    """
    Configure logging for simulation.

    Args:
        log_file: Path to log file (None for console only)
        level: Logging level
        format_string: Custom format string
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        )

    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers
    )

# main.py
import logging
from logging_config import setup_logging

logger = logging.getLogger(__name__)

def run_panel(iter: int, config: SimulationConfig) -> PanelResults:
    """Run one panel iteration."""
    logger.info(f"Starting iteration {iter}")

    # Simulate panel
    logger.debug(f"Creating arrays for {config.N} firms, {config.T} periods")
    arr_tuple = panels[config.model].create_arrays(config.N, config.T + config.burnin)

    logger.debug("Creating panel DataFrame")
    panel = panels[config.model].create_panel(config.N, config.T + config.burnin, arr_tuple)

    logger.info(f"Panel shape: {panel.shape}")
    logger.info(f"Missing values: {panel.isnull().sum().sum()}")

    # ... rest of simulation ...

    logger.info(f"Completed iteration {iter}")
    return results
```

#### 3.3 Add Progress Tracking

```python
# progress.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

@dataclass
class ProgressTracker:
    """Track progress of long-running simulation."""

    total_iterations: int
    current_iteration: int = 0
    start_time: Optional[datetime] = None

    def start(self):
        """Start tracking."""
        self.start_time = datetime.now()
        self.current_iteration = 0

    def update(self, iter: int):
        """Update progress."""
        self.current_iteration = iter

        elapsed = datetime.now() - self.start_time
        per_iter = elapsed / (iter + 1)
        remaining_iters = self.total_iterations - (iter + 1)
        eta = datetime.now() + per_iter * remaining_iters

        pct = 100 * (iter + 1) / self.total_iterations

        print(f"Iteration {iter + 1}/{self.total_iterations} "
              f"({pct:.1f}%) - "
              f"Elapsed: {elapsed} - "
              f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

    def summary(self):
        """Print final summary."""
        elapsed = datetime.now() - self.start_time
        per_iter = elapsed / self.total_iterations
        print(f"\nCompleted {self.total_iterations} iterations")
        print(f"Total time: {elapsed}")
        print(f"Average per iteration: {per_iter}")

# Usage
progress = ProgressTracker(total_iterations=config.num_iters)
progress.start()

for iter in range(config.num_iters):
    results = run_panel(iter, config)
    progress.update(iter)

progress.summary()
```

---

## Suggested Refactoring Roadmap

### Phase 1: Foundation (Week 1-2)
1. Add configuration management system
2. Create base classes for models
3. Extract constants
4. Add type hints to main functions

### Phase 2: Restructuring (Week 3-4)
1. Break up `run_panel()` into smaller functions
2. Break up `run_month()` into strategy pattern
3. Separate I/O from computation
4. Add logging infrastructure

### Phase 3: Quality (Week 5-6)
1. Add unit tests for core functions
2. Add integration tests
3. Improve error handling
4. Add documentation

### Phase 4: Performance (Week 7-8)
1. Profile code
2. Optimize bottlenecks
3. Consider caching expensive computations
4. Parallelize more operations

---

## Performance Considerations

### Current Bottlenecks

1. **SDF Computation:** The `sdf_compute()` function is extremely expensive
   - Nested loops over time and firms
   - Sparse matrix operations
   - Gauss-Hermite integration

   **Suggestions:**
   - Consider pre-computing more integrals
   - Use numba JIT compilation for inner loops
   - Cache repeated calculations

2. **IPCA Estimation:** Alternating least squares can be slow
   - Many iterations for convergence
   - Repeated matrix multiplications

   **Suggestions:**
   - Warm-start with previous period's solution (already done)
   - Consider faster optimization algorithms
   - Use GPU acceleration for large problems

3. **Parallel Processing:** Already used, but could be improved
   - Load balancing across months
   - Reduce overhead of spawning processes

   **Suggestions:**
   - Use shared memory for large arrays
   - Batch smaller tasks together

### Memory Usage

**Current Issues:**
- Large matrices stored in memory
- Results accumulated before writing

**Suggestions:**
- Stream results to disk instead of accumulating
- Use memory-mapped arrays for large data
- Implement chunked processing

---

## Testing Strategy

### Unit Tests

Test individual functions in isolation:

```python
# Core transformations
test_rank_standardize()
test_rff_transformation()

# Portfolio construction
test_fama_french_portfolios()
test_fama_macbeth_portfolios()

# Numerical utilities
test_ridge_regression()
test_matrix_inversion()
```

### Integration Tests

Test workflows:

```python
test_full_panel_generation()
test_factor_estimation_pipeline()
test_portfolio_evaluation()
```

### Regression Tests

Ensure refactoring doesn't change results:

```python
def test_refactored_vs_original():
    """Compare refactored code output to original."""
    np.random.seed(42)

    # Run original
    results_original = run_original_code(config)

    # Run refactored
    results_refactored = run_refactored_code(config)

    # Compare
    pd.testing.assert_frame_equal(
        results_original,
        results_refactored,
        check_exact=False,
        rtol=1e-10
    )
```

---

## Documentation Needs

### Code Documentation

1. **Module docstrings:** Explain what each module does
2. **Function docstrings:** All public functions need docs
3. **Class docstrings:** Explain class purpose and usage
4. **Inline comments:** For complex algorithms only

### Project Documentation

1. **README.md:** Installation, quick start, overview
2. **THEORY.md:** Explanation of the models and methods
3. **API.md:** Public API reference
4. **EXAMPLES.md:** Usage examples
5. **CHANGELOG.md:** Track changes over time

### Example README Structure

```markdown
# Asset Pricing Model Comparison

## Overview
This project compares different asset pricing models...

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from simulation import run_simulation, SimulationConfig

config = SimulationConfig(model='gs', num_iters=10)
results = run_simulation(config)
```

## Models
- BGN: Bansal-Gallant-Navarro (2019)
- KP14: Kelly-Pruitt (2014)
- GS21: Gospodinov-Shi (2021)

## Methods Compared
- Fama-French factors
- Fama-MacBeth
- IPCA (Kelly-Pruitt-Su 2019)
- DKKM (Deep Kernel)

## Performance Metrics
- Hansen-Jagannathan Distance (HJD)
- Sharpe Ratio
- ...

## Project Structure
...
```

---

## Conclusion

This codebase implements sophisticated financial econometrics research but suffers from common technical debt issues:

1. **Code duplication** across model implementations
2. **God functions** that do too much
3. **Mixed concerns** (computation + I/O)
4. **Limited testability** due to global state
5. **Inadequate documentation**

The suggested refactorings follow a clear priority:

**Priority 1 (Critical):** Focus on architectural improvements that will make future work easier
**Priority 2 (Important):** Add type safety, documentation, and error handling
**Priority 3 (Nice-to-have):** Add testing, logging, and monitoring

By following this roadmap, the codebase will become:
- **More maintainable:** Clear structure, separated concerns
- **More reliable:** Better error handling, comprehensive tests
- **More performant:** Identified and optimized bottlenecks
- **More extensible:** Easy to add new models and methods
- **More understandable:** Type hints and documentation

The key is to refactor incrementally while maintaining backward compatibility through regression testing.
