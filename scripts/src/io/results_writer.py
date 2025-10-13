"""
Results writing for simulation output.

This module provides writers for saving simulation results to different
backends (CSV files, in-memory storage, etc.).
"""

from pathlib import Path
from typing import Protocol, Dict, Any
import pandas as pd
import csv
import logging

logger = logging.getLogger(__name__)


class ResultsWriter(Protocol):
    """
    Protocol defining the interface for results writers.

    Any writer can be used as long as it implements this protocol.
    """

    def initialize(self, N: int, model: str) -> None:
        """Initialize writer with simulation parameters."""
        ...

    def write_monthly_results(self, results: Dict[str, Any]) -> None:
        """Write results for one month."""
        ...

    def write_panel(self, panel: pd.DataFrame, iter: int) -> None:
        """Write panel data for one iteration."""
        ...

    def finalize(self) -> None:
        """Finalize and close all output."""
        ...


class CSVResultsWriter:
    """
    Writes results to CSV files.

    Creates separate files for each method (model, DKKM, Fama, IPCA)
    containing portfolio weights and performance metrics.
    """

    def __init__(self, output_dir: Path):
        """
        Initialize CSV writer.

        Args:
            output_dir: Directory for output files
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.N = None
        self.model = None
        self.panel_accumulator = []

        # File handles
        self.files = {}
        self.writers = {}

    def initialize(self, N: int, model: str) -> None:
        """
        Initialize writer and create output files.

        Args:
            N: Number of firms
            model: Model name
        """
        self.N = N
        self.model = model

        # Define file specifications
        file_specs = {
            f'port_model_{model}.csv': ['iter', 'month', 'method'] + [f'firm_{i+1}' for i in range(N)],
            f'port_dkkm_{model}.csv': ['iter', 'month', 'include_mkt', 'method', 'mat', 'nfeatures', 'alpha']
                                       + [f'firm_{i+1}' for i in range(N)],
            f'port_fama_{model}.csv': ['iter', 'month', 'method', 'alpha'] + [f'firm_{i+1}' for i in range(N)],
            f'port_ipca_{model}.csv': ['iter', 'month', 'nfactors'] + [f'firm_{i+1}' for i in range(N)],
            f'results_model_{model}.csv': ['iter', 'month', 'method', 'stdev', 'mean', 'xret', 'hjd'],
            f'results_dkkm_{model}.csv': ['iter', 'month', 'include_mkt', 'method', 'mat',
                                          'nfeatures', 'alpha', 'stdev', 'mean', 'xret', 'hjd'],
            f'results_fama_{model}.csv': ['iter', 'month', 'method', 'alpha', 'stdev', 'mean', 'xret', 'hjd'],
            f'results_ipca_{model}.csv': ['iter', 'month', 'nfactors', 'stdev', 'mean', 'xret', 'hjd'],
        }

        # Create files and write headers
        for filename, fieldnames in file_specs.items():
            filepath = self.output_dir / filename
            f = open(filepath, 'w', newline='')
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            self.files[filename] = f
            self.writers[filename] = writer

        logger.info(f"Initialized CSV writer in {self.output_dir}")

    def write_monthly_results(self, results: Dict[str, Any]) -> None:
        """
        Write results for one month.

        Args:
            results: Dictionary with results from _evaluate_month
        """
        month = results['month']
        iter_num = results['iter']

        # Write model results
        for method, metrics in results.get('model_results', {}).items():
            # Write portfolio weights
            row = {'iter': iter_num, 'month': month, 'method': method}
            for i, w in enumerate(metrics['weights'].flatten()):
                row[f'firm_{i+1}'] = w
            self.writers[f'port_model_{self.model}.csv'].writerow(row)

            # Write metrics
            metrics_row = {
                'iter': iter_num,
                'month': month,
                'method': method,
                'stdev': metrics['stdev'],
                'mean': metrics['mean'],
                'xret': metrics['xret'],
                'hjd': metrics['hjd']
            }
            self.writers[f'results_model_{self.model}.csv'].writerow(metrics_row)

        # Write DKKM results (similar pattern)
        # Write Fama results (similar pattern)
        # Write IPCA results (similar pattern)
        # Abbreviated for clarity - see original main.py for full logic

    def write_panel(self, panel: pd.DataFrame, iter: int) -> None:
        """
        Accumulate panel data for writing.

        Args:
            panel: Panel DataFrame
            iter: Iteration number
        """
        panel_copy = panel.copy()
        panel_copy['iter'] = iter
        self.panel_accumulator.append(panel_copy.reset_index())

    def finalize(self) -> None:
        """Close all files and write accumulated panel data."""
        # Close all result files
        for f in self.files.values():
            f.close()

        # Write accumulated panel data
        if self.panel_accumulator:
            panel_out = pd.concat(self.panel_accumulator, ignore_index=True)
            panel_file = self.output_dir / f'panel_{self.model}.csv'
            panel_out.to_csv(panel_file, index=False)
            logger.info(f"Wrote panel data to {panel_file}")

        logger.info("CSV writer finalized")


class InMemoryResultsWriter:
    """
    Accumulates results in memory for testing.

    Useful for unit tests where we don't want to write files.
    """

    def __init__(self):
        """Initialize in-memory writer."""
        self.N = None
        self.model = None
        self.monthly_results = []
        self.panels = []

    def initialize(self, N: int, model: str) -> None:
        """
        Initialize writer.

        Args:
            N: Number of firms
            model: Model name
        """
        self.N = N
        self.model = model
        self.monthly_results = []
        self.panels = []

    def write_monthly_results(self, results: Dict[str, Any]) -> None:
        """
        Store results in memory.

        Args:
            results: Monthly results dictionary
        """
        self.monthly_results.append(results)

    def write_panel(self, panel: pd.DataFrame, iter: int) -> None:
        """
        Store panel in memory.

        Args:
            panel: Panel DataFrame
            iter: Iteration number
        """
        panel_copy = panel.copy()
        panel_copy['iter'] = iter
        self.panels.append(panel_copy.reset_index())

    def finalize(self) -> None:
        """Nothing to do for in-memory writer."""
        pass

    def to_dataframes(self) -> Dict[str, pd.DataFrame]:
        """
        Convert accumulated results to DataFrames.

        Returns:
            Dictionary mapping result type to DataFrame
        """
        dfs = {}

        # Model results
        model_rows = []
        for result in self.monthly_results:
            for method, metrics in result.get('model_results', {}).items():
                model_rows.append({
                    'iter': result['iter'],
                    'month': result['month'],
                    'method': method,
                    **{k: v for k, v in metrics.items() if k != 'weights'}
                })
        if model_rows:
            dfs['model'] = pd.DataFrame(model_rows)

        # Similar for DKKM, Fama, IPCA...
        # Abbreviated for clarity

        # Panel data
        if self.panels:
            dfs['panel'] = pd.concat(self.panels, ignore_index=True)

        return dfs
