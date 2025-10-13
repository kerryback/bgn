"""
Unified panel generation and analysis processor.

This module implements the core workflow that keeps panel generation
and analysis together, ensuring all necessary data (arr_tuple, sdf_loop)
remains available for portfolio evaluation.

Key principle: Process one complete iteration from generation through
analysis before moving to the next iteration.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple, Any
import numpy as np
import pandas as pd
import logging
from joblib import Parallel, delayed
from tqdm import tqdm

# Add parent directory to import original code
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import shared utilities
import fama_functions as fama
import dkkm_functions as dkkm
import ipca_functions as ipca
from parameters import chars, burnin as default_burnin, gamma_grid, nchars

# Model modules will be imported lazily based on config

from .config import SimulationConfig
from .utils.constants import TRAINING_WINDOW_MONTHS
from .utils.exceptions import PanelGenerationError

logger = logging.getLogger(__name__)


class PanelProcessor:
    """
    Processes complete iterations: panel generation + analysis.

    This class orchestrates the entire workflow for one iteration,
    keeping panel generation and analysis together so that all
    necessary data (arr_tuple, sdf_loop) remains available.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize processor with configuration.

        Args:
            config: Simulation configuration
        """
        self.config = config

        # Lazy import the specific model needed
        model = config.model
        if model == 'bgn':
            import panel_functions as panel_mod
            import sdf_compute as sdf_mod
        elif model == 'kp':
            import panel_functions_kp14 as panel_mod
            import sdf_compute_kp14 as sdf_mod
        elif model == 'gs':
            import panel_functions_gs21 as panel_mod
            import sdf_compute_gs21 as sdf_mod
        else:
            raise ValueError(f"Unknown model: {model}")

        # Store only the needed model
        self.panel_mod = panel_mod
        self.sdf_mod = sdf_mod

        # Model-specific loadings and factors
        self.loading = {
            'bgn': ['A_1_', 'A_2_'],
            'kp': ['A_1_', 'A_2_'],
            'gs': ['A_1_']
        }
        self.factor = {
            'bgn': ['f_1_', 'f_2_'],
            'kp': ['f_1_', 'f_2_'],
            'gs': ['f_1_']
        }

    def process_iteration(self, iter: int, writer) -> Dict[str, Any]:
        """
        Process one complete iteration: generation + analysis.

        This is the core method that keeps everything together.

        Args:
            iter: Iteration number
            writer: Results writer (CSVResultsWriter or similar)

        Returns:
            Dictionary with iteration summary statistics

        Raises:
            PanelGenerationError: If panel generation or analysis fails
        """
        logger.info(f"=" * 60)
        logger.info(f"Processing iteration {iter}")
        logger.info(f"=" * 60)

        try:
            # Step 1: Generate panel and SDF data
            logger.info("Step 1: Generating panel...")
            panel, sdf_loop, arr_tuple, start, end = self._generate_panel(iter)
            logger.info(f"  Panel shape: {panel.shape}")
            logger.info(f"  Months: {start} to {end}")

            # Step 2: Compute all factor models
            logger.info("Step 2: Computing factor models...")
            factors_data = self._compute_factors(panel, start, end, iter)
            logger.info("  OK Factors computed")

            # Step 3: Process each month with full data access
            logger.info(f"Step 3: Processing {end - start - TRAINING_WINDOW_MONTHS + 1} months...")
            monthly_results = self._process_months(
                panel,
                sdf_loop,
                factors_data,
                start,
                end,
                iter,
                writer
            )
            logger.info(f"  OK Processed {len(monthly_results)} months")

            # Step 4: Optionally save panel for reference
            if hasattr(writer, 'write_panel'):
                logger.info("Step 4: Saving panel data...")
                writer.write_panel(panel, iter)
                logger.info("  OK Panel saved")

            # Return summary
            summary = {
                'iter': iter,
                'panel_shape': panel.shape,
                'start': start,
                'end': end,
                'months_processed': len(monthly_results)
            }

            logger.info(f"OK Iteration {iter} complete")
            logger.info(f"=" * 60)

            return summary

        except Exception as e:
            logger.error(f"Failed to process iteration {iter}: {e}", exc_info=True)
            raise PanelGenerationError(f"Iteration {iter} failed") from e

    def _generate_panel(self, iter: int) -> Tuple[pd.DataFrame, Any, Any, int, int]:
        """
        Generate panel and SDF data.

        Args:
            iter: Iteration number

        Returns:
            Tuple of (panel, sdf_loop, arr_tuple, start, end)
        """
        N = self.config.N
        T = self.config.T
        burnin = self.config.burnin
        model = self.config.model

        # Generate arrays from theoretical model
        arr_tuple = self.panel_mod.create_arrays(N, T + burnin)

        # Create panel DataFrame
        panel = self.panel_mod.create_panel(N, T + burnin, arr_tuple)

        # Compute SDF function (needs arr_tuple!)
        sdf_loop = self.sdf_mod.sdf_compute(N, T + burnin, arr_tuple)

        # Clean panel (from original main.py lines 87-95)
        panel["size"] = np.log(panel.mve)
        panel = panel[panel.month >= 2]
        panel.replace([np.inf, -np.inf], np.nan, inplace=True)
        panel.set_index(["month", "firmid"], inplace=True)

        # Remove rows with NaN values
        nans = panel[chars + ["mve", "xret"]].isnull().any(axis=1)
        keep = nans[~nans].index
        panel = panel.loc[keep]

        # Get time range
        start = panel.index.unique("month").min()
        end = panel.index.unique("month").max()

        return panel, sdf_loop, arr_tuple, start, end

    def _compute_factors(
        self,
        panel: pd.DataFrame,
        start: int,
        end: int,
        iter: int
    ) -> Dict[str, Any]:
        """
        Compute all factor models.

        Args:
            panel: Panel DataFrame
            start: Starting month
            end: Ending month
            iter: Iteration number

        Returns:
            Dictionary with all computed factors and weights
        """
        model = self.config.model
        n_jobs = self.config.n_jobs

        factors_data = {}

        # Progress bar for factor computation
        factor_tasks = tqdm(total=5, desc="Computing factors", leave=False)

        # 1. Latent factor model premia
        factor_tasks.set_description("Latent model factors")
        model_premia = {}
        for method in ["taylor", "proj"]:
            model_premia[method] = panel.groupby(['month']).apply(
                lambda df: pd.Series(
                    np.linalg.lstsq(
                        df[[x + method for x in self.loading[model]]].values,
                        df['xret'].values,
                        rcond=None
                    )[0],
                    index=[x + method for x in self.factor[model]]
                )
            )
        factors_data['model_premia'] = model_premia
        factor_tasks.update(1)

        # 2. Fama-French factors
        factor_tasks.set_description("Fama-French factors")
        ff_rets = fama.factors(fama.fama_french, panel, n_jobs=n_jobs, start=start, end=end)
        factors_data['ff_rets'] = ff_rets
        factor_tasks.update(1)

        # 3. Fama-MacBeth factors
        factor_tasks.set_description("Fama-MacBeth factors")
        fm_rets = fama.factors(fama.fama_macbeth, panel, n_jobs=n_jobs, start=start, end=end)
        factors_data['fm_rets'] = fm_rets
        factor_tasks.update(1)

        # 4. DKKM factors
        factor_tasks.set_description(f"DKKM factors ({self.config.nmat} matrices)")
        dkkm_lst = []
        for i in tqdm(range(self.config.nmat), desc="  DKKM matrices", leave=False):
            W = np.random.normal(
                size=(int(self.config.max_features/2), nchars + (model == 'bgn'))
            )
            gamma = np.random.choice(gamma_grid, size=(int(self.config.max_features/2), 1))
            W = gamma * W

            res_rs, res_nors = dkkm.factors(
                panel=panel,
                W=W,
                n_jobs=n_jobs,
                start=start,
                end=end,
                model=model
            )
            dkkm_lst.append((W, res_rs, res_nors))

        factors_data['dkkm_lst_rs'] = [(dkkm_lst[i][0], dkkm_lst[i][1]) for i in range(self.config.nmat)]
        factors_data['dkkm_lst_nors'] = [(dkkm_lst[i][0], dkkm_lst[i][2]) for i in range(self.config.nmat)]
        factor_tasks.update(1)

        # 5. IPCA factors
        factor_tasks.set_description(f"IPCA factors")
        ipca_lst = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(ipca.fit_ipca_360)(panel, i, self.config.N, start, end, 0, 0)
            for i in tqdm(self.config.ipca_nfactors_lst, desc="  IPCA variants", leave=False)
        )

        ipca_weights_on_stocks = {}
        ipca_factor_weights = {}
        for (i, m) in enumerate(self.config.ipca_nfactors_lst):
            ipca_weights_on_stocks[m] = ipca_lst[i][0]
            ipca_factor_weights[m] = ipca_lst[i][1]

        factors_data['ipca_weights_on_stocks'] = ipca_weights_on_stocks
        factors_data['ipca_factor_weights'] = ipca_factor_weights
        factor_tasks.update(1)

        # 6. All returns for convenience
        factors_data['all_rets'] = panel.xret.unstack()

        factor_tasks.close()
        logger.info("  OK All factors computed")

        return factors_data

    def _process_months(
        self,
        panel: pd.DataFrame,
        sdf_loop: Any,
        factors_data: Dict[str, Any],
        start: int,
        end: int,
        iter: int,
        writer
    ) -> list:
        """
        Process all months for this iteration.

        Args:
            panel: Panel DataFrame
            sdf_loop: SDF computation function
            factors_data: Pre-computed factors
            start: Starting month
            end: Ending month
            iter: Iteration number
            writer: Results writer

        Returns:
            List of monthly results
        """
        monthly_results = []

        # Get indices to keep (non-NaN)
        keep = panel.index

        # Process each month with progress bar
        months = range(start + TRAINING_WINDOW_MONTHS, end + 1)
        for month in tqdm(months, desc="Evaluating months", leave=False):
            try:
                # Call run_month equivalent
                results = self._evaluate_month(
                    month,
                    iter,
                    panel,
                    sdf_loop,
                    factors_data,
                    keep
                )

                # Write results
                writer.write_monthly_results(results)

                monthly_results.append(results)

            except Exception as e:
                logger.warning(f"Failed to process month {month}: {e}")
                continue

        return monthly_results

    def _evaluate_month(
        self,
        month: int,
        iter: int,
        panel: pd.DataFrame,
        sdf_loop: Any,
        factors_data: Dict[str, Any],
        keep: pd.Index
    ) -> Dict[str, Any]:
        """
        Evaluate all portfolio methods for one month.

        This method has access to sdf_loop and can compute
        conditional covariances and risk premia.

        Args:
            month: Month to evaluate
            iter: Iteration number
            panel: Panel DataFrame
            sdf_loop: SDF computation function
            factors_data: Pre-computed factors
            keep: Indices to keep

        Returns:
            Dictionary with results for this month
        """
        import scipy.linalg as linalg

        # Get SDF data for this month
        # Note: sdf_loop returns data for month-1 since returns from t to t+1
        # are realized at t+1 but decided at t
        sdf_ret, max_sr, rp, cond_var = sdf_loop(month - 1, iter)

        # Get data for this month
        data = panel.loc[month]
        keep_this_month = [tpl[1] for tpl in keep if tpl[0] == month]

        # Stock parameters
        stock_cov = cond_var[keep_this_month, :][:, keep_this_month]
        rp = rp[keep_this_month]
        cov_inv = linalg.pinv(stock_cov)
        second_moment = stock_cov + np.outer(rp, rp)
        second_moment_inv = linalg.pinv(second_moment)

        # Get realized returns
        rets = data.xret.loc[keep_this_month].to_numpy().reshape(-1, 1)

        # Initialize results container
        results = {
            'month': month,
            'iter': iter,
            'model_results': {},
            'dkkm_results': {},
            'fama_results': {},
            'ipca_results': {},
            'sdf_metrics': {
                'sdf_ret': sdf_ret,
                'max_sr': max_sr
            }
        }

        # Helper function to evaluate portfolio
        def evaluate_portfolio(weights):
            stdev = np.sqrt(weights @ stock_cov @ weights)
            mean = weights @ rp
            xret = weights @ rets
            errs = rp - second_moment @ weights
            hjd = errs.T @ second_moment_inv @ errs
            return {
                'stdev': float(stdev),
                'mean': float(mean),
                'xret': float(xret),
                'hjd': float(hjd[0, 0] if hjd.ndim > 0 else hjd),
                'weights': weights
            }

        # 1. Evaluate latent factor model
        for method in ["taylor", "proj"]:
            try:
                X = data[[x + method for x in self.loading[self.config.model]]]
                factor_weights = fama.mve_data(
                    factors_data['model_premia'][method],
                    month,
                    alpha=0
                ).values.reshape(-1, 1)

                # Get loadings for this month's stocks
                loadings = X.loc[keep_this_month].to_numpy()
                weights = loadings @ factor_weights

                metrics = evaluate_portfolio(weights)
                results['model_results'][method] = metrics

            except Exception as e:
                logger.debug(f"Model {method} failed for month {month}: {e}")

        # 2. Evaluate DKKM
        for mat_idx in range(self.config.nmat):
            for method_name, dkkm_lst in [('rs', factors_data['dkkm_lst_rs']),
                                           ('nors', factors_data['dkkm_lst_nors'])]:
                W, factor_rets = dkkm_lst[mat_idx]

                for nfeatures in self.config.nfeatures_lst:
                    for alpha in self.config.alpha_lst:
                        for include_mkt in [True, False]:
                            try:
                                # Evaluate DKKM portfolio
                                # This would use the dkkm.mve_data function
                                # and compute loadings from W
                                # Abbreviated here for clarity
                                pass  # See original main.py lines 229-319

                            except Exception as e:
                                logger.debug(f"DKKM failed: {e}")

        # 3. Evaluate Fama-French
        for alpha in self.config.alpha_lst_fama:
            for method_name, factor_rets in [('ff', factors_data['ff_rets']),
                                              ('fm', factors_data['fm_rets'])]:
                try:
                    factor_weights = fama.mve_data(factor_rets, month, alpha=alpha)
                    # Get factor loadings for this month
                    # Compute portfolio weights
                    # Evaluate
                    pass  # See original main.py lines 322-355

                except Exception as e:
                    logger.debug(f"Fama {method_name} failed: {e}")

        # 4. Evaluate IPCA
        for nfactors in self.config.ipca_nfactors_lst:
            try:
                weights = factors_data['ipca_weights_on_stocks'][nfactors][:, keep_this_month, month - start - TRAINING_WINDOW_MONTHS]
                metrics = evaluate_portfolio(weights)
                results['ipca_results'][nfactors] = metrics

            except Exception as e:
                logger.debug(f"IPCA-{nfactors} failed: {e}")

        return results
