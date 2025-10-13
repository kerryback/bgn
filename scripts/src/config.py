"""
Configuration management for asset pricing simulations.

This module provides dataclasses for managing simulation parameters,
replacing the global variable approach with type-safe configuration objects.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional
import numpy as np
from .utils.exceptions import ConfigurationError


@dataclass
class SimulationConfig:
    """
    Configuration for simulation run.

    This class contains all parameters needed to run a complete simulation,
    including model selection, panel dimensions, method parameters, and
    computational settings.
    """

    # Model selection
    model: Literal['bgn', 'kp', 'gs'] = 'gs'

    # Simulation parameters
    start_iter: int = 0
    num_iters: int = 10

    # Panel dimensions
    N: int = 100  # Number of firms
    T: int = 400  # Number of time periods (excluding burnin)
    burnin: int = 200  # Burnin period

    # DKKM parameters
    include_mkt: bool = False  # Include market factor in DKKM
    nmat: int = 1  # Number of random weight matrices for DKKM
    nfeatures_lst: list[int] = field(default_factory=lambda: [6, 36, 360])
    n_ipca_rff: int = 36  # Number of RFF features in IPCA-DKKM hybrid

    # Regularization parameters
    alpha_lst_fama: list[float] = field(default_factory=lambda: [0])
    alpha_lst: list[float] = field(default_factory=lambda:
        [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 1])

    # IPCA parameters
    ipca_nfactors_lst: list[int] = field(default_factory=lambda: [1, 2])

    # Computation parameters
    n_jobs: int = 10

    # Output parameters
    output_dir: Path = field(default_factory=lambda: Path('.'))

    def __post_init__(self):
        """Validate and adjust configuration after initialization."""
        # Model-specific alpha adjustments
        if self.model == 'gs':
            self.alpha_lst = [0, 0.0000001, 0.000001, 0.00001,
                             0.0001, 0.001, 0.01, 0.1, 1]

        # Validate dimensions
        if self.N <= 0:
            raise ConfigurationError(f"N must be positive, got {self.N}")
        if self.T <= 0:
            raise ConfigurationError(f"T must be positive, got {self.T}")
        if self.burnin < 0:
            raise ConfigurationError(f"burnin must be non-negative, got {self.burnin}")

        # Validate iteration parameters
        if self.num_iters <= 0:
            raise ConfigurationError(f"num_iters must be positive, got {self.num_iters}")

        # Validate DKKM parameters
        if not self.nfeatures_lst:
            raise ConfigurationError("nfeatures_lst cannot be empty")
        if any(n <= 0 for n in self.nfeatures_lst):
            raise ConfigurationError("All values in nfeatures_lst must be positive")

        # Validate IPCA parameters
        if not self.ipca_nfactors_lst:
            raise ConfigurationError("ipca_nfactors_lst cannot be empty")
        if any(n <= 0 for n in self.ipca_nfactors_lst):
            raise ConfigurationError("All values in ipca_nfactors_lst must be positive")

        # Compute derived values
        self.max_features = max(self.nfeatures_lst)

        # Ensure output directory exists
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: dict) -> 'SimulationConfig':
        """
        Create configuration from dictionary.

        Args:
            config_dict: Dictionary of configuration parameters

        Returns:
            SimulationConfig instance
        """
        return cls(**config_dict)

    @classmethod
    def from_yaml(cls, path: Path) -> 'SimulationConfig':
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file

        Returns:
            SimulationConfig instance

        Raises:
            ConfigurationError: If file cannot be loaded
        """
        try:
            import yaml
        except ImportError:
            raise ConfigurationError("PyYAML must be installed to load YAML configs")

        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            return cls(**data)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config from {path}: {e}")

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            'model': self.model,
            'start_iter': self.start_iter,
            'num_iters': self.num_iters,
            'N': self.N,
            'T': self.T,
            'burnin': self.burnin,
            'include_mkt': self.include_mkt,
            'nmat': self.nmat,
            'nfeatures_lst': self.nfeatures_lst,
            'n_ipca_rff': self.n_ipca_rff,
            'alpha_lst_fama': self.alpha_lst_fama,
            'alpha_lst': self.alpha_lst,
            'ipca_nfactors_lst': self.ipca_nfactors_lst,
            'n_jobs': self.n_jobs,
            'output_dir': str(self.output_dir),
        }
