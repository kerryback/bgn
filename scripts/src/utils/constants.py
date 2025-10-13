"""
Constants used throughout the simulation.

This module centralizes all magic numbers and constant values used in the
asset pricing simulation, making them easier to understand and modify.
"""

# Rolling windows
ROLLING_WINDOW_MONTHS = 360  # 30 years of monthly data for factor estimation
TRAINING_WINDOW_MONTHS = 360  # Training window for portfolio construction

# Panel construction
MIN_MONTH_FOR_LAGS = 2  # Minimum month after computing lags (momentum, etc.)
MOMENTUM_LOOKBACK_START = 2  # Start of momentum calculation window (months ago)
MOMENTUM_LOOKBACK_END = 13  # End of momentum calculation window (months ago)

# Rank standardization (DKKM)
RANK_STD_MIN = -0.5  # Minimum value after rank standardization
RANK_STD_MAX = 0.5   # Maximum value after rank standardization
RANK_STD_CENTER = 0.0  # Center value after rank standardization

# Firm characteristics
DEFAULT_CHARACTERISTICS = ["size", "bm", "agr", "roe", "mom"]

# Fama-French portfolio construction
FF_SIZE_BREAKPOINT = 0.5  # Median for size split (big/small)
FF_CHAR_LOW_BREAKPOINT = 0.3  # 30th percentile for low characteristic
FF_CHAR_HIGH_BREAKPOINT = 0.7  # 70th percentile for high characteristic

# Sorted portfolios
N_QUINTILES = 5  # Number of quintiles for sorted portfolios

# Numerical stability
MATRIX_PERTURBATION = 1e-6  # Small value to add to diagonal for stability
CONDITION_NUMBER_THRESHOLD = 1e10  # Threshold for ill-conditioned matrices
MAX_PORTFOLIO_WEIGHT = 1e3  # Maximum absolute portfolio weight (safety check)

# Output formats
FLOAT_PRECISION = 6  # Decimal places for float output
