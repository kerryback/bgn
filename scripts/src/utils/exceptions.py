"""
Custom exceptions for the asset pricing simulation.

This module defines custom exception types for better error handling
and debugging throughout the simulation.
"""


class SimulationError(Exception):
    """Base exception for simulation errors."""
    pass


class ConfigurationError(SimulationError):
    """Raised when configuration is invalid."""
    pass


class NumericalInstabilityError(SimulationError):
    """Raised when numerical computation fails."""
    pass


class InsufficientDataError(SimulationError):
    """Raised when not enough data for estimation."""
    pass


class PanelGenerationError(SimulationError):
    """Raised when panel generation fails."""
    pass


class StrategyError(SimulationError):
    """Raised when a portfolio strategy fails."""
    pass
