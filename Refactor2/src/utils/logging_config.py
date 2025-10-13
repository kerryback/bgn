"""
Logging configuration for the simulation.

This module provides centralized logging setup with support for both
console and file output.
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(
    log_file: Optional[Path] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> None:
    """
    Configure logging for simulation.

    Args:
        log_file: Path to log file (None for console only)
        level: Logging level (default: INFO)
        format_string: Custom format string (None for default)

    Example:
        >>> setup_logging(Path('simulation.log'), level=logging.DEBUG)
        >>> logger = logging.getLogger(__name__)
        >>> logger.info("Simulation started")
    """
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '%(funcName)s:%(lineno)d - %(message)s'
        )

    # Create handlers
    handlers = [logging.StreamHandler()]
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=format_string,
        handlers=handlers,
        force=True  # Override any existing configuration
    )


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the given name.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
