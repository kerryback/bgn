"""
Progress tracking for long-running simulations.

This module provides utilities for tracking and reporting progress
during Monte Carlo simulations.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    """
    Track progress of long-running simulation.

    Attributes:
        total_iterations: Total number of iterations to complete
        current_iteration: Current iteration number (0-indexed)
        start_time: Time when tracking started
    """

    total_iterations: int
    current_iteration: int = 0
    start_time: Optional[datetime] = None
    _last_update_time: Optional[datetime] = field(default=None, init=False, repr=False)

    def start(self) -> None:
        """Start tracking progress."""
        self.start_time = datetime.now()
        self.current_iteration = 0
        self._last_update_time = self.start_time
        logger.info(f"Starting {self.total_iterations} iterations")

    def update(self, iter_num: int) -> None:
        """
        Update progress for given iteration.

        Args:
            iter_num: Current iteration number (0-indexed)
        """
        self.current_iteration = iter_num
        now = datetime.now()

        if self.start_time is None:
            logger.warning("ProgressTracker.update() called before start()")
            return

        # Calculate timing statistics
        elapsed = now - self.start_time
        completed = iter_num + 1
        per_iter = elapsed / completed
        remaining_iters = self.total_iterations - completed
        eta = now + per_iter * remaining_iters

        # Calculate percentage
        pct = 100 * completed / self.total_iterations

        # Format output
        msg = (f"Iteration {completed}/{self.total_iterations} "
               f"({pct:.1f}%) - "
               f"Elapsed: {self._format_timedelta(elapsed)} - "
               f"ETA: {eta.strftime('%Y-%m-%d %H:%M:%S')}")

        logger.info(msg)
        self._last_update_time = now

    def summary(self) -> None:
        """Print final summary of timing statistics."""
        if self.start_time is None:
            logger.warning("ProgressTracker.summary() called before start()")
            return

        elapsed = datetime.now() - self.start_time
        per_iter = elapsed / self.total_iterations

        logger.info("")
        logger.info("=" * 60)
        logger.info(f"Completed {self.total_iterations} iterations")
        logger.info(f"Total time: {self._format_timedelta(elapsed)}")
        logger.info(f"Average per iteration: {self._format_timedelta(per_iter)}")
        logger.info("=" * 60)

    @staticmethod
    def _format_timedelta(td: timedelta) -> str:
        """
        Format timedelta in human-readable format.

        Args:
            td: Timedelta to format

        Returns:
            Formatted string (e.g., "2h 15m 30s")
        """
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        parts = []
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0 or hours > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")

        return " ".join(parts)
