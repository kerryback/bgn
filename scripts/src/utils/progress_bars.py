"""
Enhanced progress tracking with tqdm progress bars.

This module provides utilities for displaying progress bars during
slow operations in the simulation.
"""

from typing import Optional, Iterator, Any
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class ProgressBarManager:
    """
    Manages nested progress bars for complex workflows.

    This class helps coordinate multiple progress bars at different
    levels (e.g., iterations, months, factors).
    """

    def __init__(self, disable: bool = False):
        """
        Initialize progress bar manager.

        Args:
            disable: If True, disable all progress bars (useful for logging)
        """
        self.disable = disable
        self.bars = {}

    def create_bar(
        self,
        total: int,
        desc: str,
        position: int = 0,
        leave: bool = True,
        unit: str = "it"
    ) -> tqdm:
        """
        Create a new progress bar.

        Args:
            total: Total number of iterations
            desc: Description text
            position: Position for nested bars (0 = outermost)
            leave: Whether to leave bar after completion
            unit: Unit name for progress

        Returns:
            tqdm progress bar object
        """
        bar = tqdm(
            total=total,
            desc=desc,
            position=position,
            leave=leave,
            unit=unit,
            disable=self.disable
        )
        self.bars[desc] = bar
        return bar

    def update_bar(self, desc: str, n: int = 1, **kwargs) -> None:
        """
        Update a progress bar.

        Args:
            desc: Description of bar to update
            n: Number of steps to advance
            **kwargs: Additional attributes to set (e.g., postfix)
        """
        if desc in self.bars:
            bar = self.bars[desc]
            bar.update(n)
            if kwargs:
                bar.set_postfix(kwargs)

    def close_bar(self, desc: str) -> None:
        """
        Close a specific progress bar.

        Args:
            desc: Description of bar to close
        """
        if desc in self.bars:
            self.bars[desc].close()
            del self.bars[desc]

    def close_all(self) -> None:
        """Close all progress bars."""
        for bar in self.bars.values():
            bar.close()
        self.bars.clear()


def iterate_with_progress(
    iterable: Iterator[Any],
    desc: str,
    total: Optional[int] = None,
    leave: bool = True,
    unit: str = "it",
    disable: bool = False
) -> Iterator[Any]:
    """
    Wrap an iterable with a progress bar.

    Args:
        iterable: Iterable to wrap
        desc: Description text
        total: Total iterations (if known)
        leave: Whether to leave bar after completion
        unit: Unit name
        disable: If True, disable progress bar

    Yields:
        Items from iterable with progress tracking

    Example:
        >>> for month in iterate_with_progress(range(100), desc="Processing months"):
        >>>     process_month(month)
    """
    yield from tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        unit=unit,
        disable=disable
    )


def parallel_progress(
    func,
    items,
    n_jobs: int = -1,
    desc: str = "Processing",
    backend: str = "loky"
):
    """
    Execute parallel computation with progress bar.

    This wraps joblib.Parallel to add progress tracking.

    Args:
        func: Function to apply
        items: Items to process
        n_jobs: Number of parallel jobs
        desc: Description for progress bar
        backend: Joblib backend to use

    Returns:
        List of results

    Example:
        >>> def process(x):
        >>>     return x ** 2
        >>> results = parallel_progress(process, range(100), n_jobs=4, desc="Computing")
    """
    from joblib import Parallel, delayed

    # Note: Progress tracking with joblib is tricky because processes
    # run in parallel. We use a simple wrapper here.
    with tqdm(total=len(items), desc=desc) as pbar:
        results = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(func)(item) for item in items
        )
        pbar.update(len(items))

    return results


class ProgressContext:
    """
    Context manager for progress bars.

    This provides a clean way to ensure progress bars are properly
    closed even if exceptions occur.

    Example:
        >>> with ProgressContext(total=100, desc="Processing") as pbar:
        >>>     for i in range(100):
        >>>         # ... do work ...
        >>>         pbar.update(1)
    """

    def __init__(self, total: int, desc: str, **kwargs):
        """
        Initialize progress context.

        Args:
            total: Total iterations
            desc: Description text
            **kwargs: Additional tqdm arguments
        """
        self.total = total
        self.desc = desc
        self.kwargs = kwargs
        self.pbar = None

    def __enter__(self) -> tqdm:
        """Enter context and create progress bar."""
        self.pbar = tqdm(total=self.total, desc=self.desc, **self.kwargs)
        return self.pbar

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context and close progress bar."""
        if self.pbar is not None:
            self.pbar.close()
        return False  # Don't suppress exceptions


def log_and_progress(
    iterable,
    desc: str,
    logger_obj: Optional[logging.Logger] = None,
    log_every: int = 10,
    **tqdm_kwargs
):
    """
    Combine logging and progress bars.

    This wraps an iterable with both a progress bar and periodic logging,
    useful for long-running processes where you want both visual progress
    and log records.

    Args:
        iterable: Items to iterate
        desc: Description for both log and progress bar
        logger_obj: Logger to use (None = module logger)
        log_every: Log progress every N items
        **tqdm_kwargs: Additional arguments for tqdm

    Yields:
        Items from iterable

    Example:
        >>> logger = logging.getLogger(__name__)
        >>> for item in log_and_progress(items, "Processing items", logger):
        >>>     process(item)
    """
    if logger_obj is None:
        logger_obj = logger

    logger_obj.info(f"Starting: {desc}")

    for i, item in enumerate(tqdm(iterable, desc=desc, **tqdm_kwargs), 1):
        if i % log_every == 0:
            logger_obj.debug(f"{desc}: {i} items processed")
        yield item

    logger_obj.info(f"Completed: {desc}")
