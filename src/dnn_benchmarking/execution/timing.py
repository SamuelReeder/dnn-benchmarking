"""Timing utilities for benchmark execution."""

import time
from types import TracebackType
from typing import Optional, Type


class Timer:
    """Context manager for measuring wall-clock execution time.

    Uses time.perf_counter() for high-resolution timing.

    Example:
        with Timer() as t:
            # code to time
            pass
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self) -> None:
        """Initialize timer with zero elapsed time."""
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop timing."""
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self._end - self._start
