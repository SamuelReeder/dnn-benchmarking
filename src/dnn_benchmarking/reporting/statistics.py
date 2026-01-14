"""Benchmark statistics calculation."""

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class BenchmarkStats:
    """Statistics from benchmark execution.

    Attributes:
        mean_ms: Mean execution time in milliseconds.
        std_ms: Standard deviation of execution time in milliseconds.
        min_ms: Minimum execution time in milliseconds.
        max_ms: Maximum execution time in milliseconds.
        p95_ms: 95th percentile execution time in milliseconds.
        p99_ms: 99th percentile execution time in milliseconds.
    """

    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p95_ms: float
    p99_ms: float

    @classmethod
    def from_timings(cls, timings: List[float]) -> "BenchmarkStats":
        """Calculate statistics from a list of timing values.

        Args:
            timings: List of execution times in milliseconds.

        Returns:
            BenchmarkStats with calculated statistics.

        Raises:
            ValueError: If timings list is empty.
        """
        if not timings:
            raise ValueError("timings list cannot be empty")

        arr = np.array(timings)

        return cls(
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr)),
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
        )
