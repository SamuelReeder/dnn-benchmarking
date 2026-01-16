"""Benchmark statistics calculation."""

from dataclasses import dataclass
from typing import List, Optional

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


@dataclass
class BenchmarkResult:
    """Raw benchmark timing results.

    Holds both E2E (wall-clock) and optional kernel (GPU event) timings.

    Attributes:
        e2e_timings: List of end-to-end execution times in milliseconds.
        kernel_timings: Optional list of GPU kernel times in milliseconds.
    """

    e2e_timings: List[float]
    kernel_timings: Optional[List[float]] = None

    @property
    def has_kernel_timings(self) -> bool:
        """Check if kernel timings are available."""
        return self.kernel_timings is not None and len(self.kernel_timings) > 0


@dataclass
class CombinedBenchmarkStats:
    """Combined statistics for E2E and kernel timing.

    Attributes:
        e2e_stats: Statistics from wall-clock timing.
        kernel_stats: Optional statistics from GPU kernel timing.
    """

    e2e_stats: BenchmarkStats
    kernel_stats: Optional[BenchmarkStats] = None

    @classmethod
    def from_result(cls, result: BenchmarkResult) -> "CombinedBenchmarkStats":
        """Create combined stats from a BenchmarkResult.

        Args:
            result: BenchmarkResult with E2E and optional kernel timings.

        Returns:
            CombinedBenchmarkStats with calculated statistics.
        """
        e2e = BenchmarkStats.from_timings(result.e2e_timings)
        kernel = (
            BenchmarkStats.from_timings(result.kernel_timings)
            if result.has_kernel_timings
            else None
        )
        return cls(e2e_stats=e2e, kernel_stats=kernel)
