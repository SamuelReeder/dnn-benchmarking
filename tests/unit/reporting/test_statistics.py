# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for BenchmarkStats."""

import pytest

from dnn_benchmarking.reporting import BenchmarkStats


class TestBenchmarkStats:
    """Tests for BenchmarkStats dataclass."""

    def test_from_timings_basic(self) -> None:
        """Test basic stats calculation."""
        timings = [1.0, 2.0, 3.0, 4.0, 5.0]
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 3.0
        assert stats.min_ms == 1.0
        assert stats.max_ms == 5.0

    def test_from_timings_single_value(self) -> None:
        """Test stats with single value."""
        timings = [5.0]
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 5.0
        assert stats.std_ms == 0.0
        assert stats.min_ms == 5.0
        assert stats.max_ms == 5.0
        assert stats.p95_ms == 5.0
        assert stats.p99_ms == 5.0

    def test_from_timings_empty_raises(self) -> None:
        """Test that empty timings raises ValueError."""
        with pytest.raises(ValueError, match="timings list cannot be empty"):
            BenchmarkStats.from_timings([])

    def test_from_timings_std_calculation(self) -> None:
        """Test standard deviation calculation."""
        timings = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 5.0
        assert stats.std_ms == pytest.approx(2.0, rel=0.01)

    def test_from_timings_percentiles(self) -> None:
        """Test percentile calculations."""
        # 100 values from 1 to 100
        timings = list(range(1, 101))
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 50.5
        assert stats.min_ms == 1.0
        assert stats.max_ms == 100.0
        # For 100 uniformly spaced values, p95 should be around 95.05
        assert stats.p95_ms == pytest.approx(95.05, rel=0.01)
        assert stats.p99_ms == pytest.approx(99.01, rel=0.01)

    def test_from_timings_uniform_values(self) -> None:
        """Test with all identical values."""
        timings = [10.0] * 100
        stats = BenchmarkStats.from_timings(timings)

        assert stats.mean_ms == 10.0
        assert stats.std_ms == 0.0
        assert stats.min_ms == 10.0
        assert stats.max_ms == 10.0
        assert stats.p95_ms == 10.0
        assert stats.p99_ms == 10.0
