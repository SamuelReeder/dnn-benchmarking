"""Tests for Timer."""

import time

import pytest

from dnn_benchmarking.execution import Timer


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_measures_elapsed_time(self) -> None:
        """Test that timer measures elapsed time correctly."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms sleep

        # Should be at least 10ms, allow for some variance
        assert t.elapsed_ms >= 10.0
        assert t.elapsed_ms < 100.0  # Sanity check

    def test_timer_elapsed_seconds(self) -> None:
        """Test elapsed_s property."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms sleep

        assert t.elapsed_s >= 0.01
        assert t.elapsed_s < 0.1

    def test_timer_zero_when_not_started(self) -> None:
        """Test timer returns zero before use."""
        t = Timer()
        assert t.elapsed_ms == 0.0
        assert t.elapsed_s == 0.0

    def test_timer_reusable(self) -> None:
        """Test that timer can be reused."""
        t = Timer()

        with t:
            time.sleep(0.005)  # 5ms
        first_elapsed = t.elapsed_ms

        with t:
            time.sleep(0.01)  # 10ms
        second_elapsed = t.elapsed_ms

        # Second measurement should be longer
        assert second_elapsed > first_elapsed

    def test_timer_with_exception(self) -> None:
        """Test that timer still records time when exception occurs."""
        t = Timer()

        with pytest.raises(ValueError):
            with t:
                time.sleep(0.005)
                raise ValueError("test error")

        # Time should still be recorded
        assert t.elapsed_ms >= 5.0
