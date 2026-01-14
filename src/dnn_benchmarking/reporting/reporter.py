"""Console output formatting for benchmark results."""

import sys
from pathlib import Path
from typing import TextIO

from ..config.benchmark_config import ABTestConfig, BenchmarkConfig
from .statistics import BenchmarkStats


class Reporter:
    """Formats and prints benchmark results to console.

    Handles all console output including:
    - Configuration header
    - Initialization timing
    - Execution statistics
    - Validation results
    """

    WIDTH = 80

    def __init__(self, output: TextIO = sys.stdout) -> None:
        """Initialize reporter with output stream.

        Args:
            output: Output stream (default: stdout).
        """
        self._output = output

    def print_header(self, config: BenchmarkConfig, graph_name: str) -> None:
        """Print benchmark configuration header.

        Args:
            config: Benchmark configuration.
            graph_name: Name of the graph being benchmarked.
        """
        self._print_line("=")
        self._print(f"hipDNN Benchmark: {graph_name}")
        self._print_line("=")
        self._print(f"Graph:      {config.graph_path}")
        self._print(f"Engine ID:  {config.engine_id} (MIOpen)")
        self._print(f"Warmup:     {config.warmup_iters} iterations")
        self._print(f"Benchmark:  {config.benchmark_iters} iterations")
        self._print_line("-")
        self._print("")

    def print_init_time(self, init_time_ms: float) -> None:
        """Print initialization timing.

        Args:
            init_time_ms: Graph initialization time in milliseconds.
        """
        self._print("Initialization:")
        self._print(f"  Graph build time:     {init_time_ms:.2f} ms")
        self._print("")

    def print_stats(self, stats: BenchmarkStats) -> None:
        """Print execution statistics.

        Args:
            stats: Benchmark statistics.
        """
        self._print("Execution Statistics:")
        self._print(f"  Mean:                 {stats.mean_ms:.3f} ms")
        self._print(f"  Std Dev:              {stats.std_ms:.3f} ms")
        self._print(f"  Min:                  {stats.min_ms:.3f} ms")
        self._print(f"  Max:                  {stats.max_ms:.3f} ms")
        self._print(f"  P95:                  {stats.p95_ms:.3f} ms")
        self._print(f"  P99:                  {stats.p99_ms:.3f} ms")
        self._print("")

    def print_validation(self, passed: bool, message: str) -> None:
        """Print validation result.

        Args:
            passed: Whether validation passed.
            message: Validation message.
        """
        status = "PASSED" if passed else "FAILED"
        if "skipped" in message.lower() or "stubbed" in message.lower():
            status = "SKIPPED"

        self._print(f"Validation: {status} ({message})")

    def print_footer(self) -> None:
        """Print benchmark footer."""
        self._print_line("=")

    def print_error(self, message: str) -> None:
        """Print error message.

        Args:
            message: Error message.
        """
        self._print(f"ERROR: {message}")

    def _print(self, text: str) -> None:
        """Print a line of text.

        Args:
            text: Text to print.
        """
        print(text, file=self._output)

    def _print_line(self, char: str) -> None:
        """Print a horizontal line.

        Args:
            char: Character to use for the line.
        """
        print(char * self.WIDTH, file=self._output)

    # A/B Testing Methods

    def print_ab_header(
        self, config: BenchmarkConfig, ab_config: ABTestConfig, graph_name: str
    ) -> None:
        """Print A/B test configuration header.

        Args:
            config: Benchmark configuration.
            ab_config: A/B test configuration.
            graph_name: Name of the graph being benchmarked.
        """
        self._print_line("=")
        self._print(f"hipDNN A/B Test: {graph_name}")
        self._print_line("=")
        self._print(f"Graph:      {config.graph_path}")
        self._print(f"Warmup:     {config.warmup_iters} iterations")
        self._print(f"Benchmark:  {config.benchmark_iters} iterations")
        self._print_line("-")
        self._print("Configuration A:")
        if ab_config.a_path:
            self._print(f"  Plugin Path: {ab_config.a_path}")
        else:
            self._print("  Plugin Path: (default)")
        self._print(f"  Engine ID:   {ab_config.a_id}")
        self._print("Configuration B:")
        if ab_config.b_path:
            self._print(f"  Plugin Path: {ab_config.b_path}")
        else:
            self._print("  Plugin Path: (default)")
        self._print(f"  Engine ID:   {ab_config.b_id}")
        self._print_line("-")
        self._print("")

    def print_ab_stats(
        self,
        stats_a: BenchmarkStats,
        stats_b: BenchmarkStats,
        init_time_a_ms: float,
        init_time_b_ms: float,
    ) -> None:
        """Print side-by-side comparison of A vs B statistics.

        Args:
            stats_a: Statistics for configuration A.
            stats_b: Statistics for configuration B.
            init_time_a_ms: Init time for A in milliseconds.
            init_time_b_ms: Init time for B in milliseconds.
        """
        # Header
        self._print(f"{'':20} {'A':>15} {'B':>15}")
        self._print_line("-")

        # Init times
        self._print(
            f"{'Init Time:':20} {init_time_a_ms:>12.2f} ms {init_time_b_ms:>12.2f} ms"
        )

        # Execution stats
        self._print(
            f"{'Mean:':20} {stats_a.mean_ms:>12.3f} ms {stats_b.mean_ms:>12.3f} ms"
        )
        self._print(
            f"{'Std Dev:':20} {stats_a.std_ms:>12.3f} ms {stats_b.std_ms:>12.3f} ms"
        )
        self._print(
            f"{'Min:':20} {stats_a.min_ms:>12.3f} ms {stats_b.min_ms:>12.3f} ms"
        )
        self._print(
            f"{'Max:':20} {stats_a.max_ms:>12.3f} ms {stats_b.max_ms:>12.3f} ms"
        )
        self._print(
            f"{'P95:':20} {stats_a.p95_ms:>12.3f} ms {stats_b.p95_ms:>12.3f} ms"
        )
        self._print(
            f"{'P99:':20} {stats_a.p99_ms:>12.3f} ms {stats_b.p99_ms:>12.3f} ms"
        )
        self._print_line("-")

        # Calculate speedup
        if stats_a.mean_ms > 0 and stats_b.mean_ms > 0:
            if stats_a.mean_ms > stats_b.mean_ms:
                speedup = (stats_a.mean_ms - stats_b.mean_ms) / stats_a.mean_ms * 100
                self._print(f"Speedup:            B is {speedup:.1f}% faster")
            elif stats_b.mean_ms > stats_a.mean_ms:
                speedup = (stats_b.mean_ms - stats_a.mean_ms) / stats_b.mean_ms * 100
                self._print(f"Speedup:            A is {speedup:.1f}% faster")
            else:
                self._print("Speedup:            A and B are equal")

        self._print("")

    def print_ab_comparison(
        self, passed: bool, max_abs_diff: float, max_rel_diff: float, rtol: float, atol: float
    ) -> None:
        """Print A/B accuracy comparison result.

        Args:
            passed: Whether comparison passed.
            max_abs_diff: Maximum absolute difference.
            max_rel_diff: Maximum relative difference.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
        """
        status = "PASSED" if passed else "FAILED"
        self._print(f"Accuracy Comparison: {status}")
        self._print(f"  (rtol={rtol:.0e}, atol={atol:.0e})")
        if not passed:
            self._print(f"  Max abs diff: {max_abs_diff:.2e}")
            self._print(f"  Max rel diff: {max_rel_diff:.2e}")

    # Reference Validation Methods

    def print_reference_validation(
        self,
        provider_name: str,
        passed: bool,
        max_abs_diff: float,
        max_rel_diff: float,
        rtol: float,
        atol: float,
    ) -> None:
        """Print reference validation result.

        Args:
            provider_name: Name of the reference provider used.
            passed: Whether validation passed.
            max_abs_diff: Maximum absolute difference.
            max_rel_diff: Maximum relative difference.
            rtol: Relative tolerance used.
            atol: Absolute tolerance used.
        """
        status = "PASSED" if passed else "FAILED"
        self._print(f"Reference Validation: {status}")
        self._print(f"  Provider: {provider_name}")
        self._print(f"  (rtol={rtol:.0e}, atol={atol:.0e})")
        if not passed:
            self._print(f"  Max abs diff: {max_abs_diff:.2e}")
            self._print(f"  Max rel diff: {max_rel_diff:.2e}")
