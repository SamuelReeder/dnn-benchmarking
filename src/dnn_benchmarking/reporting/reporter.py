# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Console output formatting for benchmark results."""

import sys
from pathlib import Path
from typing import TextIO

from ..config.benchmark_config import BenchmarkConfig
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
