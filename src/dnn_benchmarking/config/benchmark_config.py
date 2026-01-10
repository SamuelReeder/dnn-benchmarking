# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Benchmark configuration dataclass."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        graph_path: Path to the JSON-serialized hipDNN graph file.
        warmup_iters: Number of warmup iterations before benchmarking.
        benchmark_iters: Number of benchmark iterations for timing.
        engine_id: Engine ID to use (1 = MIOpen).
    """

    graph_path: Path
    warmup_iters: int = 10
    benchmark_iters: int = 100
    engine_id: int = 1

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if isinstance(self.graph_path, str):
            self.graph_path = Path(self.graph_path)

        if self.warmup_iters < 0:
            raise ValueError("warmup_iters must be non-negative")

        if self.benchmark_iters <= 0:
            raise ValueError("benchmark_iters must be positive")

        if self.engine_id < 0:
            raise ValueError("engine_id must be non-negative")
