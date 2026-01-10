# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Tests for BenchmarkConfig."""

from pathlib import Path

import pytest

from dnn_benchmarking.config import BenchmarkConfig


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_default_values(self) -> None:
        """Test that defaults are applied correctly."""
        config = BenchmarkConfig(graph_path=Path("/test/graph.json"))

        assert config.warmup_iters == 10
        assert config.benchmark_iters == 100
        assert config.engine_id == 1

    def test_custom_values(self) -> None:
        """Test that custom values are stored correctly."""
        config = BenchmarkConfig(
            graph_path=Path("/test/graph.json"),
            warmup_iters=20,
            benchmark_iters=200,
            engine_id=2,
        )

        assert config.graph_path == Path("/test/graph.json")
        assert config.warmup_iters == 20
        assert config.benchmark_iters == 200
        assert config.engine_id == 2

    def test_string_path_converted_to_path(self) -> None:
        """Test that string path is converted to Path object."""
        config = BenchmarkConfig(graph_path="/test/graph.json")  # type: ignore

        assert isinstance(config.graph_path, Path)
        assert config.graph_path == Path("/test/graph.json")

    def test_negative_warmup_raises(self) -> None:
        """Test that negative warmup_iters raises ValueError."""
        with pytest.raises(ValueError, match="warmup_iters must be non-negative"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), warmup_iters=-1)

    def test_zero_warmup_allowed(self) -> None:
        """Test that zero warmup_iters is allowed."""
        config = BenchmarkConfig(graph_path=Path("/test/graph.json"), warmup_iters=0)
        assert config.warmup_iters == 0

    def test_zero_benchmark_iters_raises(self) -> None:
        """Test that zero benchmark_iters raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_iters must be positive"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), benchmark_iters=0)

    def test_negative_benchmark_iters_raises(self) -> None:
        """Test that negative benchmark_iters raises ValueError."""
        with pytest.raises(ValueError, match="benchmark_iters must be positive"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), benchmark_iters=-1)

    def test_negative_engine_id_raises(self) -> None:
        """Test that negative engine_id raises ValueError."""
        with pytest.raises(ValueError, match="engine_id must be non-negative"):
            BenchmarkConfig(graph_path=Path("/test/graph.json"), engine_id=-1)
