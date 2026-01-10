# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration tests for CLI."""

import subprocess
import sys
from pathlib import Path

import pytest


class TestCLIIntegration:
    """Integration tests for CLI invocation."""

    @pytest.fixture
    def sample_graph_path(self) -> Path:
        """Get path to sample graph."""
        return Path(__file__).parent.parent.parent / "graphs" / "sample_conv_fwd.json"

    def test_cli_help(self) -> None:
        """Test that --help works."""
        result = subprocess.run(
            [sys.executable, "-m", "dnn_benchmarking", "--help"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode == 0
        assert "dnn-benchmark" in result.stdout or "dnn_benchmarking" in result.stdout
        assert "--graph" in result.stdout
        assert "--warmup" in result.stdout
        assert "--iters" in result.stdout

    def test_cli_missing_required_arg(self) -> None:
        """Test that missing --graph arg gives error."""
        result = subprocess.run(
            [sys.executable, "-m", "dnn_benchmarking"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert "graph" in result.stderr.lower() or "required" in result.stderr.lower()

    def test_cli_nonexistent_graph(self) -> None:
        """Test error handling for nonexistent graph file."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                "/nonexistent/path/graph.json",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        assert result.returncode != 0
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    @pytest.mark.gpu
    def test_cli_full_run(self, sample_graph_path: Path) -> None:
        """Test full CLI run with sample graph (requires GPU)."""
        if not sample_graph_path.exists():
            pytest.skip(f"Sample graph not found: {sample_graph_path}")

        # Check if hipdnn is available
        try:
            import hipdnn_frontend

            hipdnn_frontend.Handle()
        except Exception as e:
            pytest.skip(f"hipdnn_frontend not available or no GPU: {e}")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "dnn_benchmarking",
                "--graph",
                str(sample_graph_path),
                "--warmup",
                "1",
                "--iters",
                "2",
            ],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Check output contains expected sections
        assert "hipDNN Benchmark" in result.stdout
        assert "Execution Statistics" in result.stdout
        assert "Mean" in result.stdout
        assert "Validation" in result.stdout

        # Should succeed
        assert result.returncode == 0, f"CLI failed with: {result.stderr}"


class TestCLIParser:
    """Unit tests for CLI parser."""

    def test_parse_default_values(self) -> None:
        """Test parsing with default values."""
        from dnn_benchmarking.cli.parser import parse_args

        config = parse_args(["--graph", "/test/graph.json"])

        assert config.graph_path == Path("/test/graph.json")
        assert config.warmup_iters == 10
        assert config.benchmark_iters == 100
        assert config.engine_id == 1

    def test_parse_custom_values(self) -> None:
        """Test parsing with custom values."""
        from dnn_benchmarking.cli.parser import parse_args

        config = parse_args(
            [
                "--graph",
                "/test/graph.json",
                "--warmup",
                "20",
                "--iters",
                "200",
                "--engine-id",
                "2",
            ]
        )

        assert config.graph_path == Path("/test/graph.json")
        assert config.warmup_iters == 20
        assert config.benchmark_iters == 200
        assert config.engine_id == 2

    def test_parse_short_options(self) -> None:
        """Test parsing with short option names."""
        from dnn_benchmarking.cli.parser import parse_args

        config = parse_args(["-g", "/test/graph.json", "-w", "5", "-i", "50", "-e", "3"])

        assert config.graph_path == Path("/test/graph.json")
        assert config.warmup_iters == 5
        assert config.benchmark_iters == 50
        assert config.engine_id == 3
