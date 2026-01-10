# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI argument parsing for dnn-benchmarking."""

import argparse
from pathlib import Path

from ..config.benchmark_config import BenchmarkConfig


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for dnn-benchmark CLI.

    Returns:
        Configured ArgumentParser.
    """
    parser = argparse.ArgumentParser(
        prog="dnn-benchmark",
        description="Benchmarking and validation tool for hipDNN graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  dnn-benchmark --graph ./graphs/conv1_fwd.json
  dnn-benchmark --graph ./graphs/conv1_fwd.json --warmup 20 --iters 200
  dnn-benchmark -g ./graphs/conv1_fwd.json -e 1
        """,
    )

    parser.add_argument(
        "--graph",
        "-g",
        type=Path,
        required=True,
        metavar="PATH",
        help="Path to JSON-serialized hipDNN graph file",
    )

    parser.add_argument(
        "--warmup",
        "-w",
        type=int,
        default=10,
        metavar="N",
        help="Number of warmup iterations (default: 10)",
    )

    parser.add_argument(
        "--iters",
        "-i",
        type=int,
        default=100,
        metavar="N",
        help="Number of benchmark iterations (default: 100)",
    )

    parser.add_argument(
        "--engine-id",
        "-e",
        type=int,
        default=1,
        metavar="ID",
        help="Engine ID to use (default: 1 for MIOpen)",
    )

    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=None,
        metavar="SEED",
        help="Random seed for reproducible input data (default: None)",
    )

    return parser


def parse_args(args=None) -> BenchmarkConfig:
    """Parse command line arguments and return BenchmarkConfig.

    Args:
        args: Command line arguments (default: sys.argv).

    Returns:
        BenchmarkConfig with parsed values.

    Raises:
        SystemExit: If arguments are invalid.
    """
    parser = create_parser()
    parsed = parser.parse_args(args)

    return BenchmarkConfig(
        graph_path=parsed.graph,
        warmup_iters=parsed.warmup,
        benchmark_iters=parsed.iters,
        engine_id=parsed.engine_id,
    )
