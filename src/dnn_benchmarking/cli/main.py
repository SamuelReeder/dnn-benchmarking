# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Main entry point for dnn-benchmark CLI."""

import json
import sys
from typing import Optional

from ..common.exceptions import ExecutionError, GraphLoadError
from ..config.benchmark_config import ABTestConfig, BenchmarkConfig
from ..execution.ab_runner import ABRunner
from ..execution.buffer_manager import BufferManager
from ..execution.executor import Executor
from ..graph.loader import GraphLoader
from ..reporting.reporter import Reporter
from ..reporting.statistics import BenchmarkStats
from ..validation.validator import Validator
from .parser import create_parser


def run_benchmark(config: BenchmarkConfig, seed: Optional[int] = None) -> int:
    """Run the benchmark workflow.

    Args:
        config: Benchmark configuration.
        seed: Optional random seed for reproducibility.

    Returns:
        Exit code (0 for success, 1 for error).
    """
    reporter = Reporter()

    try:
        # Load and validate graph
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)
        loader.validate(graph_json)

        graph_name = loader.get_graph_name(graph_json)
        tensor_infos = loader.extract_tensor_info(graph_json)

        # Print header
        reporter.print_header(config, graph_name)

        # Import hipdnn after validation to give better error messages
        try:
            import hipdnn_frontend as hipdnn
        except ImportError:
            reporter.print_error(
                "hipdnn_frontend not available. "
                "Install hipDNN Python bindings first."
            )
            return 1

        # Create handle
        handle = hipdnn.Handle()

        # Prepare executor
        graph_json_str = json.dumps(graph_json)
        executor = Executor(graph_json_str, config)
        executor.prepare(handle)

        reporter.print_init_time(executor.init_time_ms)

        # Allocate buffers
        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)
            buffer_manager.zero_outputs()

            variant_pack = buffer_manager.create_variant_pack()

            # Run warmup
            executor.warmup(handle, variant_pack)

            # Run benchmark
            timings = executor.benchmark(handle, variant_pack)

            # Calculate statistics
            stats = BenchmarkStats.from_timings(timings)
            reporter.print_stats(stats)

            # Validation (stubbed)
            validator = Validator()
            passed, message = validator.validate_stub()
            reporter.print_validation(passed, message)

        reporter.print_footer()
        return 0

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except ExecutionError as e:
        reporter.print_error(f"Execution error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


def run_ab_test(
    config: BenchmarkConfig, ab_config: ABTestConfig, seed: Optional[int] = None
) -> int:
    """Run A/B comparison workflow.

    Args:
        config: Benchmark configuration.
        ab_config: A/B test configuration.
        seed: Optional random seed for reproducibility.

    Returns:
        Exit code (0 for success, 1 for error, 2 for comparison failure).
    """
    reporter = Reporter()

    try:
        # Validate plugin paths if specified
        ab_config.validate_paths()

        # Load and validate graph
        loader = GraphLoader()
        graph_json = loader.load_json(config.graph_path)
        loader.validate(graph_json)

        graph_name = loader.get_graph_name(graph_json)

        # Print header
        reporter.print_ab_header(config, ab_config, graph_name)

        # Run A/B comparison
        runner = ABRunner(graph_json, config, ab_config)
        result = runner.run(seed=seed)

        # Print results
        reporter.print_ab_stats(
            result.stats_a,
            result.stats_b,
            result.init_time_a_ms,
            result.init_time_b_ms,
        )

        reporter.print_ab_comparison(
            result.passed,
            result.max_abs_diff,
            result.max_rel_diff,
            ab_config.rtol,
            ab_config.atol,
        )

        reporter.print_footer()

        # Return 0 for pass, 2 for comparison failure
        return 0 if result.passed else 2

    except GraphLoadError as e:
        reporter.print_error(f"Graph load error: {e}")
        return 1

    except ExecutionError as e:
        reporter.print_error(f"Execution error: {e}")
        return 1

    except ValueError as e:
        reporter.print_error(f"Configuration error: {e}")
        return 1

    except Exception as e:
        reporter.print_error(f"Unexpected error: {e}")
        return 1


def main() -> int:
    """CLI entry point.

    Returns:
        Exit code.
    """
    parser = create_parser()
    args = parser.parse_args()

    try:
        config = BenchmarkConfig(
            graph_path=args.graph,
            warmup_iters=args.warmup,
            benchmark_iters=args.iters,
            engine_id=args.engine_id,
        )
    except ValueError as e:
        print(f"Configuration error: {e}", file=sys.stderr)
        return 1

    # Check if A/B testing mode is enabled (either AId or BId specified)
    if args.AId is not None or args.BId is not None:
        # Both AId and BId should be specified for A/B testing
        if args.AId is None or args.BId is None:
            print(
                "A/B testing requires both --AId and --BId to be specified",
                file=sys.stderr,
            )
            return 1

        try:
            ab_config = ABTestConfig(
                a_path=args.APath,
                a_id=args.AId,
                b_path=args.BPath,
                b_id=args.BId,
                rtol=args.rtol,
                atol=args.atol,
            )
        except ValueError as e:
            print(f"A/B configuration error: {e}", file=sys.stderr)
            return 1

        return run_ab_test(config, ab_config, seed=args.seed)

    return run_benchmark(config, seed=args.seed)


if __name__ == "__main__":
    sys.exit(main())
