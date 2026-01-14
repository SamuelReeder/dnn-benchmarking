"""A/B testing runner for comparing plugin/engine configurations."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import ABTestConfig, BenchmarkConfig
from ..graph.loader import GraphLoader
from ..reporting.statistics import BenchmarkStats
from .buffer_manager import BufferManager
from .executor import Executor


@dataclass
class ABTestResult:
    """Result of A/B test comparison.

    Attributes:
        stats_a: Benchmark statistics for configuration A.
        stats_b: Benchmark statistics for configuration B.
        init_time_a_ms: Graph initialization time for A in milliseconds.
        init_time_b_ms: Graph initialization time for B in milliseconds.
        passed: Whether outputs match within tolerance.
        max_abs_diff: Maximum absolute difference between outputs.
        max_rel_diff: Maximum relative difference between outputs.
    """

    stats_a: BenchmarkStats
    stats_b: BenchmarkStats
    init_time_a_ms: float
    init_time_b_ms: float
    passed: bool
    max_abs_diff: float
    max_rel_diff: float


class ABRunner:
    """Runs A/B comparison between two plugin/engine configurations.

    This class handles:
    - Setting plugin paths for each configuration
    - Executing the same graph with different engines
    - Comparing outputs using np.allclose
    - Collecting timing statistics for both configurations
    """

    def __init__(
        self,
        graph_json: Dict[str, Any],
        config: BenchmarkConfig,
        ab_config: ABTestConfig,
    ) -> None:
        """Initialize A/B runner.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            config: Benchmark configuration (warmup/iters).
            ab_config: A/B test configuration (paths, engine IDs, tolerances).
        """
        self._graph_json = graph_json
        self._config = config
        self._ab_config = ab_config

    def _set_plugin_path(self, plugin_path: Optional[Path]) -> None:
        """Set plugin path using hipdnn_frontend API.

        Args:
            plugin_path: Path to plugin directory, or None for default.
        """
        import hipdnn_frontend as hipdnn

        if plugin_path is not None:
            # Use ABSOLUTE mode to ensure only this plugin is used
            hipdnn.set_engine_plugin_paths(
                [str(plugin_path)], hipdnn.PluginLoadingMode.ABSOLUTE
            )

    def _run_single(
        self,
        plugin_path: Optional[Path],
        engine_id: int,
        buffer_manager: BufferManager,
    ) -> Tuple[np.ndarray, List[float], float]:
        """Execute graph with specific plugin/engine configuration.

        Args:
            plugin_path: Path to plugin directory, or None for default.
            engine_id: Engine ID to use.
            buffer_manager: Buffer manager with allocated tensors.

        Returns:
            Tuple of (output_data, timings, init_time_ms).
        """
        import hipdnn_frontend as hipdnn

        # Set plugin path before creating Handle
        self._set_plugin_path(plugin_path)

        handle = hipdnn.Handle()
        executor = Executor(json.dumps(self._graph_json), self._config)
        executor.prepare(handle, engine_id=engine_id)
        init_time_ms = executor.init_time_ms

        variant_pack = buffer_manager.create_variant_pack()
        executor.warmup(handle, variant_pack)
        timings = executor.benchmark(handle, variant_pack)

        # Get output data - copy to avoid overwriting
        output_tensors = buffer_manager.get_output_tensors()
        if not output_tensors:
            raise ExecutionError("No output tensors found in graph")

        output_data = buffer_manager.get_output_data(output_tensors[0].uid)
        if output_data is None:
            raise ExecutionError("Failed to retrieve output data")

        return output_data.copy(), timings, init_time_ms

    def run(self, seed: Optional[int] = 42) -> ABTestResult:
        """Run A/B comparison.

        Args:
            seed: Random seed for reproducible input data.

        Returns:
            ABTestResult with statistics and comparison results.
        """
        loader = GraphLoader()
        tensor_infos = loader.extract_tensor_info(self._graph_json)

        with BufferManager(tensor_infos) as buffer_manager:
            buffer_manager.allocate_all()
            buffer_manager.fill_inputs_random(seed=seed)

            # Run configuration A
            buffer_manager.zero_outputs()
            output_a, timings_a, init_a = self._run_single(
                self._ab_config.a_path, self._ab_config.a_id, buffer_manager
            )
            stats_a = BenchmarkStats.from_timings(timings_a)

            # Run configuration B (same inputs)
            buffer_manager.zero_outputs()
            output_b, timings_b, init_b = self._run_single(
                self._ab_config.b_path, self._ab_config.b_id, buffer_manager
            )
            stats_b = BenchmarkStats.from_timings(timings_b)

        # Compare outputs
        passed = np.allclose(
            output_a, output_b, rtol=self._ab_config.rtol, atol=self._ab_config.atol
        )

        # Calculate differences
        abs_diff = np.abs(output_a - output_b)
        max_abs_diff = float(np.max(abs_diff))

        # Handle division by zero for relative difference
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / (np.abs(output_b) + 1e-10)
            max_rel_diff = float(np.max(rel_diff))

        # Check for NaN/Inf in outputs
        if np.any(np.isnan(output_a)) or np.any(np.isinf(output_a)):
            passed = False
            max_abs_diff = float("inf")
            max_rel_diff = float("inf")

        if np.any(np.isnan(output_b)) or np.any(np.isinf(output_b)):
            passed = False
            max_abs_diff = float("inf")
            max_rel_diff = float("inf")

        return ABTestResult(
            stats_a=stats_a,
            stats_b=stats_b,
            init_time_a_ms=init_a,
            init_time_b_ms=init_b,
            passed=passed,
            max_abs_diff=max_abs_diff,
            max_rel_diff=max_rel_diff,
        )
