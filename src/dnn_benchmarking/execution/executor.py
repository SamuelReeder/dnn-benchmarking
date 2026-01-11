# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Graph execution with timing for benchmarks."""

from typing import Any, Dict, List, Optional, Union

from ..common.exceptions import ExecutionError
from ..config.benchmark_config import BenchmarkConfig
from .timing import Timer


class Executor:
    """Executes hipDNN graphs with warmup and timed benchmark loops.

    This class handles:
    - Loading graph from JSON into hipdnn
    - Setting engine preferences
    - Building the operation graph
    - Running warmup iterations
    - Running timed benchmark iterations
    """

    def __init__(self, graph_json_str: str, config: BenchmarkConfig) -> None:
        """Initialize executor with graph JSON and configuration.

        Args:
            graph_json_str: The graph as a JSON string.
            config: Benchmark configuration.
        """
        self._graph_json_str = graph_json_str
        self._config = config
        self._graph: Any = None
        self._workspace: Any = None
        self._workspace_ptr: int = 0
        self._init_time_ms: float = 0.0

    def prepare(self, handle: Any, engine_id: Optional[int] = None) -> None:
        """Build the operation graph and prepare for execution.

        Args:
            handle: hipdnn.Handle instance.
            engine_id: Optional engine ID to use. If specified, overrides
                       any engine ID in the graph JSON.

        Raises:
            ExecutionError: If graph building fails.
        """
        try:
            import hipdnn_frontend as hipdnn
        except ImportError as e:
            raise ExecutionError(
                "hipdnn_frontend not available. Install hipDNN Python bindings."
            ) from e

        with Timer() as t:
            # Create and configure graph
            self._graph = hipdnn.Graph()
            self._graph.set_io_data_type(hipdnn.DataType.FLOAT)
            self._graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
            self._graph.set_compute_data_type(hipdnn.DataType.FLOAT)

            # Deserialize from JSON
            result = self._graph.from_json(self._graph_json_str)
            if result.is_bad():
                raise ExecutionError(f"Failed to deserialize graph: {result.get_message()}")

            # Set engine preference if specified
            if engine_id is not None:
                self._graph.set_preferred_engine_id_ext(engine_id)

            # Validate
            result = self._graph.validate()
            if result.is_bad():
                raise ExecutionError(f"Graph validation failed: {result.get_message()}")

            # Build operation graph
            result = self._graph.build_operation_graph(handle)
            if result.is_bad():
                raise ExecutionError(
                    f"Failed to build operation graph: {result.get_message()}"
                )

            # Create execution plans
            result = self._graph.create_execution_plans()
            if result.is_bad():
                raise ExecutionError(
                    f"Failed to create execution plans: {result.get_message()}"
                )

            # Check support
            result = self._graph.check_support()
            if result.is_bad():
                raise ExecutionError(
                    f"Backend support check failed: {result.get_message()}"
                )

            # Build plans
            result = self._graph.build_plans()
            if result.is_bad():
                raise ExecutionError(f"Failed to build plans: {result.get_message()}")

            # Allocate workspace
            workspace_size = self._graph.get_workspace_size()
            if workspace_size > 0:
                self._workspace = hipdnn.DeviceBuffer(workspace_size)
                self._workspace_ptr = self._workspace.ptr()

        self._init_time_ms = t.elapsed_ms

    def warmup(self, handle: Any, variant_pack: Dict[int, int]) -> None:
        """Run warmup iterations (timing discarded).

        Args:
            handle: hipdnn.Handle instance.
            variant_pack: Mapping of tensor UIDs to device pointers.

        Raises:
            ExecutionError: If graph not prepared or execution fails.
        """
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")

        for _ in range(self._config.warmup_iters):
            result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
            if result.is_bad():
                raise ExecutionError(f"Warmup execution failed: {result.get_message()}")

    def benchmark(self, handle: Any, variant_pack: Dict[int, int]) -> List[float]:
        """Run benchmark iterations and collect timing.

        Args:
            handle: hipdnn.Handle instance.
            variant_pack: Mapping of tensor UIDs to device pointers.

        Returns:
            List of execution times in milliseconds.

        Raises:
            ExecutionError: If graph not prepared or execution fails.
        """
        if self._graph is None:
            raise ExecutionError("Graph not prepared. Call prepare() first.")

        timings = []

        for _ in range(self._config.benchmark_iters):
            with Timer() as t:
                result = self._graph.execute(handle, variant_pack, self._workspace_ptr)
                if result.is_bad():
                    raise ExecutionError(
                        f"Benchmark execution failed: {result.get_message()}"
                    )

            timings.append(t.elapsed_ms)

        return timings

    @property
    def init_time_ms(self) -> float:
        """Get graph initialization time in milliseconds."""
        return self._init_time_ms

    @property
    def graph(self) -> Any:
        """Get the underlying hipdnn graph object."""
        return self._graph
