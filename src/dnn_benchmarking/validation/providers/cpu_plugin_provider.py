"""CPU reference plugin provider (STUBBED).

This provider will use the hipDNN CPU reference plugin when available.
Currently stubbed as CPU reference is not yet available in Python bindings.
"""

from typing import Any, Dict

import numpy as np

from ..reference_provider import (
    ReferenceOutput,
    ReferenceProvider,
    ReferenceProviderRegistry,
)


@ReferenceProviderRegistry.register("cpu_plugin")
class CPUPluginReferenceProvider(ReferenceProvider):
    """Reference provider using hipDNN CPU reference plugin (STUBBED).

    When available, this provider will:
    1. Use a special CPU reference engine ID
    2. Execute the same graph JSON through hipDNN
    3. Return outputs computed on CPU

    This allows validation using the same execution path as GPU,
    but with deterministic CPU-based reference computation.
    """

    # Placeholder - actual engine ID TBD when CPU plugin is available
    CPU_ENGINE_ID = 0

    @property
    def name(self) -> str:
        """Provider name."""
        return "cpu_plugin"

    def is_available(self) -> bool:
        """Check if CPU reference plugin is available.

        Returns:
            False - CPU reference plugin not yet available in Python.
        """
        # STUB: CPU reference plugin not yet available in Python bindings
        # When available, this would check:
        # 1. hipdnn_frontend is installed
        # 2. CPU reference plugin exists
        # 3. Engine ID is valid
        return False

    def compute_reference(
        self,
        graph_json: Dict[str, Any],
        input_data: Dict[int, np.ndarray],
    ) -> Dict[int, ReferenceOutput]:
        """Compute reference using CPU plugin.

        This would use the same execution path as GPU but with CPU engine.
        Similar to how ABRunner runs two configurations.

        Args:
            graph_json: The graph as a parsed JSON dictionary.
            input_data: Mapping of tensor UID to input numpy arrays.

        Returns:
            Mapping of output tensor UID to ReferenceOutput.

        Raises:
            NotImplementedError: Always - CPU plugin not available.
        """
        if not self.is_available():
            raise NotImplementedError(
                "CPU reference plugin not available in Python bindings. "
                "Use 'pytorch' provider instead, or wait for CPU plugin support."
            )

        # Future implementation would:
        # 1. Create buffers from input_data
        # 2. Set engine to CPU_ENGINE_ID
        # 3. Execute graph through hipdnn
        # 4. Return outputs as ReferenceOutput objects
        #
        # The implementation would be similar to ABRunner._run_single()
        # but with CPU-specific configuration.
        raise NotImplementedError("CPU plugin execution not implemented")
