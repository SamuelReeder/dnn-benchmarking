"""Reference provider implementations.

Import providers here to register them with the ReferenceProviderRegistry.
"""

from .cpu_plugin_provider import CPUPluginReferenceProvider
from .pytorch_provider import PyTorchReferenceProvider

__all__ = ["CPUPluginReferenceProvider", "PyTorchReferenceProvider"]
