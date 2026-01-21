"""Execution module for dnn-benchmarking."""

from .ab_runner import ABRunner, ABTestResult
from .buffer_manager import BufferManager
from .executor import Executor
from .pytorch_buffer_manager import PyTorchCudaBufferManager
from .pytorch_executor import PyTorchCudaExecutor, PyTorchExecutionError
from .timing import (
    CudaGpuTimer,
    GpuTimer,
    GpuTimerInterface,
    HipGpuTimer,
    Timer,
    create_gpu_timer,
    get_available_backends,
    is_gpu_timing_available,
)

__all__ = [
    "ABRunner",
    "ABTestResult",
    "BufferManager",
    "CudaGpuTimer",
    "Executor",
    "GpuTimer",
    "GpuTimerInterface",
    "HipGpuTimer",
    "PyTorchCudaBufferManager",
    "PyTorchCudaExecutor",
    "PyTorchExecutionError",
    "Timer",
    "create_gpu_timer",
    "get_available_backends",
    "is_gpu_timing_available",
]
