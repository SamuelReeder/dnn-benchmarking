"""Execution module for dnn-benchmarking."""

from .ab_runner import ABRunner, ABTestResult
from .buffer_manager import BufferManager
from .executor import Executor
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
    "Timer",
    "create_gpu_timer",
    "get_available_backends",
    "is_gpu_timing_available",
]
