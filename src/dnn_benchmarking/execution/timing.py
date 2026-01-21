"""Timing utilities for benchmark execution."""

import ctypes
import time
from abc import ABC, abstractmethod
from ctypes import POINTER, byref, c_float, c_int, c_void_p
from types import TracebackType
from typing import List, Literal, Optional, Type

# HIP library handle (lazy loaded)
_hip_lib: Optional[ctypes.CDLL] = None


def _get_hip_lib() -> Optional[ctypes.CDLL]:
    """Get the HIP library handle, loading it if necessary.

    Returns:
        The HIP library handle, or None if not available.
    """
    global _hip_lib
    if _hip_lib is not None:
        return _hip_lib

    # Try common HIP library locations
    hip_paths = [
        "/opt/rocm/lib/libamdhip64.so",
        "libamdhip64.so",
        "/opt/rocm/lib/libamdhip64.so.6",
        "/opt/rocm/lib/libamdhip64.so.7",
    ]

    for path in hip_paths:
        try:
            _hip_lib = ctypes.CDLL(path)
            # Set up function signatures
            _hip_lib.hipEventCreate.argtypes = [POINTER(c_void_p)]
            _hip_lib.hipEventCreate.restype = c_int

            _hip_lib.hipEventRecord.argtypes = [c_void_p, c_void_p]
            _hip_lib.hipEventRecord.restype = c_int

            _hip_lib.hipEventSynchronize.argtypes = [c_void_p]
            _hip_lib.hipEventSynchronize.restype = c_int

            _hip_lib.hipEventElapsedTime.argtypes = [POINTER(c_float), c_void_p, c_void_p]
            _hip_lib.hipEventElapsedTime.restype = c_int

            _hip_lib.hipEventDestroy.argtypes = [c_void_p]
            _hip_lib.hipEventDestroy.restype = c_int

            return _hip_lib
        except OSError:
            continue

    return None


def _is_hip_available() -> bool:
    """Check if HIP runtime is available.

    Returns:
        True if HIP library can be loaded, False otherwise.
    """
    return _get_hip_lib() is not None


def _is_cuda_available() -> bool:
    """Check if PyTorch CUDA is available.

    Returns:
        True if torch.cuda.is_available() returns True, False otherwise.
    """
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def get_available_backends() -> List[str]:
    """Return list of available GPU timer backends.

    Returns:
        List of backend names (e.g., ["hip"], ["cuda"], ["hip", "cuda"], or []).
    """
    backends = []
    if _is_hip_available():
        backends.append("hip")
    if _is_cuda_available():
        backends.append("cuda")
    return backends


def is_gpu_timing_available() -> bool:
    """Check if any GPU timing backend is available.

    Returns:
        True if HIP or CUDA timing is available, False otherwise.
    """
    return _is_hip_available() or _is_cuda_available()


class GpuTimerInterface(ABC):
    """Abstract interface for GPU kernel timing.

    Supports context manager protocol for convenient timing blocks,
    as well as explicit start/stop control for fine-grained timing.
    """

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the backend name (e.g., 'hip', 'cuda')."""
        ...

    @abstractmethod
    def start(self) -> None:
        """Record the start timestamp on the GPU stream."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Record the stop timestamp on the GPU stream."""
        ...

    @abstractmethod
    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed time in milliseconds.

        Must be called after stop(). Blocks until GPU operations complete.
        """
        ...

    def __enter__(self) -> "GpuTimerInterface":
        """Context manager entry - records start."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Context manager exit - records stop."""
        self.stop()


class HipGpuTimer(GpuTimerInterface):
    """GPU kernel timing using HIP events via ctypes.

    Directly calls HIP runtime APIs for event-based timing, bypassing
    any PyTorch/ROCm version conflicts.

    Example:
        timer = HipGpuTimer()
        timer.start()
        # GPU kernel execution
        timer.stop()
        elapsed = timer.elapsed_ms()
    """

    @property
    def backend_name(self) -> str:
        """Return 'hip' as the backend name."""
        return "hip"

    def __init__(self) -> None:
        """Initialize GPU timer with HIP events.

        Raises:
            RuntimeError: If HIP library is not available.
        """
        self._hip = _get_hip_lib()
        if self._hip is None:
            raise RuntimeError("HIP library not available for GPU timing")

        self._start_event = c_void_p()
        self._stop_event = c_void_p()

        err = self._hip.hipEventCreate(byref(self._start_event))
        if err != 0:
            raise RuntimeError(f"Failed to create HIP start event: error {err}")

        err = self._hip.hipEventCreate(byref(self._stop_event))
        if err != 0:
            self._hip.hipEventDestroy(self._start_event)
            raise RuntimeError(f"Failed to create HIP stop event: error {err}")

    def start(self) -> None:
        """Record the start event on the default stream."""
        err = self._hip.hipEventRecord(self._start_event, None)
        if err != 0:
            raise RuntimeError(f"Failed to record HIP start event: error {err}")

    def stop(self) -> None:
        """Record the stop event on the default stream."""
        err = self._hip.hipEventRecord(self._stop_event, None)
        if err != 0:
            raise RuntimeError(f"Failed to record HIP stop event: error {err}")

    def elapsed_ms(self) -> float:
        """Synchronize on the stop event and return elapsed time.

        Returns:
            Elapsed time in milliseconds between start and stop events.
        """
        err = self._hip.hipEventSynchronize(self._stop_event)
        if err != 0:
            raise RuntimeError(f"Failed to synchronize HIP stop event: error {err}")

        elapsed = c_float()
        err = self._hip.hipEventElapsedTime(byref(elapsed), self._start_event, self._stop_event)
        if err != 0:
            raise RuntimeError(f"Failed to get HIP elapsed time: error {err}")

        return float(elapsed.value)

    # Backward compatibility aliases
    def record_start(self) -> None:
        """Alias for start() for backward compatibility."""
        self.start()

    def record_stop(self) -> None:
        """Alias for stop() for backward compatibility."""
        self.stop()

    def synchronize_and_get_elapsed(self) -> float:
        """Alias for elapsed_ms() for backward compatibility."""
        return self.elapsed_ms()

    def __del__(self) -> None:
        """Clean up HIP events."""
        if hasattr(self, "_hip") and self._hip is not None:
            if hasattr(self, "_start_event"):
                self._hip.hipEventDestroy(self._start_event)
            if hasattr(self, "_stop_event"):
                self._hip.hipEventDestroy(self._stop_event)


class CudaGpuTimer(GpuTimerInterface):
    """GPU kernel timing using PyTorch CUDA events.

    Uses torch.cuda.Event for timing on NVIDIA GPUs.

    Example:
        timer = CudaGpuTimer()
        timer.start()
        # GPU kernel execution
        timer.stop()
        elapsed = timer.elapsed_ms()
    """

    @property
    def backend_name(self) -> str:
        """Return 'cuda' as the backend name."""
        return "cuda"

    def __init__(self) -> None:
        """Initialize CUDA timer with PyTorch events.

        Raises:
            RuntimeError: If PyTorch CUDA is not available.
        """
        if not _is_cuda_available():
            raise RuntimeError("PyTorch CUDA not available for GPU timing")

        import torch

        self._start_event = torch.cuda.Event(enable_timing=True)
        self._stop_event = torch.cuda.Event(enable_timing=True)

    def start(self) -> None:
        """Record the start event on the current CUDA stream."""
        self._start_event.record()

    def stop(self) -> None:
        """Record the stop event on the current CUDA stream."""
        self._stop_event.record()

    def elapsed_ms(self) -> float:
        """Synchronize and return elapsed time in milliseconds.

        Returns:
            Elapsed time in milliseconds between start and stop events.
        """
        self._stop_event.synchronize()
        return self._start_event.elapsed_time(self._stop_event)


def create_gpu_timer(
    backend: Optional[Literal["hip", "cuda", "auto"]] = "auto",
) -> GpuTimerInterface:
    """Create a GPU timer for the specified or detected backend.

    Args:
        backend: Timer backend to use:
            - "hip": Force HIP backend (AMD GPUs)
            - "cuda": Force CUDA/PyTorch backend (NVIDIA GPUs)
            - "auto": Auto-detect (prefers HIP if both available)

    Returns:
        GpuTimerInterface implementation.

    Raises:
        RuntimeError: If requested backend is not available.
        ValueError: If invalid backend is specified.
    """
    if backend == "auto" or backend is None:
        if _is_hip_available():
            return HipGpuTimer()
        elif _is_cuda_available():
            return CudaGpuTimer()
        else:
            raise RuntimeError(
                "No GPU timing backend available. "
                "Requires either HIP runtime or PyTorch with CUDA."
            )

    elif backend == "hip":
        if not _is_hip_available():
            raise RuntimeError("HIP runtime not available")
        return HipGpuTimer()

    elif backend == "cuda":
        if not _is_cuda_available():
            raise RuntimeError("PyTorch CUDA not available")
        return CudaGpuTimer()

    else:
        raise ValueError(f"Unknown backend: {backend}")


# Backward compatibility alias
GpuTimer = HipGpuTimer


class Timer:
    """Context manager for measuring wall-clock execution time.

    Uses time.perf_counter() for high-resolution timing.

    Example:
        with Timer() as t:
            # code to time
            pass
        print(f"Elapsed: {t.elapsed_ms:.2f} ms")
    """

    def __init__(self) -> None:
        """Initialize timer with zero elapsed time."""
        self._start: float = 0.0
        self._end: float = 0.0

    def __enter__(self) -> "Timer":
        """Start timing."""
        self._start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        """Stop timing."""
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        """Get elapsed time in milliseconds."""
        return (self._end - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        """Get elapsed time in seconds."""
        return self._end - self._start
