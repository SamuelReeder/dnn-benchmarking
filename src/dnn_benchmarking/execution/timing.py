"""Timing utilities for benchmark execution."""

import ctypes
import time
from ctypes import POINTER, byref, c_float, c_int, c_void_p
from types import TracebackType
from typing import Any, Optional, Type

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


def is_gpu_timing_available() -> bool:
    """Check if GPU timing is available (requires HIP runtime).

    Returns:
        True if HIP library is available, False otherwise.
    """
    return _get_hip_lib() is not None


class GpuTimer:
    """GPU kernel timing using HIP events via ctypes.

    Directly calls HIP runtime APIs for event-based timing, bypassing
    any PyTorch/ROCm version conflicts.

    Example:
        timer = GpuTimer()
        timer.record_start()
        # GPU kernel execution
        timer.record_stop()
        elapsed_ms = timer.synchronize_and_get_elapsed()
    """

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

    def record_start(self) -> None:
        """Record the start event on the default stream."""
        err = self._hip.hipEventRecord(self._start_event, None)
        if err != 0:
            raise RuntimeError(f"Failed to record HIP start event: error {err}")

    def record_stop(self) -> None:
        """Record the stop event on the default stream."""
        err = self._hip.hipEventRecord(self._stop_event, None)
        if err != 0:
            raise RuntimeError(f"Failed to record HIP stop event: error {err}")

    def synchronize_and_get_elapsed(self) -> float:
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

    def __del__(self) -> None:
        """Clean up HIP events."""
        if hasattr(self, "_hip") and self._hip is not None:
            if hasattr(self, "_start_event"):
                self._hip.hipEventDestroy(self._start_event)
            if hasattr(self, "_stop_event"):
                self._hip.hipEventDestroy(self._stop_event)


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
