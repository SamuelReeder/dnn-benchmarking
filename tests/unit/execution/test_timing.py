"""Tests for Timer and GPU timing utilities."""

import time

import pytest

from dnn_benchmarking.execution.timing import (
    CudaGpuTimer,
    GpuTimerInterface,
    HipGpuTimer,
    Timer,
    create_gpu_timer,
    get_available_backends,
    is_gpu_timing_available,
)


class TestTimer:
    """Tests for Timer context manager."""

    def test_timer_measures_elapsed_time(self) -> None:
        """Test that timer measures elapsed time correctly."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms sleep

        # Should be at least 10ms, allow for some variance
        assert t.elapsed_ms >= 10.0
        assert t.elapsed_ms < 100.0  # Sanity check

    def test_timer_elapsed_seconds(self) -> None:
        """Test elapsed_s property."""
        with Timer() as t:
            time.sleep(0.01)  # 10ms sleep

        assert t.elapsed_s >= 0.01
        assert t.elapsed_s < 0.1

    def test_timer_zero_when_not_started(self) -> None:
        """Test timer returns zero before use."""
        t = Timer()
        assert t.elapsed_ms == 0.0
        assert t.elapsed_s == 0.0

    def test_timer_reusable(self) -> None:
        """Test that timer can be reused."""
        t = Timer()

        with t:
            time.sleep(0.005)  # 5ms
        first_elapsed = t.elapsed_ms

        with t:
            time.sleep(0.01)  # 10ms
        second_elapsed = t.elapsed_ms

        # Second measurement should be longer
        assert second_elapsed > first_elapsed

    def test_timer_with_exception(self) -> None:
        """Test that timer still records time when exception occurs."""
        t = Timer()

        with pytest.raises(ValueError):
            with t:
                time.sleep(0.005)
                raise ValueError("test error")

        # Time should still be recorded
        assert t.elapsed_ms >= 5.0


class TestGpuTimerInterface:
    """Tests for the unified GPU timer interface."""

    def test_interface_is_abstract(self) -> None:
        """Verify GpuTimerInterface cannot be instantiated directly."""
        with pytest.raises(TypeError):
            GpuTimerInterface()  # type: ignore

    def test_interface_defines_required_methods(self) -> None:
        """Verify interface defines required abstract methods."""
        # Check that abstract methods are defined
        assert hasattr(GpuTimerInterface, "start")
        assert hasattr(GpuTimerInterface, "stop")
        assert hasattr(GpuTimerInterface, "elapsed_ms")
        assert hasattr(GpuTimerInterface, "backend_name")

    def test_interface_has_context_manager_protocol(self) -> None:
        """Verify interface supports context manager."""
        assert hasattr(GpuTimerInterface, "__enter__")
        assert hasattr(GpuTimerInterface, "__exit__")


class TestBackendDetection:
    """Tests for backend availability detection."""

    def test_get_available_backends_returns_list(self) -> None:
        """Test that get_available_backends returns a list."""
        backends = get_available_backends()
        assert isinstance(backends, list)

    def test_get_available_backends_only_valid_values(self) -> None:
        """Test that only valid backend names are returned."""
        backends = get_available_backends()
        for backend in backends:
            assert backend in ("hip", "cuda")

    def test_is_gpu_timing_available_matches_backends(self) -> None:
        """Test consistency between availability functions."""
        backends = get_available_backends()
        assert is_gpu_timing_available() == (len(backends) > 0)


class TestFactoryFunction:
    """Tests for create_gpu_timer factory."""

    def test_invalid_backend_raises_error(self) -> None:
        """Test that invalid backend name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_gpu_timer("invalid")  # type: ignore

    def test_hip_backend_unavailable_raises_error(self) -> None:
        """Test that requesting unavailable hip backend raises RuntimeError."""
        if "hip" in get_available_backends():
            pytest.skip("HIP backend is available")
        with pytest.raises(RuntimeError, match="HIP runtime not available"):
            create_gpu_timer("hip")

    def test_cuda_backend_unavailable_raises_error(self) -> None:
        """Test that requesting unavailable cuda backend raises RuntimeError."""
        if "cuda" in get_available_backends():
            pytest.skip("CUDA backend is available")
        with pytest.raises(RuntimeError, match="PyTorch CUDA not available"):
            create_gpu_timer("cuda")

    def test_auto_no_backend_raises_error(self) -> None:
        """Test that auto with no backends raises RuntimeError."""
        if is_gpu_timing_available():
            pytest.skip("GPU backend is available")
        with pytest.raises(RuntimeError, match="No GPU timing backend available"):
            create_gpu_timer("auto")

    @pytest.mark.gpu
    def test_auto_creates_timer_when_available(self) -> None:
        """Test auto-detection creates a working timer."""
        if not is_gpu_timing_available():
            pytest.skip("No GPU backend available")
        timer = create_gpu_timer("auto")
        assert isinstance(timer, GpuTimerInterface)
        assert timer.backend_name in ("hip", "cuda")

    @pytest.mark.gpu
    def test_hip_timer_implements_interface(self) -> None:
        """Test that HipGpuTimer implements the interface."""
        if "hip" not in get_available_backends():
            pytest.skip("HIP backend not available")
        timer = create_gpu_timer("hip")
        assert isinstance(timer, GpuTimerInterface)
        assert isinstance(timer, HipGpuTimer)
        assert timer.backend_name == "hip"

    @pytest.mark.gpu
    def test_cuda_timer_implements_interface(self) -> None:
        """Test that CudaGpuTimer implements the interface."""
        if "cuda" not in get_available_backends():
            pytest.skip("CUDA backend not available")
        timer = create_gpu_timer("cuda")
        assert isinstance(timer, GpuTimerInterface)
        assert isinstance(timer, CudaGpuTimer)
        assert timer.backend_name == "cuda"


class TestHipGpuTimerBackwardCompat:
    """Tests for backward compatibility aliases."""

    @pytest.mark.gpu
    def test_record_start_alias(self) -> None:
        """Test record_start alias for start."""
        if "hip" not in get_available_backends():
            pytest.skip("HIP backend not available")
        timer = HipGpuTimer()
        # Should not raise
        timer.record_start()
        timer.record_stop()
        _ = timer.synchronize_and_get_elapsed()

    @pytest.mark.gpu
    def test_gputimer_alias(self) -> None:
        """Test that GpuTimer is alias for HipGpuTimer."""
        from dnn_benchmarking.execution.timing import GpuTimer

        assert GpuTimer is HipGpuTimer
