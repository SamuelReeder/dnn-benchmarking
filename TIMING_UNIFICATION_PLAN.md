# Kernel timing unification plan (PyTorch events for CUDA + ROCm)

## Goals
- Use PyTorch CUDA/ROCm events as the single kernel-timing backend for both NVIDIA and AMD.
- Remove HIP ctypes timing and HIP backend selection.
- Align E2E timing semantics across hipDNN and PyTorch backends: execution-only (includes device sync).
- Keep validation CPU-only even when ROCm PyTorch is installed.

## Assumptions
- hipDNN executes on the default stream (torch events on the current/default stream are valid).
- ROCm nightly PyTorch wheels are required on AMD systems.

## Primary changes (files to edit)

### Timing backend
- `src/dnn_benchmarking/execution/timing.py`
  - Remove HIP ctypes loader and `HipGpuTimer` implementation.
  - Keep a single PyTorch-event timer (rename `CudaGpuTimer` to `TorchGpuTimer` or keep the name but document it works on ROCm).
  - Simplify `create_gpu_timer()` to return only the torch-based timer.
  - Simplify availability checks to `torch.cuda.is_available()` only.
  - Remove `get_available_backends()` references to `hip`.

### hipDNN executor E2E synchronization
- `src/dnn_benchmarking/execution/executor.py`
  - Insert `torch.cuda.synchronize()` inside the timed region after `graph.execute(...)` so E2E reflects actual execution.
  - Continue to record kernel timing via torch events around the same execute call.
  - Guard the sync behind torch availability (ImportError => no sync, or raise if GPU timing requested).

### PyTorch executor E2E semantics
- `src/dnn_benchmarking/execution/pytorch_executor.py`
  - Keep the existing sync in the timed region (already execution-only).
  - Switch to the unified torch-based timer class name if it changes in `timing.py`.

### CLI and reporting
- `src/dnn_benchmarking/cli/parser.py`
  - Change `--gpu-backend` choices to `auto | torch | none`.
  - Update help text to reflect unified torch timing for CUDA/ROCm.
- `src/dnn_benchmarking/cli/main.py`
  - Update typing and defaults to match `torch/auto/none`.
- `src/dnn_benchmarking/reporting/reporter.py`
  - Update kernel-timing availability message to reference PyTorch GPU availability.

### Requirements / install guidance
- `requirements-rocm.txt`
  - Replace CPU-only torch pin with ROCm nightly wheel guidance (torch/torchaudio/torchvision) from the gfx90X-dcgpu index URL.
  - Note that ROCm torch is required for kernel timing on AMD.
- `requirements-cuda.txt`
  - No functional change required, but ensure documentation reflects that torch events are used for timing.
- `README.md`
  - Update install instructions for ROCm nightly wheels and clarify that torch timing is used on both vendors.

### ROCm nightlies (implementation notes)
- Python 3.12 wheels were only available on the gfx90X-dcgpu **staging** index at the time of implementation.
  - Install command used:
    - `python -m pip install --index-url https://rocm.nightlies.amd.com/v2-staging/gfx90X-dcgpu/ --pre --upgrade --force-reinstall torch`
  - `torchaudio`/`torchvision` are optional for benchmarking.
- To avoid LLVM symbol mismatches (e.g., `libamd_comgr.so.3: undefined symbol ... LLVM_22.0`) when mixing hipDNN/MIOpen with ROCm torch:
  - Prefer the venv ROCm SDK libraries first in `LD_LIBRARY_PATH`:
    - `$VENV_SITE/_rocm_sdk_core/lib`
    - `$VENV_SITE/_rocm_sdk_libraries_gfx90X_dcgpu/lib`
    - `$VENV_SITE/triton/backends/amd/lib`

## Items to remove or simplify
- `src/dnn_benchmarking/execution/timing.py`
  - Remove `ctypes`, `_get_hip_lib`, `_is_hip_available`, `HipGpuTimer`, `GpuTimer = HipGpuTimer` alias.
  - Remove backend selection branching for `hip`.
- CLI backend selection
  - Remove `hip` and `cuda` options; keep only `torch`, `auto`, `none`.
- Reporter messaging
  - Replace HIP-runtime-specific messaging with PyTorch GPU availability checks.

## E2E timing model (final behavior)
- hipDNN backend:
  - E2E: wall-clock timing of `graph.execute(...)` + `torch.cuda.synchronize()` inside the timed block.
  - Kernel: torch events around `graph.execute(...)`.
- PyTorch backend:
  - E2E: wall-clock timing of `execute_graph(...)` + `torch.cuda.synchronize()` inside the timed block.
  - Kernel: torch events around `execute_graph(...)`.

## Tests to add or update

### Unit tests (CPU-friendly, mock-based)
- `tests/unit/test_timing_backend.py`
  - Verify `create_gpu_timer()` returns the torch-based timer when torch is importable and `torch.cuda.is_available()` is True (mocked).
  - Verify `create_gpu_timer()` raises when torch is missing or `torch.cuda.is_available()` is False.
  - Verify `BenchmarkMetadata.gpu_backend` is set to `torch` when timing is enabled.
- `tests/unit/test_executor_e2e_sync.py`
  - Mock torch and assert `torch.cuda.synchronize()` is called inside the timed section for hipDNN execution.
  - Ensure E2E timings are recorded even when GPU timing is disabled.

### GPU integration tests (gated by marker)
- `tests/integration/test_gpu_timing_torch.py` (marker: `gpu`)
  - Run a small graph and assert kernel timings are present and positive.
  - Assert E2E timings are >= kernel timings (allowing small tolerance).

### NVIDIA-only tests
- `tests/integration/test_gpu_timing_cuda.py` (marker: `gpu`, `nvidia`)
  - Run with CUDA wheel; verify kernel timing works.
  - Optional: compare timing stability with multiple iterations.

### AMD-only tests
- `tests/integration/test_gpu_timing_rocm.py` (marker: `gpu`, `amd`)
  - Run with ROCm nightly wheel; verify kernel timing works for hipDNN execution using torch events.
  - Validate that CPU-only reference provider still runs (no GPU tensors used in validation path).

## Open questions (to resolve before implementation)
- If torch is required on AMD, do we want to fail fast when torch is missing, or allow E2E-only without kernel timing?
- Do we want `gpu_backend` metadata to report `torch` explicitly, or keep `cuda` for compatibility?
- Is the hipDNN handle guaranteed to use the default stream, or should we explicitly control stream selection later?
