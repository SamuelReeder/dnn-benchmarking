# dnn-benchmarking

Benchmarking and validation tool for hipDNN graphs.

## Overview

This tool loads serialized hipDNN graphs, executes them via the MIOpen plugin, captures performance metrics, and validates correctness against a reference baseline.

## Requirements

- Python 3.8+
- numpy
- hipdnn_frontend (installed hipDNN Python bindings)
- AMD GPU with ROCm + MIOpen plugin

## Installation

### Using Virtual Environment (Recommended)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies and package
pip install -r requirements.txt
pip install -e .

# Install hipDNN Python bindings (from your hipDNN build)
cd /path/to/hipdnn/python && pip install -e . && cd -
```

For development (includes pytest):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements-dev.txt
pip install -e .
```

To deactivate the virtual environment later:
```bash
deactivate
```

### Direct Installation (No venv)

```bash
pip install -e .  # Basic installation
pip install -e ".[dev]"  # With development tools
```

**Note**: hipDNN Python bindings (`hipdnn_frontend`) must be installed separately.

## Usage

### Basic Benchmarking

```bash
# Run benchmark on a serialized graph
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --warmup 10 --iters 100

# With custom engine ID
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --engine-id 1

# With reproducible random seed
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --seed 42
```

### A/B Testing

Compare two different plugin/engine configurations and validate accuracy:

```bash
# Compare two different engines on the default plugin
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --AId 1 --BId 2

# Compare two different plugins with specific engine IDs
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json \
  --APath /path/to/pluginA --AId 1 \
  --BPath /path/to/pluginB --BId 2

# With custom tolerance for accuracy comparison
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json \
  --AId 1 --BId 2 --rtol 1e-3 --atol 1e-6
```

### CLI Options

#### Basic Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graph`, `-g` | Path to JSON-serialized hipDNN graph file | Required |
| `--warmup`, `-w` | Number of warmup iterations | 10 |
| `--iters`, `-i` | Number of benchmark iterations | 100 |
| `--engine-id`, `-e` | Engine ID (1 = MIOpen) | 1 |
| `--seed` | Random seed for reproducibility | None |

#### A/B Testing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--APath` | Plugin path for configuration A | None (default) |
| `--AId` | Engine ID for configuration A | Required for A/B |
| `--BPath` | Plugin path for configuration B | None (default) |
| `--BId` | Engine ID for configuration B | Required for A/B |
| `--rtol` | Relative tolerance for accuracy comparison | 1e-5 |
| `--atol` | Absolute tolerance for accuracy comparison | 1e-8 |

**Note**: A/B testing mode is enabled when both `--AId` and `--BId` are specified.

## Output

### Basic Benchmark Output

```
================================================================================
hipDNN Benchmark: sample_conv_fwd_16x16x16x16_k16_3x3
================================================================================
Graph:      ./graphs/sample_conv_fwd.json
Engine ID:  1 (MIOpen)
Warmup:     10 iterations
Benchmark:  100 iterations
--------------------------------------------------------------------------------

Initialization:
  Graph build time:     45.23 ms

Execution Statistics:
  Mean:                 1.234 ms
  Std Dev:              0.045 ms
  Min:                  1.156 ms
  Max:                  1.456 ms
  P95:                  1.312 ms
  P99:                  1.398 ms

Validation: SKIPPED (CPU reference not available)
================================================================================
```

### A/B Testing Output

```
================================================================================
hipDNN A/B Test: sample_conv_fwd_16x16x16x16_k16_3x3
================================================================================
Graph:      ./graphs/sample_conv_fwd.json
Warmup:     10 iterations
Benchmark:  100 iterations
--------------------------------------------------------------------------------
Configuration A:
  Plugin Path: /path/to/pluginA
  Engine ID:   1
Configuration B:
  Plugin Path: /path/to/pluginB
  Engine ID:   2
--------------------------------------------------------------------------------

                        A                   B
Init Time:          45.23 ms            42.18 ms
Mean:                1.234 ms            1.156 ms
Std Dev:             0.045 ms            0.038 ms
Min:                 1.156 ms            1.098 ms
Max:                 1.456 ms            1.312 ms
P95:                 1.312 ms            1.234 ms
P99:                 1.398 ms            1.289 ms
--------------------------------------------------------------------------------
Speedup:            B is 6.3% faster

Accuracy Comparison: PASSED
  (rtol=1e-05, atol=1e-08)
================================================================================
```

### Exit Codes

| Code | Description |
|------|-------------|
| 0 | Success (benchmark passed or A/B comparison passed) |
| 1 | Error (graph load error, execution error, configuration error) |
| 2 | A/B comparison failed (accuracy mismatch) |

## Running Tests

### Quick Start

```bash
# Activate venv
source .venv/bin/activate

# All non-GPU tests (no hipDNN required)
pytest -m "not gpu"

# All tests including GPU (requires hipDNN bindings)
pytest

# Only GPU tests
pytest -m gpu
```

### Detailed Setup for GPU Tests

GPU tests require hipDNN Python bindings. See **[SETUP.md](SETUP.md)** for complete installation instructions.

**Quick version:**
```bash
source .venv/bin/activate
export CMAKE_PREFIX_PATH=/path/to/hipdnn/build/lib/cmake
cd /path/to/hipdnn/python && pip install -e .
```

Then run:
```bash
pytest  # Runs all 111 tests (96 passed, 15 GPU)
```

## Supported Operations

The benchmark tool accepts any valid hipDNN graph operation. However, **actual execution depends on available engine plugins**.

**Currently Working with MIOpen Plugin:**
- **Convolution**: Forward, Data Gradient, Weight Gradient ✓

**Defined but Unsupported by MIOpen Plugin (need other engines):**
- **Matrix Multiplication**: Standard matmul operations
- **Pointwise**: Activations (ReLU, Sigmoid, Tanh, etc.) and element-wise operations (Add, Mul, etc.)
- **Batch Normalization**: Training and inference modes

Sample graphs are provided in the `graphs/` directory:
- ✓ `sample_conv_fwd.json` - Convolution forward (16x16x16x16, k=16, 3x3) - **Works**
- ✗ `sample_matmul.json` - Matrix multiplication (256x512 × 512x1024) - Needs matmul engine
- ✗ `sample_relu.json` - ReLU activation (64x128x56x56) - Needs pointwise engine
- ✗ `sample_add.json` - Element-wise addition (128x256x14x14) - Needs pointwise engine
- ✗ `sample_batchnorm.json` - Batch normalization inference (32x64x28x28) - Needs batchnorm engine

**Note**: The tool validates and loads all graph types. Execution requires appropriate engine plugins. Tests for unsupported operations are marked as xfail and will pass once engines become available.

## Limitations

- CPU reference validation is stubbed (CPU reference plugin not yet available in Python bindings)
- A/B testing uses `np.allclose()` for accuracy comparison between configurations
