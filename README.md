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

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Run benchmark on a serialized graph
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --warmup 10 --iters 100

# With custom engine ID
python -m dnn_benchmarking --graph ./graphs/conv1_fwd.json --engine-id 1
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--graph`, `-g` | Path to JSON-serialized hipDNN graph file | Required |
| `--warmup`, `-w` | Number of warmup iterations | 10 |
| `--iters`, `-i` | Number of benchmark iterations | 100 |
| `--engine-id`, `-e` | Engine ID (1 = MIOpen) | 1 |

## Output

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

## Running Tests

```bash
# All tests
pytest

# Unit tests only
pytest tests/unit

# Skip GPU tests
pytest -m "not gpu"
```

## MVP Limitations

- Supports Convolution Forward Propagation (Conv Fwd) graphs only
- Validation is stubbed (CPU reference not yet available in Python)
- A/B engine comparison deferred to post-MVP
