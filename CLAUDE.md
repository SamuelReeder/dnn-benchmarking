# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Benchmarking and validation tool for hipDNN graphs. Loads JSON-serialized hipDNN graphs, executes them via the MIOpen plugin, captures performance metrics, and supports A/B testing between different plugin/engine configurations.

## Build and Development Commands

```bash
# Install for development
pip install -r requirements-dev.txt
pip install -e .

# hipDNN bindings must be installed separately from your hipDNN build
cd /path/to/hipdnn/python && pip install -e .
```

## Running Tests

```bash
# All non-GPU tests (no hipDNN required)
pytest -m "not gpu"

# All tests including GPU tests
pytest

# Single test file
pytest tests/unit/execution/test_timing.py

# With coverage
pytest --cov=dnn_benchmarking tests/
```

Test markers: `gpu` (requires GPU), `slow` (slow integration tests).

## Running the Tool

```bash
# Basic benchmark
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100

# A/B testing (compare two engine configurations)
python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --AId 1 --BId 2
```

## Architecture

```
src/dnn_benchmarking/
├── cli/              # Entry point (main.py, parser.py)
├── config/           # BenchmarkConfig, ABTestConfig dataclasses
├── execution/        # executor.py (graph building), buffer_manager.py, ab_runner.py, timing.py
├── graph/            # loader.py (JSON loading), validator.py, tensor_info.py
├── reporting/        # reporter.py (console output), statistics.py
└── validation/       # validator.py (stubbed - CPU reference not available)
```

**Data flow:** CLI → Config → GraphLoader → Executor → BufferManager → Timing → BenchmarkStats → Reporter

**Key external dependency:** `hipdnn_frontend` - AMD's hipDNN Python bindings (requires AMD GPU + ROCm).

## Exit Codes

- 0: Success
- 1: Error (graph load, execution, configuration)
- 2: A/B comparison failed (accuracy mismatch)
