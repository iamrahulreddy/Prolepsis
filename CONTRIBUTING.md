# Contributing to Prolepsis

Thanks for your interest in contributing! Here's how to get started.

## Setup

```bash
git clone https://github.com/iamrahulreddy/Prolepsis.git
cd Prolepsis
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v -x --tb=short
```

Tests are designed to run on CPU without downloading models. The `TestTritonKernel` and `TestEndToEnd` classes are automatically skipped when Triton or GPU models are unavailable.

## Submitting Changes

1. Fork the repository and create a feature branch
2. Make your changes and ensure tests pass
3. Open a pull request with a clear description of what you changed and why

## Reporting Issues

Please open a GitHub Issue with:
- A clear description of the problem
- Steps to reproduce (if applicable)
- Your environment (Python version, PyTorch version, GPU if relevant)
