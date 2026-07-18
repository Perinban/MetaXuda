# Contributing

MetaXuda is an experimental Apple Silicon GPU runtime. Bug reports and focused improvements are welcome.

## Bug Reports

Open a bug report and include:

- What you expected to happen
- What actually happened
- A minimal reproducible example
- Full error output or logs
- macOS, Apple Silicon model, Python, Numba, and MetaXuda versions

Remove credentials, tokens, and personal paths before posting logs.

## Feature Requests

Open a feature request describing the limitation, proposed behaviour, and a practical CUDA or Numba use case.

## Examples and Integrations

Contributions may include generic Python examples, benchmarks, framework integrations, and practical Numba CUDA workloads that run through MetaXuda.

You may implement the work directly and open a pull request. The changes will be reviewed before they are merged.

If you build a separate project using MetaXuda, open an issue or discussion with a link and a short description. Useful projects may be referenced from the MetaXuda documentation.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Validation

Before submitting changes, verify that the package builds and imports:

```bash
python -m pip install build pytest
python -m build
python -m pip install dist/*.whl
python -m pytest tests
```

Keep changes focused and avoid committing virtual environments, build outputs, generated metadata, or local IDE files.
