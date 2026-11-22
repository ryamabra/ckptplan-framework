# Troubleshooting ckptplan

Use this guide to distinguish setup problems, expected limitations, and missing benchmark evidence. It summarizes the behavior documented in this checkout; it does not introduce new performance claims. Commands are usage examples, not a record of tests run while writing this guide.

## Supported Python environment

If installation rejects your Python version, use Python 3.10–3.12, the range documented by this checkout. Create a fresh virtual environment with a supported interpreter, activate it, and install from the repository root:

```sh
python -m pip install -e ".[dev]"
```

Do not remove the Python version constraint to force installation. See [contributor setup](../CONTRIBUTING.md#python-development).

## Imports use the wrong installation

Run installation and tests through the same interpreter (`python -m pip` and `python -m pytest`). If edits appear to have no effect, confirm that the active environment contains the editable checkout, rather than a separately installed release. Reinstall with `python -m pip install -e ".[dev]"` from the repository root. Preserve the actual import error when reporting a failure; reinstalling is not evidence that the error is fixed.

## CPU profiling stops before planning

`TimingOnlyProfileError` is expected when activation-based planning receives CPU profiles. CPU profiles have `timing_only=True`, `activation_bytes_estimate=None`, and no activation-byte method. Do not replace these missing measurements with zero to make planning proceed. Use CPU for timing and unit tests; collect real activation profiles on CUDA for activation-budget planning. See [CPU behavior](../README.md#cpu-is-timing-only).

## CUDA is unavailable or inputs use another device

The Python profiler supports CPU timing and CUDA activation measurement; do not assume another accelerator backend is interchangeable with CUDA. Its device and dtype arguments validate caller-owned data: they do not move or cast your model and inputs. Prepare both before profiling, and report the device, dtype, Python version, and PyTorch version if validation fails. See [device handling](../ARCHITECTURE.md#profiler).
