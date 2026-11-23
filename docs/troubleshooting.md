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

## The example does not complete all five stages

After installing the checkout, run `python examples/end_to_end.py` from the repository root. The documented CPU path declares blocks and measures timing, then stops at planning with an explanatory error. That is not a CUDA memory benchmark. On CUDA the example can exercise all five stages. Compare the device printed by the example with the device you intended to use. See [the pipeline](../README.md#the-pipeline).

## Block declaration rejects a model

Declare blocks in execution order with unique, non-empty identifiers. Blocks must belong to the model and represent disjoint module subtrees. Do not declare both a parent and its child as separate blocks, or repeat the same module under different identifiers. Choose one consistent level of granularity. Distinct blocks may share parameters, but shared registered buffers are rejected; changing a buffer to non-persistent does not remove that restriction. See [declaration evidence](../STATE.md#completed-revision-32-declaration-boundary-hardening).
