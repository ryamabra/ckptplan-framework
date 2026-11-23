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

## A block is excluded from checkpointing

Inspect the profile’s `exclusion_reason` before changing the plan. Stochastic behavior, state changes that would be repeated during recomputation, or an absence of differentiable output can make a block ineligible. Do not hand-edit its eligibility flag to force selection. Changing model behavior to make a block eligible may change the training problem; profile and validate the intended model again. See [eligibility](../README.md#blocks-that-cannot-be-checkpointed).

## A plan meets its budget but training still runs out of memory

An activation budget is not a total GPU-memory limit. Parameters, gradients, other training state, tensor lifetimes, and allocator behavior also affect the measured peak. Do not set an activation budget equal to the device’s full capacity and interpret it as an OOM guarantee. Benchmark the complete workload and retain the measured peak and OOM fields. See [system inputs](../ARCHITECTURE.md#system-boundary).
