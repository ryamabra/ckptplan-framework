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

## Choosing a planner for a comparison

Use `no_checkpoint` as the baseline and `checkpoint_all` to inspect the all-eligible-blocks tradeoff. `uniform`, `greedy`, and `dynamic_programming` provide alternative selections. Keep the workload and target consistent when comparing them. Deterministic planning means identical inputs produce an identical plan; separately measured profiles can differ and therefore produce different selections. Save the selected block IDs with benchmark evidence. See [planner overview](../README.md#what-each-step-does).

## A saved plan no longer validates

Validation re-derives execution signatures and checks the model fingerprint. If block order, model structure, input shape, device, or dtype changed, revisit the declaration and profiling steps rather than bypassing validation. A plan that worked for one workload is not automatically valid for another. Keep the workload description alongside saved plans. See [plan application](../ARCHITECTURE.md#plan-application).

## Does applying a plan replace the parameters?

`apply_plan` reuses the original module instances and preserves parameter identity, so existing optimizers keep referring to those parameters. The returned object is a sequential execution container for the declared blocks, not a transparent wrapper for arbitrary model control flow. Use that container for the planned execution path and verify that the declared order represents the intended computation. See [API notes](../README.md#what-each-step-does).

## Predicted savings do not match measured savings

The profiler’s isolated per-block activation estimates are not an end-to-end peak-memory prediction. Summing them does not account for all allocation reuse, tensor lifetimes, or fixed training memory. Report both the estimates and measured peaks, with the workload, instead of adjusting one to force agreement. Small peak savings can coexist with correctly applied checkpointing. See [measurement limits](../README.md#measured-and-not-claimed).

## Checkpointing uses less memory but runs slower

Checkpointing discards selected activations and recomputes them during backward execution. Extra latency is a tradeoff to measure, not by itself a correctness failure. Compare the same batch shape, dtype, and model with warm-up and repeated trials. Report memory and latency together; a larger feasible batch is a separate result from faster execution at the original batch size. See [measured scaling](../README.md#progressive-scaling-on-a10g).

## A correctness field is null

A missing or null correctness value is not a passing check. The `no_checkpoint` reference skips comparison against itself by design. A correctness check that cannot finish, including an OOM path, also cannot establish a pass. Read the correctness-check flag, verdict, error message, and OOM status together. See [verified evidence](../README.md#verified-evidence) for the configurations actually validated.

## The checkpointed model fits but the reference does not

If the no-checkpoint reference cannot execute even in isolation, there is no gradient comparison at that configuration. A successful checkpointed run establishes feasibility, not equivalence to an unexecuted reference. Report the missing comparison explicitly and validate correctness at a fitting workload separately; do not transfer that verdict to the OOM boundary. See [the documented limitation](../README.md#verified-evidence).

## The saved-result report cannot import ckptplan

Activate the development environment and run the module from the repository root:

```sh
python -m benchmarks.report benchmarks/matrix_a10g_result.json --reference no_checkpoint
```

Directly executing `benchmarks/report.py` may omit the sibling package from the import path in an uninstalled checkout. This command reads saved results; it does not start a GPU job or create new benchmark evidence. See [reporting instructions](../CONTRIBUTING.md#reporting-saved-results).

## Relative comparisons are missing

Choose a `--reference` name present in the saved artifact. A missing reference is different from a present reference that OOMed: the latter cannot provide valid relative peak-memory or latency comparisons. Zero reference denominators also make affected ratios undefined. Retain absolute measurements and the reference identity when sharing a table; do not convert undefined ratios to zero. See [comparison semantics](../STATE.md#completed-benchmark-comparison-reporting-slice).

## What do n/r, an em dash, and a tilde mean?

In saved-report output, `n/r` means the needed measurement was not recorded. `—` means the value is unavailable or undefined for that row. A throughput value prefixed with `~` is derived from batch size and latency rather than independently measured. Preserve these markers when copying tables. When batch size is constant, derived throughput is another expression of latency, not independent performance evidence. See [saved-result reporting](../CONTRIBUTING.md#reporting-saved-results).

## The dashboard is not producing new benchmark results

The dashboard displays saved JSON; it does not run training or GPU profiling. From `dashboard/`, use `npm ci` and `npm run dev`, then load a saved artifact. For dashboard changes, run `npm test` and `npm run build` there. A displayed result retains the hardware and workload limitations of its source artifact. See [dashboard documentation](../dashboard/README.md) for supported input shapes and setup requirements.

## Native build fails without CUDA

The default native C++20 planner build does not require CUDA:

```sh
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

CUDA support is opt-in and requires its own toolchain and hardware. Do not interpret a successful native CPU build as CUDA validation or Python integration evidence. See [native documentation](../native/README.md) before enabling the optional path.
