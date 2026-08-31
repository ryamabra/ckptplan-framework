# Project State

## Project

Working name: `ckptplan`

An open-source PyTorch library that profiles checkpointable model blocks and selects a gradient-checkpointing plan for a user-specified GPU-memory budget while minimizing estimated recomputation overhead.

## Status

Phase: implementation (`declare_blocks`, profiling, planning, application, and benchmarking complete)

MVP_SPEC.md Revision 3.4 is approved. The repository foundation,
`declare_blocks`, declaration-boundary hardening, and CPU-testable
`profile_blocks` and deterministic `plan_checkpoints` slices are complete,
independently reviewed, and passing.
CUDA activation-memory measurement is implemented and verified; benchmark-scale
calibration and comparison runs remain to be completed.

## Completed (CI and documentation)

Added `.github/workflows/ci.yml`: CPU-only pytest across Python 3.10/3.12 x
PyTorch 2.5.0/2.13.0 on ubuntu-latest, `fail-fast: false`, PyTorch pulled from
the CPU wheel index so the CUDA runtime is never downloaded. All four cells were
confirmed buildable before writing the matrix (torch 2.5.0 ships cp310-cp313;
torch 2.13.0 requires Python >=3.10 and ships cp310), matching pyproject's
`requires-python = ">=3.10,<3.13"` and `torch>=2.5.0,<2.14.0`.

**Verified, not assumed:** run 33300397003 on GitHub completed in 47 s with all
four jobs green — py3.10/torch 2.5.0, py3.10/torch 2.13.0, py3.12/torch 2.5.0,
py3.12/torch 2.13.0.

Rewrote `README.md`, which still claimed "repository-foundation phase. No public
API is implemented yet" and cited Revision 3.2. It now documents the five-call
pipeline with a runnable example, a per-call reference table, the
checkpoint-eligibility rules, and the "Measured, and not claimed" section
recording that no percentage is gated and that gradient correctness is currently
unverified pending a GPU re-run.

Added `examples/end_to_end.py`, runnable on either device: on CUDA it completes
declare -> profile -> plan -> apply -> benchmark; on CPU it runs through
`profile_blocks`, then stops at `plan_checkpoints` and prints the
`TimingOnlyProfileError` explaining that activation-based planning is CUDA-only
because CPU profiles carry no activation-byte measurements. The README's pasted
CPU transcript is that script's real output. A smoke test executes the script so
the documentation cannot rot silently.

## Correctness Evidence — CORRECTED

**Every `correctness_passed: true` recorded in this repository is unverified for
gradients.** Two independent defects in `run_benchmark`, both now fixed:

1. **Shared-parameter self-comparison (present from `c0acf30`, the original
   benchmark commit).** The gradient comparison was
   `zip(container.parameters(), reference.parameters())` computing
   `(a.grad - b.grad)`. `apply_plan` preserves parameter identity and both
   containers are built from the same `blocks`, so `a is b` for every pair —
   the expression was a tensor minus itself and `grad_diff` was structurally
   `0.0` regardless of whether checkpointing produced correct gradients.
   Verified on CPU: all 4 parameter pairs satisfy `a is b`.
2. **Gradient comparison relocated into the OOM handler (`bd72423`, "capture
   OOM during benchmark correctness").** That commit inserted `try:` and an
   `except torch.cuda.OutOfMemoryError:` clause into the middle of the
   correctness block. Adding 6 lines silently moved the 8 following lines — the
   two backward passes, the gradient comparison, and the verdict assignment —
   out of the `try` body and into the handler. From that commit on, a run that
   completed normally computed only `output_diff`, leaving `grad_diff` and
   `correctness_passed` as `None`; a verdict could only be produced by first
   raising a CUDA OOM. The handler also used `reference` before it was bound
   (an OOM inside the reference `apply_plan` raised `UnboundLocalError` instead
   of returning a structured result) and overwrote its own
   `correctness_passed = False` two lines later.

**Chronology, from `git log --date-order`:** `c0acf30` (benchmark added) →
`d0d1615` (five-planner matrix run) → `f936da1` → `bd72423` (defect 2
introduced) → `1ec4024` → `8a834be` (boundary correctness run).

**`benchmarks/matrix_a10g_result.json` — what its `correctness_passed: true`
means.** The matrix run predates `bd72423`, so its verdicts came from the
normal path, *not* through the OOM handler. But it carries defect 1: the output
comparison was real and passed, while the gradient comparison could not return
anything but `0.0`. The four `true` values are therefore evidence about
forward outputs only. **MVP_SPEC.md §12.5 hard gate 1 (100% pass rate on the
declared numerical-correctness tolerance) is not satisfied by that artifact**,
and the same caveat applies to every other pre-`bd72423` correctness pass in the
repository, including `benchmarks/billion_a10g_result.json` and the calibration
run.

**`benchmarks/boundary_correctness_result.json` — the previous interpretation
was wrong.** Its `max_output_diff: 0.0, max_grad_diff: null,
correctness_passed: null, oom: false` is precisely the signature of defect 2 on
a run that did *not* OOM: `output_diff` was assigned, so both co-located forward
passes completed successfully, and the `except` branch never executed, so the
gradient comparison was never attempted. The recorded interpretation — that the
co-located reference "cannot retain gradients reliably at this model scale", a
memory-pressure harness limitation — is **not supported by that artifact**.

What is established: no OOM occurred in the correctness section, and the
reference model was built and forward-evaluated co-resident with the container.
What is *not* established: whether the gradient comparison would have OOMed had
it run. Two full backward passes at 1.2B parameters is genuinely memory-heavy,
so the memory hypothesis remains plausible — it is **untested, not disproven**.

**Consequence for `benchmarks/modal_isolated_correctness.py`.** That script was
written to work around a co-location OOM that has never actually been
demonstrated. Its design is unaffected and still sound, but the cheapest next
step is to re-run `benchmarks/modal_boundary_correctness.py` against the fixed
`run_benchmark` and find out whether co-location OOMs at all. Per-container
isolation may not be needed.

## Fix applied

The gradient comparison and the verdict now run on the normal path. Because the
container and reference share `.grad` buffers, the two backward passes run
sequentially with the checkpointed gradients snapshotted and the buffers zeroed
in between, so the comparison is between two genuinely distinct gradient sets.
The OOM handler now only records that the check could not be completed: it
leaves `correctness_passed` as `None` (never `False` — "no evidence" is not
"failed"), notes the reason in `error_message` without masking the benchmark's
own OOM, and cleans up in a `finally` that tolerates an unbound reference.
**Tolerances and the comparison formula are unchanged.**

Added `tests/unit/test_run_benchmark.py` — the first tests `run_benchmark` has
ever had, which is how both defects survived. 10 CPU tests; **6 of them fail
against the unfixed tree**, verified by checking out `HEAD -- ckptplan/api.py`
and re-running. They cover: a verdict reached with no OOM; gradient diff
populated on the normal path; a deliberately divergent backward producing a
non-zero diff and a `False` verdict (guarding against a vacuous check); the
shared-parameter identity that makes snapshotting necessary; OOM while building
the reference returning a structured result instead of `UnboundLocalError`; OOM
mid-correctness; a correctness OOM not masking a benchmark OOM; and the
opt-out paths. No configuration genuinely fails correctness on CPU —
`grad_diff` is `0.0` because recomputation replays identical ops in identical
order, and the divergence test proves that `0.0` is now measured rather than
structural.

**Pending:** no GPU re-run. Whether the matrix configurations actually pass a
real gradient comparison on an A10G is unknown and requires re-running
`benchmarks/modal_matrix.py` against the fixed code.

## Current Task

The deterministic pure-planning `plan_checkpoints` slice, plan
validation/application slice, CUDA profiling implementation slice, and the
benchmark comparison reporting slice are complete, and saved runs can now be
re-reported locally via `benchmarks/report.py`. CUDA execution still requires
verification in the separate A10G workflow; an actual five-planner A10G matrix
run under the expanded schema remains next.

## Completed (per-config process isolation for correctness) — WRITTEN, NOT RUN

Added `benchmarks/modal_isolated_correctness.py` to attack the harness
limitation recorded in `benchmarks/boundary_correctness_result.json`: at ~1.2B
parameters the co-located no-checkpoint correctness reference cannot hold its
gradients alongside the checkpointed run, so `correctness_passed` and
`max_grad_diff` came back null. The approach is one Modal container per
configuration, so the reference and the configuration under test never contend
for memory.

**Reference exchange without co-residency.** The reference container writes its
gradients to a Modal Volume, one file per parameter. Each candidate container
reloads them **one parameter at a time** onto the CPU and compares against its
own gradient. Peak extra memory is two copies of the single largest parameter
(tens of MB), not two whole models, and the reference gradients are never
co-resident in GPU memory with the candidate's model.

**What crosses the Modal boundary.** Not gradient tensors — a full set is
~4.8 GB per configuration. Each container returns only per-parameter scalars:
`max_abs_diff`, `reference_max_abs` for scale, and an exact `torch.allclose`
verdict at the declared tolerance. This is sound because the comparison is
performed in-container against the real reference tensors streamed from the
volume; the scalars are the *result* of an exact elementwise comparison, not a
lossy proxy for one. A test asserts the returned payload is scalars-only and
JSON-serializable.

**Determinism of cross-container initialization.** Two mitigations and one
stated limit:

1. The CUDA RNG is removed from the initialization path. Every layer is built
   on the **CPU** under a seeded global RNG and only then moved to the GPU, so
   initialization never depends on CUDA launch configuration, device
   architecture, or driver version; the transfer is a bit copy. Layers are
   built, fingerprinted, and moved one at a time, bounding peak host memory to
   one layer.
2. Determinism is checked at run time rather than assumed. Every container
   returns an `init_fingerprint` (SHA-256 over each parameter's name, shape,
   dtype, element count, and 256 `float.hex()` elements sampled at fixed
   strides). `reconcile` marks the run `correctness_valid: false` and claims no
   verdict if any fingerprint differs or is missing.

**How determinism was verified, and what is still unproven.** Verified: the
CPU-initialization path is bit-identical across separate OS *processes* on this
machine at a fixed torch version — three independent processes produced
identical fingerprints (`425165a9da03357d…`), and a `@pytest.mark.slow` test
spawns two subprocesses and asserts they agree with the in-process value. A
further test confirms the fingerprint detects a single perturbed weight, so it
is not vacuous. **Not proven:** bit-identical initialization across separate
A10G *containers*, which requires running Modal. The GPU is not involved in
initialization and the image pins Python 3.12 and `torch==2.13.0`, so the
argument is strong — but it is an argument, not a proof, which is exactly why
the run-time fingerprint gate exists.

**Scope limit worth recording.** Isolation rescues the
`boundary_correctness_result.json` configuration (seq_len 512, batch 1), where
no-checkpoint fits *alone* but not *alongside*. It cannot rescue the harder
`oom_boundary_a10g.json` point (seq_len 4096, batch 4), where no-checkpoint
OOMs even alone: no reference can exist there at all, by definition.

**Proven:** spec round-tripping and key stability, fingerprint reproducibility
and sensitivity, cross-process initialization determinism, input determinism
and independence from global RNG state, gradient save/stream/compare round trip
including exact `max_abs_diff`, within-tolerance and out-of-tolerance cases,
missing-parameter reporting, scalars-only JSON payloads, budget selection, and
all five `reconcile` verdict paths — 25 CPU-only tests in
`tests/unit/test_isolated_correctness.py`. The module imports and is fully
unit-testable without Modal installed, via an offline app stub. Registered a
`slow` pytest marker in `pyproject.toml` for the subprocess test. Full suite:
123 passed before, 148 passed after.

**Cost, against verified pricing.** Modal publishes A10/A10G at
$0.000306/sec ($1.102/hr); memory is $0.00000222/GiB/sec and CPU
$0.0000131/core/sec (0.125 core minimum). With this script's `memory=16384`
request the effective container rate is ~$0.000343/sec ≈ $1.235/hr — the 16 GiB
request adds ~12% over the bare GPU rate, which the first estimate omitted. At
~8 min/container: the two-container default is ~$0.33 (range $0.21–$0.49) and
the five-planner variant ~$0.82 (range $0.51–$1.23), both ~16 min wall clock
since candidates run in parallel after the reference. The 4.8 GB gradient set
is free (1 TiB/month allowance). `timeout` was lowered from 5400 s to 1800 s,
still ~3.75x the expected runtime, cutting the worst-case hang ceiling from
$3.71/$9.26 to $1.24/$3.09 — the timeout, not the estimate, is what actually
caps spend.

**Two review findings, verified by reading the code and recorded in the module
docstring.**

1. *Fingerprint checking is late, not early.* `reconcile` runs locally on the
   entrypoint after every container has returned. No container compares its own
   fingerprint against the reference's, so an initialization mismatch is caught
   only after all profiling, benchmark, and gradient work has completed. The run
   is correctly invalidated and no false verdict is claimed, but no compute is
   saved. Failing fast would require the reference to publish its fingerprint to
   the volume and candidates to check it immediately after their own build.

2. *Plans are recomputed per container.* `run_isolated_config` calls
   `profile_blocks` and `plan_checkpoints` inside each container. For the
   two-container default this is harmless: `no_checkpoint` selects the empty set
   and `checkpoint_all` selects every eligible block
   (`ckptplan/planning/planner.py`), neither consulting any measured estimate,
   so both plans are identical across containers by construction. It is *not*
   harmless for the five-planner variant: `greedy` and `dynamic_programming`
   rank blocks by `recompute_time_upper_bound_ms_mean`, and `uniform` depends on
   measured activation bytes and the derived budget, so all three may select
   different blocks in different containers. Each container's gradient
   comparison remains a valid correctness check of the plan it actually ran, and
   `checkpointed_block_ids` records that plan — but the planner *label* would
   not identify a reproducible plan, and cross-planner comparison would not be
   controlled. The five-planner variant should therefore not be run until plan
   decisions are computed once and passed to each container.

**Pending — nothing has been measured.** Modal was not run. No A10G container
has executed this script; there is no output artifact, no gradient comparison,
and no correctness verdict at the boundary. Awaiting owner approval to run.

## Completed (report script: predictions, derived throughput, blank markers)

Follow-up on `benchmarks/report.py`, closing the gap that
`throughput_change_pct` and `prediction_gap_pct` — the two columns that
motivated the reporting slice — rendered blank for every row on the only real
multi-config artifact.

**`--predicted`.** Accepts inline `name=bytes` pairs and/or paths to a JSON
object mapping config names to byte counts, repeatable, later values winning,
and overriding any `predicted_activation_bytes_after` embedded in the artifact.
Non-integer, negative, and empty-name specs are rejected with exit 2. A name
matching no config in the artifact warns on stderr but still exits 0, so one
predictions file can be reused across artifacts with differing config sets.

**Derived throughput — decision: derive it, marked.** `ckptplan/api.py:232`
computes `throughput_samples_per_sec` as `samples * 1000 / step_latency_ms_mean`.
It is therefore not an independent measurement but a pure algebraic function of
batch size and mean latency, so reproducing it from an artifact that records
batch size and mean latency is exact, not an approximation. Derived cells are
marked `~`. The honest caveat, recorded in the module docstring: because batch
size is constant across configurations in these runs, a derived
`throughput_change_pct` is a monotone restatement of `latency_overhead_pct` and
carries no independent information. It is shown for schema parity with real
runs, not as separate evidence.

**Two blanks instead of one.** `n/r` now means the artifact never recorded the
input; the em dash `—` means the input exists but the value is undefined for
that row (config OOMed, reference OOMed, zero reference denominator, or fewer
than two latency trials). A ratio is "recorded" only when both sides of it were
recorded. This separates a gap in an old artifact from a genuinely undefined
quantity — previously one em dash meant both. It also distinguishes a
`correctness_passed` key that is absent (`n/r`) from one present but null
(`—`, i.e. checked but no verdict, as on a reference row).

Re-run against `benchmarks/matrix_a10g_result.json` with predictions supplied,
the gap column reads +55.21% for no-checkpoint and +100.00% for checkpoint-all,
with +77.58% for uniform, greedy, and dynamic-programming.

**Proven:** derivation formula parity with `run_benchmark`, derived-marker
rendering, non-derivation when throughput is recorded or batch size or latency
is missing, `n/r` versus `—` for every affected column, prediction parsing
(inline, file, precedence, all five rejection paths), unknown-name warning, and
CLI exit codes — 19 further CPU-only tests. Full suite: 104 passed before, 123
passed after.

**Pending:** the prediction-gap values above were produced from predictions
supplied on the command line, **not** from saved measurements. The A10G matrix
run predates per-config `predicted_activation_bytes_after`, so the reference
value used (4,433,117,184) is the artifact's own recorded
`profile_activation_total`, checkpoint-all's is 0 by construction, and the three
budget planners' are the artifact's `profile_activation_total * target_fraction`
target rather than each planner's actual discrete predicted-after. The column
is therefore proven as plumbing and illustrative in magnitude, not measured.
Only a fresh `modal_matrix.py` run under the expanded schema will make it a
measured result.

## Completed (local benchmark report script)

Added `benchmarks/report.py`, a standalone local CPU-only script that reads any
saved benchmark result JSON and prints an aligned comparison table via
`ckptplan.compare_results`. It deliberately does **not** run inside a Modal
function: saved runs can be re-reported for free, and the Modal functions stay
pure measurement jobs that only emit JSON.

It takes a JSON path plus an optional `--reference` (default `no_checkpoint`),
and recognizes three saved-result shapes rather than the two originally
anticipated:

1. `{"results": [ {"planner": ...}, ... ]}` — the expanded schema
   `benchmarks/modal_matrix.py` writes today.
2. `{"results": {"<config>": {...}}}` — the older matrix schema in
   `benchmarks/matrix_a10g_result.json`, which predates `step_latency_ms`,
   `peak_reserved_bytes`, throughput, and the predicted fields.
3. **No `"results"` key at all** — config dicts at the top level beside scalar
   and string metadata, which is the actual shape of
   `benchmarks/oom_boundary_a10g.json`. This third shape was not in the task
   description; it was found by inspecting the artifact and is handled rather
   than crashed on.

Missing fields default neutrally and render as an em dash, never as `0` or
`"None"`. A `LoadedConfig.source_keys` set records which keys the saved dict
actually had, so "not recorded" is distinguishable from "recorded as zero" —
that is why the old matrix artifact shows an em dash for `latency_std` instead
of a fabricated `0.00`. The `predicted_activation_bytes` mapping is built from
each config's `predicted_activation_bytes_after` where present.

Verified against both existing artifacts. `matrix_a10g_result.json` reports
+0.10% peak-allocated reduction for all four checkpointed planners against
no-checkpoint, with latency overhead from +19.61% (uniform) to +34.51%
(checkpoint-all); `latency_std`, `throughput_change_pct`, and
`prediction_gap_pct` are em dashes because that older artifact never recorded
the inputs. `oom_boundary_a10g.json` reports every ratio as an em dash because
the `no_checkpoint` reference OOMed and has no valid peak or latency —
the documented OOM-reference path, not a failure.

The module docstring records that `prediction_gap_pct` is necessarily 100% for
`checkpoint_all` (its `predicted_activation_bytes_after` is 0 by construction,
so the gap is `|0 - peak| / peak`), and that the gap is a reported diagnostic
per MVP_SPEC.md §12.5, not a gate.

**Proven:** JSON-to-`BenchmarkResult` rebuild across all three schemas
(including the old-schema path), neutral defaults, em-dash rendering, table
alignment, reference marking, and CLI exit codes, via 16 CPU-only tests in
`tests/unit/test_report_script.py`. Two of those tests run the script against
the real committed artifacts. Full suite: 88 passed before, 104 passed after.

**Pending:** no new measurement. The script only re-reports already-saved runs;
`benchmarks/modal_matrix.py`'s expanded result dict is still unverified against
a real A10G execution.

## Completed (benchmark comparison reporting slice)

Added `ckptplan/reporting.py` with `compare_results(results,
reference_config_name, predicted_activation_bytes=None)` returning a tuple of
frozen `ConfigComparison` records, one per input result in input order, each
carrying `peak_allocated_reduction_pct`, `peak_reserved_reduction_pct`,
`latency_overhead_pct`, `throughput_change_pct`, `latency_ms_std` (sample stdev
over the `step_latency_ms` trial tuple), and `prediction_gap_pct`. All
quantities are reported, not gated, per MVP_SPEC.md §12.4/§12.5.

Predictions are supplied as an explicit `config_name -> predicted bytes`
mapping rather than by passing `CheckpointPlan` objects: `BenchmarkResult` does
not carry the prediction, and a mapping lets results reloaded from a saved JSON
benchmark run be reported without reconstructing plans.

OOM and zero-denominator cases are handled explicitly instead of dividing by
zero: an OOMed configuration reports `None` for every ratio and for
`latency_ms_std`; an OOMed *reference* makes every relative field `None` for
all configurations while leaving `latency_ms_std` intact; a zero reference
latency, zero reference throughput, or zero measured peak yields `None` for the
affected field only. Duplicate `config_name`s and a missing reference raise
`ValueError`.

`benchmarks/modal_matrix.py`'s saved result dict now retains the full
`step_latency_ms` trial tuple and uses the spec field names
`predicted_activation_bytes_after` and
`predicted_recompute_time_upper_bound_ms`; `peak_reserved_bytes`,
`latency_ms_p50`, and `latency_ms_p95` were already retained.

No planner behaviour changed. The profiler's additive isolated activation
estimate still legitimately exceeds the measured whole-model peak reduction;
`prediction_gap_pct` reports that gap rather than correcting it.

**Proven:** `compare_results` arithmetic, ordering, OOM handling, zero-latency
and zero-peak edge cases, and error paths, via 11 new CPU-only unit tests in
`tests/unit/test_reporting.py` over synthetic `BenchmarkResult` objects. Full
suite: 77 passed before, 88 passed after.

**Pending:** `benchmarks/modal_matrix.py` was not executed — the expanded
result dict is unverified against a real A10G run, and no measured five-planner
comparison numbers exist yet.

## Completed (CUDA profiling implementation slice)

Extended `profile_blocks` to accept CUDA devices when CUDA is available. CUDA
profiles use synchronized allocator readings for isolated forward activation
deltas, CUDA events for forward and full checkpoint-recomputation timing, and
report `timing_only=False` with `activation_bytes_method="isolated_forward_delta"`.
CPU behavior remains timing-only and unchanged. Requests for unsupported devices
or unavailable CUDA fail before caller modules execute.

The local environment has no CUDA device, so the A10G integration tests remain
an external verification task. Local verification: `.venv/bin/pytest -q` passes
**77 tests, 0 failures** and `git diff --check` passes.

## Completed (Modal A10G CUDA smoke verification)

Added `benchmarks/modal_cuda.py`, a reproducible Modal runner pinned to Python
3.12, PyTorch 2.13.0, and an A10G GPU. The cloud smoke test completed
successfully on an NVIDIA A10 using CUDA 13.0 and confirmed CUDA profile output
(`timing_only=False`, isolated activation fields populated, forward and
recomputation timings recorded). The intentionally tiny smoke model produced
zero allocator-delta activation bytes; benchmark-scale directional memory and
OOM tests remain part of the later benchmarking workflow.

## Completed (benchmark execution and reporting slice)

Added the frozen `BenchmarkResult` schema and public `run_benchmark` API. The
runner validates and applies a plan, performs optional no-checkpoint numerical
correctness comparison, executes warm-up and measured forward/backward trials,
records CUDA allocated/reserved peaks when available, measures latency and
throughput, and returns explicit OOM/error fields plus environment metadata.

Local verification: a correctness-enabled CPU benchmark completed successfully;
`.venv/bin/pytest -q` passes **77 tests, 0 failures** and `git diff --check`
passes. The calibrated Modal benchmark remains to be run after the model/config
calibration workflow is added.

## Completed (plan validation and application slice)

## Completed (initial Modal toy-transformer calibration)

## Completed (Modal ~1B-parameter benchmark)

Dedicated correctness testing is recorded in
`benchmarks/boundary_correctness_result.json`. The checkpointed path fits, but
co-locating the no-checkpoint reference for this 1.2B model exhausts memory
before gradients can be compared. Independent checkpoint correctness passed in
the five-planner matrix; the exact OOM-boundary correctness comparison remains
unavailable and is not claimed as a release result.

The isolated OOM-boundary runs are recorded in `benchmarks/oom_boundary_a10g.json`:
at 24 layers, hidden 2048, sequence length 4096, and batch 4, no-checkpoint
returned a structured CUDA OOM while checkpoint-all completed with 12,505,498,624
peak allocated bytes. Exact-boundary correctness comparison is unavailable
because the no-checkpoint reference cannot execute; checkpoint correctness was
validated at lower fitting configurations.

Focused A10G instrumentation confirmed all 24 blocks were checkpointed. It
measured 4,834,394,112 parameter bytes and 4,433,117,184 profiled isolated
activation bytes; checkpoint-all reduced forward peak by 18,137,088 bytes and
full backward peak by 9,739,776 bytes. The additive isolated activation total
is therefore not a whole-model peak prediction: tensor lifetimes, allocator
reuse, and fixed parameter/gradient memory dominate the end-to-end peak. The
diagnosis is recorded in `benchmarks/memory_diagnosis_a10g.json`; checkpoint
behavior was not changed based on this evidence.

Completed the five-planner A10G matrix at sequence length 2048 with 20 measured
trials per configuration. All configurations fit; all checkpointed planners
passed correctness. The result is recorded in
`benchmarks/matrix_a10g_result.json`. Checkpointed planners measured 9,887,221,760
peak allocated bytes versus 9,896,961,536 for no-checkpoint, with mean latency
from 1,949.998 ms (uniform) to 2,192.905 ms (checkpoint-all).

Diagnosis: the 1.208B float32 parameters plus their gradients consume about
9.67 GB, nearly the entire 9.70 GB measured peak. With sequence length 512 and
batch size 1, activation memory is too small relative to that fixed training
state to produce a visible peak reduction. The next experiment should increase
activation volume while remaining within the A10G budget.

Ran `benchmarks/modal_billion.py` on an A10G using a 24-layer transformer with
1,208,598,528 parameters, hidden size 2048, sequence length 512, batch size 1,
and float32. Both configurations fit without OOM and checkpoint correctness
passed. Peak allocated memory was effectively unchanged (9,702,605,824 versus
9,702,606,848 bytes), while checkpoint-all latency was 484.96 ms versus 360.24
ms. Results are recorded in `benchmarks/billion_a10g_result.json`; the required
directional memory-reduction gate is not claimed at this configuration.

Added `benchmarks/modal_calibrate.py` and recorded the first calibrated
configuration in `benchmarks/calibrated_config.json`: toy-transformer with
`N=2`, `B=1`, hidden size 1024, 16 heads, sequence length 1024, float32. The
Modal A10G run completed without OOM for both no-checkpoint and checkpoint-all;
peak allocated memory was 289,961,472 bytes versus 285,739,520 bytes,
respectively, establishing the required directional reduction at this bounded
calibration point. The run also found and fixed an output-signature mismatch
between profiling and plan validation. The full calibration search and five-way
baseline matrix remain future benchmark work.

Implemented `validate_plan`, `CheckpointedSequential`, and `apply_plan` using
the canonical execution helpers. Validation now checks the plan format version,
decision-list structure and eligibility consistency, re-derives every block's
execution signature under state-preserving `no_grad`, and verifies the model
fingerprint. Application composes the original module instances, checkpoints
exactly the selected blocks, preserves parameter identity, enforces the entry
signature on every runtime call, and applies the shared boundary conversion
rules.

Added focused application tests covering successful execution, parameter
identity, runtime shape rejection, model-fingerprint rejection, and unsupported
plan versions. Verification: `.venv/bin/pytest -q` passes **77 tests, 0
failures**.

## Senior Decisions Recorded

- **MVP_SPEC.md Revision 3.1 is approved for implementation.**
- **MVP_SPEC.md Revision 3.2 is approved for implementation.** Declared block
  subtrees must have disjoint recursive module-identity sets. Sharing any
  registered buffer across declared blocks is rejected, including buffers with
  `persistent=False`, because persistence controls serialization only—not
  runtime mutation or checkpoint-recomputation side effects.
- **MVP_SPEC.md Revision 3.3 is approved and implemented.** CPU profiling
  performs genuine forward and full-checkpoint-recomputation timing, keeps
  activation-memory fields explicitly unavailable, treats device/dtype as
  validation-only, and restores caller-owned module state on success and error.
- **The disclosed self-consistent plan-tampering limitation (§14) is accepted for
  v0.1.** Serialized `CheckpointPlan`s are trusted local artifacts, not a security
  boundary; `validate_plan` is not required to detect a decision tampered
  consistently with itself, and `validate_plan`'s signature will not be widened to
  accept `profiles` to close this gap.
- **The five judgment calls recorded in `MVP_SPEC.md` §14 are approved as
  written** (the `profile_fingerprint` field inclusion/exclusion list; greedy
  fallback over adaptive-bucket retry for the DP sub-bucket fix; `ExecutionSignature`
  exposed as an explicit `CheckpointPlan` field; the reserved-but-unreachable
  `exclusion_reason="shared_mutable_buffer"` enum value; `plan_format_version`
  `"3.1"` as a minor, not major, bump).

`MVP_SPEC.md` is now the accepted source of truth for v0.1's design; the
specification phase is closed and implementation has begun.

## Completed (Revision 3.3 CPU `profile_blocks` slice)

Implemented the frozen `BlockProfile` schema, public `profile_blocks` API,
profiling errors, canonical private chain/boundary/signature/state helpers in
`ckptplan/_execution.py`, and the CPU timing path in
`ckptplan/profiling/profiler.py`. The profiler walks actual declared block
boundaries, supports positional/keyword and nested tensor inputs/outputs,
measures genuine non-reentrant checkpoint recomputation with early stopping
disabled, reports CPU profiles as `timing_only=True`, and never fabricates CUDA
activation-memory results.

Three independent read-only reviews covered spec acceptance, autograd/state
correctness, and the public API boundary. Their findings were reconciled into
the implementation: intermediate device/dtype failures are rejected before a
downstream block executes; repeated tensor arguments preserve alias identity;
pre-existing gradients are restored even when forward mutates `.grad`; buffer
registration, persistence flags, object identity, resized metadata, sparse COO
metadata, quantized/sparse-COO values, and overlapping storage views are preserved;
and restoration failures cannot mask the original forward exception or skip
mode/gradient cleanup. Revision 3.3 also corrects §10.1 so
`no_differentiable_output` is explicitly a profile-level exclusion. Compressed
sparse buffer layouts are rejected in preflight before caller code executes;
PyTorch does not provide one reversible metadata-restoration operation across
those layouts.

Verification: `.venv/bin/pytest -q` passes **58 tests, 0 failures**;
`git diff --check` and `python -m compileall -q ckptplan tests` pass. Warning
categories are environment/API notices (NumPy absent in the local torch wheel,
quantized-constructor deprecation, sparse-invariant-check guidance, and
PyTorch's compressed-sparse beta notice), not test failures. Graphify's
incremental code update rebuilt the graph to **285 nodes, 537 edges, and 27
communities**; only changed code was structurally re-extracted, with no LLM
semantic-extraction spend.

## Completed (Revision 3.2 declaration-boundary hardening)

A prior implementation agent began this correction and was interrupted during
the semantic portion of an incremental Graphify update. The reasoning agent
audited the uncommitted working tree and completed the handoff without
discarding work.

Two boundary gaps are now closed:

1. `declare_blocks` rejects overlapping declared module subtrees by comparing
   the complete recursive `named_modules()` identity sets for every pair. This
   covers parent/descendant declarations and sibling wrappers that share a
   descendant module, while still allowing distinct modules to share an
   individual `nn.Parameter`.
2. The buffer rule is stated precisely and conservatively: sharing any
   registered buffer across blocks is rejected, regardless of the
   `persistent` serialization flag. `persistent=False` removes a buffer from
   `state_dict`; it does not prevent runtime mutation or recomputation side
   effects.

Three read-only reviewers independently checked specification consistency,
PyTorch/boundary correctness, and API/test scope. They correctly found stale
"persistent buffer" error prose and that the initial ancestor-only overlap
check did not detect sibling wrappers sharing a descendant; both were fixed.
They also objected that the all-registered-buffer rule differed from the
initial correction prompt. Senior review considered that objection and retained
the conservative rule as an explicit safety decision for the reason above,
rather than preserving an unsafe serialization-based exemption.

Verification: `.venv/bin/pytest -q` passes all 31 tests after the reviewer-driven
shared-descendant regression test; `git diff --check` is clean. The incremental
Graphify update re-extracted four changed Python files structurally and three
changed documents semantically, then replaced stale nodes from those sources:
the resulting graph has 151 nodes, 200 edges, and 21 communities. Its health
check reports zero missing endpoints, dangling endpoints, self-loops, duplicate
edges, or collapsed edges.

## Completed (baseline commit)

Per explicit approval, created the initial baseline commit (`6ddf5b8`) on top of
the previously-initialized git repository (`git init -b main`, no remote —
see the now-superseded "Completed (repository initialization)" entry below for
that step's own verification). The commit contains: `MVP_SPEC.md`
(Revision 3.1) and `ARCHITECTURE.md` as the accepted design; the repository
skeleton and pytest harness from the foundation slice; `STATE.md`'s full
decision history; the graphify skill's own project-local files,
checked for secrets before committing — only a hook-guard config, nothing
sensitive); and, per instruction, only the graphify outputs needed for future
bounded queries — `graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.json`.
`graphify-out/cache/`, `cost.json`, `manifest.json`, and every `.graphify_*`
pointer file were added to `.gitignore` and confirmed absent from `git status`
before committing (temporary/machine-local/regenerable, not needed to query).

Verified before and after: `git rev-parse --show-toplevel` →
the project root exactly, both before this
commit and again after the second commit below — the repository root has not
drifted. No remote was added at any point (`git remote -v` empty throughout).

## Completed (declare_blocks slice)

Implemented `declare_blocks(model, blocks) -> tuple[CheckpointableBlock, ...]`
(`ckptplan/api.py`), the frozen `CheckpointableBlock` dataclass
(`ckptplan/types.py`), and `BlockDeclarationError` (`ckptplan/errors.py`), all
re-exported from `ckptplan/__init__.py`. Committed as `112e632`.

**Query-first research performed before implementing:** ran
`graphify query "declare_blocks CheckpointableBlock block declaration
invariants"` and `graphify query "shared parameters persistent buffer
restriction declare_blocks"` against the graph built in the prior round. The
graph correctly confirmed `declare_blocks`'s existence, its source file, and
its structural relationships to the rest of the data model, but — as
expected of a concept/relationship graph rather than a full-text index — did
not surface exact rule wording. Per instruction, only the precise sections
needed were then read directly: MVP_SPEC.md's "## 3. Declaring ordered
checkpointable blocks", the `CheckpointableBlock` dataclass in "## 5. Core
schemas", and "### 10.3 Shared parameters and buffers" — not the rest of the
1300+ line document.

**A real spec-internal tension was found and resolved during that reading,
not invented after the fact:** Sec 10.3's prose says "`declare_blocks`
computes `parameter_alias_groups`... and records them," but `CheckpointableBlock`'s
schema (Sec 5) has exactly three fields (`block_id, order, module`) with
nowhere to store that, and `compute_parameter_alias_groups` is referenced
exactly once in the whole document, inside Sec 9.1's fingerprinting machinery
(used by `plan_checkpoints`/`validate_plan`, both explicitly out of scope for
this slice). Resolution: `declare_blocks` permits shared parameters (does not
raise) but does not compute or store alias groups — deferred to the planning
slice. All three reviewers agreed with this resolution on independent
inspection (see below); the first reviewer explicitly called it "correct on
closer reading — not just a scope-boundary rationalization."

Implements every required behavior: non-empty/unique string `block_id`;
caller order preserved with `order` assigned `0..n-1` from list position
(verified distinct from the model's own definition order); every module
reachable from `model.named_modules()` by identity (`id()`-based, not name);
repeated module instances rejected; shared registered-buffer identity across
declared blocks rejected; shared `nn.Parameter` instances permitted; returns
`tuple[CheckpointableBlock, ...]`; never calls any module's `forward()`; and
takes no `example_inputs` parameter. 22 unit tests in
`tests/unit/test_declare_blocks.py` cover every one of these plus negative-space
cases added during review (two blocks each with their own distinct buffer;
one tensor registered under two names within one block).

### Commands run and results

- `.venv/bin/pytest -v` → **22 passed, 0 failed** (17 `declare_blocks` tests +
  3 pre-existing smoke tests + 2 negative-space tests added during review),
  run three times across the implement → review → fix cycle, all green on
  the final run.
- `git add -A && git commit` (baseline, `6ddf5b8`) and again after this slice
  (`112e632`) — `git log --oneline` shows both; `git status` clean after each.
- Incremental `graphify update` against
  project root (not the unrelated
  shell working directory) — see "Graphify update result" below.

### Reviewer findings and resolutions

Three independent read-only reviewer agents ran in parallel (spec/acceptance,
correctness/edge-cases, packaging/API-boundary).

- **Spec/acceptance reviewer:** all 9 required behaviors DELIVERED with
  file:line evidence; confirmed the `parameter_alias_groups` resolution above
  is correct, not just defensible; confirmed `tuple[CheckpointableBlock, ...]`
  return type and dataclass immutability; confirmed zero out-of-scope
  implementations. No blocking issues.
- **Correctness/edge-case reviewer:** no bugs in the 8 shipped invariants, but
  found two real test-coverage gaps — **fixed**: added
  `test_two_blocks_with_distinct_own_buffers_does_not_raise` and
  `test_same_buffer_tensor_under_two_names_within_one_block_does_not_raise`
  (the latter documents reliance on `named_buffers()`'s
  `remove_duplicate=True` default; added a matching code comment in
  `api.py`). Also flagged at the time, **subsequently fixed by Revision 3.2**:
  declaring a parent module
  and its own nested submodule as two separate, overlapping blocks is only
  accidentally caught (via a misleading buffer-collision error) when the
  child happens to have a buffer, and silently succeeds otherwise. Revision
  3.2 now enforces complete recursive module-subtree disjointness.
- **Packaging/API-boundary reviewer:** confirmed public exports, no scope
  creep, `pyproject.toml` unchanged. **Corrected a mistaken premise of mine**:
  I had told this reviewer that MVP_SPEC.md says exceptions live in
  `ckptplan.errors`; the reviewer found zero occurrences of the word "errors"
  anywhere in MVP_SPEC.md and I verified this myself directly (`grep -n
  "errors" MVP_SPEC.md` → no output). `errors.py` is therefore an
  implementer judgment call (reasonable and minimal — keeps exceptions out of
  the data-schema file, adds no subpackage), not a spec-grounded one as I'd
  claimed; recorded honestly below rather than left uncorrected. Also
  suggested (fixed, cheap) clarifying `CheckpointableBlock`'s docstring that
  `frozen=True` protects field reassignment only, not the referenced module's
  internal mutability — already true of the design, just undocumented.

### Graphify update result

Ran the incremental `--update` flow (per `references/update.md`) against
  project root — explicitly not the
session's unrelated shell working directory. `detect_incremental` found 5
changed files, all code (`ckptplan/__init__.py`, `api.py`, `errors.py`,
`types.py`, `tests/unit/test_declare_blocks.py`) — code-only, so semantic
extraction (LLM/subagents) was correctly skipped entirely; only structural
AST extraction ran. Merged via `build_merge` (replaces re-extracted files'
nodes rather than duplicating): **98 → 140 nodes, 105 → 189 edges** (42 new
nodes, 84 new edges — `declare_blocks`, `CheckpointableBlock`,
`BlockDeclarationError`, `api.py`/`errors.py`/`types.py` and their docstrings,
every new test function). Graph health check: **OK, no dangling/missing/
collapsed edges** this time (clean AST-only extraction, unlike the earlier
semantic-extraction round which had 6 dangling edges). `graphify-out/graph.json`
and `GRAPH_REPORT.md` were regenerated and committed as part of this slice's
bookkeeping; `cache/`/`cost.json`/`manifest.json` remain git-ignored as before.

## Completed (implementation phase — repository-foundation slice)

Implemented exactly the five items authorized, and nothing else: (1) a minimal
`ckptplan` package skeleton (`ckptplan/__init__.py` with `__version__`,
`ckptplan/py.typed` for PEP 561); (2) packaging/dev configuration
(`pyproject.toml` — `requires-python = ">=3.10,<3.13"` and
`torch>=2.5.0,<2.14.0"`, matching `MVP_SPEC.md` §1's claimed range exactly;
`[project.optional-dependencies].dev` with pytest; `README.md`; `.gitignore`);
(3) a pytest test harness (`tests/__init__.py`, `tests/conftest.py`,
`tests/unit/`, `tests/integration/`, `tests/correctness/`, each a real package;
`[tool.pytest.ini_options]` with `testpaths=["tests"]`, `pythonpath=["."]`); (4)
the tiny deterministic sequential-model fixture in `tests/conftest.py`
(`TinySequentialModel`, `build_tiny_sequential_model`, `build_tiny_batch`) —
verified to match `MVP_SPEC.md` §11.1's fixture text exactly: 4 blocks, each
`nn.Sequential(nn.Linear(64,64), nn.ReLU())`, `float32`, CPU, seed 0, batch
shape `(8, 64)`; (5) a smoke test (`tests/unit/test_smoke.py`) proving the
package imports, its version string is internally consistent with
`pyproject.toml` (added during review, see below), and the fixture produces
bit-identical output across two independently constructed model+batch pairs.

Deliberately not implemented (per instruction): `declare_blocks`,
`profile_blocks`, `plan_checkpoints`, `validate_plan`, `apply_plan`,
`run_benchmark`, `CheckpointedSequential`, `BlockProfile`, `CheckpointPlan`, or
any other part of `MVP_SPEC.md` §4/§5's public API. Also not pre-created:
ARCHITECTURE.md's proposed `profiling/`, `planning/`, `application/`,
`reporting/`, `benchmarks/` subpackages and `api.py`/`types.py` — a deliberate
choice (see "Remaining Blockers" below), not an oversight.

### Commands run and results

- `.venv/bin/pip install "torch>=2.5.0,<2.14.0" --index-url https://download.pytorch.org/whl/cpu`
  → installed torch 2.13.0 (CPU wheel), inside `.venv` (Python 3.11.16 via
  Homebrew — the machine's default `python3` is 3.14, outside the supported
  range, so a compatible interpreter was located and used instead).
- `.venv/bin/pip install -e ".[dev]"` → succeeded; `pip show ckptplan` confirms
  an editable install pointing at this directory.
- `.venv/bin/pytest -v` → **3 passed, 0 failed**, run twice (once before, once
  after the review-driven fix below), both green.

### Reviewer findings and resolutions

Three independent read-only reviewer agents ran in parallel (spec/acceptance,
correctness/test-quality, packaging/architecture-boundary). Consolidated
findings and resolutions:

- **No defects found in the delivered files themselves** by any reviewer — the
  fixture matches `MVP_SPEC.md` §11.1 exactly, the test-harness import wiring
  (`pythonpath`, package `__init__.py` files, `from tests.conftest import ...`)
  is coherent and verified working, packaging metadata matches `MVP_SPEC.md`
  §1 exactly, and no profiling/planning/application/benchmarking concepts leak
  into `ckptplan/`.
- **Fixed:** the correctness reviewer flagged that
  `test_package_imports_and_reports_a_version` only checked the version string
  was a non-empty `str`, not that it matched `pyproject.toml`'s declared
  version — allowing future drift between the two declarations to go
  undetected. Fixed by asserting
  `ckptplan.__version__ == importlib.metadata.version("ckptplan")` in that
  test; reran the full suite (3 passed) to confirm.
- **STATE.md itself was stale (caught independently by two reviewers):** at
  review time, "Current Task" still pointed at a non-existent "In Progress"
  section and "Completed"/"Next After Current Task" hadn't been updated for
  this slice at all. Resolved by this rewrite.
- **Flagged, not resolved unilaterally (see "Remaining Blockers"):** whether a
  git repository should be initialized for this project, and whether
  ARCHITECTURE.md's remaining proposed subpackages/`api.py`/`types.py` should
  be pre-created as empty stubs now. Both reviewers treated these as
  legitimate open judgment calls rather than defects; left for the human/next
  task rather than decided silently.

## Completed (repository initialization)

Per explicit senior approval, initialized a git repository via `git init -b main`
in the project root, with **no remote added**
and **no subpackage stubs created** (both explicitly excluded from this step).

Verified before acting: `git rev-parse --show-toplevel` from this directory,
run *before* `git init`, reported `/Users/ryanabraham` — confirming this
directory previously fell under the pre-existing, unrelated `~`-rooted repo
flagged in the prior round (an empty repo with no commits, not something to
build on). No `.git` existed in this directory yet.

Verified after acting:

- `git rev-parse --show-toplevel` → the project root
  exactly — the new repository's root is this project directory, not a parent
  or the pre-existing `~` repo.
- `git branch --show-current` → `main`.
- `git remote -v` → empty (no remote configured).
- `git status` → all project files correctly appear as untracked (nothing
  staged, nothing committed yet); `.venv/`, `__pycache__/`, `ckptplan.egg-info/`,
  and `.pytest_cache/` correctly do **not** appear, confirming `.gitignore` is
  effective from the very first `git status`.

**No commit was made.** Only initialization was requested; per standing
practice, commits are made only when explicitly asked. The working tree is
fully untracked at this point.

## Completed (specification phase)

`MVP_SPEC.md` Revision 3 was patched to Revision 3.1 to fix one remaining blocking
defect found in review: `validate_plan`'s documented defense-in-depth eligibility
check ("any block with `checkpointed=True` is one §10 requires excluded... —
defense-in-depth re-check") was unimplementable, because neither `blocks`
(`CheckpointableBlock` carries only `block_id, order, module`) nor `plan.decisions`
(`CheckpointDecision` carried only `block_id, checkpointed`) gave `validate_plan`
any eligibility data to check against.

Fixed by: (1) adding `eligible_for_checkpoint: bool` and
`exclusion_reason: Optional[ExclusionReason]` to `CheckpointDecision`; (2) every
planner now builds its `decisions` tuple through one new shared helper,
`_build_decisions` (§8.6), which copies these two fields verbatim from the source
`BlockProfile` — no planner re-derives or leaves them blank; (3) `validate_plan`
was rewritten (§9.3) with a dedicated `_validate_decisions_structure` function that
checks, directly from `plan.decisions` (no new dependency on `blocks` internals or
`profiles`): `checkpointed=True` paired with `eligible_for_checkpoint=False`
(unsafe), both directions of eligibility/exclusion-reason inconsistency, and — now
individually distinguishable, where before there was one generic length/position
check — missing, duplicate, reordered, and unknown decision block IDs; (4)
`profile_fingerprint`'s inputs were extended so `compute_profile_fingerprint` now
takes both `profiles` and `decisions`, hashing the decision-level eligibility echo
alongside the profile fields, so a decision hand-edited independently of its
source profile changes the fingerprint; (5) `plan_format_version` bumped
`"3.0"` → `"3.1"` (breaking: two new required `CheckpointDecision` fields).

Five new tests were added to §11.1 (eligibility copy-through for every planner;
`validate_plan` rejecting the unsafe checkpointed+ineligible case; both
eligibility/exclusion-reason inconsistency directions; all four structural
decision-list tampering cases individually; `compute_profile_fingerprint`
detecting decision-level tampering). The document explicitly discloses one
residual, accepted limitation this patch does not claim to close: `validate_plan`
still cannot detect a decision tampered *consistently* (eligibility, exclusion
reason, and checkpointed flag all changed together) without cross-referencing the
original `profiles` via the manual auditing tool — closing that would require
widening `validate_plan`'s signature to accept `profiles`, a larger change than
this patch, and is flagged in §14 for explicit sign-off rather than silently
addressed. No production source files were created; Graphify was not run.

## Files Changed

**This round (declare_blocks slice + baseline commit + graphify update):**

- `ckptplan/api.py` (new): `declare_blocks`.
- `ckptplan/types.py` (new): `CheckpointableBlock`.
- `ckptplan/errors.py` (new): `BlockDeclarationError`.
- `ckptplan/__init__.py` (modified): re-exports the three above.
- `tests/unit/test_declare_blocks.py` (new): 22 tests.
- `.gitignore` (modified): added `graphify-out/cache/`, `cost.json`,
  `manifest.json`, `.graphify_*` exclusions.
- `graphify-out/GRAPH_REPORT.md`, `graphify-out/graph.json` (modified,
  committed): regenerated by the incremental update above.
- `STATE.md` (this update).
- Two commits: `6ddf5b8` (baseline: spec, foundation, tests, STATE.md, graph
  outputs) and `112e632` (declare_blocks implementation + tests). No remote
  added at any point.

**Prior round (repository initialization, now folded into the baseline
commit above):**

- `.git/` (new): repository initialized via `git init -b main`, no remote.

**Earlier round (repository-foundation implementation):**

- `ckptplan/__init__.py` (new), `ckptplan/py.typed` (new).
- `pyproject.toml` (new), `README.md` (new), `.gitignore` (new).
- `tests/__init__.py`, `tests/conftest.py`, `tests/unit/__init__.py`,
  `tests/unit/test_smoke.py`, `tests/integration/__init__.py`,
  `tests/correctness/__init__.py` (all new).
- `STATE.md` (this update — senior decisions recorded, then this implementation
  slice's completion, files, commands, and reviewer findings).
- `MVP_SPEC.md`, `ARCHITECTURE.md`: not modified this round.
- `.venv/` (new, local-only, git-ignored): Python 3.11 virtualenv with `torch`
  and `ckptplan` (editable) installed, used to run the commands above.

**Prior round (specification patch, unchanged from last entry):**

- `MVP_SPEC.md` (revised in place — "Revision 3.1").
- `ARCHITECTURE.md`: not modified that round either.

## Validation Performed (prior round — specification patch)

No PyTorch source re-verification was needed — this patch is entirely about
ckptplan's own internal data flow (what `validate_plan` receives and checks), not
a PyTorch API claim. Verification was a targeted consistency pass: grepped every
`plan_format_version` literal in the document to confirm all read `"3.1"`
consistently (found and fixed two that still said `"3.0"`/an old signature
reference); confirmed `CheckpointDecision` is defined exactly once and
`_validate_decisions_structure`/`_build_decisions` are each referenced
consistently; counted code-fence markers to confirm they balance after every
edit. No code was run that round — it was a specification-only deliverable. (See
"Commands run and results" above for this round's actual code execution.)

## Remaining Blockers / Unresolved Decisions (Senior Review)

**Resolved this round:**

1. ~~No git repository initialized.~~ **Resolved:** approved and done — see
   "Completed (repository initialization)" above. Repository root verified as
   exactly this project directory; branch `main`; no remote; no commit yet
   (not requested).

**Still open, carried forward:**

2. **ARCHITECTURE.md's remaining proposed layout (`profiling/`, `planning/`,
   `application/`, `reporting/`, `benchmarks/` subpackages) remains
   deliberately not pre-created as empty stubs.** `api.py` and `types.py`
   themselves now exist (with real content, for `declare_blocks`), so this
   item narrows to just the five out-of-scope subsystem subpackages.

**New this round, flagged by the reviewer agents:**

3. **`ckptplan/errors.py`'s location is an implementer judgment call, not a
   spec-grounded one — correcting a claim I made to a reviewer.** ARCHITECTURE.md's
   proposed layout names only `api.py`/`types.py` at the top level, and
   MVP_SPEC.md contains no reference to a `ckptplan.errors` module anywhere
   (verified directly: `grep -n "errors" MVP_SPEC.md` returns nothing). Keeping
   exceptions in their own module is reasonable and minimal, but should be
   confirmed as a deliberate deviation from the literal proposed layout, not
   assumed pre-approved.
4. **Overlapping block-subtree declarations (a parent module and its own
   nested submodule declared as two separate blocks) are not governed by any
   invariant in this slice.** They are only accidentally rejected when the
   child happens to have a buffer (via a misleadingly-worded buffer-collision
   error), and silently accepted otherwise. Not one of the 9 required
   behaviors, so not implemented — flagged for a future slice or explicit
   "not needed" sign-off rather than left undisclosed.

`MVP_SPEC.md` §14 separates the specification-phase items into tiers. All items
from the first two review rounds remain resolved/approved. That round's
additions were:

**Disclosed, accepted residual limitation (flagged for explicit sign-off, not a
newly discovered defect):** `validate_plan` cannot catch a self-consistently
tampered decision without independently auditing against the original `profiles`
via `compute_profile_fingerprint` — it does not do this automatically, by the
already-approved design choice not to require `profiles` at validation time.
Confirm this is acceptable, or direct that `validate_plan` be widened to accept
`profiles`.

**Provisional parameters needing empirical calibration during implementation
(not review-blocking, unchanged from last round):** the `activation_bucket_bytes`
default (1 MiB), the `dp_scale_guard_cells` default (5,000,000), and the
`toy-transformer` calibration script's own search parameters.

**New design judgment calls made while executing this revision, recommended for
explicit confirmation:**

1. The exact `profile_fingerprint` field inclusion/exclusion list (planning
   values in; variance/std and run metadata out) is one reasonable line, not the
   only defensible one.
2. Greedy fallback (not an adaptive-bucket retry) was chosen for the sub-bucket
   false-infeasibility fix, for simplicity and because it needs no new tunable
   parameters.
3. `ExecutionSignature` is exposed as an explicit `CheckpointPlan` field, not only
   folded into the fingerprint hash — a deliberate debuggability trade-off (a
   slightly larger serialized plan) beyond the letter of the governing correction.
4. `exclusion_reason="shared_mutable_buffer"` is reserved on the enum but
   structurally unreachable in v0.1 (such blocks are rejected earlier, at
   `declare_blocks`) — kept for forward compatibility rather than omitted.
5. `plan_format_version` bumped to `"3.1"` (minor), not `"4.0"` (major) — this
   patch's schema change is narrower in scope than Revision 3's, but is still a
   breaking change to the serialized format. Confirm this versioning granularity.

## MVP Direction

The first release should support models represented as an ordered sequence of explicit checkpointable blocks. It should profile those blocks, choose a subset to checkpoint under a target memory constraint, apply the selection through supported PyTorch checkpoint APIs, and report measured memory and runtime results.

The MVP must compare against:

1. no checkpointing;
2. checkpointing every eligible block;
3. a simple uniform checkpointing strategy;
4. the planned strategy.

## Required Evidence

- Forward outputs and gradients remain correct within declared numerical tolerances.
- Peak GPU memory is measured using a reproducible protocol.
- Step time or throughput includes warm-up and repeated trials.
- Profiling overhead is reported separately from steady-state training performance.
- Results distinguish estimated savings from measured savings.
- Claims such as "optimal," "first," and "model-agnostic" are not used publicly without evidence.

## Constraints

- Primary development machine: MacBook Air M2 with 8 GB unified memory.
- Local work should use CPU and small deterministic models.
- CUDA validation and benchmarks may run on a Modal A10G with 24 GB VRAM.
- The core library must not require paid API services.
- GPU expenses should be controlled with short, reproducible benchmark jobs.

## Decisions

- Start at module/block granularity, not arbitrary tensor-level autograd scheduling.
- Use Python first. Rust or C++ requires a demonstrated performance or integration need.
- Prefer supported public PyTorch APIs over fragile internal hooks.
- Treat scheduling estimates as a cost model that must be checked against real measurements.
- Optimize for a credible, tested systems project rather than the broadest feature set.
- Use the project-scoped Graphify integration after a meaningful code skeleton exists; prefer bounded graph queries and incremental updates over repeated full-repository reads.

## Open Questions

- Which model structures can be wrapped safely without changing user model code?
- Should v0.1 target only sequential stacks or also repeated transformer blocks?
- How will activation lifetimes and peak memory be estimated accurately at block granularity?
- Is the initial scheduler a dynamic program, greedy heuristic, or both?
- How will shared parameters, stochastic layers, stateful modules, and nested outputs be handled?
- Which small model is appropriate for CI, and which larger models are appropriate for A10G benchmarks?

## Collaboration Protocol

- The human project owner approves scope and major design decisions.
- The implementation agent owns execution following the project owner's
  explicit direction; work may resume later from this file.
- The reasoning agent also owns architecture reasoning and coordinates independent
  read-only reviewers for each completed slice.
- Project assistants should read this file and `ARCHITECTURE.md` before proposing or making project changes.
- Both assistants should also follow `WORKFLOW.md` for token-efficient handoffs.
- Record accepted decisions here before relying on them in implementation.
- Do not overwrite another contributor's active changes; use focused branches or clearly separated tasks.
- Every completed feature requires tests and a short update to this file.

## Next After Current Task

**Proposed next slice:** implement plan application and validation using the
shared execution helpers. CUDA activation measurement, reporting, and
benchmarks remain deferred.

## Completed (Revision 3.4 pure planning)

`plan_checkpoints(profiles, blocks, ...)` now validates correspondence,
assembles execution signatures, computes parameter aliases and fingerprints,
and implements deterministic greedy, dynamic-programming, uniform, and
baseline planners with infeasibility policies. Review findings were resolved:
timing-only profiles with unavailable activation bytes now reach the documented
`TimingOnlyProfileError`; device indices are compatible when the profile omits
an index; and the public fingerprint helper rejects malformed decision lists.
Verification: `.venv/bin/python -m pytest -q` passes **74 tests** with 4
environment/API warnings; `git diff --check` passes. The three independent
reviews found no remaining algorithmic or specification blockers. Incremental
Graphify code update produced **336 nodes and 810 edges**.

## Completed (boundary correctness re-run against the fixed `run_benchmark`, A10G-verified)

Re-ran `benchmarks/modal_boundary_correctness.py` on Modal against the now-fixed
`run_benchmark` (the shared-parameter self-comparison and OOM-handler
mis-indentation defects recorded above are both corrected in the current tree).
This directly answers the question left open in the "Correctness Evidence —
CORRECTED" section above: whether co-locating the no-checkpoint reference with
the checkpointed container at this 1.2B-parameter scale actually OOMs.

**Exact command run:** `modal run benchmarks/modal_boundary_correctness.py`
(plain `python benchmarks/modal_boundary_correctness.py` does *not* work — the
script has no `__main__` guard and depends on Modal's CLI to invoke the
`@app.local_entrypoint()`; running it as a bare Python script is a silent no-op,
exits 0, and prints nothing. This was verified directly during this task before
switching to `modal run`.)

**Configuration (unchanged from the script, all on one A10G):** 24-layer
`nn.TransformerEncoderLayer` stack, hidden size 2048, 16 heads, feedforward
8192, dropout 0.0, sequence length 512, batch size 1, float32, seed 0.
`checkpoint_all` planner (`target_kind="activation_budget_bytes",
target_value=0`), `check_correctness=True`, `correctness_rtol=1e-3`,
`correctness_atol=1e-5`, 2 warm-up / 3 measured trials.

**Result (real Modal output, not fabricated), also saved to
`benchmarks/boundary_correctness_result.json`:**

```json
{
  "parameters": 1208598528,
  "profiled_activation_total": 1108279296,
  "correctness_checked": true,
  "correctness_passed": true,
  "max_output_diff": 0.0,
  "max_grad_diff": 0.0,
  "oom": false,
  "peak_allocated_bytes": 9702605824
}
```

**This is a genuine, non-null gradient-correctness pass, not a repeat of the
prior defects:** `correctness_passed` is the boolean `true` (not `null`), and
`max_grad_diff` is populated as `0.0` (not `null`), both output and gradient
tolerances (`rtol=1e-3, atol=1e-5`) pass, and `oom: false` confirms the result
came from the normal completion path, not the OOM handler — under the fixed
code, the OOM handler can only ever leave `correctness_passed` as `None`, never
`True`, so a `True` value is only reachable through the real backward-pass
comparison. This resolves the open question from the "Correctness Evidence —
CORRECTED" section: **no co-location OOM occurred** at this configuration
(seq_len 512, batch 1); the no-checkpoint reference and the checkpoint-all
container both fit and both produced comparable gradients. Per-container
isolation (`benchmarks/modal_isolated_correctness.py`) was therefore not
needed for this run and was not executed.

One caveat worth stating plainly: `max_grad_diff` and `max_output_diff` both
landing at exactly `0.0` (not merely within tolerance) is consistent with
non-reentrant checkpoint recomputation replaying identical deterministic ops in
identical order on this model (no dropout, `preserve_rng_state=True`), matching
the same zero-diff pattern already observed and explained on CPU in
`tests/unit/test_run_benchmark.py`. It is expected exactness for this
architecture, not a suspicious result.

**Runtime and cost.** Wall-clock time for the whole `modal run` invocation
(image resolution from cache, mount creation, function execution, and
teardown) was **18.46 s** end-to-end (`time` around the command). The image
was already built from a prior run, so this is effectively pure function
execution plus Modal orchestration overhead, not a cold image build. Using
Modal's published A10G rate of $0.000306/s ($1.102/hr) as a bound on the whole
wall-clock window: **≈$0.0056, well under one cent**, and almost certainly an
overestimate since it includes non-GPU-billed orchestration time around the
actual container execution. This is an **estimate**, not a Modal billing
export — precise per-container billed seconds were not independently queried
from the Modal dashboard.

**CPU test suite:** `.venv/bin/pytest -q` → **159 passed, 0 failed**, 4
environment/API warnings (NumPy absent in the local torch wheel, quantized
constructor deprecation, sparse-invariant-check guidance, compressed-sparse
beta notice) — same warning categories as prior rounds, not new failures.

**Remaining blockers/caveats, stated plainly:**

1. This result is specific to seq_len 512 / batch 1 / checkpoint_all. It does
   not by itself confirm the five-planner matrix's gradient correctness on
   real A10G hardware under the fixed code — `benchmarks/modal_matrix.py` has
   still not been re-run since the fix, per the "Pending" note under the
   original "Fix applied" section above.
2. `oom_boundary_a10g.json`'s harder configuration (seq_len 4096, batch 4)
   still has no possible no-checkpoint reference at all — no-checkpoint OOMs
   even in isolation there, so no correctness comparison can exist for that
   point, isolated or otherwise.
3. Cost figure above is a wall-clock-based estimate against Modal's published
   rate, not a reconciled billing statement.

## Completed (five-planner matrix re-run against the fixed `run_benchmark`, A10G-verified)

Re-ran `benchmarks/modal_matrix.py` on Modal to close the "Pending" item left
open above: whether all five planner configurations in the matrix actually
pass real gradient correctness on A10G hardware under the fixed
`run_benchmark`, not just the single `checkpoint_all` / seq_len 512 point
already verified.

**Bug found and fixed before running.** `measure()`'s per-planner result dict
only included `correctness_passed`, `oom`, and `error_message` — it never read
`result.correctness_checked`, `result.correctness_max_abs_output_diff`, or
`result.correctness_max_abs_grad_diff` off the returned `BenchmarkResult`. As
written, the script could not have reported `max_grad_diff` at all, which
would have made per-planner gradient-correctness verification impossible from
its output. Added the three missing fields to the appended dict; no other
behavior changed (tolerances, planners, model config, and the benchmark call
itself are untouched).

**Exact command run:** `modal run benchmarks/modal_matrix.py` (again via
Modal's CLI entrypoint mechanism, not bare `python`, for the same reason
recorded in the boundary-correctness section above — the script has no
`__main__` guard).

**Configuration (unchanged from the script, all five planners on one A10G,
single container, sequential):** 24-layer `nn.TransformerEncoderLayer` stack,
hidden size 2048, 16 heads, feedforward 8192, dropout 0.0, **sequence length
2048**, batch size 1, float32, seed 0 — 1,208,598,528 parameters. Profiled
activation total 4,433,117,184 bytes; `uniform`, `greedy`, and
`dynamic_programming` targeted 50% of that (2,216,558,592 bytes budget).
5 warm-up / 20 measured trials per planner, `check_correctness=True`,
`correctness_rtol=1e-3`, `correctness_atol=1e-5`.

**Per-planner results (real Modal output, saved verbatim to
`benchmarks/matrix_a10g_result.json`, replacing the pre-fix, defect-1-tainted
artifact of the same name):**

| planner | correctness_checked | correctness_passed | max_output_diff | max_grad_diff | oom | peak_allocated_bytes | latency_ms_mean |
|---|---|---|---|---|---|---|---|
| no_checkpoint | false | null | null | null | false | 9,896,961,536 | 1639.32 |
| checkpoint_all | true | **true** | 0.0 | **7.105e-15** | false | 9,887,220,736 | 2217.86 |
| uniform | true | **true** | 0.0 | **6.217e-15** | false | 9,887,220,736 | 1978.76 |
| greedy | true | **true** | 0.0 | **5.329e-15** | false | 9,887,220,736 | 1996.41 |
| dynamic_programming | true | **true** | 0.0 | **5.329e-15** | false | 9,887,220,736 | 2032.46 |

`no_checkpoint`'s correctness fields are correctly `null`/`false`: it *is* the
no-checkpoint reference plan, so `run_benchmark` skips the correctness check
for it by design (`correctness_checked = check_correctness and
plan.planner_name != "no_checkpoint"`), not because anything failed or was
unverified. Its own gradients are trivially the reference.

**All four checkpointing planners are genuine, non-null, non-OOM-path
gradient-correctness passes.** For each: `correctness_passed` is the boolean
`true` (never `null`), `max_grad_diff` is populated with a real nonzero
floating-point value (never `null`), both are below the `rtol=1e-3,
atol=1e-5` tolerance, and `oom: false` on every planner confirms every result
came from the normal completion path — under the fixed code the OOM handler
can only ever leave `correctness_passed` as `None`, so a `true` value could
only be reached through the real sequential backward-pass comparison
described in the fix above. Checkpoint plans (`plan_id`,
`predicted_activation_bytes_after`, `predicted_recompute_time_upper_bound_ms`)
and measured memory (`peak_allocated_bytes`, `peak_reserved_bytes`) are
recorded per planner in the saved JSON.

**Note on the diff magnitudes.** Unlike the boundary-correctness run (seq_len
512), where `max_grad_diff` landed at exact `0.0`, this run's four checkpointed
planners show small nonzero diffs (5.3e-15 to 7.1e-15) — consistent with
ordinary floating-point non-associativity from checkpoint recomputation at a
larger sequence length (2048 vs. 512), not a correctness failure; all four are
many orders of magnitude under the declared tolerance.

**Runtime and cost.** Wall-clock time for the whole `modal run` invocation
(image build/resolution, mount creation, one profiling pass, five
plan+benchmark cycles at 5 warmup + 20 measured trials each, and teardown):
**5 min 2.9 s** end-to-end (`time` around the command). Using Modal's
published A10G rate of $0.000306/s ($1.102/hr) as a bound on the whole
wall-clock window: **≈$0.0925 (about 9 cents)**. This is a wall-clock-based
**estimate**, not a reconciled Modal billing export, and likely a slight
overestimate since it includes non-GPU-billed orchestration time (image
resolution, mount upload, container teardown) around the actual GPU
execution.

**CPU test suite:** `.venv/bin/python -m pytest -q` → see result recorded
below in this same task round.

**Remaining blockers/caveats, stated plainly:**

1. This is a single run per planner (no repeated seeds/trials across separate
   Modal invocations), so it establishes that this particular execution
   passed at these tolerances — it is not a statistical guarantee across
   arbitrary reruns, though the mechanism (deterministic recomputation,
   `preserve_rng_state=True`, no dropout) makes flakiness unlikely.
2. `no_checkpoint`'s correctness fields are `null` by design, not by
   omission — flagged explicitly here so it is not mistaken for an
   unverified planner.
3. Cost figure is a wall-clock estimate against Modal's published rate, not a
   billing-dashboard reconciliation.
4. `oom_boundary_a10g.json`'s harder configuration (seq_len 4096, batch 4)
   remains untouched by this run and still has no possible no-checkpoint
   reference at all, per the caveat already recorded above.
