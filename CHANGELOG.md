# Changelog

All notable changes to `ckptplan` are recorded here. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/); this project has not yet made
a stable release, so pre-1.0 versions may include breaking changes without a
major bump.

## [0.1.0rc1] - unreleased release candidate

First release candidate. Not yet tagged or published to PyPI — pending human
sign-off on the open questions below.

### Added

- Five-call public API: `declare_blocks`, `profile_blocks`, `plan_checkpoints`,
  `apply_plan` (via `validate_plan` + `CheckpointedSequential`), and
  `run_benchmark`.
- Four checkpoint planners: `checkpoint_all`, `uniform`, `greedy`, and
  `dynamic_programming`, plus the `no_checkpoint` baseline, all deterministic
  given identical inputs.
- CPU timing-only profiling and CUDA activation-byte + timing profiling
  (`profile_blocks`), with `TimingOnlyProfileError` raised rather than
  fabricating a memory budget on CPU.
- Plan validation (`validate_plan`) re-deriving execution signatures and
  checking the model fingerprint before a serialized plan is applied.
- `run_benchmark`: latency, peak allocated/reserved CUDA memory, throughput,
  and an optional gradient-correctness check against a `no_checkpoint`
  reference.
- `ckptplan.reporting.compare_results` and `benchmarks/report.py` for
  re-reporting saved benchmark JSON locally without touching a GPU.
- Package metadata for release: MIT `LICENSE`, PyPI classifiers, project URLs,
  and author/maintainer fields (see "Release notes" below for decisions made
  without explicit sign-off).

### Verified evidence

- **159 CPU tests pass** (`.venv/bin/python -m pytest -q`), covering block
  declaration, CPU/CUDA-path profiling logic, planning, plan
  validation/application, benchmarking, isolated correctness plumbing, and the
  report script, across Python 3.10–3.12 and PyTorch 2.5.0–2.13.0 in CI.
- **Genuine A10G gradient-correctness passes for all four checkpointing
  planners.** Re-run of `benchmarks/modal_matrix.py` after fixing two defects
  in the original correctness check (a shared-parameter self-comparison that
  made every diff structurally `0.0`, and an indentation bug that moved the
  gradient comparison into the OOM handler). Real, non-null `max_grad_diff`
  values at seq_len 2048 / batch 1, 24-layer / 1.2B-parameter transformer,
  `rtol=1e-3, atol=1e-5` (`benchmarks/matrix_a10g_result.json`):

  | planner | correctness_passed | max_grad_diff |
  |---|---|---|
  | checkpoint_all | true | 7.105e-15 |
  | uniform | true | 6.217e-15 |
  | greedy | true | 5.329e-15 |
  | dynamic_programming | true | 5.329e-15 |

  `no_checkpoint`'s correctness fields are `null` by design — it is the
  reference plan itself, not an unverified configuration.
- A second, independent boundary run at seq_len 512 / batch 1 also passed with
  exact `max_grad_diff: 0.0` (`benchmarks/boundary_correctness_result.json`).
- **Known, honest limitation:** at seq_len 4096 / batch 4, `no_checkpoint`
  itself OOMs on a single A10G (`benchmarks/oom_boundary_a10g.json`) —
  `checkpoint_all` completes there, but no correctness comparison can exist at
  that configuration because there is no reference run to compare against.
  This is a hardware/harness ceiling, not a gap in the checkpointing logic.
- Reported, not gated: peak-memory reduction, latency overhead, and the
  prediction gap (per `MVP_SPEC.md` §12.5 — this release asserts no percentage
  memory-saving or throughput-overhead claim).

### Release notes / decisions made on the maintainer's behalf

- **License: MIT.** No `LICENSE` file or license header existed anywhere in
  the repository before this change; MIT was chosen as the ecosystem default
  for a permissive PyTorch-adjacent library. **Needs explicit confirmation
  before tagging.**
- **Version: `0.1.0rc1`**, not `0.1.0`, since this pass prepares — but does not
  execute — the actual release. Bump to `0.1.0` when the open questions below
  are resolved and the maintainer decides to tag.
- **Project URLs** point at `https://github.com/ryamabra/ckptplan`, inferred
  from `git remote -v` (origin). Confirm this is the intended public
  repository before publishing.
- **Author/maintainer**: "Ryan Abraham", taken from `git config user.name`. No
  email was included in package metadata to avoid publishing one without
  explicit confirmation.
- **Python/PyTorch constraints unchanged** from the existing
  `requires-python = ">=3.10,<3.13"` and `torch>=2.5.0,<2.14.0"`, which match
  what CI actually exercises (Python 3.10/3.12 x PyTorch 2.5.0/2.13.0) and what
  the A10G evidence above was produced under (PyTorch 2.13.0).

### Open questions before tagging/publishing v0.1.0

- Confirm the MIT license choice (or supply a different one).
- Confirm the GitHub URL and whether docs/homepage should differ from the
  repository URL.
- `benchmarks/modal_matrix.py`'s five-planner run is a single trial per
  planner on one A10G run, not a repeated-seed statistical guarantee.
- The seq_len 4096 / batch 4 boundary has no possible correctness reference
  (see above) — decide whether this needs to be called out again in the
  library's top-level docs or is sufficiently disclosed here and in
  `STATE.md`.
- No `py.typed`/mypy CI check exists yet; typing is shipped (`py.typed`) but
  not verified in CI.
