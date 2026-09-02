<p align="center">
  <img src="assets/ckptplan-wordmark.svg" alt="ckptplan logo" width="420">
</p>

# ckptplan

Python library for cost-aware activation checkpoint placement using GPU memory
profiling and dynamic programming for PyTorch training.

Gradient checkpointing trades compute for memory: a checkpointed block discards
its activations during the forward pass and recomputes them during the backward
pass. Checkpointing *everything* is the usual default and is rarely what you
want — it pays the maximum recompute cost for memory you may not need to save.
`ckptplan` measures each block, then picks the subset to checkpoint that meets
your memory budget at the lowest estimated recompute cost.

**Status: v0.1.0rc1 is available on PyPI.** The full v0.1 API works, is covered
by a 163-test CPU suite, and its four
checkpointing planners have passed genuine gradient-correctness checks on real
A10G hardware — see [Verified evidence](#verified-evidence). No memory-saving
or throughput percentage is claimed as a gate; see
[Measured, and not claimed](#measured-and-not-claimed).
[`MVP_SPEC.md`](./MVP_SPEC.md) (Revision 3.4) is the accepted design;
[`STATE.md`](./STATE.md) tracks progress and open blockers;
[`CHANGELOG.md`](./CHANGELOG.md) lists what this release candidate includes and
the metadata decisions still awaiting the maintainer's confirmation.

## Built with

- **Python** — public PyTorch API, model profiling, application, and benchmarks
- **PyTorch** — tensor execution, autograd, and gradient checkpointing
- **C++20** — standalone deterministic planning core for native integrations
- **CUDA** — device-side activation accounting plus A10G validation
- **React + TypeScript** — local benchmark explorer for saved result JSON
- **Modal** — reproducible cloud GPU benchmark workflows
- **pytest** — CPU and correctness test suite
- **GitHub Actions** — Python/PyTorch compatibility CI

GitHub reports React source as **TypeScript** in its language bar. The dashboard
keeps its styling in TypeScript, while `.gitattributes` limits the language bar
to the three added implementation languages and the Python public API surface:
TypeScript, C++, CUDA, and Python. Supporting HTML, configuration, documentation,
and data remain available without affecting those percentages.

## Install

```bash
pip install ckptplan
```

For local development:

```bash
pip install -e ".[dev]"
```

Or, from a built wheel:

```bash
python -m build
pip install dist/ckptplan-*.whl
```

Python 3.10–3.12, PyTorch >=2.5.0,<2.14.0. Licensed under MIT (see
[`LICENSE`](./LICENSE)).

## The pipeline

Five calls, in order. A runnable version of exactly this is in
[`examples/end_to_end.py`](./examples/end_to_end.py).

```python
import torch
from ckptplan import (
    declare_blocks, profile_blocks, plan_checkpoints, apply_plan, run_benchmark,
)

device = "cuda"
model = torch.nn.ModuleList([
    torch.nn.TransformerEncoderLayer(
        d_model=64, nhead=4, dim_feedforward=256,
        dropout=0.0,          # a stochastic block cannot be checkpointed
        batch_first=True, device=device, dtype=torch.float32,
    )
    for _ in range(4)
])
example_inputs = (torch.randn(2, 32, 64, device=device),)

# 1. Declare the checkpointable blocks, in execution order.
blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])

# 2. Measure each block: isolated activation bytes and recomputation time.
profiles = profile_blocks(blocks, example_inputs, device=device, dtype=torch.float32)

# 3. Choose which blocks to checkpoint, under an activation budget.
activation_total = sum(p.activation_bytes_estimate or 0 for p in profiles)
plan = plan_checkpoints(
    profiles, blocks,
    target_kind="activation_budget_bytes",
    target_value=activation_total // 2,     # keep at most half the activations
    planner="dynamic_programming",
)

# 4. Build a runnable module that checkpoints exactly the selected blocks.
container = apply_plan(blocks, plan, example_inputs, None)
output = container(*example_inputs)

# 5. Optionally, measure it — including a gradient correctness check against
#    an equivalent no-checkpoint plan.
result = run_benchmark(
    blocks, plan, example_inputs, None, lambda out: out.float().square().mean(),
    device=device, dtype=torch.float32, check_correctness=True,
)
print(result.peak_allocated_bytes, result.step_latency_ms_mean, result.correctness_passed)
```

### What each step does

| Call | Returns | Notes |
|---|---|---|
| `declare_blocks` | `tuple[CheckpointableBlock, ...]` | Blocks must be disjoint module subtrees with unique ids. Never calls `forward()`. |
| `profile_blocks` | `tuple[BlockProfile, ...]` | Measures activation bytes (CUDA only) and genuine full-recomputation timing. Restores all caller-owned module state, including on error. |
| `plan_checkpoints` | `CheckpointPlan` | Deterministic: identical inputs give a bit-identical plan. Planners: `dynamic_programming`, `greedy`, `uniform`, `checkpoint_all`, `no_checkpoint`. |
| `apply_plan` | `CheckpointedSequential` | Reuses the original module instances and preserves parameter identity, so existing optimizers keep working. |
| `run_benchmark` | `BenchmarkResult` | Latency, peak allocated/reserved memory, and an optional correctness check against a no-checkpoint reference. |

Use `validate_plan` to check a serialized plan against a model before applying
it; it re-derives every block's execution signature and verifies the model
fingerprint.

## CPU is timing-only

**Activation-based planning requires CUDA.** On CPU, PyTorch exposes no
allocator counters equivalent to `torch.cuda.max_memory_allocated`, so
`profile_blocks` cannot measure activation bytes. Rather than invent a number,
it reports:

```python
profile.timing_only               # True
profile.activation_bytes_estimate # None
profile.activation_bytes_method   # None
```

and `plan_checkpoints` refuses those profiles outright:

```
TimingOnlyProfileError: activation-based planning requires real activation-byte
profiles; CPU timing_only profiles are not valid planner inputs
```

This is a deliberate guard: the planner optimizes recompute cost subject to a
*memory* constraint, and it will not pretend to satisfy a budget it cannot
measure. On CPU you can still use `declare_blocks` and `profile_blocks` for
timing, and `apply_plan`/`run_benchmark` work with any plan you already have.

`examples/end_to_end.py` runs on either device: on CUDA it completes all five
steps; on CPU it stops at step 3 and prints the reason.

```
$ python examples/end_to_end.py
device: cpu  (torch 2.13.0)

1. declare_blocks -> 4 blocks: ['layer0', 'layer1', 'layer2', 'layer3']

2. profile_blocks:
     timing_only               = True
     activation_bytes_estimate = None
     activation_bytes_method   = None
     forward_time_ms_mean      = 0.3812
     eligible_for_checkpoint   = True

3. plan_checkpoints -> TimingOnlyProfileError (expected on CPU)
     activation-based planning requires real activation-byte profiles; CPU
     timing_only profiles are not valid planner inputs
```

## Blocks that cannot be checkpointed

`profile_blocks` marks a block ineligible rather than silently producing wrong
gradients. A block is excluded when it is stochastic (dropout and friends —
recomputation would not reproduce the forward pass), stateful in a way
recomputation would re-apply, or has no differentiable output. The reason is
recorded in `profile.exclusion_reason` and carried into the plan.

## Verified evidence

### Progressive scaling on A10G

In a real A10G run with a 24-layer, 1.2B-parameter Transformer and sequence
length 2048, the dynamic-programming planner sustained batch size 4 while the
no-checkpoint baseline ran out of memory at batch size 4:

| batch size | no checkpoint | ckptplan-DP |
|---:|:---:|:---:|
| 1 | pass | pass |
| 2 | pass, 15.7 GiB | pass, 10.3 GiB |
| 4 | **OOM** | **pass, 11.8 GiB** |

This is a memory/compute tradeoff: DP made the larger workload feasible, while
its batch-4 step took about 8.6 seconds. The raw result is available in
[`benchmarks/progressive_scaling_result.json`](./benchmarks/progressive_scaling_result.json).

- **159 CPU tests pass** (163 including packaging/metadata checks added for
  this release), run via `.venv/bin/python -m pytest -q`, across Python
  3.10/3.12 and PyTorch 2.5.0/2.13.0 in CI.
- **All four checkpointing planners passed genuine A10G gradient-correctness
  checks.** `benchmarks/matrix_a10g_result.json` (24-layer, 1.2B-parameter
  transformer, seq_len 2048, batch 1, `rtol=1e-3, atol=1e-5`), re-run after
  fixing two defects in the original correctness harness (see `STATE.md`'s
  "Correctness Evidence — CORRECTED" section for the full defect history):

  | planner | correctness_passed | max_grad_diff |
  |---|---|---|
  | checkpoint_all | true | 7.105e-15 |
  | uniform | true | 6.217e-15 |
  | greedy | true | 5.329e-15 |
  | dynamic_programming | true | 5.329e-15 |

  Every value above is a real, non-null result from the normal completion
  path (`oom: false`) — not a null placeholder and not a value produced by the
  OOM-fallback path, which under the fixed code can only ever leave
  `correctness_passed` as `None`. A second, independent boundary run at
  seq_len 512 also passed with exact `max_grad_diff: 0.0`
  (`benchmarks/boundary_correctness_result.json`).
- `no_checkpoint`'s correctness fields (`correctness_passed`, `max_grad_diff`)
  are `null` **by design, not by gap**: it is the reference plan itself, so
  `run_benchmark` skips the correctness check for it rather than comparing it
  against itself.
- **Known, honest limitation — not glossed over:** at seq_len 4096 / batch 4,
  `no_checkpoint` itself OOMs on a single A10G
  (`benchmarks/oom_boundary_a10g.json`). `checkpoint_all` completes at that
  configuration, but **no correctness comparison can exist there**, because
  there is no baseline run to compare its gradients against — the reference
  itself cannot execute, isolated or otherwise. This is a hardware/harness
  ceiling at this model scale, not a defect in the checkpointing logic.

## Measured, and not claimed

Per MVP_SPEC.md §12.5, no release gate asserts a percentage of memory saved or a
bound on throughput overhead. Two things are worth stating plainly:

- **Reported, not gated:** memory reduction, step-time overhead, and the
  prediction gap. The profiler's additive per-block isolated activation estimate
  legitimately exceeds the measured whole-model peak reduction, because
  parameters, gradients, and allocator reuse dominate the end-to-end peak. That
  gap is reported, not corrected.
- **Gradient correctness is now verified for the two A10G configurations that
  can run at all** (seq_len 512 and seq_len 2048, see
  [Verified evidence](#verified-evidence) above) — the two defects that
  previously made every `correctness_passed` value in this repository
  meaningless (a shared-parameter self-comparison, and an indentation bug that
  routed the real comparison through the OOM handler) are both fixed, tested,
  and re-verified against real A10G runs. What remains unverified is
  correctness at configurations where no reference can execute at all — see
  the seq_len 4096 / batch 4 limitation above. See STATE.md's "Correctness
  Evidence" section for the full defect history.

`benchmarks/report.py` re-reports any saved benchmark JSON locally, for free:

```bash
python benchmarks/report.py benchmarks/matrix_a10g_result.json
```

## Development

```bash
pip install -e ".[dev]"
pytest -q
```

### Native C++ and CUDA

The dependency-free C++20 core implements the same five planner strategies as
the Python package. It is useful for embedding ckptplan's cost model in a native
runtime without importing PyTorch:

```bash
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release
cmake --build native/build
ctest --test-dir native/build --output-on-failure
```

CUDA support is opt-in because a CUDA toolkit and GPU are not available on
normal GitHub-hosted runners:

```bash
cmake -S native -B native/build -DCKPTPLAN_ENABLE_CUDA=ON
cmake --build native/build --target ckptplan_cuda_demo
./native/build/ckptplan_cuda_demo
```

See [`native/README.md`](./native/README.md) for the API boundary and current
integration status.

### React benchmark dashboard

The dashboard reads the repository's existing progressive, list, map, and
legacy top-level benchmark JSON shapes locally in the browser. Files never
leave the machine.

```bash
cd dashboard
npm install
npm run dev
```

Run `npm test` for schema normalization tests or `npm run build` for a static
production bundle. See [`dashboard/README.md`](./dashboard/README.md).

CI runs the Python suite on CPU across Python 3.10/3.12 and PyTorch 2.5/2.13,
plus the C++ tests and React test/build jobs
([`.github/workflows/ci.yml`](./.github/workflows/ci.yml)). The GPU benchmarks
under `benchmarks/` require Modal and an A10G and are run separately, never in
CI.

## Documents

- [`MVP_SPEC.md`](./MVP_SPEC.md) — accepted v0.1 design (Revision 3.4).
- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — component structure.
- [`STATE.md`](./STATE.md) — progress, decisions, and what is proven vs pending.
