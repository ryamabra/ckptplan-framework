# ckptplan v0.1 MVP Specification (Revision 3.4)

Status: approved for implementation. Revision 3.4 records the senior-approved
profiles-plus-blocks planning boundary. Revision 3.3 records the senior-approved
CPU profiling scope and strengthens profiling state safety without changing any
serialized-plan schema. Revision 3.1 was judged to contain two declaration-boundary
defects in `declare_blocks` — fixed as Revision 3.2. Revision 3 was judged to contain one remaining blocking defect —
`validate_plan`'s stated eligibility re-check was unimplementable with the data it
actually receives — fixed as Revision 3.1. Revision 2 was judged close but not yet
approved and had already applied 6 required corrections plus 5 approved design
judgments; Revision 1 was rejected with 25 numbered blockers, all addressed in
Revision 2. `ARCHITECTURE.md` required no further changes this round — nothing
here contradicts it.

## Revision 3.4 Notes (planning provenance boundary)

`plan_checkpoints` receives both `profiles` and `blocks`: profiles own measured
cost, eligibility, and execution-shape data; blocks own module structure,
parameter metadata, and parameter identity. It assembles `ExecutionSignature`
from profiles, computes `parameter_alias_groups` and `model_fingerprint` from
blocks plus that signature, and does not add model-provenance fields to
`BlockProfile` or leave required `CheckpointPlan` fields unbound. Before any
selection it requires equal non-empty lengths, unique and matching IDs, matching
order, and compatible device/dtype metadata.

## Revision 3.3 Notes (CPU profiling and state-safety clarification)

This focused clarification was approved before the CPU `profile_blocks` slice:

1. CPU profiles include real forward and full-checkpoint-recomputation timing
   using `time.perf_counter()`, non-reentrant checkpointing,
   `preserve_rng_state=True`, and checkpoint early stopping disabled. CPU remains
   `timing_only=True`; no CUDA activation measurement is fabricated.
2. Canonical chain walking, boundary conversion, and shape signatures live in
   private top-level `ckptplan/_execution.py`. Profiling and future application
   code must reuse them.
3. `device`/`dtype` are validation-only. `profile_blocks` never moves or casts
   caller-owned modules, parameters, buffers, or tensor leaves. A requested dtype
   must match all floating values. With `dtype=None`, a single execution dtype is
   inferred from actual floating values; ambiguous mixed floating dtypes raise a
   clear `ValueError` rather than producing a misleading profile.
4. State restoration is exact across success and exception: every submodule's
   individual `.training` flag, every named buffer value, and every pre-existing
   parameter gradient (including `None` versus a populated tensor) is preserved.
   Restoration assigns individual training flags directly; it must not call
   `module.train(original_root_flag)`, which would flatten a mixed train/eval tree.
   v0.1 supports registered buffers with strided (including quantized) and sparse
   COO layouts. Other sparse layouts are rejected during preflight before any
   caller module executes because PyTorch does not expose a uniform reversible
   metadata-restoration operation for them.
5. A block with no differentiable output remains ineligible. Its forward and
   full-recomputation timings are still genuinely measured by adding a private,
   zero-valued differentiable probe around (not into) the block output; this probe
   forces the checkpoint recomputation but contributes no block gradient or output.
   If every declared block lacks differentiable output, `profile_blocks` raises
   `NoDifferentiableOutputError` as before.

## Revision 3.2 Notes (this round's two corrections)

Both defects are in `declare_blocks` (§3) — no schema or serialized-plan-format
change, so `plan_format_version` stays `"3.1"`.

1. **The buffer-sharing terminology was too narrow.** Earlier revisions said
   "persistent buffer," but PyTorch's `persistent` registration flag controls
   `state_dict` serialization only. It does not make a buffer immutable or
   prevent a block's `forward` (including checkpoint recomputation) from
   mutating it. The safety rule therefore applies to every registered buffer
   returned by `named_buffers(recurse=True)`, including buffers registered with
   `persistent=False`. The implementation's original identity check was the
   safe behavior; Revision 3.2 makes the specification match it and adds
   explicit non-persistent and mixed-persistence regression tests. See §3,
   §10.3.
2. **Overlapping module subtrees were never rejected**, contradicting §2's own
   topology statement that declared blocks are "a linear sequence of
   **disjoint**... submodules" — that word existed in prose but was never a
   checked invariant. Declaring a parent module and one of its own descendants
   as two separate blocks previously succeeded (or was only accidentally
   caught, with a misleading error, if the two happened to share a buffer).
   Fixed by an explicit disjointness check: for every pair of declared blocks,
   their recursive `named_modules()` identity sets must not intersect. This
   catches both ancestor/descendant declarations and sibling wrappers sharing
   a descendant module, symmetrically and independent of declaration order.
   New §10.7. §2, §3
   updated to state this as a checked invariant, not just descriptive prose.

## Revision 3.1 Notes (this round's blocking correction)

`validate_plan(plan, blocks, example_inputs, example_kwargs)` §9.3 claimed a
defense-in-depth check: "any block with `checkpointed=True` is one §10 requires
excluded... — defense-in-depth re-check independent of whether `plan_checkpoints`
already excluded it." This was unimplementable: `blocks: Sequence[CheckpointableBlock]`
carries only `block_id, order, module`; `plan.decisions: tuple[CheckpointDecision, ...]`
carried only `block_id, checkpointed`; nothing accessible to `validate_plan` carried
eligibility data at all. Fixed by:

1. Adding `eligible_for_checkpoint: bool` and `exclusion_reason: Optional[ExclusionReason]`
   to `CheckpointDecision` (§5), copied verbatim from the corresponding `BlockProfile`
   by every planner via one shared helper (§8.6, new).
2. `validate_plan` now checks these fields directly from `plan.decisions` — no new
   dependency on `blocks` or `profiles` was needed (§9.3, rewritten).
3. `validate_plan` also now explicitly distinguishes missing/duplicate/reordered/
   unknown decision block IDs (previously folded into one generic length/position
   check) with individually identifiable error messages.
4. `profile_fingerprint`'s inputs are extended to include the decision-level
   eligibility echo, via `compute_profile_fingerprint(profiles, decisions)` (§9.1),
   so the manual auditing tool can detect a decision hand-edited independently of
   its source profile — see §9.1 and §14 for the honestly disclosed limit of this
   (it is not automatically checked by `validate_plan`, which still does not take
   `profiles`).
5. `plan_format_version` bumped `"3.0"` → `"3.1"`: `CheckpointDecision` gained two
   required fields, a breaking change to the serialized plan schema.

## Revision 2 Notes (what changed and why, for context)

The single largest change is architectural: **v0.1 no longer attempts to wrap an
arbitrary existing `nn.Module.forward`.** Revision 1's `apply_plan` implicitly
promised that undeclared model code before/between/after declared blocks would
"execute unchanged," which cannot be implemented without tracing or mutating the
original model (blocker 1). The fix chosen here is narrower and honest: **the
declared blocks must span the entire computation ckptplan manages**, composed by a
new library-owned `CheckpointedSequential` container (§9). This has a large,
beneficial side effect: `profile_blocks`, `validate_plan`, `apply_plan`, and
`run_benchmark` no longer need a `model` argument or any forward hooks at all,
because ckptplan's own definition of "the computation" is now exactly the declared
block chain — there is no separate arbitrary model execution to observe or diverge
from. Only `declare_blocks` still takes `model` (to check the blocks are real,
reachable submodules).

Every other blocker and open-question decision is folded into the sections below in
place, not as a changelog — a reader encountering this document cold should see one
coherent v0.1 design, not a diff. §14 lists what is now closed versus what remains
genuinely open or provisional.

## Revision 3 Notes (this round's 6 corrections)

Five design judgments from Revision 2 §14 are now explicitly approved unchanged:
dropping `model` from every public function except `declare_blocks`; using the
`"no_checkpoint"` plan's container over the same blocks as the sole correctness
reference; the `(j*m)//k` uniform-spacing formula; greedy-order DP repair; and
hard-rejecting `timing_only=True` profiles from activation planning via
`TimingOnlyProfileError`.

Six corrections were required:

1. **Final-block boundary handling (§6.1, §9.2).** `CheckpointedSequential.forward`
   called `_boundary_convert` after every block including the last, contradicting
   its own stated claim that the final output is returned raw — a final block
   returning, e.g., a `list` was incorrectly rejected. The identical bug existed in
   §6.1's chain-walk. Both are fixed the same way, by only converting the boundary
   when a subsequent block exists, and both now share one canonical chain-walking
   helper (`_run_block_chain`) so the rule is specified exactly once.
2. **`ExecutionSignature` (§9.1, §9.2, §9.3).** The fingerprint section previously
   claimed the entry-input signature alone "implicitly" covered every intermediate
   block boundary; it did not, because nothing re-executed the chain to derive
   intermediate shapes. Replaced with an explicit `ExecutionSignature` type (entry
   signature, every block's real input/output signature, block order/identity), a
   precisely specified derivation function built on `_run_block_chain` with the
   same state-preservation contract as §6.8, and an explicit description of when
   `profile_blocks` assembles it for free (from profiles it already computed)
   versus when `validate_plan` must re-derive it by actually running the chain once.
3. **`profile_fingerprint` now includes planning-relevant measurements (§9.1).** It
   previously excluded exactly the values (`activation_bytes_estimate`,
   `recompute_time_upper_bound_ms_mean`, eligibility state) that determine planner
   decisions, so a hand-edited or differently measured profile could retain the
   same fingerprint. Fixed by specifying the exact field list the fingerprint now
   covers, and what it deliberately still excludes.
4. **Explicit eligibility fields (§5, §6.3, §7.1, §10.2).** `BlockProfile` gains
   `eligible_for_checkpoint: bool` and `exclusion_reason: Optional[...]`, set
   explicitly at the point of each exclusion. Planner logic now reads
   `eligible_for_checkpoint` directly instead of inferring eligibility from
   `is_stateful`/`is_stochastic`/`warnings`.
5. **DP false infeasibility on sub-bucket blocks (§8.2).** Conservative floor
   bucketing can make every eligible block round to zero bucket-units, so the
   bucketed search reports infeasible even when the exact byte sum proves a real
   solution exists. Fixed by comparing the exact byte sum against the target before
   ever declaring infeasibility, and deterministically falling back to greedy
   selection when the bucketed search is the sole obstacle — recorded via a new
   `dp_fallback_reason` field, distinct from the existing `dp_repair_applied`
   mechanism (a different case: DP found a selection that then failed
   verification, rather than finding no selection at all).
6. **Runtime static-shape enforcement (§9.2, §9.4).** Construction-time validation
   alone does not stop a caller from later invoking the same, already-built
   container with a different-shaped input. `CheckpointedSequential.forward` now
   checks the cheap entry-boundary signature on every call, before running any
   block, raising `PlanIncompatibleError` immediately on mismatch.

---

## 0. PyTorch API verification

Unchanged from revision 1; no blocker touched PyTorch API facts. Repeated here for a
single reading surface.

Checked: `torch/utils/checkpoint.py` at tags `v2.4.0` through `v2.13.0`, `torch/cuda/memory.py` and `torch/cuda/__init__.py` at `v2.13.0` — https://github.com/pytorch/pytorch/blob/v2.13.0/torch/utils/checkpoint.py, https://github.com/pytorch/pytorch/blob/v2.13.0/torch/cuda/memory.py, https://github.com/pytorch/pytorch/blob/v2.13.0/torch/cuda/__init__.py — plus rendered docs (https://docs.pytorch.org/docs/stable/checkpoint.html) and PyPI metadata for `torch` (latest stable observed: `2.13.0`, 2026-07-08; `Requires-Python: >=3.10`; classifiers list 3.10–3.14).

Findings:

1. `checkpoint(function, *args, use_reentrant=None, context_fn=noop_context_fn, determinism_check="default", debug=False, **kwargs)` is stable `v2.4.0`–`v2.8.0`; `early_stop: bool = True` was added as an explicit keyword only at `v2.9.0`. `preserve_rng_state` is popped from `**kwargs` (default `True`) at every checked version, never a named parameter.
2. `use_reentrant=None` prints a `FutureWarning` and falls back to `True` at every checked version including `v2.13.0`. `use_reentrant=False` is the documented recommendation.
3. Non-reentrant (`use_reentrant=False`) vs. reentrant: non-reentrant records the autograd graph during the real forward, supports arbitrary nested output structures participating in autograd, supports `**kwargs`, has no requires-grad requirement on inputs/outputs, supports `torch.autograd.grad`, and stops recomputation early by default (`early_stop`). **v0.1 always calls `checkpoint(..., use_reentrant=False)` explicitly.**
4. `checkpoint_sequential` only accepts a single `Tensor` at input/segment boundaries and partitions into contiguous equal-length chunks — unusable for arbitrary block subset selection. **v0.1 calls `checkpoint()` per selected block instead.**
5. `torch.cuda.memory_allocated/max_memory_allocated/memory_reserved/max_memory_reserved/reset_peak_memory_stats/empty_cache` and `torch.cuda.synchronize` all exist with expected signatures at `v2.13.0`.
6. `set_checkpoint_early_stop(enable: bool)` (context manager) exists from `v2.4.0` onward; v0.1 uses it instead of the `early_stop` kwarg to stay compatible with PyTorch <2.9.

---

## 1. Supported Python and PyTorch versions

**Corrected per directive 19.** A specification cannot claim a version is "tested" before any CI exists. The published support range is narrowed to exactly what the CI plan (once implementation begins) commits to running, not a wider range chosen for "safety margin."

- **Python:** 3.10 and 3.12 are the two versions CI must run (floor and ceiling of the claimed range). 3.11 is claimed compatible by extrapolation (it sits strictly between two tested versions and PyTorch's own checkpoint code is unchanged across this range, per §0) but is **not** independently run in CI for v0.1. 3.13/3.14 are not claimed.
- **PyTorch:** 2.5.0 and 2.13.0 are the two versions CI must run. Versions strictly between are claimed compatible by extrapolation (§0 confirms the relevant API surface is identical across `v2.4.0`–`v2.13.0`) but are not independently run.
- **CI legs (exactly two, both required before v0.1 is released):** (Python 3.10, PyTorch 2.5.0, CPU wheel) and (Python 3.12, PyTorch 2.13.0, CPU wheel).
- **CUDA:** validated separately on a Modal A10G (24 GB) running (Python 3.12, PyTorch 2.13.0) with a matching CUDA build. This is not a CI leg (per `ARCHITECTURE.md`'s Compatibility Strategy, CUDA runs as a separate reproducible workflow); it is still required before v0.1 is released. No other GPU is tested or claimed.
- **Backends:** CUDA only for accelerated execution. ROCm, MPS, XPU are untested and unsupported.
- If, during implementation, only one CI leg can actually be stood up, the published range must be narrowed further to just that leg — the range in this document is a target for CI, not yet a fact.

---

## 2. Exact supported model topology

**Narrowed per directive 1.** v0.1 supports exactly one topology: **a linear sequence of disjoint, explicitly declared submodules that together constitute the entire computation ckptplan manages.** There is no supported non-block computation before the first block, between blocks, or after the last block, inside the region ckptplan executes.

A model is eligible if:

1. The user provides an ordered list of `(block_id, submodule)` pairs (§3), each `submodule` a distinct `torch.nn.Module` reachable from a root `model`.
2. No registered buffer is shared by identity between two declared blocks (§10.3), regardless of its `persistent` serialization flag. **Shared parameters are permitted** (§10.3, revised per directive 11). **Declared blocks must additionally be disjoint module subtrees: the recursive module-identity sets of two blocks may not intersect** (§10.7, new in Revision 3.2) — this was already implied by "disjoint" above but is now a checked invariant, not just descriptive prose.
3. Each declared block is called exactly once per forward pass, in declared order; a block's own internal loop (e.g., an RNN run for T steps inside one block) is fine because it is opaque to ckptplan.
4. **The first declared block's call signature is exactly the arguments the user will supply to the container** (§9) — there is no pre-block code. **The last declared block's return value is exactly what the container returns** — there is no post-block code. Any embeddings, heads, or loss computation the user needs must be called by the user's own code, outside the container, using the container's return value as an ordinary Python value. ckptplan does not see, profile, or checkpoint that code, and does not claim to run it.
5. Between blocks, block `i`'s return value becomes block `i+1`'s entire input, converted per the fixed boundary rule in §9.2. There is no side-channel: block `i+1` does not receive anything from block `i-1` or from the container's original call arguments, only from block `i`'s return value.
6. The block sequence, count, and identity are static across calls — no data-dependent branching.

Repeated transformer blocks are supported only by declaring each instance individually — no auto-repetition shortcut (§13).

**Rejected, not silently degraded:**

- Any model where meaningful computation must happen between/before/after declared blocks (the user must either fold that computation into an adjacent block's own `forward`, or accept that ckptplan only manages the sub-region expressible as pure block composition, calling the container as one step inside their own larger, ckptplan-unaware training step).
- Arbitrary/unstructured `nn.Module` graphs (no tracing is performed, so this is a documented user responsibility with the guardrails in §9.2, §9.4 — not something ckptplan proves).
- Branching control flow, `nn.DataParallel`/`DistributedDataParallel`-wrapped models, and everything `ARCHITECTURE.md`'s non-goals exclude.

---

## 3. Declaring ordered checkpointable blocks

Unchanged in mechanics from revision 1, with the parameter-sharing rule relaxed (§10.3) and buffer-sharing rule kept.

```python
from ckptplan import declare_blocks

blocks = declare_blocks(
    model=my_model,
    blocks=[
        ("encoder.layer0", my_model.encoder.layer0),
        ("encoder.layer1", my_model.encoder.layer1),
        ("encoder.layer2", my_model.encoder.layer2),
        ("encoder.layer3", my_model.encoder.layer3),
    ],
)
```

Rules, enforced at call time (`BlockDeclarationError` on violation):

- `block_id`: non-empty, unique `str`.
- List order **is** execution order; `CheckpointableBlock.order` is `0..n-1` from position.
- Every `submodule` must be reachable from `model` via `model.named_modules()` identity.
- No two entries reference the same module instance (`id(submodule)` uniqueness).
- **Declared blocks must be disjoint subtrees (new, Revision 3.2, §10.7):** the
  sets of module identities returned recursively by each declared module's
  `named_modules()` call may not intersect. This rejects ancestor/descendant
  declarations and sibling wrappers that share a descendant module, while
  still permitting distinct modules to share individual `nn.Parameter`
  instances. The check is symmetric and independent of declaration order and
  raises `BlockDeclarationError` naming both block IDs.
- No two entries' modules share a registered buffer instance (`id(b)`
  uniqueness across `named_buffers(recurse=True)`, including buffers registered
  with `persistent=False`; see §10.3). Shared **parameters** are permitted and are
  recorded, not rejected (§10.3).
- `declare_blocks` does not run the model and does not require example inputs.

`declare_blocks` returns an immutable `tuple[CheckpointableBlock, ...]`, the required input to every downstream stage.

---

## 4. Public API signatures

**Signatures changed per directives 1, 2, 3, 13, 14, 18, 22.** `model` is removed from every function except `declare_blocks` (§ "Revision Notes"). `example_kwargs` is threaded consistently everywhere `example_inputs` appears (directive 22). `memory_budget_bytes`/`memory_saving_fraction` are renamed to `activation_budget_bytes`/`activation_saving_fraction` (directive 3). `memory_bucket_bytes` is renamed to `activation_bucket_bytes` for the same reason. `recompute_time_ms_*` is renamed to `recompute_time_upper_bound_ms_*` (directive 13). The DP scale guard is an explicit, configurable parameter (directive 14), not a hardcoded constant.

```python
from __future__ import annotations

from typing import Any, Literal, Mapping, Sequence

import torch

from ckptplan.types import (
    BenchmarkResult,
    CheckpointableBlock,
    CheckpointPlan,
    BlockProfile,
    ExecutionSignature,
)

PlannerName = Literal["greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"]
TargetKind = Literal["activation_budget_bytes", "activation_saving_fraction"]


def declare_blocks(
    model: torch.nn.Module,
    blocks: Sequence[tuple[str, torch.nn.Module]],
) -> tuple[CheckpointableBlock, ...]:
    """Validate and freeze an ordered set of checkpointable blocks. See spec §3.
    The only function in the public API that takes `model` — see §"Revision Notes"."""
    ...


def profile_blocks(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    num_warmup: int = 3,
    num_trials: int = 10,
) -> tuple[BlockProfile, ...]:
    """Directly execute the declared block chain (no hooks, no `model`; see §6) to
    capture each block's real received inputs, then profile each block in isolation.
    On CPU, every returned BlockProfile has timing_only=True and
    activation_bytes_estimate=None (§6.5, §10 item 18)."""
    ...


def plan_checkpoints(
    profiles: Sequence[BlockProfile],
    blocks: Sequence[CheckpointableBlock],
    *,
    target_kind: TargetKind,
    target_value: float,
    planner: PlannerName = "dynamic_programming",
    activation_bucket_bytes: int = 1 << 20,   # 1 MiB, dynamic_programming only, provisional (§8.2, §14)
    dp_scale_guard_cells: int = 5_000_000,    # dynamic_programming only, provisional (§8.2, §14)
    on_infeasible: Literal["raise", "best_effort"] = "raise",
) -> CheckpointPlan:
    """Select which profiled blocks to checkpoint. See §7-8. Raises
    TimingOnlyProfileError if any profile has timing_only=True and target_kind
    requires real activation-byte accounting (§10 item 18)."""
    ...


def validate_plan(
    plan: CheckpointPlan,
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Raise PlanIncompatibleError if `plan` cannot be safely applied to `blocks`
    given `example_inputs`/`example_kwargs`. Recomputes the model fingerprint,
    including a fresh ExecutionSignature (§9.1) derived by actually running the
    declared block chain once under torch.no_grad() (state-preserving per §6.8),
    and compares it to plan.model_fingerprint/plan.execution_signature. This
    construction-time check is thorough (covers every block's real input/output
    shape) but is not the only shape check in v0.1: CheckpointedSequential.forward
    also performs a cheap entry-only signature check on every call (§9.2, §9.4) as
    a runtime safety net against a later call with different-shaped inputs on an
    already-constructed container. Does not permanently mutate anything (training
    mode and buffers are restored around the chain execution). See §9."""
    ...


def apply_plan(
    blocks: Sequence[CheckpointableBlock],
    plan: CheckpointPlan,
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
) -> "CheckpointedSequential":
    """Call validate_plan, then return a new CheckpointedSequential (§9) composing
    `blocks` in order, applying torch.utils.checkpoint.checkpoint(...,
    use_reentrant=False, preserve_rng_state=plan.preserve_rng_state) around each
    block selected by plan.decisions. The returned container checks the cheap
    entry-boundary signature on every subsequent forward() call (§9.2, §9.4).
    Shares model parameters by reference (§9.4); does not guarantee state_dict
    compatibility with any other module (§9.4)."""
    ...


def run_benchmark(
    blocks: Sequence[CheckpointableBlock],
    plan: CheckpointPlan,
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
    make_target: Any,  # Callable[[Any], torch.Tensor]: scalar loss from the container's output
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    num_warmup: int = 5,
    num_trials: int = 20,
    check_correctness: bool = True,
    correctness_rtol: float = 1e-5,
    correctness_atol: float = 1e-6,
) -> BenchmarkResult:
    """Build apply_plan(blocks, plan, example_inputs, example_kwargs) and run
    warm-up + measured training steps. If check_correctness and
    plan.planner_name != "no_checkpoint", also builds the "no_checkpoint" plan's
    container from the same blocks as the correctness reference (§9.5) — there is
    no other "original model" to compare against, by design (see "Revision
    Notes"). See §6, §11-12."""
    ...
```

Example, end to end:

```python
import torch
from ckptplan import (
    declare_blocks, profile_blocks, plan_checkpoints, apply_plan, run_benchmark,
)

model = build_transformer_encoder(num_layers=8).to("cuda")
example_inputs = (torch.randn(32, 512, 1024, device="cuda"),)

blocks = declare_blocks(
    model=model,
    blocks=[(f"layer{i}", model.layers[i]) for i in range(8)],
)

profiles = profile_blocks(
    blocks, example_inputs,
    device="cuda", dtype=torch.float32, num_warmup=3, num_trials=10,
)

plan = plan_checkpoints(
    profiles,
    target_kind="activation_budget_bytes",
    target_value=2 * (1 << 30),  # 2 GiB of block activation memory
    planner="dynamic_programming",
)

container = apply_plan(blocks, plan, example_inputs)

result = run_benchmark(
    blocks, plan, example_inputs, None,
    make_target=lambda out: out.float().pow(2).mean(),
    device="cuda", dtype=torch.float32,
)
print(result.peak_allocated_bytes, result.peak_reserved_bytes, result.step_latency_ms_mean)

# The container itself is an ordinary nn.Module the user's own training loop can
# also drive directly:
out = container(*example_inputs)
```

---

## 5. Core schemas

```python
from __future__ import annotations

import dataclasses
from typing import Any, Literal, Mapping, Optional, Sequence

import torch

BlockId = str
ExclusionReason = Literal[
    "stateful_mutation_in_train_mode",
    "shared_mutable_buffer",   # reserved; unreachable on a BlockProfile in v0.1
                               # since declare_blocks already rejects buffer
                               # sharing before profile_blocks ever runs (§10.3)
    "no_differentiable_output",
]


@dataclasses.dataclass(frozen=True)
class CheckpointableBlock:
    block_id: BlockId
    order: int
    module: torch.nn.Module


@dataclasses.dataclass(frozen=True)
class BlockProfile:
    block_id: BlockId
    order: int
    device: str
    dtype: str
    input_shape_signature: str      # over (args, kwargs) jointly, see §6.4
    output_shape_signature: str
    param_count: int
    trainable_param_count: int

    timing_only: bool                              # True on CPU (§10 item 18)
    activation_bytes_estimate: Optional[int]        # a_i; None iff timing_only=True
    activation_bytes_method: Optional[Literal["isolated_forward_delta"]]  # None iff timing_only=True

    forward_time_ms_mean: float
    forward_time_ms_std: float
    recompute_time_upper_bound_ms_mean: float       # renamed, §6.2, §13
    recompute_time_upper_bound_ms_std: float
    recompute_time_source: Literal["measured_full_recompute_early_stop_disabled"]

    num_warmup: int
    num_trials: int

    is_stochastic: Optional[bool]
    is_stateful: Optional[bool]
    stochastic_submodules: tuple[str, ...]
    stateful_submodules: tuple[str, ...]

    eligible_for_checkpoint: bool                    # explicit decision field, §10, correction 4
    exclusion_reason: Optional[ExclusionReason]      # set iff eligible_for_checkpoint is False

    warnings: tuple[str, ...]

    pytorch_version: str
    profiler_version: str


@dataclasses.dataclass(frozen=True)
class CheckpointDecision:
    block_id: BlockId
    checkpointed: bool
    eligible_for_checkpoint: bool                 # copied verbatim from the source BlockProfile at
                                                   # plan-construction time (§8.6) — Revision 3.1 fix
    exclusion_reason: Optional[ExclusionReason]   # copied verbatim from the source BlockProfile


@dataclasses.dataclass(frozen=True)
class ExecutionSignature:
    """§9.1, §9.2, §9.3 — correction 2. Captures the real, executed shape of the
    entire declared block chain, not just the entry call."""
    entry_signature: str                                   # _io_signature(example_inputs, example_kwargs)
    block_signatures: tuple[tuple[BlockId, str, str], ...]  # (block_id, input_signature, raw_output_signature), in order
    block_order: tuple[BlockId, ...]


@dataclasses.dataclass(frozen=True)
class CheckpointPlan:
    plan_id: str
    plan_format_version: str       # "3.1" for this revision (breaking: CheckpointDecision gained 2 required fields; was "3.0")
    created_at: str

    planner_name: Literal["greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"]
    planner_version: str

    target_kind: Literal["activation_budget_bytes", "activation_saving_fraction"]
    target_value: float
    activation_bucket_bytes: Optional[int]     # dynamic_programming only
    dp_repair_applied: bool                    # True iff §8.2's top-up repair fired (DP found a selection, real check failed)
    dp_fallback_reason: Optional[Literal["exact_bytes_feasible_bucketed_infeasible"]]
                                                # True case iff §8.2's greedy fallback fired (DP found no selection, exact bytes did) — correction 5

    decisions: tuple[CheckpointDecision, ...]
    feasible: bool

    predicted_activation_bytes_before: int              # A0
    predicted_activation_bytes_after: int               # A(x), from REAL (non-bucketed) a_i of the final selection
    predicted_recompute_time_upper_bound_ms: float       # C(x), renamed

    parameter_alias_groups: tuple[tuple[str, ...], ...]  # §9.1, §10.3, directive 11
    execution_signature: ExecutionSignature              # §9.1, §9.2, §9.3 — correction 2
    profile_fingerprint: str
    model_fingerprint: str

    use_reentrant: bool          # always False
    preserve_rng_state: bool     # always True

    assumptions: tuple[str, ...]


@dataclasses.dataclass(frozen=True)
class BenchmarkResult:
    config_name: str
    plan_id: str

    device_name: str
    pytorch_version: str
    python_version: str
    dtype: str
    batch_shape: str

    warmup_trials: int
    measured_trials: int

    peak_allocated_bytes: int      # primary metric, §12.4 (directive 8)
    peak_reserved_bytes: int       # secondary; not required to decrease monotonically

    step_latency_ms: tuple[float, ...]
    step_latency_ms_mean: float
    step_latency_ms_p50: float
    step_latency_ms_p95: float
    throughput_samples_per_sec: float

    correctness_checked: bool
    correctness_reference: Literal["no_checkpoint_plan", "none"]
    correctness_max_abs_output_diff: Optional[float]
    correctness_max_abs_grad_diff: Optional[float]
    correctness_passed: Optional[bool]

    oom: bool
    error_message: Optional[str]

    environment: Mapping[str, Any]
```

---

## 6. Profiling methodology

**Redesigned per directives 1, 6, 21, 22.** `profile_blocks` no longer takes `model` and uses no hooks. It derives each block's real input by directly executing the declared chain itself — this is not an approximation of what the container will do, it is *the same computation* the container performs (§9), so there is nothing to reconcile between profiling and application.

Revision 3.3 makes `device` and `dtype` validation-only inputs. The requested
device is normalized before any block executes. Parameters, buffers, and every
tensor leaf encountered at the entry or an executed boundary must already be on
that device; nothing is moved. When `dtype` is supplied, every floating parameter,
buffer, input, and output leaf must match it; nothing is cast. When `dtype=None`,
the profiler infers one dtype from those actual floating values and rejects mixed
floating dtypes as ambiguous. Errors name the offending block/value.

### 6.1 Deriving each block's real inputs (replaces hook-based capture)

**Corrected per correction 1.** The last-block boundary bug fixed in
`CheckpointedSequential.forward` (§9.2) was present here too, identically — both are
now built on one shared, canonical chain-walking helper, so the "only convert the
boundary when a next block exists" rule is specified exactly once and cannot drift
between profiling and application:

```python
def _run_block_chain(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
):
    """Execute the declared block chain exactly once, under torch.no_grad(),
    yielding (block, input_args, input_kwargs, output) per block in order.
    State-preserving per §6.8 around each block's call: every submodule training
    flag, every named buffer, and every pre-existing parameter gradient are
    preserved and restored in a try/finally, whether or not forward raises.
    _boundary_convert (§9.2) is applied
    only when a subsequent block exists — the last block's yielded `output` is its
    raw, unconverted return value, exactly mirroring CheckpointedSequential.forward
    (§9.2) so the two can never diverge on what a "final output" is allowed to be."""
    current_args, current_kwargs = example_inputs, dict(example_kwargs or {})
    n = len(blocks)
    for idx, block in enumerate(blocks):
        with _preserve_module_state(block.module, isolate_gradients=False):
            with torch.no_grad():
                output = block.module(*current_args, **current_kwargs)
        yield block, current_args, current_kwargs, output
        if idx < n - 1:
            current_args, current_kwargs = _boundary_convert(output)
```

`profile_blocks` derives `captured_inputs` from this generator:

```python
captured_inputs: list[tuple[tuple[Any, ...], dict[str, Any]]] = [
    (args, kwargs) for _, args, kwargs, _ in _run_block_chain(blocks, example_inputs, example_kwargs)
]
```

`compute_execution_signature` (§9.1, correction 2) is built on the same generator,
used by `validate_plan`/`apply_plan` (which do not have profiles to read signatures
from, unlike `profile_blocks`):

```python
def compute_execution_signature(
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
) -> "ExecutionSignature":
    block_signatures = []
    for block, args, kwargs, output in _run_block_chain(blocks, example_inputs, example_kwargs):
        block_signatures.append((block.block_id, _io_signature(args, kwargs), _io_signature((output,), {})))
    return ExecutionSignature(
        entry_signature=_io_signature(example_inputs, dict(example_kwargs or {})),
        block_signatures=tuple(block_signatures),
        block_order=tuple(b.block_id for b in blocks),
    )
```

`_run_block_chain` is the single function shared between input capture (here),
`compute_execution_signature` (§9.1), and — for the boundary-conversion rule
specifically — `CheckpointedSequential.forward` (§9.2). This sharing is what
guarantees profiling, fingerprinting, and application never diverge on chain
semantics. (`_io_signature`/`_signature` are defined in §6.4, below.)

### 6.2 Activation-memory estimate (`activation_bytes_estimate`, i.e. `a_i`)

Unchanged mechanically from revision 1 (§6.1 there), operating on `captured_inputs[i]` instead of hook-captured values:

1. `torch.cuda.empty_cache()` once before profiling this block (CUDA only), and again once after (directive 21 — prevents cross-block fragmentation from skewing later baselines).
2. Per trial (after warm-up): `reset_peak_memory_stats`; `mem_before = memory_allocated`; fresh `requires_grad_(True)` leaf copies of the captured args/kwargs; `output = block.module(*args, **kwargs)` under `enable_grad()`, no backward; `synchronize`; `mem_after = memory_allocated`; `trial_estimate = max(0, (mem_after - mem_before) - input_bytes)`, where `input_bytes` sums floating-point leaf tensor bytes across both `args` and `kwargs` (directive 22: kwargs count too). Discard `output` before the next trial.
3. `activation_bytes_estimate = round(mean(trial_estimate))`, `activation_bytes_method = "isolated_forward_delta"`, `timing_only = False`.

On CPU: `timing_only = True`, `activation_bytes_estimate = None`, `activation_bytes_method = None` (§10 item 18 — corrected from revision 1's misleading `= 0`).

### 6.3 Forward and recomputation-upper-bound timing

**Renamed per directive 13**, and the nested-output backward call is **fixed per directive 6**.

```python
class _TimedCall:
    def __init__(self, block):
        self.block = block
        self.call_times_ms: list[float] = []

    def __call__(self, *args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.block(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        self.call_times_ms.append(start.elapsed_time(end))
        return out


def _collect_differentiable_leaves(value: Any) -> list[torch.Tensor]:
    """Public-API-only recursive walk collecting tensor leaves. Reused by §6.3's
    backward step, §10.4's nested-output test, and nowhere else — it is distinct
    from §9.2's boundary conversion, which is a one-level, Tensor/tuple/dict-only
    convention for inter-block data flow. This collector allows arbitrary nesting
    depth because it exists only to drive torch.autograd.backward correctly for
    profiling measurement, not to define block-to-block data flow."""
    if torch.is_tensor(value):
        return [value] if (value.requires_grad and value.is_floating_point()) else []
    if isinstance(value, (list, tuple)):
        leaves: list[torch.Tensor] = []
        for v in value:
            leaves.extend(_collect_differentiable_leaves(v))
        return leaves
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise UnsupportedBoundaryError(
                    f"non-string dict key {k!r} of type {type(k).__name__}; "
                    "see spec §9.2/§10.4 (directive 23)"
                )
        leaves = []
        for k in sorted(value.keys()):
            leaves.extend(_collect_differentiable_leaves(value[k]))
        return leaves
    return []
```

Per trial: build fresh leaf input copies; with `torch.utils.checkpoint.set_checkpoint_early_stop(False)` active (§0.6):

```python
timed = _TimedCall(block.module)
out = torch.utils.checkpoint.checkpoint(
    timed, *args, **kwargs, use_reentrant=False, preserve_rng_state=True,
)
leaves = _collect_differentiable_leaves(out)
if not leaves:
    raise NoDifferentiableOutputError(block_id=block.block_id)
torch.autograd.backward(leaves, [torch.ones_like(t) for t in leaves])
```

`timed.call_times_ms[0]` is the forward-pass time; `timed.call_times_ms[1]` (present because `early_stop` is disabled) is `recompute_time_upper_bound_ms` for that trial. `recompute_time_source = "measured_full_recompute_early_stop_disabled"` — the source label itself now states the caveat, not just prose elsewhere.

On CPU, `_TimedCall` uses `time.perf_counter()` immediately before and after
the block call, converting elapsed seconds to milliseconds. Every timing trial
uses `checkpoint(..., use_reentrant=False, preserve_rng_state=True)` inside
`set_checkpoint_early_stop(False)`, exactly as CUDA does apart from the timing
clock and synchronization mechanics.

`NoDifferentiableOutputError` (directive 6): a block whose recursive output walk
contains no tensor with `requires_grad=True` and floating dtype is never silently
given zero cost. Revision 3.3 wraps every timed call with a separate zero-valued
differentiable probe. Backward through that probe forces the same genuine full
checkpoint recomputation even for an otherwise non-differentiable output, so the
non-optional timing fields remain measured rather than fabricated. `profile_blocks`
records such an individual block with `eligible_for_checkpoint=False`,
`exclusion_reason="no_differentiable_output"`, and `is_stochastic=is_stateful=None`,
then continues when another valid block remains. It raises
`NoDifferentiableOutputError` only when every declared block fails this test.

### 6.4 Shape/structure signatures

Extended per directive 22 (kwargs) and restricted per directive 23 (string-only dict keys):

```python
def _signature(value: Any) -> str:
    if torch.is_tensor(value):
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, device={value.device.type})"
    if isinstance(value, (list, tuple)):
        kind = "List" if isinstance(value, list) else "Tuple"
        return f"{kind}[{', '.join(_signature(v) for v in value)}]"
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise UnsupportedBoundaryError(f"non-string dict key {k!r}")
        items = ", ".join(f"{k}={_signature(value[k])}" for k in sorted(value))
        return f"Dict{{{items}}}"
    return f"Const({type(value).__name__})"

def _io_signature(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return f"Args{_signature(tuple(args))}, Kwargs{_signature(dict(sorted((kwargs or {}).items())))}"
```

`input_shape_signature`/`output_shape_signature` are computed with `_io_signature`, applied to `(args, {})` for outputs (a block's return value is a single positional-style value for signature purposes, before boundary conversion). This restriction to string keys (directive 23) applies uniformly everywhere a dict is walked by this module: block outputs, `example_kwargs`, and nested structures inside them. A non-`str` key raises `UnsupportedBoundaryError` at the point it is discovered (profiling time for outputs; `example_kwargs` keys are always valid Python identifiers already, so no separate check is needed there).

### 6.5 CPU profiling

Unchanged in spirit from revision 1 but corrected per directive 18: `timing_only = True`, `activation_bytes_estimate = None` (not `0`). `plan_checkpoints` given any `timing_only=True` profile with an activation-based `target_kind` raises `TimingOnlyProfileError` immediately (§8, §10 item 18) — CPU profiles exist only for CPU correctness/planner-logic tests that either don't need real memory numbers or construct synthetic `BlockProfile`s by hand.

### 6.6 CUDA synchronization

Unchanged from revision 1: synchronize before the first trial of a measurement sequence; CUDA events for all latency numbers, `elapsed_time()` only after synchronization; memory readings only immediately after a synchronize call; CPU uses `time.perf_counter()` and no memory statistics.

### 6.7 Allocated versus reserved memory

Unchanged in mechanics from revision 1, **re-ordered per directive 8**: `peak_allocated_bytes` is the metric this document treats as primary throughout (§12.4); `peak_reserved_bytes` remains a recorded secondary metric, useful because it is closer to what `nvidia-smi` reports, but is explicitly **not** required to move monotonically with `peak_allocated_bytes` (allocator caching/fragmentation can hide or delay a reserved-memory change even when allocated memory clearly drops).

### 6.8 State preservation during profiling

**Strengthened in Revision 3.3.** For each block, profiling wraps steps 6.2–6.3 in a `try/finally`:

1. Before profiling block `k`, snapshot every recursive submodule's individual
   `.training` flag, every named buffer's tensor value, and every parameter's
   existing `.grad` object (including `None`). Profiling gradients are isolated
   from those caller-owned gradients.
2. In `finally` (runs on success **and** on exception), restore buffer values
   byte-for-byte, assign every submodule's own saved `.training` flag directly,
   and reattach every original gradient object. Do **not** call
   `block.module.train(original_root_flag)`: that recursively overwrites mixed
   child modes. Clear only gradients and tensor references created by profiling.
3. There are no hooks to remove in this design (§ "Revision Notes" — hooks were eliminated entirely, not just cleaned up better). If a future revision reintroduces hooks for other instrumentation, they must be registered and removed inside the same `try/finally`.
4. `profile_blocks` re-raises the original exception after `finally` runs — state restoration never suppresses an error, it only ensures the error doesn't leave the real block instances (which will later be composed live into `CheckpointedSequential`, §9, since they are never copied) corrupted for whatever runs next.
5. §6.1's `_run_block_chain` (used for input capture here and for
   `compute_execution_signature`, §9.1) uses the same private state guard around
   its single no-measurement pass. Canonical chain, boundary, signature, and state
   helpers live in `ckptplan/_execution.py`; later application code must import
   them rather than duplicate their semantics.

---

## 7. Scheduling formulation

### 7.1 Decision variables

Unchanged structurally, renamed per directive 13: for eligible block `i` — **defined as `BlockProfile.eligible_for_checkpoint is True`, read directly, never re-derived from `is_stateful`/`is_stochastic`/`warnings` (§10, correction 4)** — `a_i := BlockProfile.activation_bytes_estimate`, `c_i := BlockProfile.recompute_time_upper_bound_ms_mean`, `x_i ∈ {0,1}` means "checkpoint block `i`." Ineligible blocks (`eligible_for_checkpoint is False`) are fixed at `x_i = 0` for every planner, unconditionally (§10, §8).

### 7.2 Objective and constraint

**Renamed per directive 3** (`memory_budget_bytes` → `activation_budget_bytes`, `memory_saving_fraction` → `activation_saving_fraction`); math otherwise unchanged:

```
A0   := Σ over all declared blocks of a_i
A(x) := A0 - Σ over eligible i of x_i · a_i
C(x) := Σ over eligible i of x_i · c_i
```

- `target_kind = "activation_budget_bytes"`, value `B`: `S := max(0, A0 - B)`.
- `target_kind = "activation_saving_fraction"`, value `f ∈ (0,1]`: `S := f · A0`.

```
minimize    C(x) = Σ x_i · c_i
subject to  Σ x_i · a_i  ≥  S
            x_i ∈ {0, 1}
```

**Hard invariant, unconditional in code, not just by construction (directive 4):** any `CheckpointPlan` with `feasible=True` satisfies `Σ_{i: x_i=1} a_i ≥ S` using the real, undiscretized `a_i` values — this is checked explicitly after planning (§8.2), not merely implied by the discretization scheme.

### 7.3 Assumptions (recorded verbatim in `CheckpointPlan.assumptions`)

1. Activation memory is additive/independent across blocks (unchanged rationale from revision 1).
2. Recompute cost is additive across checkpointed blocks (unchanged rationale).
3. "`recompute_time_upper_bound_ms_mean` is a conservative cost-model input, not an unbiased runtime prediction: it was measured with early stopping disabled. Real training, which uses PyTorch's default early-stop behavior, typically incurs recompute time less than or equal to this value." (Directive 13 — reworded from "worst-case estimate" framing to explicit "upper bound, not unbiased predictor" framing.)
4. "The activation budget/target concerns block-local activation memory only (`Σ a_i`), not total process GPU memory, and not `peak_allocated_bytes`/`peak_reserved_bytes` directly. Parameters, gradients, optimizer state, non-block activations, and CUDA context overhead are excluded." (Directive 3.)
5. Profiles are valid only for the exact input/output shape+kwargs signature captured during profiling (§6.4, §10.5).
6. "If dynamic-programming planning required the deterministic top-up repair (§8.2) to satisfy the exact byte constraint, `dp_repair_applied=True` and this plan's `predicted_recompute_time_upper_bound_ms` may exceed the value the discretized surrogate model alone would predict." (New, directive 4 — only present when repair actually fired.)
7. "If dynamic-programming planning required the greedy fallback (§8.2 step 4) because every eligible block's activation size was smaller than one bucket, `dp_fallback_reason=\"exact_bytes_feasible_bucketed_infeasible\"` and this plan's decisions were produced by greedy selection over the real, non-bucketed bytes, not by the dynamic-programming search itself." (New, correction 5 — only present when the fallback actually fired.)

### 7.4 Where the result is not globally optimal for real GPU peak memory

Unchanged from revision 1 (caching-allocator fragmentation, non-block memory outside the model, independence-assumption error, discretized-surrogate exactness only). One addition: the surrogate model now explicitly does **not** claim to optimize `peak_reserved_bytes` or `peak_allocated_bytes` at all — it optimizes `C(x)` subject to the `a_i`-based surrogate constraint (directive 9). Any observed relationship between the plan and measured peak memory is empirical and reported, not designed-for.

---

## 8. Greedy, uniform, and dynamic-programming planner behavior

All three "real" planners plus the two trivial baselines (§8.4) consume the same eligible-block list `[(a_i, c_i, order_i)]` and must be deterministic.

### 8.1 Greedy (`planner="greedy"`)

Unchanged from revision 1: density `d_i = a_i/c_i` (or `+inf` if `c_i=0`), sort by `d_i` desc, `a_i` desc, `c_i` asc, `order_i` asc; accumulate until `Σ x_i·a_i ≥ S`. Not guaranteed cost-optimal; stated in `assumptions` when selected.

### 8.2 Dynamic programming (`planner="dynamic_programming"`)

**Rewritten per directive 4; extended per correction 5 (this revision).** Revision 1
used `ā_i = ceil(a_i/bucket)`, which can overstate a selection's real savings and
return a `feasible=True` plan that fails the real byte constraint. Revision 2 fixed
that with conservative floor/ceil discretization, but left a different gap open:
when every eligible block's `a_i` is smaller than one bucket, `ā_i =
floor(a_i/bucket) = 0` for all of them, so the bucketed search can *never* find a
nonzero-savings selection and reports infeasible even when the real, undiscretized
bytes are sufficient. Correction 5 closes this with an explicit exact-byte check
before ever declaring infeasibility.

1. `ā_i = floor(a_i / activation_bucket_bytes)` (a lower bound on each block's real contribution: `ā_i · bucket ≤ a_i`). `S̄ = ceil(S / activation_bucket_bytes)` (an upper bound on the target: `S̄ · bucket ≥ S`).

   **Proof this combination is safe:** if the DP finds a selection with bucketed savings `Σ ā_i ≥ S̄`, then real savings `Σ a_i ≥ Σ ā_i · bucket ≥ S̄ · bucket ≥ S`. So any bucket-feasible selection is automatically real-feasible. The floor/ceil choice can only make the DP wrongly call something *infeasible* that a finer-grained method would find feasible (a false negative, the safe direction of error) — never wrongly call something feasible that isn't.

2. Scale guard, a configurable parameter, not a hardcoded constant (directive 14): before running, compute `n × (S̄ + 1)`; if it exceeds `dp_scale_guard_cells` (default `5_000_000`, a provisional default requiring empirical calibration during implementation — not a validated number), raise `PlannerScaleError` recommending a larger `activation_bucket_bytes` or `planner="greedy"`.

3. Same 0/1 "minimum cost to reach at least a target" DP as before, over `ā_i`/`S̄`, processing blocks in execution order, with the same strict-improvement (`<`) tie-break rule (prefer earlier-order blocks among equal-cost alternatives). `O(n × S̄)` time/space with a full `(n+1)×(S̄+1)` table for exact backtracking. This produces `dp[n][S̄]`, possibly `+inf` (no bucket-feasible selection found).

4. **Exact-byte check before declaring infeasibility (new, correction 5):** if `dp[n][S̄]` is `+inf`, do **not** immediately raise/report infeasible. First compute `exact_total := Σ a_i` over **all** eligible blocks, using real, non-bucketed values.
   - If `exact_total < S`: genuine infeasibility — proceed to §8.5.
   - If `exact_total ≥ S`: the bucketed search itself is the sole obstacle, not the underlying problem (this is exactly the sub-bucket-blocks case). Deterministically **fall back to `"greedy"` selection** (§8.1) using the real, non-bucketed `a_i`/`c_i` and the same target `S` — greedy's own stopping rule guarantees it finds a feasible selection whenever `exact_total ≥ S`, since it accumulates real bytes directly and never discretizes. Set `decisions` from greedy's result, `feasible=True`, and `dp_fallback_reason="exact_bytes_feasible_bucketed_infeasible"` on the returned `CheckpointPlan` (`planner_name` remains `"dynamic_programming"`, since that is what was requested; the fallback is disclosed via `dp_fallback_reason`, not by silently relabeling the plan). Append the assumptions note from §7.3 item 7.
   - **Why greedy, not an adaptive-bucket retry:** both were valid per the governing directive; greedy is chosen because it is already fully specified (§8.1), needs no new tunable retry parameters (candidate bucket sizes, a retry limit, a convergence argument), and is provably sufficient exactly when `exact_total ≥ S` — an adaptive retry would need to reach the same guarantee with more moving parts for no additional correctness benefit.

5. **Mandatory post-hoc verification of a *found* bucket-feasible selection (directive 4 — a different case from step 4 above: here `dp[n][S̄] < +inf`, a selection was found):** after reconstructing the selected set `x`, compute the real (non-bucketed) `achieved = Σ_{i: x_i=1} a_i`. Given step 1's proof, `achieved ≥ S` always holds; this is asserted in code, not only claimed on paper.
   - If the assertion ever fails (should be unreachable given the proof; its presence guards against implementation bugs, not a known gap), apply **deterministic top-up repair**: extend `x` by adding not-yet-selected eligible blocks in the same order as greedy's sort key (§8.1: `d_i` desc, `a_i` desc, `c_i` asc, `order_i` asc) until `achieved ≥ S` or all eligible blocks are selected. Set `dp_repair_applied=True` and append the assumptions note from §7.3 item 6. If topping up with every eligible block still does not reach `S`, this is genuine infeasibility (§8.5) — this can only happen if step 1's proof was violated by an implementation bug.
   - `dp_repair_applied` (step 5) and `dp_fallback_reason` (step 4) are mutually exclusive and describe different situations: repair patches a selection the DP *did* find; fallback replaces a search that found *no* selection at all.

6. `predicted_activation_bytes_after` and `predicted_recompute_time_upper_bound_ms` in the returned `CheckpointPlan` are always computed from the **final, real, non-bucketed** `a_i`/`c_i` values of the selected set (after any repair or fallback), never from bucketed values — bucketing is purely an internal search-space discretization, invisible in the plan's reported numbers.

7. Within this surrogate model, DP's `C(x)` is `≤` greedy's for any feasible target when neither repair nor fallback fires — an implementation-correctness property, tested in §11, not an empirical claim. When either fires, this inequality is not guaranteed (both intentionally sacrifice optimality for the unconditional real-feasibility guarantee) — `dp_repair_applied`/`dp_fallback_reason` make this visible rather than silently comparing an already-different plan to greedy as if it were still DP-optimal.

`activation_bucket_bytes` default `1 << 20` (1 MiB) is a **provisional default**, not validated against real block activation sizes (directive 15) — see §14.

### 8.3 Uniform (`planner="uniform"`)

**Rewritten per directive 5.** Revision 1's "uniform" checkpointed a prefix, which is not uniform selection. Fixed:

1. Let `E` = eligible blocks sorted by `order_i` ascending, `m = len(E)`.
2. For `k = 1, 2, ..., m`: compute the evenly spaced index set of size `k` into `E`:
   `positions(k) = { (j * m) // k  for j in 0..k-1 }` (a standard even-spacing formula, equivalent to the boundaries `numpy.array_split` would choose). This set always has exactly `k` distinct elements for `1 ≤ k ≤ m`.

   **Worked example (documented so the formula is unambiguous to implement):** `m=8, k=4` → `positions = {0, 2, 4, 6}`. `m=8, k=1` → `positions = {0}`. `m=8, k=3` → `positions = {0, 2, 5}`.
3. Set `x_i = 1` for blocks of `E` at those positions, `x_i = 0` otherwise; compute real (non-bucketed) `Σ x_i · a_i`. Stop at the **smallest** `k` for which this sum `≥ S`.
4. If `k = m` and the full-checkpoint sum is still `< S`: infeasible (§8.5).
5. This formula is the sole, exact, deterministic selection and tie-break rule for `"uniform"` — there is no secondary tie-break needed because `positions(k)` is a pure function of `(m, k)` with no ties to resolve.

### 8.4 Baseline "planners" `"no_checkpoint"` / `"checkpoint_all"`

Unchanged from revision 1: `x_i = 0` for all `i` / `x_i = 1` for all eligible `i`, respectively.

### 8.5 Infeasibility policy

**Confirmed accepted, directive 10** — unchanged from revision 1's proposal: `on_infeasible="raise"` (default) raises `InfeasibleTargetError(required_savings_bytes=S, max_achievable_bytes=Σ a_i over eligible blocks)`. `on_infeasible="best_effort"` returns a plan checkpointing every eligible block, `feasible=False`, `predicted_activation_bytes_after` reflecting the best achievable reduction. No longer flagged as an open question.

### 8.6 Constructing `CheckpointDecision`s (Revision 3.1 fix, shared across every planner)

Every planner — `greedy`, `dynamic_programming` (including its fallback and repair
paths), `uniform`, `checkpoint_all`, `no_checkpoint` — produces its final
`decisions` tuple through one shared helper, so eligibility copy-through can never
drift between planners:

```python
def _build_decisions(
    profiles: Sequence[BlockProfile],
    checkpointed_ids: set[BlockId],
) -> tuple[CheckpointDecision, ...]:
    """checkpointed_ids is the set of block_ids a planner selected (x_i = 1).
    eligible_for_checkpoint/exclusion_reason are always copied verbatim from the
    profile that produced this decision, never re-derived or left blank — this is
    the fix for the Revision 3.1 blocking defect: validate_plan (§9.3) can only
    check what plan_checkpoints actually persisted here."""
    return tuple(
        CheckpointDecision(
            block_id=p.block_id,
            checkpointed=p.block_id in checkpointed_ids,
            eligible_for_checkpoint=p.eligible_for_checkpoint,
            exclusion_reason=p.exclusion_reason,
        )
        for p in profiles
    )
```

A planner never sets `checkpointed=True` for a `block_id` whose source profile has
`eligible_for_checkpoint=False` — this was already true of every planner's own
selection logic (§8.1–§8.4 all operate only over the eligible subset), and
`_build_decisions` does not change that; it only guarantees the resulting
`CheckpointDecision` also *records* eligibility, so a later, separate call to
`validate_plan` (§9.3) — which has no access to `profiles` — can still check it.

---

## 9. Plan validation and application

### 9.1 Fingerprints and `ExecutionSignature`

**Extended per directives 1, 11, 22; corrected per corrections 2 and 3 (this revision).**

**`ExecutionSignature` (new, correction 2)** replaces the false claim that an
entry-input signature alone "implicitly" covered every intermediate block
boundary — it did not, because nothing re-executed the chain to derive
intermediate shapes. Its schema is defined in §5; its derivation has two distinct,
precisely specified paths:

- **At `plan_checkpoints` time (cheap — no execution):** `profiles` already contain
  each block's real `input_shape_signature`/`output_shape_signature` (§6.4,
  computed once during `profile_blocks`'s chain walk, §6.1). `plan_checkpoints`
  receives the matching `blocks` sequence as well. It assembles
  `ExecutionSignature` directly from `profiles` — `entry_signature =
  profiles[0].input_shape_signature`, `block_signatures = tuple((p.block_id,
  p.input_shape_signature, p.output_shape_signature) for p in profiles)` — with no
  additional model execution. It computes `parameter_alias_groups` and the
  module-structure portion of `model_fingerprint` from `blocks` plus that
  signature.
- **At `validate_plan`/`apply_plan` time (one real execution):** these functions do
  not receive `profiles`, only `blocks` + `example_inputs`/`example_kwargs`. They
  call `compute_execution_signature(blocks, example_inputs, example_kwargs)` (§6.1),
  which runs `_run_block_chain` — the declared chain, once, under
  `torch.no_grad()`, with the identical state-preservation contract as §6.8
  (training mode and every buffer snapshotted and restored around each block's
  call, in a `try/finally`, whether or not it raises) — and builds the same
  `ExecutionSignature` shape from the real, freshly observed input/output
  signatures. This is the "inexpensive" execution referred to in
  §9.3/§9.4/correction 6: one forward pass per declared block, no backward, no
  repeated trials, no memory/timing instrumentation.

`validate_plan` compares the freshly derived `ExecutionSignature` against
`plan.execution_signature` **directly** (dataclass equality — an exact
string/tuple comparison, not just a hash), which lets a mismatch report name the
specific block and whether it was the input or output side that changed. This
direct comparison is in addition to, not instead of, folding the signature into
the overall `model_fingerprint` hash below (used for the coarser pass/fail check
and for detecting a hand-edited or incompatible serialized plan, §10.6).

`model_fingerprint` is `sha256` over a canonical JSON object with:

- one entry per declared block, in order: `{block_id, order, module_qualified_class_name, param_shapes (ordered list of (name, tuple(shape), dtype) for module.named_parameters(recurse=True)), device, dtype}`;
- the full `ExecutionSignature` (entry signature, every block's input/output signature, block order/identity) — no longer just the entry signature;
- `parameter_alias_groups`: the output of `compute_parameter_alias_groups(blocks)` (directive 11) — a sorted tuple of sorted-tuples, each inner tuple the qualified dotted names (`"{block_id}.{param_name}"`) of parameters that are the *same* `nn.Parameter` instance across two or more declared blocks. Computed once by scanning `id(p)` across every declared block's `named_parameters(recurse=True)` and grouping matches. Folding this into the fingerprint means a plan built against one aliasing structure cannot be silently applied to a differently-tied model.

**`profile_fingerprint` (corrected, correction 3)** is `sha256` over, for each
`BlockProfile` in order, exactly the fields that determine planner decisions:
`block_id`, `order`, `device`, `dtype`, `input_shape_signature`,
`output_shape_signature`, `timing_only`, `activation_bytes_estimate`,
`activation_bytes_method`, `recompute_time_upper_bound_ms_mean`,
`recompute_time_source`, `eligible_for_checkpoint`, `exclusion_reason`. It
explicitly **excludes** variance/std fields (`forward_time_ms_std`,
`recompute_time_upper_bound_ms_std`) and run metadata (`num_warmup`, `num_trials`,
`pytorch_version`, `profiler_version`, `warnings`) — none of these determine what a
planner selects, so including them would make the fingerprint fragile to
irrelevant re-runs (e.g., re-profiling with more trials). Because the fingerprint
now covers every value the planner actually reads, a hand-edited or differently
measured profile (e.g., a manually inflated `activation_bytes_estimate`, or a
profile whose `eligible_for_checkpoint` was flipped) cannot retain the same
`profile_fingerprint`.

**Extended again, Revision 3.1:** since `CheckpointDecision` now also carries an
`eligible_for_checkpoint`/`exclusion_reason` echo (§5, §8.6), `compute_profile_fingerprint`
takes **both** `profiles` and `decisions` and additionally hashes, per block in
order, `(checkpointed, eligible_for_checkpoint, exclusion_reason)` from the
matching `CheckpointDecision` alongside the profile fields above. This closes the
specific gap that motivated this patch: without it, someone could hand-edit a
decision's eligibility echo (e.g., flip `eligible_for_checkpoint` from `False` to
`True` consistently, clearing `exclusion_reason` and setting `checkpointed=True`)
without the stored `profile_fingerprint` changing, since a fingerprint computed
only from `profiles` never looks at `decisions` at all.

`compute_profile_fingerprint(profiles, decisions)` is available as a public
function for provenance auditing — comparing a freshly recomputed value against a
saved `CheckpointPlan.profile_fingerprint`, given the original `profiles` list, to
detect tampering or drift in either the profiles or the decisions derived from
them. It is **not** automatically re-checked by `validate_plan`, which still does
not take `profiles` as input (an intentional Revision 2/3 design choice to avoid
requiring re-profiling at every application site); only `model_fingerprint`/
`ExecutionSignature` (derived from `blocks` + `example_inputs`/`example_kwargs`)
and the decision-internal consistency checks (§9.3) are automatically re-validated
on every `validate_plan`/`apply_plan` call.

**Disclosed residual limitation (Revision 3.1):** `validate_plan`'s new checks
(§9.3) catch a decision that is *internally* inconsistent (`checkpointed=True`
with `eligible_for_checkpoint=False`, or a mismatched exclusion-reason pairing) and
catch *structural* tampering of the decision list (missing/duplicate/reordered/
unknown block IDs). They do **not** catch a decision that was tampered
*consistently* — e.g., an attacker who flips `eligible_for_checkpoint` to `True`,
clears `exclusion_reason`, and sets `checkpointed=True` all together, producing an
internally self-consistent but factually false decision. Detecting that requires
cross-referencing the original `profiles` via `compute_profile_fingerprint`, which
`validate_plan` does not do automatically. This is an accepted, disclosed gap
given the already-approved decision not to require `profiles` at validation time —
not something this patch claims to close — and is listed again in §14 for
explicit sign-off.

### 9.2 The boundary conversion rule and `CheckpointedSequential`

**New, per directives 1 and 23** — this is the concrete container replacing revision 1's under-specified "wrapper."

```python
def _boundary_convert(value: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """The single fixed rule for turning one block's return value into the next
    block's call arguments. Shared verbatim between profiling (§6.1) and
    CheckpointedSequential.forward (below) — this identity is what guarantees the
    two never diverge."""
    if torch.is_tensor(value):
        return (value,), {}
    if isinstance(value, tuple):
        return value, {}
    if isinstance(value, dict):
        for k in value:
            if not isinstance(k, str):
                raise UnsupportedBoundaryError(f"non-string dict key {k!r} at a block boundary")
        return (), dict(value)
    raise UnsupportedBoundaryError(
        f"block output of type {type(value).__name__} is not a Tensor, tuple, or "
        "dict[str, Any] and cannot be used as the next block's input (§9.2). A "
        "block may still internally use richer nested structures (§10.4) as long "
        "as its return value at the boundary is one of these three shapes."
    )


class CheckpointedSequential(torch.nn.Module):
    """A library-owned nn.Module whose forward is exactly the ordered composition
    of the declared blocks under `plan`. It does not execute, wrap, or reproduce
    any other code from wherever the blocks originally came from — see spec §2,
    §"Revision Notes"."""

    def __init__(self, blocks: Sequence[CheckpointableBlock], plan: CheckpointPlan):
        super().__init__()
        self._module_list = torch.nn.ModuleList([b.module for b in blocks])
        self._block_ids = tuple(b.block_id for b in blocks)
        self._checkpointed = tuple(d.checkpointed for d in plan.decisions)
        self._entry_signature = plan.execution_signature.entry_signature
        self.plan = plan

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        # Correction 6: cheap, entry-only runtime check on every call, before any
        # block executes. This is in addition to the thorough, construction-time
        # ExecutionSignature check in validate_plan (§9.1, §9.3), which covers every
        # block's shape but only runs once, at apply_plan time. This per-call check
        # exists specifically to catch a later call to this same, already-built
        # container with a different-shaped input. Cost: one _signature() walk over
        # the caller's own args/kwargs (shape/dtype/device attribute reads and
        # string formatting only, no tensor data touched, no GPU sync) — negligible
        # next to any real block's computation, but not literally zero; see §9.4.
        if _io_signature(args, kwargs) != self._entry_signature:
            raise PlanIncompatibleError(
                "input shape/dtype/device at the entry boundary does not match the "
                "signature this plan was validated against (§10.5); this container "
                "was built for a specific shape and does not support dynamic shapes"
            )
        current_args, current_kwargs = args, kwargs
        num_blocks = len(self._module_list)
        output = None
        for idx, (module, checkpointed) in enumerate(zip(self._module_list, self._checkpointed)):
            if checkpointed:
                output = torch.utils.checkpoint.checkpoint(
                    module, *current_args, **current_kwargs,
                    use_reentrant=False, preserve_rng_state=self.plan.preserve_rng_state,
                )
            else:
                output = module(*current_args, **current_kwargs)
            # Correction 1: only convert the boundary when a next block will
            # consume it. The final block's raw output — which may legitimately be
            # something _boundary_convert would reject, e.g. a list — is returned
            # untouched below, matching _run_block_chain's identical rule (§6.1).
            if idx < num_blocks - 1:
                current_args, current_kwargs = _boundary_convert(output)
        return output
```

Precisely: **only block 0 may receive both positional and keyword arguments simultaneously** (from the container's own `forward(*args, **kwargs)` call, i.e. from `example_inputs`/`example_kwargs`). Blocks `1..n-1` receive input derived solely from the previous block's single return value via `_boundary_convert`, which produces either pure positional args (Tensor or tuple case) or pure keyword args (dict case) — never both at once.

### 9.3 `validate_plan` — construction-time shape validation, plus §9.2's per-call entry check

**Decided per directive 2; extended per correction 6.** Two construction-time
mechanisms were considered for the *thorough* check: (A) require
`example_inputs`/`example_kwargs` at `validate_plan`/`apply_plan` call time and
check the full `ExecutionSignature` once, at construction; (B) defer the full
check into the container's first real `forward()` call. **v0.1 implements (A) for
the thorough, every-block check.** Rationale: it fails fast, before any training
step runs, and is consistent with `profile_blocks` already requiring
representative inputs; running the full chain on every forward call would add real
compute overhead (a second execution of every block) to every training step for a
shape that v0.1 already assumes is static (§10.5).

**Separately, correction 6 adds a cheap, entry-only check that does run on every
`forward()` call** (§9.2) — this is not mechanism (B): it does not re-derive or
compare every block's shape (that remains construction-time-only, per (A)), it
only compares the caller's own entry arguments against the signature recorded at
construction, which requires no block execution at all, just inspecting the
caller's own tensors. The two mechanisms are complementary: (A) is thorough but
one-time; the entry check is cheap but partial, covering exactly the case a
one-time check cannot — reuse of an already-built container with different inputs
later.

`validate_plan(plan, blocks, example_inputs, example_kwargs)` raises `PlanIncompatibleError` if:

- `plan.plan_format_version` is unrecognized → `UnsupportedPlanVersionError` (§9.6).
- The freshly computed `ExecutionSignature` (§9.1, via `compute_execution_signature`, §6.1, which runs the chain once under `torch.no_grad()` with §6.8's state-preservation contract) does not equal `plan.execution_signature` — reported with the specific block and side (input/output) that differs, not just a generic mismatch.
- Recomputing `model_fingerprint` (§9.1) from `blocks` + the freshly computed `ExecutionSignature` does not match `plan.model_fingerprint` exactly — this covers block set/order/class/param shapes/device/dtype and the parameter-alias-group structure alongside the `ExecutionSignature` check above.
- `_validate_decisions_structure(plan, blocks)` fails (below) — **rewritten, Revision 3.1**, this replaces the previous, unimplementable claim that this function could check eligibility against `blocks`/`§10`, since neither `blocks` nor `plan.decisions` carried eligibility data before this patch (§8.6, §5).

```python
def _validate_decisions_structure(
    plan: CheckpointPlan,
    blocks: Sequence[CheckpointableBlock],
) -> None:
    decision_ids = [d.block_id for d in plan.decisions]
    block_ids = [b.block_id for b in blocks]

    duplicates = {i for i in decision_ids if decision_ids.count(i) > 1}
    if duplicates:
        raise PlanIncompatibleError(f"plan.decisions has duplicate block_id(s): {sorted(duplicates)}")

    unknown = set(decision_ids) - set(block_ids)
    if unknown:
        raise PlanIncompatibleError(f"plan.decisions references block_id(s) not in blocks: {sorted(unknown)}")

    missing = set(block_ids) - set(decision_ids)
    if missing:
        raise PlanIncompatibleError(f"plan.decisions is missing block_id(s) present in blocks: {sorted(missing)}")

    if decision_ids != block_ids:
        raise PlanIncompatibleError(
            f"plan.decisions order does not match blocks order: expected {block_ids}, got {decision_ids}"
        )

    for d in plan.decisions:
        if d.checkpointed and not d.eligible_for_checkpoint:
            raise PlanIncompatibleError(
                f"block {d.block_id!r} is checkpointed=True but eligible_for_checkpoint=False "
                f"(exclusion_reason={d.exclusion_reason!r}) — refusing to apply an unsafe plan"
            )
        if d.eligible_for_checkpoint and d.exclusion_reason is not None:
            raise PlanIncompatibleError(
                f"block {d.block_id!r}: eligible_for_checkpoint=True but exclusion_reason="
                f"{d.exclusion_reason!r} is set — inconsistent decision"
            )
        if not d.eligible_for_checkpoint and d.exclusion_reason is None:
            raise PlanIncompatibleError(
                f"block {d.block_id!r}: eligible_for_checkpoint=False but exclusion_reason=None "
                f"— inconsistent decision, an exclusion reason is required"
            )
```

This single function precisely covers every case directive 6 (this round) named —
missing, duplicate, reordered, and unknown decision block IDs, each with a
distinguishable error message, plus both directions of eligibility/exclusion-reason
inconsistency and the `checkpointed=True` + `eligible_for_checkpoint=False` unsafe
case. It reads only `plan.decisions` and `blocks`' `block_id`s — no `profiles`
dependency was introduced, preserving the already-approved decision that
`validate_plan` does not require re-profiling. See §9.1 for the disclosed residual
limitation this does **not** close (a self-consistently tampered decision).

### 9.4 `apply_plan`

1. Calls `validate_plan` first; propagates its exception unchanged. **This is the one point where `apply_plan` runs real computation:** `validate_plan`'s `ExecutionSignature` re-derivation (§9.1, §9.3) executes the declared chain once, under `torch.no_grad()`, with §6.8's state-preservation contract — one forward pass per declared block, no backward, no repeated trials. This is a one-time construction cost, not a per-training-step cost.
2. Returns `CheckpointedSequential(blocks, plan)` (§9.2). `CheckpointedSequential.__init__` itself is `O(n)` bookkeeping (building the `ModuleList`, copying `plan.decisions`/`plan.execution_signature.entry_signature`) and does not execute any block — the one real execution already happened in step 1, inside `validate_plan`.
3. `early_stop` is left at the library default (enabled) — only profiling (§6.3) disables it.
4. **Parameter sharing, corrected per directive 24:** the container holds the *same* module instances as `blocks` (no copy), so `container.parameters()` yields the exact same `nn.Parameter` tensors as the blocks' original owner. Training via an optimizer over `container.parameters()` updates the same weights as training through the blocks directly. **`container.state_dict()` keys do not match the original model's `state_dict()` keys** (the container's module nesting — `_module_list.0`, `_module_list.1`, ... — is different from wherever the blocks lived in the original model's hierarchy). State-dict cross-compatibility between the container and the original model is **not guaranteed and not tested** in v0.1 (§13); users who need to persist checkpoints should do so via the original model object, which remains a fully usable, independent `nn.Module` — declaring blocks and applying a plan does not detach or copy anything out of it.
5. **Per-call runtime check (new, correction 6):** every subsequent `container(*args, **kwargs)` call re-checks only the cheap entry-boundary signature (§9.2) before running any block — see §9.2 for the exact mechanism and cost, and §9.3 for why this is deliberately narrower than the construction-time check.

### 9.5 Correctness reference

**New, replaces revision 1's `check_correctness_against` parameter.** Since there is no longer a separate "original arbitrary model" concept once blocks are declared (§ "Revision Notes"), the natural, and only, correctness reference is the `"no_checkpoint"` plan (§8.4) built from the *same* `blocks`. `run_benchmark(..., check_correctness=True)` on a plan whose `planner_name != "no_checkpoint"` builds that reference container internally, runs one forward+backward on both with identical inputs and identical initial parameter values (no optimizer step in between), and compares outputs/gradients within `correctness_rtol`/`correctness_atol`. If `plan.planner_name == "no_checkpoint"` itself, there is nothing to compare against; `correctness_checked=False`, `correctness_reference="none"`.

### 9.6 Serialization

Unchanged mechanism from revision 1: `dataclasses.asdict` to JSON; `load_plan(path)` reads only `plan_format_version` first and raises `UnsupportedPlanVersionError` before constructing the full dataclass on any value other than the current version (`"3.1"`).

---

## 10. Rejection and handling rules

### 10.1 Stochastic modules — reconciled, directive 7

Revision 1 contained a latent contradiction: §9.2 spoke of rejecting "stochastic-unverified" blocks while the actual profiling policy allowed them. **Corrected: there is no rejection based on stochasticity anywhere in v0.1.**

- Static detection (unchanged mechanism): `profile_blocks` flags any instance of `{nn.Dropout, nn.Dropout1d/2d/3d, nn.AlphaDropout, nn.FeatureAlphaDropout}`, setting `is_stochastic=True` and populating `stochastic_submodules`.
- **`is_stochastic=True` blocks are eligible for checkpointing.** `preserve_rng_state=True` (fixed in v0.1) stashes/restores torch's CPU and per-device CUDA RNG state around each checkpoint's recompute, making `nn.Dropout`'s mask reproduce identically. A warning is recorded: "checkpoint RNG replay only covers torch's CPU/CUDA RNG; other randomness sources (Python `random`, NumPy's global RNG, etc.) are not protected."
- **`is_stochastic=None` (no known stochastic module found) blocks are also eligible** — this is the default for most blocks, since arbitrary custom `forward` code calling `torch.rand`/`random`/NumPy directly cannot be statically detected with public APIs. v0.1 does not and cannot categorically reject "unverified" stochasticity; it relies on (a) `preserve_rng_state=True`'s torch-RNG replay for the common case, and (b) the CPU/CUDA correctness tests (§11) to empirically catch any resulting numerical mismatch.
- **The only profile-level exclusions from checkpoint eligibility in v0.1 are
  §10.2 (stateful-in-train-mode) and §6.3/§10.4 (no differentiable output).**
  Shared registered buffers are rejected earlier by `declare_blocks` (§10.3),
  so their reserved exclusion enum never appears on a v0.1 `BlockProfile`.
  Nothing else is rejected on eligibility grounds.

### 10.2 Stateful modules — confirmed unchanged, directive 12

Unchanged from revision 1: static detection of `{nn.BatchNorm1d/2d/3d, nn.SyncBatchNorm, nn.InstanceNorm1d/2d/3d}` with `track_running_stats=True` **and** `module.training=True` at profiling time sets `is_stateful=True` (descriptive) **and, per correction 4, explicitly sets `eligible_for_checkpoint=False`, `exclusion_reason="stateful_mutation_in_train_mode"`** (decision fields) — the planner reads only the latter two, never re-deriving eligibility from `is_stateful`. No override in v0.1 (directive 12 explicitly confirms: keep hard rejection, no unsafe override). The same module in `.eval()` mode is not flagged: `is_stateful=False`, `eligible_for_checkpoint=True`, `exclusion_reason=None`.

### 10.3 Shared parameters and buffers — relaxed for parameters, directive 11

**Parameters:** no longer rejected. `declare_blocks` permits shared parameters;
`plan_checkpoints` computes `parameter_alias_groups` (§9.1) by `id()`-matching
across the declared blocks' `named_parameters(recurse=True)` and records them.
PyTorch's autograd correctly accumulates gradient contributions from every usage
site of a shared leaf `nn.Parameter`, including across checkpoint boundaries
(each recomputation site's backward contributes to the same `AccumulateGrad`
node) — this is a property of autograd itself, not something ckptplan implements,
and §11 adds a test (directive 25e) confirming ckptplan's checkpointing does not
break it (no double-counting or dropped contributions when one of two blocks
sharing a parameter is checkpointed).

**Buffers:** hard-rejected when shared. If any registered buffer instance (`id(b)` over `named_buffers(recurse=True)`) appears under more than one declared block, `declare_blocks` raises `BlockDeclarationError`. This includes buffers registered with `persistent=False`: PyTorch's persistence flag controls whether a buffer is serialized in `state_dict`, not whether it is mutable or participates in forward execution. A non-persistent buffer can still be mutated as a side effect of `forward`, so sharing it across blocks has the same checkpoint-recomputation hazard. There is no override in v0.1.

Because buffer-sharing is enforced at `declare_blocks` time, such a block never reaches `profile_blocks`, so `exclusion_reason="shared_mutable_buffer"` (§5, correction 4) is **reserved but structurally unreachable on a `BlockProfile` in v0.1** — it exists for forward compatibility, in case a future revision moves this check later in the pipeline, not because it is exercised today.

### 10.4 Nested outputs — two distinct, now-disambiguated notions

Revision 1 conflated two different things under "nested outputs"; this revision separates them:

1. **Within a single block's own `forward`,** arbitrary nesting (dicts, lists, tuples of tensors) is fine for `torch.utils.checkpoint` correctness, because v0.1 always uses `use_reentrant=False` (§0.3). §6.3's `_collect_differentiable_leaves` handles arbitrary nesting depth when measuring recompute cost during profiling.
2. **At an inter-block boundary** (one declared block's return value becoming the next declared block's call arguments, inside `CheckpointedSequential`), only `Tensor` / `tuple[Any, ...]` / `dict[str, Any]` are supported, per `_boundary_convert` (§9.2) — a narrower, one-level convention chosen for implementability. A block whose *internal* computation is richly nested is fine as long as what it *returns* fits one of these three shapes (or it is the last declared block, whose return value is passed through to the caller unconverted, §9.2). **This exemption is now actually enforced by the code (correction 1) — Revision 2's `CheckpointedSequential.forward` and §6.1's chain walk both called `_boundary_convert` on the last block too, contradicting this paragraph; both are fixed to only convert when a subsequent block exists.** For example, a final block may validly return a plain `list` of tensors (not itself a supported boundary type), since nothing downstream needs to consume it through `_boundary_convert`. Dict keys at a boundary must be `str` (directive 23); a non-`str` key raises `UnsupportedBoundaryError` at profiling or construction time, not a silent stringification.

### 10.5 Shape changes

Unchanged mechanism from revision 1, extended to cover kwargs (directive 22): a plan is valid only for the exact `_io_signature(example_inputs, example_kwargs)` captured when it was validated (§9.1, §9.3); any different call shape produces a different `model_fingerprint`, rejected by `validate_plan`/`apply_plan` via `PlanIncompatibleError`.

### 10.6 Incompatible serialized plans

Unchanged: §9.3 (`validate_plan`) and §9.6 (`load_plan` version gate). No partial or best-effort application of an incompatible or unrecognized-version plan.

### 10.7 Overlapping block subtrees (new, Revision 3.2)

§2's topology statement — "a linear sequence of **disjoint**, explicitly
declared submodules" — was, before Revision 3.2, descriptive prose only:
nothing in `declare_blocks` actually checked it. A parent module and one of
its own descendant submodules—or two sibling wrappers sharing a descendant
module—could be declared as separate blocks. The only thing that might reject
such a declaration accidentally, with a confusing error, was §10.3's
registered-buffer-sharing check. An overlap containing no buffer silently
succeeded.

**Rule:** for every pair of declared blocks `i ≠ j`, the recursive identity
sets `{id(m) for _, m in blocks[i].module.named_modules()}` and
`{id(m) for _, m in blocks[j].module.named_modules()}` must be disjoint. The
earlier duplicate-root check remains separately reported; this subtree check
also catches ancestor/descendant pairs and sibling wrappers that share any
descendant module.

**Determinism and order-independence:** set intersection is symmetric, so it
produces the same result regardless of which of the two overlapping blocks
was declared first in the list.

**Efficiency:** implementations should compute each declared block's
subtree-id set once (`O(n)` calls to `named_modules()` for `n` declared
blocks, each call costing time proportional to that module's own subtree
size) rather than recomputing it per pair, then do `O(n²)` set-intersection
checks—not re-walk `named_modules()` `O(n²)` times.

**Error:** raises `BlockDeclarationError` naming both block IDs (not just one),
so a caller can identify exactly which declared pair overlaps.

**What remains allowed:** sibling submodules with distinct recursive module
identities are unaffected, even though they share an undeclared parent in the
root model. Distinct declared modules may also share individual
`nn.Parameter` objects (§10.3); parameters are not modules and do not make the
module-subtree identity sets intersect.

---

## 11. CPU correctness tests and CUDA integration tests

### 11.1 CPU correctness and planner-logic tests (default CI, no GPU)

Fixture unchanged from revision 1: 4 declared blocks, each `nn.Sequential(nn.Linear(64,64), nn.ReLU())`, `float32`, CPU, `torch.manual_seed(0)`, batch `(8, 64)`.

Tests 1–3, 6–11 are carried over from revision 1 with signature updates (no `model` argument to `profile_blocks`/`validate_plan`/`apply_plan`; `target_kind="activation_budget_bytes"`; `recompute_time_upper_bound_ms_mean` field name) and one policy correction:

1. Output/gradient equivalence (checkpoint-all vs. `"no_checkpoint"`; `rtol=1e-5, atol=1e-6`).
2. No dropped gradients.
3. Planner determinism (`plan_checkpoints` called twice with identical inputs ⇒ identical `decisions`).
4. **(Corrected, directive 7)** Stochastic handling: declare a block containing `nn.Dropout`; assert `is_stochastic is True` **and that it is eligible** (`plan_checkpoints` may set `checkpointed=True` for it; assert this happens for at least one target where it is the density-optimal choice) — this replaces revision 1's now-incorrect "assert exclusion" test.
5. Stateful rejection (`nn.BatchNorm1d`, `track_running_stats=True`; `.train()` ⇒ excluded, `.eval()` ⇒ eligible). Unchanged.
6. **(Corrected, directive 11)** Shared-parameter **acceptance**: two block wrappers around the same `nn.Linear` instance; assert `declare_blocks` **succeeds**, `parameter_alias_groups` records the pair, and `model_fingerprint` changes if the sharing structure is later broken.
7. Shared-buffer rejection (new counterpart to 6): two blocks sharing a `register_buffer`d tensor instance; assert `declare_blocks` raises `BlockDeclarationError`.
8. Incompatible-model rejection via shape mismatch (directive 25d): call `validate_plan`/`apply_plan` with `example_inputs` of a different batch size than what the plan's fingerprint was built from; assert `PlanIncompatibleError`.
9. Version-gated plan loading (`plan_format_version="99.0"` ⇒ `UnsupportedPlanVersionError`).

New tests, mapped 1:1 to directive 25:

10. **Exact-byte feasibility after DP discretization (25a).** Hand-construct a `BlockProfile` list where naive `ceil`-based bucketing would have accepted a selection whose real `Σ a_i < S` (e.g., several blocks each with `a_i` just over half of `activation_bucket_bytes`); run `plan_checkpoints(planner="dynamic_programming")`; assert `feasible=True` implies real achieved savings `≥ S`, for every such constructed case. Separately unit-test the repair function directly by calling it with a deliberately insufficient hand-built selection and asserting it deterministically tops up in greedy order until the real constraint holds.
11. **Sub-bucket activation sizes (25b).** A block with `a_i = activation_bucket_bytes // 4` (so `ā_i = floor(...) = 0`); assert `plan_checkpoints` does not crash and does not report a false `feasible=True`, and assert `"greedy"`/`"uniform"` (which do not discretize) can still select this block using its real `a_i`.
12. **Exact bucket-boundary sizes.** A block with `a_i` exactly divisible by `activation_bucket_bytes`; assert `ā_i = a_i / activation_bucket_bytes` exactly (no off-by-one from floor/ceil at an exact boundary).
13. **Uniformly spaced baseline selection (25c).** Directly test `positions(k)` for `m=8` against the worked example in §8.3 (`k=1 → {0}`, `k=3 → {0,2,5}`, `k=4 → {0,2,4,6}`); then test `plan_checkpoints(planner="uniform")` picks the smallest `k` meeting a given target on a hand-built profile set.
14. **Tied parameters and gradient accumulation (25e).** Two declared blocks sharing one `nn.Linear`'s weight `nn.Parameter` (via `declare_blocks`, now accepted per test 6); build a `CheckpointedSequential` where exactly one of the two blocks is checkpointed; run forward+backward; assert the shared parameter's accumulated `.grad` equals the sum of each usage site's individually-computed contribution (computed via a separate, unchecked reference calculation using `torch.autograd.grad` on each site in isolation, then summed).
15. **No differentiable output leaves (25f).** A block whose `forward` returns only integer-dtype tensors (or an empty structure); assert `profile_blocks` raises `NoDifferentiableOutputError` for that block specifically (and, in a model with at least one other valid block, that `profile_blocks` still succeeds for the rest, per §6.3).
16. **State restoration under exception (25g — reframed; there are no hooks to clean up in this design, see §"Revision Notes", so this test targets the actual try/finally guarantees of §6.8 instead).** Construct a block whose `forward` raises on its second profiling trial; assert the exception propagates out of `profile_blocks`, and separately assert (by inspecting the block object after catching the exception) that its `.training` mode and buffer values are exactly what they were before profiling began.
17. **State restoration, happy path (25h).** Profile a block in `.eval()` mode with specific non-default buffer values; assert mode and buffer values are byte-identical after `profile_blocks` returns normally, and assert no parameter in the block has a non-`None` `.grad` left over.
18. **Kwargs and nested boundary structures (25i).** A model whose block 0 accepts `(*args, **kwargs)` and whose block 1 is fed via a `dict[str, Tensor]` boundary (block 0 returns such a dict, unpacked into block 1's keyword arguments per `_boundary_convert`); assert `profile_blocks`/`apply_plan` with `example_kwargs` populated work end to end and gradients reach every parameter.
19. **CPU profiles are `timing_only` (directive 18).** Assert every `BlockProfile` from `profile_blocks(..., device="cpu")` has `timing_only=True` and `activation_bytes_estimate is None`; assert `plan_checkpoints` on such profiles with `target_kind="activation_budget_bytes"` raises `TimingOnlyProfileError` rather than silently treating the target as trivially satisfied or infeasible.

New tests, Revision 3 (mapped to the 6 required corrections):

20. **Final-block boundary exemption (correction 1).** A 3-block model whose final block's `forward` returns a plain `list[Tensor]` (not a supported boundary type per `_boundary_convert`, §9.2); assert `apply_plan(...)` succeeds and `container(*example_inputs)` returns that list untouched — no `UnsupportedBoundaryError`. Separately assert that if the *same* list-returning module were placed as a *non-final* block (with something after it), `UnsupportedBoundaryError` is raised — confirming the exemption is genuinely position-dependent, not a blanket allowance.
21. **`ExecutionSignature` catches an intermediate-only change (correction 2).** Build a plan from a 3-block model. Mutate block 1's module in a way that changes its *output* shape without changing the container's entry `example_inputs` shape (e.g., swap in a `Linear` with a different `out_features`). Assert `validate_plan`/`apply_plan` raise `PlanIncompatibleError` — proving the check is not fooled by an unchanged entry signature, unlike the false claim corrected in §9.1.
22. **`profile_fingerprint` detects a hand-edited measurement (correction 3).** Take a real `BlockProfile` list, hand-edit one `activation_bytes_estimate` (simulating tampering or a differently measured re-run) without changing any shape/identity field, and assert `compute_profile_fingerprint` on the edited list differs from the original — and that it does **not** differ when only `forward_time_ms_std`/`num_trials` are changed (confirming the deliberate exclusion list in §9.1).
23. **Explicit eligibility fields, not inferred (correction 4).** A block with no differentiable output; assert its `BlockProfile` has `eligible_for_checkpoint=False, exclusion_reason="no_differentiable_output"` and that `is_stateful is None, is_stochastic is None` (genuinely unevaluated, not a repurposed sentinel). Assert `plan_checkpoints` never sets `checkpointed=True` for this block under any planner or target by constructing a hand-built profile where `warnings` is deliberately empty and `is_stateful`/`is_stochastic` are `True`-ish decoys, confirming the planner ignores them and only honors `eligible_for_checkpoint`.
24. **Sub-bucket blocks jointly meet target (correction 5).** Hand-construct several eligible blocks each with `a_i < activation_bucket_bytes` whose sum exceeds a target `S` that no single block can reach alone; assert `plan_checkpoints(planner="dynamic_programming")` (the default) does **not** raise `InfeasibleTargetError`, returns `feasible=True` with `dp_fallback_reason="exact_bytes_feasible_bucketed_infeasible"`, and that the real (non-bucketed) achieved savings of its `decisions` is `≥ S`.
25. **Runtime shape enforcement, matching shape (correction 6).** Build a container via `apply_plan`; call it twice with two different tensors of the *same* shape/dtype/device; assert both calls succeed and produce outputs (no `PlanIncompatibleError`), confirming the entry check does not false-positive on distinct-but-conforming inputs.
26. **Runtime shape enforcement, changed shape (correction 6).** Call the same already-built container from test 25 with a different batch size; assert `PlanIncompatibleError` is raised **before** any block's `forward` executes (verified via a call-counting wrapper on block 0 that must show zero additional calls after the exception).

New tests, Revision 3.1 (mapped to this round's blocking correction):

27. **Eligibility is copied verbatim into `CheckpointDecision` (§8.6).** For every planner (`greedy`, `dynamic_programming`, `uniform`, `checkpoint_all`, `no_checkpoint`), assert every `CheckpointDecision.eligible_for_checkpoint`/`exclusion_reason` in the returned plan exactly equals the corresponding source `BlockProfile`'s fields — including for a model with at least one excluded (stateful-in-train-mode) block, so the copy-through is exercised for both eligible and ineligible cases.
28. **`validate_plan` rejects `checkpointed=True` with `eligible_for_checkpoint=False` (§9.3).** Take a valid plan, hand-edit one decision to `checkpointed=True, eligible_for_checkpoint=False` (leaving `exclusion_reason` set, so this is *only* the unsafe-checkpoint violation, not also an inconsistency violation); assert `validate_plan`/`apply_plan` raise `PlanIncompatibleError` naming that block.
29. **`validate_plan` rejects inconsistent eligibility/exclusion pairs (§9.3).** Two hand-edited cases on an otherwise-valid plan: (a) a decision with `eligible_for_checkpoint=True` and a non-`None` `exclusion_reason`; (b) a decision with `eligible_for_checkpoint=False` and `exclusion_reason=None`. Assert `PlanIncompatibleError` for both, with distinguishable messages.
30. **`validate_plan` rejects structural decision-list tampering (§9.3).** Four hand-edited cases on a 4-block plan's `decisions`: a duplicate block_id (with another omitted), an unknown block_id not present in `blocks`, a missing block_id (list one short), and a reordered-but-complete list (same set, wrong order). Assert `PlanIncompatibleError` for all four, each identifiable from its message as the specific failure mode (not a single generic "mismatch").
31. **`compute_profile_fingerprint` detects decision-level tampering (§9.1).** Take a real `(profiles, decisions)` pair, hand-edit one decision's `checkpointed`/`eligible_for_checkpoint` without touching `profiles`; assert `compute_profile_fingerprint(profiles, edited_decisions)` differs from `compute_profile_fingerprint(profiles, original_decisions)` — confirming the fingerprint is bound to decisions, not just profiles, closing the specific gap this patch addresses (while §9.1 documents that this is a manual auditing check, not one `validate_plan` runs automatically).

Numerical tolerance for all CPU correctness assertions: `rtol=1e-5, atol=1e-6` on `float32`.

### 11.2 CUDA integration tests (separate workflow, Modal A10G — not default CI)

1. Re-run test 1 (§11.1) on CUDA with `rtol=1e-3, atol=1e-5` (documented as accounting for GPU floating-point reduction-order nondeterminism, not a checkpointing defect).
2. **Directional `peak_allocated_bytes` check (directive 8/9 — corrected metric).** On the calibrated benchmark model (§12.1), assert `run_benchmark(..., plan=checkpoint_all_plan).peak_allocated_bytes < run_benchmark(..., plan=no_checkpoint_plan).peak_allocated_bytes`. `peak_reserved_bytes` is recorded for the same runs but this assertion is **not** made about it (directive 8).
3. OOM boundary: unchanged from revision 1 — a config where `no_checkpoint` OOMs and `checkpoint_all` does not; both outcomes asserted explicitly, `BenchmarkResult.oom`/`error_message` populated for the failing configuration.
4. Warm-up/steady-state observation: unchanged, reported not gated.

---

## 12. Benchmarks

### 12.1 Models and calibration

**Rewritten per directive 16.** Revision 1 proposed a fixed, unvalidated 24-layer/hidden-1024/batch-16 transformer configuration. That fixed configuration is removed. In its place, v0.1 specifies a **calibration procedure** that must be run once on the actual A10G before the benchmark suite's model configuration is considered final, and whose output is checked into the repository (e.g. `benchmarks/calibrated_config.json`) as the fixed input to all subsequent benchmark runs.

**`toy-mlp`** (sanity/calibration-of-the-profiler model): unchanged from revision 1 — `N` identical `nn.Sequential(nn.Linear(W,W), nn.GELU())` blocks sized so `activation_bytes_estimate` is analytically predictable, checked within a tolerance to be confirmed empirically (not asserted in this document).

**`toy-transformer` calibration procedure** (replaces the fixed config):

1. Fix hidden size `1024`, `16` heads, MLP expansion `4x`, sequence length `1024`, `float32`, and vary only layer count `N` and batch size `B` — a 2-parameter search, not a fixed point.
2. **Step A — find a no-checkpoint-fits config:** starting from `N=8, B=16`, run the `"no_checkpoint"` plan; if it OOMs, halve `N` (down to a floor of `N=2`) or halve `B` (down to a floor of `B=1`) — try `N` first, then `B` — until it fits; record this as `(N_fit, B_fit)`.
3. **Step B — find a no-checkpoint-OOMs-but-checkpoint-all-fits config:** starting from `(N_fit, B_fit)`, double `N` until `"no_checkpoint"` OOMs; at that `N`, confirm `"checkpoint_all"` fits (if it also OOMs, halve `N` by one step and retry `"checkpoint_all"` — never `B`, to keep the comparison at fixed batch size). Record this as `(N_boundary, B_fit)`.
4. **Step C — find a config where the memory change exceeds measurement noise:** at 2–3 candidate `N` values between `N_fit` and `N_boundary` (inclusive), run `"no_checkpoint"` and `"checkpoint_all"` each for `num_trials=20` repeated trials (fresh process per configuration, per §12.2 protocol) and record the sample standard deviation of `peak_allocated_bytes` across those 20 trials for each. Select the smallest `N` (to minimize A10G cost) at which `|mean(peak_allocated_bytes, no_checkpoint) - mean(peak_allocated_bytes, checkpoint_all)|` exceeds `5×` the larger of the two measured standard deviations — this multiplier is a documented, adjustable starting choice, not a validated statistical guarantee.
5. The resulting `(N, B, hidden=1024, heads=16, seq_len=1024, dtype=float32)` plus the measured `peak_allocated_bytes` for `no_checkpoint` and `checkpoint_all` at that config are written to `benchmarks/calibrated_config.json` and become the fixed configuration for §12.2's baseline comparisons and §11.2's tests. Re-running calibration is required if the benchmark model's architecture changes; it is not required per-commit.

This procedure is precise enough to implement without guessing, even though its **output** (the specific `N`, `B` values) is not yet known and is explicitly not asserted by this document (directive 16, 17).

### 12.2 Baselines

Unchanged from revision 1: `no_checkpoint`, `checkpoint_all`, `uniform`, `greedy`, `dynamic_programming` — five configurations per model per target.

### 12.3 A10G protocol

Unchanged from revision 1: single GPU, fresh process per configuration, fixed seed, `num_warmup=5`/`num_trials=20` defaults, `reset_peak_memory_stats` once after warm-up, one correctness check per configuration before timed trials, `float32` required (fp16/bf16 optional and separately labeled).

### 12.4 Metrics

**Reordered and corrected per directives 8 and 9:**

- **Primary:** measured `peak_allocated_bytes` (reduction vs. `no_checkpoint`, and absolute value), per configuration — the metric most directly attributable to checkpointing's mechanism (tensor memory, not allocator reservation behavior).
- **Secondary:** measured `peak_reserved_bytes` — reported for every configuration, but **not required to decrease monotonically** with `peak_allocated_bytes`; allocator caching/fragmentation can obscure or delay a `peak_allocated_bytes` improvement in `peak_reserved_bytes`, and this is not treated as a failure or a contradiction.
- Step-time overhead (%) vs. `no_checkpoint`.
- Prediction gap: `|predicted_activation_bytes_after − peak_allocated_bytes| / peak_allocated_bytes`, reported, not gated.
- **DP vs. greedy/uniform recompute overhead at comparable measured allocated-memory reduction (directive 9):** for configurations where two planners achieve similar measured `peak_allocated_bytes` reduction (within a documented band, e.g. ±10% of each other's reduction — a starting choice, not validated), compare their `step_latency_ms_mean` / `predicted_recompute_time_upper_bound_ms` as the fairer, apples-to-apples planner-quality comparison, since DP explicitly optimizes recompute cost subject to *activation savings*, not reserved memory.

### 12.5 Release gates and reported (not gated) results

**Rewritten per directive 17** — replaces revision 1's invalid reserved-memory superiority gate (removed per directive 9) and its "proposed, not senior-approved" percentage placeholder with the four gate categories directive 17 specifies:

**Hard gates (all required before v0.1 is released):**

1. **Correctness.** 100% pass rate on the declared numerical-correctness tolerance (§11.2 item 1) across every benchmark model and configuration. Zero tolerance for failures.
2. **Deterministic planning.** `plan_checkpoints` is bit-identical across repeated calls with identical inputs (§11.1 test 3), on every benchmark model.
3. **Exact surrogate-target feasibility.** Every `feasible=True` `CheckpointPlan` from `planner="dynamic_programming"` satisfies the real (undiscretized) savings constraint (§7.2, §8.2, §11.1 test 10) on every benchmark model and target tried.
4. **Demonstrated directional peak-allocated-memory reduction.** On the calibrated `toy-transformer` configuration (§12.1), at least `checkpoint_all` shows measured `peak_allocated_bytes` strictly less than `no_checkpoint`'s (§11.2 item 2). This is directional only — no percentage is promised or asserted here.

**Reported, not gated (directive 17 — no invented percentage):**

- Magnitude of `peak_allocated_bytes`/`peak_reserved_bytes` reduction for every configuration.
- Step-time/throughput overhead for every configuration.
- Prediction gap.
- `dynamic_programming` vs. `greedy` vs. `uniform` recompute overhead at comparable measured allocated-memory reduction (§12.4).

No v0.1 release gate asserts a specific percentage of memory savings or an upper bound on throughput overhead — doing so before measurement would be exactly the kind of unverified claim `STATE.md`'s "Required Evidence" section and directive 17 forbid.

---

## 13. Explicit v0.1 non-goals

Restates `ARCHITECTURE.md`'s non-goals (unchanged there) plus items specific to this spec, **updated for this revision:**

From `ARCHITECTURE.md` (unchanged): arbitrary operation/tensor-level checkpoint placement; automatic traversal/rewriting of arbitrary autograd graphs; distributed/pipeline/tensor/data-parallel scheduling; CPU or storage offloading; `torch.compile`/CUDA graph guarantees unless separately validated; mixture-of-experts and data-dependent control flow; a Rust/C++ optimization core without measured justification; guaranteed global optimality for real peak GPU memory unless the implemented model proves that guarantee.

Additional, specific to this spec (updated):

- No automatic block discovery or per-layer-type profile reuse (§2).
- No support for `nn.DataParallel`/`DistributedDataParallel`-wrapped models (§2).
- **No support for non-block computation before, between, or after declared blocks** — the declared blocks must span the entire region ckptplan manages (§2, replaces revision 1's implicit, unimplemented "transparent wrapping" claim).
- No override for stateful-in-train-mode rejection (§10.2) — confirmed final, directive 12.
- **No support for shared *mutable buffers* across declared blocks** (§10.3) — narrower than revision 1's blanket "no parameter sharing" non-goal, since parameter sharing is now supported (directive 11).
- **No guaranteed `state_dict()` compatibility between `CheckpointedSequential` and the original model** (§9.4, directive 24) — persist checkpoints via the original model object.
- No variable-shape (dynamic batch/sequence length) support (§10.5) — enforced both at `validate_plan`/`apply_plan` construction time and, cheaply, on every `CheckpointedSequential.forward()` call via the entry-signature check (§9.2, §9.4, correction 6).
- No migration tooling between plan format versions (§9.6).
- No GUI or HTML report; structured data plus terminal output.
- No `float16`/`bfloat16` requirement for required benchmark evidence (§12.3).
- No CPU memory planning — CPU is for correctness/scheduler tests only, marked `timing_only=True` (§10 item 18, directive 18).
- No claim of "first," "model-agnostic," or unqualified "optimal" anywhere in documentation (directive 20) — the working name `ckptplan` is retained.

---

## 14. Risks and open questions for senior review

### Resolved through Revision 2 (directives 10–20) — no longer open

Infeasible-target default (raise, with best-effort retained); shared-parameter handling (allowed, tracked via alias groups); stateful-module override (none, confirmed final); early-stop asymmetry framing (accepted, field renamed to make the upper-bound nature explicit); reserved-memory superiority gate (removed as invalid, replaced with directional/reported metrics); numeric success-criteria percentage (explicitly not asserted); CPU scope (CUDA-only for memory planning, CPU marked `timing_only`); PyTorch/Python version claims (narrowed to exactly what CI is specified to run); public-claims discipline (confirmed, name retained).

### Approved in this round — no longer open

The five design judgment calls raised at the end of Revision 2 are now explicitly approved, unchanged: dropping `model` from every public function except `declare_blocks`; using the `"no_checkpoint"` plan's container over the same blocks as the sole correctness reference; the `(j*m)//k` uniform-spacing formula; greedy-order DP repair; and `TimingOnlyProfileError` as a hard rejection for `timing_only=True` profiles fed to an activation-based target.

### Resolved by Revision 3's 6 required corrections — no longer open

Final-block boundary handling (§6.1, §9.2); truthful, implementable execution-signature validation via the new `ExecutionSignature` type (§9.1, §9.3); `profile_fingerprint` now binds to the actual planning-relevant measurements (§9.1); explicit `eligible_for_checkpoint`/`exclusion_reason` fields replacing inference from `is_stateful`/`is_stochastic`/warnings (§5, §6.3, §7.1, §10.2); DP false infeasibility on sub-bucket blocks, fixed via an exact-byte check and greedy fallback (§8.2); and runtime enforcement of the static-shape promise via a per-call entry check (§9.2, §9.4).

### Resolved by Revision 3.1's blocking correction — no longer open

`validate_plan`'s stated eligibility defense-in-depth check was unimplementable with the data it received (neither `blocks` nor `plan.decisions` carried eligibility). Fixed by persisting `eligible_for_checkpoint`/`exclusion_reason` onto `CheckpointDecision` (§5), copied verbatim by one shared helper for every planner (§8.6), checked directly by a rewritten `validate_plan` that also distinguishes missing/duplicate/reordered/unknown decision block IDs individually (§9.3), and bound into `profile_fingerprint`'s inputs (§9.1). `plan_format_version` bumped to `"3.1"`.

### Resolved by Revision 3.2's two declaration-boundary corrections — no longer open

(1) Earlier specification language incorrectly suggested that
`persistent=False` made a shared buffer safe. Revision 3.2 clarifies that the
flag controls serialization only and retains the conservative identity check
over every registered buffer, with explicit regression tests. (2) Overlapping
declared-block subtrees (a module and its own descendant declared as two
separate blocks) were never rejected, contradicting §2's "disjoint...
submodules" topology statement — fixed via an explicit, symmetric,
order-independent ancestor/descendant check (new §10.7). Neither fix touches
any serialized-plan schema; `plan_format_version` remains `"3.1"`.

### Disclosed, accepted residual limitation (not a defect — recorded for explicit sign-off)

`validate_plan` cannot detect a *self-consistently* tampered decision (eligibility, exclusion reason, and checkpointed flag all changed together, remaining internally consistent) without cross-referencing the original `profiles` — which it deliberately does not take as input (§9.1, §9.3), per the already-approved decision to avoid requiring re-profiling at every application site. The only defense against this specific threat is the manual `compute_profile_fingerprint(profiles, decisions)` auditing tool, which is not run automatically. Confirm this residual gap is acceptable for v0.1, or state whether `validate_plan` should be widened to accept `profiles` (a larger API change than this patch made).

### Provisional parameters requiring empirical calibration during implementation (not senior-review blockers — implementation TODOs)

1. `activation_bucket_bytes` default (`1 << 20`) — provisional (directive 15); §11.1 tests exercise its edge behavior but do not validate the default's suitability for real models.
2. `dp_scale_guard_cells` default (`5_000_000`) — provisional (directive 14), not derived from a measured budget on any target machine.
3. `toy-transformer` calibration search parameters (§12.1: candidate `N` step sizes, the `5×` noise multiplier, the `±10%` planner-comparability band in §12.4) are documented starting choices, not validated ones.

### New design judgment calls made while executing this revision (recommend explicit confirmation)

1. **Exact `profile_fingerprint` field list (§9.1).** The specific inclusion/exclusion split (planning-relevant values in; variance/std and run metadata out) is a reasonable but not the only defensible line — confirm run metadata (`num_trials`, `pytorch_version`, etc.) genuinely should not affect the fingerprint.
2. **Greedy fallback chosen over adaptive-bucket retry for correction 5 (§8.2 step 4).** Both were sanctioned by the governing correction; greedy was chosen for simplicity and because it needs no new tunable parameters. Confirm this, or state a preference for an adaptive-bucket retry instead.
3. **`ExecutionSignature` exposed as an explicit `CheckpointPlan` field (§5), not only folded into the `model_fingerprint` hash.** This goes slightly beyond the letter of the governing correction, chosen for debuggability (a mismatch can name the specific block/side that changed, not just report "hash mismatch"). Confirm this transparency trade-off (a slightly larger serialized plan) is acceptable.
4. **`exclusion_reason="shared_mutable_buffer"` is reserved but structurally unreachable in v0.1 (§10.3)** — such blocks are rejected earlier, at `declare_blocks`, before any `BlockProfile` exists. Confirm reserving the enum value for forward compatibility (rather than omitting it) is the right call.
5. **`plan_format_version` bumped to `"3.1"`, not `"4.0"` (Revision 3.1).** This patch adds two required fields to `CheckpointDecision` and changes `compute_profile_fingerprint`'s signature — arguably comparable in kind to the `"2.0"`→`"3.0"` bump. A minor-version bump was chosen since the change is narrower in scope (one dataclass, one function signature) than Revision 3's broader set of corrections. Confirm this versioning granularity, or state a preference for treating every breaking schema change as a major bump.
