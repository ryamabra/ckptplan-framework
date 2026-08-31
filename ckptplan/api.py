"""Public API functions for ckptplan. See MVP_SPEC.md Sec 4.

This module exposes declaration, profiling, planning, validation, and application.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import platform
import statistics
import sys
import time
import dataclasses
from typing import Any, Callable, Literal

import torch

from ckptplan.errors import (
    BlockDeclarationError,
    PlanIncompatibleError,
    UnsupportedPlanVersionError,
)
from ckptplan._execution import (
    _boundary_convert,
    _io_signature,
    compute_execution_signature,
)
from ckptplan.planning.planner import (
    compute_profile_fingerprint,
    plan_checkpoints as _plan_checkpoints,
)
from ckptplan.profiling.profiler import profile_blocks as _profile_blocks
from ckptplan.types import BenchmarkResult, BlockProfile, CheckpointPlan, CheckpointableBlock
from ckptplan.planning.planner import PLAN_FORMAT_VERSION, _model_fingerprint, _parameter_alias_groups


def _validate_decisions_structure(plan: CheckpointPlan, blocks: Sequence[CheckpointableBlock]) -> None:
    decision_ids = [d.block_id for d in plan.decisions]
    block_ids = [b.block_id for b in blocks]
    duplicates = sorted({item for item in decision_ids if decision_ids.count(item) > 1})
    if duplicates:
        raise PlanIncompatibleError(f"plan.decisions has duplicate block_id(s): {duplicates}")
    unknown = sorted(set(decision_ids) - set(block_ids))
    if unknown:
        raise PlanIncompatibleError(f"plan.decisions references block_id(s) not in blocks: {unknown}")
    missing = sorted(set(block_ids) - set(decision_ids))
    if missing:
        raise PlanIncompatibleError(f"plan.decisions is missing block_id(s) present in blocks: {missing}")
    if decision_ids != block_ids:
        raise PlanIncompatibleError(
            f"plan.decisions order does not match blocks order: expected {block_ids}, got {decision_ids}"
        )
    for decision in plan.decisions:
        if decision.checkpointed and not decision.eligible_for_checkpoint:
            raise PlanIncompatibleError(
                f"block {decision.block_id!r} is checkpointed=True but eligible_for_checkpoint=False"
            )
        if decision.eligible_for_checkpoint and decision.exclusion_reason is not None:
            raise PlanIncompatibleError(f"block {decision.block_id!r}: eligible_for_checkpoint=True but exclusion_reason is set")
        if not decision.eligible_for_checkpoint and decision.exclusion_reason is None:
            raise PlanIncompatibleError(f"block {decision.block_id!r}: eligible_for_checkpoint=False but exclusion_reason=None")


def validate_plan(
    plan: CheckpointPlan,
    blocks: Sequence[CheckpointableBlock],
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
) -> None:
    """Validate plan provenance, decisions, and the complete execution shape."""
    if plan.plan_format_version != PLAN_FORMAT_VERSION:
        raise UnsupportedPlanVersionError(
            f"unsupported plan_format_version {plan.plan_format_version!r}; expected {PLAN_FORMAT_VERSION!r}"
        )
    checked_blocks = tuple(blocks)
    _validate_decisions_structure(plan, checked_blocks)
    actual_signature = compute_execution_signature(checked_blocks, example_inputs, example_kwargs)
    expected = plan.execution_signature
    if actual_signature != expected:
        for index, (actual, wanted) in enumerate(zip(actual_signature.block_signatures, expected.block_signatures)):
            if actual[1] != wanted[1]:
                raise PlanIncompatibleError(f"block {actual[0]!r} input signature differs from the plan")
            if actual[2] != wanted[2]:
                raise PlanIncompatibleError(f"block {actual[0]!r} output signature differs from the plan")
        raise PlanIncompatibleError("execution signature differs from the plan")
    aliases = _parameter_alias_groups(checked_blocks)
    actual_fingerprint = _model_fingerprint(checked_blocks, actual_signature, aliases)
    if actual_fingerprint != plan.model_fingerprint:
        raise PlanIncompatibleError("model fingerprint differs from the plan")


class CheckpointedSequential(torch.nn.Module):
    """Compose declared blocks, checkpointing exactly the plan's selected blocks."""

    def __init__(self, blocks: Sequence[CheckpointableBlock], plan: CheckpointPlan) -> None:
        super().__init__()
        self._module_list = torch.nn.ModuleList([block.module for block in blocks])
        self._checkpointed = tuple(decision.checkpointed for decision in plan.decisions)
        self._entry_signature = plan.execution_signature.entry_signature
        self.plan = plan

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        if _io_signature(args, kwargs) != self._entry_signature:
            raise PlanIncompatibleError("input shape/dtype/device at the entry boundary does not match the validated plan")
        current_args, current_kwargs = args, dict(kwargs)
        output: Any = None
        for index, (module, checkpointed) in enumerate(zip(self._module_list, self._checkpointed)):
            if checkpointed:
                output = torch.utils.checkpoint.checkpoint(
                    module, *current_args, **current_kwargs,
                    use_reentrant=self.plan.use_reentrant,
                    preserve_rng_state=self.plan.preserve_rng_state,
                )
            else:
                output = module(*current_args, **current_kwargs)
            if index < len(self._module_list) - 1:
                current_args, current_kwargs = _boundary_convert(output)
        return output


def apply_plan(
    blocks: Sequence[CheckpointableBlock],
    plan: CheckpointPlan,
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None = None,
) -> CheckpointedSequential:
    validate_plan(plan, blocks, example_inputs, example_kwargs)
    return CheckpointedSequential(blocks, plan)


def _flatten_tensors(value: Any) -> list[torch.Tensor]:
    if torch.is_tensor(value):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in _flatten_tensors(item)]
    if isinstance(value, dict):
        return [tensor for key in sorted(value) for tensor in _flatten_tensors(value[key])]
    return []


def run_benchmark(
    blocks: Sequence[CheckpointableBlock],
    plan: CheckpointPlan,
    example_inputs: tuple[Any, ...],
    example_kwargs: Mapping[str, Any] | None,
    make_target: Callable[[Any], torch.Tensor],
    *,
    device: torch.device | str,
    dtype: torch.dtype | None = None,
    num_warmup: int = 5,
    num_trials: int = 20,
    check_correctness: bool = True,
    correctness_rtol: float = 1e-5,
    correctness_atol: float = 1e-6,
) -> BenchmarkResult:
    if num_warmup < 0 or num_trials < 1:
        raise ValueError("num_warmup must be non-negative and num_trials must be positive")
    normalized_device = torch.device(device)
    container = apply_plan(blocks, plan, example_inputs, example_kwargs)
    correctness_checked = bool(check_correctness and plan.planner_name != "no_checkpoint")
    output_diff = grad_diff = None
    correctness_passed = None
    correctness_error = None
    if correctness_checked:
        reference = None
        checkpointed_grads = None
        try:
            reference_plan = dataclasses.replace(
                plan,
                plan_id=plan.plan_id + "-reference",
                planner_name="no_checkpoint",
                decisions=tuple(dataclasses.replace(d, checkpointed=False) for d in plan.decisions),
            )
            reference = apply_plan(blocks, reference_plan, example_inputs, example_kwargs)
            with torch.no_grad():
                output_diff = max((float((a - b).abs().max()) for a, b in zip(_flatten_tensors(container(*example_inputs, **dict(example_kwargs or {}))), _flatten_tensors(reference(*example_inputs, **dict(example_kwargs or {}))))), default=0.0)
            # apply_plan preserves parameter identity, so the container and the
            # reference share their .grad buffers. The two backward passes must
            # therefore run sequentially with the checkpointed gradients
            # snapshotted in between: reading a.grad and b.grad at the same time
            # would compare every parameter with itself and report 0.0 whether or
            # not checkpointing is correct.
            container.zero_grad(set_to_none=True)
            make_target(container(*example_inputs, **dict(example_kwargs or {}))).backward()
            checkpointed_grads = [None if p.grad is None else p.grad.detach().clone() for p in container.parameters()]
            container.zero_grad(set_to_none=True)
            reference.zero_grad(set_to_none=True)
            make_target(reference(*example_inputs, **dict(example_kwargs or {}))).backward()
            grad_diff = max((float((saved - p.grad).abs().max()) for saved, p in zip(checkpointed_grads, reference.parameters()) if saved is not None and p.grad is not None), default=0.0)
            correctness_passed = output_diff <= correctness_atol + correctness_rtol * max(output_diff, 1.0) and grad_diff <= correctness_atol + correctness_rtol * max(grad_diff, 1.0)
        except torch.cuda.OutOfMemoryError as exc:
            # The check could not be completed. Record that and claim no verdict:
            # "no evidence" must never be reported as "failed". The benchmark's
            # own structured OOM result is left untouched.
            correctness_passed = None
            correctness_error = f"correctness check did not complete (CUDA OOM): {exc}"
            checkpointed_grads = None
            if normalized_device.type == "cuda":
                torch.cuda.empty_cache()
        finally:
            checkpointed_grads = None
            container.zero_grad(set_to_none=True)
            if reference is not None:
                reference.zero_grad(set_to_none=True)

    latencies: list[float] = []
    peak_allocated = peak_reserved = 0
    error_message = None
    oom = False
    try:
        for _ in range(num_warmup):
            container.zero_grad(set_to_none=True)
            make_target(container(*example_inputs, **dict(example_kwargs or {}))).backward()
        if normalized_device.type == "cuda":
            torch.cuda.synchronize(normalized_device)
            torch.cuda.reset_peak_memory_stats(normalized_device)
        for _ in range(num_trials):
            container.zero_grad(set_to_none=True)
            if normalized_device.type == "cuda":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                make_target(container(*example_inputs, **dict(example_kwargs or {}))).backward()
                end.record(); end.synchronize()
                latencies.append(float(start.elapsed_time(end)))
            else:
                start = time.perf_counter()
                make_target(container(*example_inputs, **dict(example_kwargs or {}))).backward()
                latencies.append((time.perf_counter() - start) * 1000.0)
        if normalized_device.type == "cuda":
            peak_allocated = torch.cuda.max_memory_allocated(normalized_device)
            peak_reserved = torch.cuda.max_memory_reserved(normalized_device)
    except torch.cuda.OutOfMemoryError as exc:
        oom, error_message = True, str(exc)
    if correctness_error is not None:
        error_message = correctness_error if error_message is None else f"{correctness_error}; {error_message}"
    mean = statistics.fmean(latencies) if latencies else 0.0
    p50 = statistics.median(latencies) if latencies else 0.0
    p95 = (statistics.quantiles(latencies, n=20, method="inclusive")[18] if len(latencies) >= 2 else p50)
    batch_tensors = _flatten_tensors(example_inputs)
    samples = batch_tensors[0].shape[0] if batch_tensors and batch_tensors[0].ndim else 1
    return BenchmarkResult(
        config_name=plan.planner_name, plan_id=plan.plan_id,
        device_name=torch.cuda.get_device_name(normalized_device) if normalized_device.type == "cuda" else "cpu",
        pytorch_version=torch.__version__, python_version=platform.python_version(),
        dtype=str(dtype or (batch_tensors[0].dtype if batch_tensors else torch.float32)),
        batch_shape=str(tuple(batch_tensors[0].shape)) if batch_tensors else "()",
        warmup_trials=num_warmup, measured_trials=len(latencies),
        peak_allocated_bytes=int(peak_allocated), peak_reserved_bytes=int(peak_reserved),
        step_latency_ms=tuple(latencies), step_latency_ms_mean=mean, step_latency_ms_p50=p50,
        step_latency_ms_p95=p95, throughput_samples_per_sec=(samples * 1000.0 / mean if mean else 0.0),
        correctness_checked=correctness_checked, correctness_reference="no_checkpoint_plan" if correctness_checked else "none",
        correctness_max_abs_output_diff=output_diff, correctness_max_abs_grad_diff=grad_diff,
        correctness_passed=correctness_passed, oom=oom, error_message=error_message,
        environment={"platform": platform.platform(), "device": str(normalized_device)},
    )


def declare_blocks(
    model: torch.nn.Module,
    blocks: Sequence[tuple[str, torch.nn.Module]],
) -> tuple[CheckpointableBlock, ...]:
    """Validate and freeze an ordered set of checkpointable blocks.

    See MVP_SPEC.md Sec 3. Does not run the model and does not require
    example inputs -- only ``model.named_modules()``/``named_buffers()`` are
    walked, never ``model.forward()`` or any block's ``forward()``.

    Per MVP_SPEC.md Sec 10.3, shared ``nn.Parameter`` instances across
    declared blocks are permitted (not rejected here); computing and
    recording the resulting parameter alias groups is deferred to the
    planning/fingerprinting stage (Sec 9.1's ``compute_parameter_alias_groups``),
    which is out of scope for this slice.

    Declared blocks must be disjoint subtrees (MVP_SPEC.md Sec 2, Sec 10.7):
    their recursive module-identity sets must not intersect.

    Raises:
        BlockDeclarationError: on any invariant violation.
    """
    # 1. block_id: non-empty, unique str.
    seen_ids: dict[str, int] = {}
    for position, (block_id, _module) in enumerate(blocks):
        if not isinstance(block_id, str) or not block_id:
            raise BlockDeclarationError(
                f"block_id at position {position} must be a non-empty str, got {block_id!r}"
            )
        if block_id in seen_ids:
            raise BlockDeclarationError(
                f"duplicate block_id {block_id!r} at positions {seen_ids[block_id]} and {position}"
            )
        seen_ids[block_id] = position

    # 2. Every submodule must be reachable from `model` via model.named_modules()
    #    identity (id()-based, not name-based).
    reachable_ids = {id(m) for _name, m in model.named_modules()}
    for position, (block_id, module) in enumerate(blocks):
        if id(module) not in reachable_ids:
            raise BlockDeclarationError(
                f"block {block_id!r} (position {position}): module is not reachable "
                "from `model` via model.named_modules() by identity"
            )

    # 3. No two entries reference the same module instance.
    module_id_to_block: dict[int, str] = {}
    for block_id, module in blocks:
        existing = module_id_to_block.get(id(module))
        if existing is not None:
            raise BlockDeclarationError(
                f"blocks {existing!r} and {block_id!r} reference the same module instance"
            )
        module_id_to_block[id(module)] = block_id

    # 4. Declared blocks must be disjoint subtrees (MVP_SPEC.md Sec 2, Sec
    #    10.7). Precompute each declared module's recursive module-identity
    #    set once so the pairwise check catches both ancestor/descendant pairs
    #    and sibling wrappers that share a descendant module.
    subtree_ids: list[frozenset[int]] = [
        frozenset(id(m) for _, m in module.named_modules())
        for _block_id, module in blocks
    ]
    for i, (block_id_i, _module_i) in enumerate(blocks):
        for j in range(i + 1, len(blocks)):
            block_id_j, _module_j = blocks[j]
            if subtree_ids[i] & subtree_ids[j]:
                raise BlockDeclarationError(
                    f"blocks {block_id_i!r} and {block_id_j!r} overlap: their "
                    "module subtrees share at least one module instance; "
                    "declared blocks must be disjoint subtrees"
                )

    # 5. No two entries' modules share a registered buffer instance. Shared
    #    parameters are permitted (MVP_SPEC.md Sec 10.3). named_buffers()'s
    #    default remove_duplicate=True means the same tensor registered under
    #    two names *within one block* is yielded only once, so it is never
    #    seen as a false cross-block collision here -- relied on deliberately,
    #    not incidentally; see the corresponding test in test_declare_blocks.py.
    #    Persistence controls state_dict serialization only; it does not make
    #    a buffer immutable or prevent forward/recomputation side effects.
    #    Therefore persistent=False is not a safety exemption.
    buffer_id_to_block: dict[int, tuple[str, str]] = {}
    for block_id, module in blocks:
        for buffer_name, buffer in module.named_buffers(recurse=True):
            existing = buffer_id_to_block.get(id(buffer))
            if existing is not None:
                other_block_id, other_buffer_name = existing
                raise BlockDeclarationError(
                    f"blocks {other_block_id!r} ({other_buffer_name!r}) and "
                    f"{block_id!r} ({buffer_name!r}) share the same registered "
                    "buffer instance; shared registered buffers are not supported"
                )
            buffer_id_to_block[id(buffer)] = (block_id, buffer_name)

    return tuple(
        CheckpointableBlock(block_id=block_id, order=order, module=module)
        for order, (block_id, module) in enumerate(blocks)
    )


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
    """Profile declared blocks on CPU using the methodology in MVP_SPEC.md Sec 6."""
    return _profile_blocks(
        blocks,
        example_inputs,
        example_kwargs,
        device=device,
        dtype=dtype,
        num_warmup=num_warmup,
        num_trials=num_trials,
    )


def plan_checkpoints(
    profiles: Sequence[BlockProfile],
    blocks: Sequence[CheckpointableBlock],
    *,
    target_kind: Literal["activation_budget_bytes", "activation_saving_fraction"],
    target_value: float,
    planner: Literal[
        "greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"
    ] = "dynamic_programming",
    activation_bucket_bytes: int = 1 << 20,
    dp_scale_guard_cells: int = 5_000_000,
    on_infeasible: Literal["raise", "best_effort"] = "raise",
) -> CheckpointPlan:
    return _plan_checkpoints(
        profiles,
        blocks,
        target_kind=target_kind,
        target_value=target_value,
        planner=planner,
        activation_bucket_bytes=activation_bucket_bytes,
        dp_scale_guard_cells=dp_scale_guard_cells,
        on_infeasible=on_infeasible,
    )
