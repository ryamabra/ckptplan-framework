"""Deterministic activation-saving planners.

No model execution, checkpoint wrapping, CUDA calls, or reporting belongs here.
The planner consumes profiles and declared blocks solely for cost selection and
provenance fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import math
import dataclasses
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal

import torch

from ckptplan.errors import InfeasibleTargetError, PlannerScaleError, TimingOnlyProfileError
from ckptplan.types import (
    BlockProfile,
    CheckpointDecision,
    CheckpointPlan,
    CheckpointableBlock,
    ExecutionSignature,
)

PlannerName = Literal["greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"]
TargetKind = Literal["activation_budget_bytes", "activation_saving_fraction"]
PLANNER_VERSION = "0.1.0.dev0"
PLAN_FORMAT_VERSION = "3.1"

_BASE_ASSUMPTIONS = (
    "Activation memory is additive/independent across blocks.",
    "Recompute cost is additive across checkpointed blocks.",
    "recompute_time_upper_bound_ms_mean is a conservative cost-model input, not an unbiased runtime prediction: it was measured with early stopping disabled. Real training, which uses PyTorch's default early-stop behavior, typically incurs recompute time less than or equal to this value.",
    "The activation budget/target concerns block-local activation memory only (Σ a_i), not total process GPU memory, and not peak_allocated_bytes/peak_reserved_bytes directly.",
    "Profiles are valid only for the exact input/output shape+kwargs signature captured during profiling.",
)


def _canonical_hash(value: Any) -> str:
    def normalize(item: Any) -> Any:
        if dataclasses.is_dataclass(item):
            return normalize(dataclasses.asdict(item))
        if isinstance(item, tuple):
            return [normalize(value) for value in item]
        if isinstance(item, list):
            return [normalize(value) for value in item]
        if isinstance(item, dict):
            return {str(key): normalize(value) for key, value in item.items()}
        return item

    encoded = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_profiles(profiles: Sequence[BlockProfile]) -> tuple[BlockProfile, ...]:
    result = tuple(profiles)
    if not result:
        raise ValueError("plan_checkpoints requires at least one BlockProfile")
    ids = [profile.block_id for profile in result]
    if len(set(ids)) != len(ids):
        raise ValueError("profiles must contain unique block_id values")
    for expected_order, profile in enumerate(result):
        if profile.order != expected_order:
            raise ValueError(
                f"profile {profile.block_id!r} has order {profile.order}, expected {expected_order}"
            )
        if not (profile.timing_only and profile.activation_bytes_estimate is None) and (
                isinstance(profile.activation_bytes_estimate, bool)
                or not isinstance(profile.activation_bytes_estimate, int)
                or profile.activation_bytes_estimate < 0):
            raise ValueError(
                f"profile {profile.block_id!r} must have a non-negative activation_bytes_estimate"
            )
        if not math.isfinite(profile.recompute_time_upper_bound_ms_mean) or profile.recompute_time_upper_bound_ms_mean < 0:
            raise ValueError(
                f"profile {profile.block_id!r} must have a finite non-negative recompute cost"
            )
    return result


def _validate_blocks(
    profiles: Sequence[BlockProfile], blocks: Sequence[CheckpointableBlock]
) -> tuple[CheckpointableBlock, ...]:
    result = tuple(blocks)
    if len(result) != len(profiles):
        raise ValueError("blocks and profiles must have the same length and order")
    block_ids = [block.block_id for block in result]
    if len(set(block_ids)) != len(block_ids):
        raise ValueError("blocks must contain unique block_id values")
    if any(block.order != index for index, block in enumerate(result)):
        raise ValueError("blocks must have contiguous order values matching their sequence")
    for profile, block in zip(profiles, result):
        if profile.block_id != block.block_id or profile.order != block.order:
            raise ValueError(
                f"block/profile mismatch at order {profile.order}: "
                f"profile={profile.block_id!r}, block={block.block_id!r}"
            )
        try:
            profile_device = torch.device(profile.device)
        except (RuntimeError, TypeError, ValueError) as exc:
            raise ValueError(f"profile {profile.block_id!r} has invalid device {profile.device!r}") from exc
        tensor_values = [
            *tuple(block.module.parameters()),
            *tuple(block.module.buffers()),
        ]
        for value in tensor_values:
            compatible_device = value.device.type == profile_device.type and (
                profile_device.index is None
                or value.device.index is None
                or value.device.index == profile_device.index
            )
            if not compatible_device:
                raise ValueError(
                    f"profile {profile.block_id!r} device {profile.device!r} does not "
                    f"match block {block.block_id!r} tensor device {value.device}"
                )
            if value.is_floating_point() and str(value.dtype) != profile.dtype:
                raise ValueError(
                    f"profile {profile.block_id!r} dtype {profile.dtype!r} does not "
                    f"match block {block.block_id!r} tensor dtype {value.dtype}"
                )
    return result


def _parameter_alias_groups(blocks: Sequence[CheckpointableBlock]) -> tuple[tuple[str, ...], ...]:
    aliases: dict[int, list[str]] = {}
    for block in blocks:
        for name, parameter in block.module.named_parameters(recurse=True):
            aliases.setdefault(id(parameter), []).append(f"{block.block_id}.{name}")
    return tuple(sorted(tuple(sorted(names)) for names in aliases.values() if len(names) > 1))


def _execution_signature(profiles: Sequence[BlockProfile]) -> ExecutionSignature:
    return ExecutionSignature(
        entry_signature=profiles[0].input_shape_signature,
        block_signatures=tuple(
            (profile.block_id, profile.input_shape_signature, profile.output_shape_signature)
            for profile in profiles
        ),
        block_order=tuple(profile.block_id for profile in profiles),
    )


def _profile_fingerprint(
    profiles: Sequence[BlockProfile], decisions: Sequence[CheckpointDecision]
) -> str:
    decision_by_id = {decision.block_id: decision for decision in decisions}
    payload = []
    for profile in profiles:
        decision = decision_by_id[profile.block_id]
        payload.append(
            {
                "block_id": profile.block_id,
                "order": profile.order,
                "device": profile.device,
                "dtype": profile.dtype,
                "input_shape_signature": profile.input_shape_signature,
                "output_shape_signature": profile.output_shape_signature,
                "timing_only": profile.timing_only,
                "activation_bytes_estimate": profile.activation_bytes_estimate,
                "activation_bytes_method": profile.activation_bytes_method,
                "recompute_time_upper_bound_ms_mean": profile.recompute_time_upper_bound_ms_mean,
                "recompute_time_source": profile.recompute_time_source,
                "eligible_for_checkpoint": profile.eligible_for_checkpoint,
                "exclusion_reason": profile.exclusion_reason,
                "decision": {
                    "checkpointed": decision.checkpointed,
                    "eligible_for_checkpoint": decision.eligible_for_checkpoint,
                    "exclusion_reason": decision.exclusion_reason,
                },
            }
        )
    return _canonical_hash(payload)


def _model_fingerprint(
    blocks: Sequence[CheckpointableBlock],
    execution_signature: ExecutionSignature,
    aliases: tuple[tuple[str, ...], ...],
) -> str:
    entries = []
    for block in blocks:
        parameters = [
            {"name": name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
            for name, parameter in block.module.named_parameters(recurse=True)
        ]
        tensors = [*block.module.parameters(), *block.module.buffers()]
        representative = tensors[0] if tensors else None
        entries.append(
            {
                "block_id": block.block_id,
                "order": block.order,
                "module_qualified_class_name": f"{type(block.module).__module__}.{type(block.module).__qualname__}",
                "param_shapes": parameters,
                "device": str(representative.device) if representative is not None else "cpu",
                "dtype": str(representative.dtype) if representative is not None else "torch.float32",
            }
        )
    return _canonical_hash(
        {
            "blocks": entries,
            "execution_signature": execution_signature,
            "parameter_alias_groups": aliases,
        }
    )


def _build_decisions(
    profiles: Sequence[BlockProfile], checkpointed_ids: set[str]
) -> tuple[CheckpointDecision, ...]:
    return tuple(
        CheckpointDecision(
            block_id=profile.block_id,
            checkpointed=profile.block_id in checkpointed_ids and profile.eligible_for_checkpoint,
            eligible_for_checkpoint=profile.eligible_for_checkpoint,
            exclusion_reason=profile.exclusion_reason,
        )
        for profile in profiles
    )


def _density_key(profile: BlockProfile) -> tuple[float, int, float, int]:
    cost = profile.recompute_time_upper_bound_ms_mean
    density = math.inf if cost == 0 else profile.activation_bytes_estimate / cost
    return (-density, -profile.activation_bytes_estimate, cost, profile.order)


def _greedy_ids(eligible: Sequence[BlockProfile], target: float) -> set[str]:
    selected: set[str] = set()
    achieved = 0.0
    if target <= 0:
        return selected
    for profile in sorted(eligible, key=_density_key):
        selected.add(profile.block_id)
        achieved += profile.activation_bytes_estimate or 0
        if achieved >= target:
            break
    return selected


def _uniform_ids(eligible: Sequence[BlockProfile], target: float) -> set[str] | None:
    ordered = sorted(eligible, key=lambda profile: profile.order)
    if target <= 0:
        return set()
    m = len(ordered)
    for k in range(1, m + 1):
        positions = {(j * m) // k for j in range(k)}
        selected = {ordered[index].block_id for index in positions}
        if sum((profile.activation_bytes_estimate or 0) for profile in ordered if profile.block_id in selected) >= target:
            return selected
    return None


def _dynamic_ids(
    eligible: Sequence[BlockProfile],
    target: float,
    bucket: int,
    guard: int,
) -> tuple[set[str] | None, bool, str | None]:
    if target <= 0:
        return set(), False, None
    target_units = math.ceil(target / bucket)
    if len(eligible) * (target_units + 1) > guard:
        raise PlannerScaleError(
            f"dynamic-programming table has {len(eligible) * (target_units + 1)} cells, "
            f"exceeding dp_scale_guard_cells={guard}; increase activation_bucket_bytes "
            "or use planner='greedy'"
        )
    inf = math.inf
    table = [[inf] * (target_units + 1) for _ in range(len(eligible) + 1)]
    table[0][0] = 0.0
    for row, profile in enumerate(eligible, start=1):
        units = (profile.activation_bytes_estimate or 0) // bucket
        cost = profile.recompute_time_upper_bound_ms_mean
        for achieved in range(target_units + 1):
            without = table[row - 1][achieved]
            with_block = inf
            if table[row - 1][max(0, achieved - units)] != inf:
                with_block = table[row - 1][max(0, achieved - units)] + cost
            table[row][achieved] = min(without, with_block)

    if table[-1][target_units] == inf:
        exact_total = sum((profile.activation_bytes_estimate or 0) for profile in eligible)
        if exact_total >= target:
            return _greedy_ids(eligible, target), False, "exact_bytes_feasible_bucketed_infeasible"
        return None, False, None

    selected: set[str] = set()
    achieved = target_units
    for row in range(len(eligible), 0, -1):
        profile = eligible[row - 1]
        without = table[row - 1][achieved]
        units = (profile.activation_bytes_estimate or 0) // bucket
        prior = table[row - 1][max(0, achieved - units)]
        if prior != inf and prior + profile.recompute_time_upper_bound_ms_mean < without:
            selected.add(profile.block_id)
            achieved = max(0, achieved - units)
    real_achieved = sum(
        profile.activation_bytes_estimate or 0
        for profile in eligible
        if profile.block_id in selected
    )
    if real_achieved >= target:
        return selected, False, None

    # Defensive repair for any future discretization change that violates the
    # floor/ceil proof. Use exactly greedy's deterministic ordering.
    for profile in sorted(eligible, key=_density_key):
        if profile.block_id not in selected:
            selected.add(profile.block_id)
            real_achieved += profile.activation_bytes_estimate or 0
            if real_achieved >= target:
                return selected, True, None
    return None, True, None


def plan_checkpoints(
    profiles: Sequence[BlockProfile],
    blocks: Sequence[CheckpointableBlock],
    *,
    target_kind: TargetKind,
    target_value: float,
    planner: PlannerName = "dynamic_programming",
    activation_bucket_bytes: int = 1 << 20,
    dp_scale_guard_cells: int = 5_000_000,
    on_infeasible: Literal["raise", "best_effort"] = "raise",
) -> CheckpointPlan:
    """Select a deterministic checkpoint subset under the §7 surrogate model."""
    checked_profiles = _validate_profiles(profiles)
    _validate_blocks(checked_profiles, blocks)
    if any(profile.timing_only for profile in checked_profiles):
        raise TimingOnlyProfileError(
            "activation-based planning requires real activation-byte profiles; "
            "CPU timing_only profiles are not valid planner inputs"
        )
    if planner not in {"greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"}:
        raise ValueError(f"unknown planner {planner!r}")
    if target_kind not in {"activation_budget_bytes", "activation_saving_fraction"}:
        raise ValueError(f"unknown target_kind {target_kind!r}")
    if not math.isfinite(target_value):
        raise ValueError("target_value must be finite")
    if on_infeasible not in {"raise", "best_effort"}:
        raise ValueError(f"unknown on_infeasible policy {on_infeasible!r}")
    if planner == "dynamic_programming":
        if isinstance(activation_bucket_bytes, bool) or not isinstance(activation_bucket_bytes, int) or activation_bucket_bytes <= 0:
            raise ValueError("activation_bucket_bytes must be a positive int")
        if isinstance(dp_scale_guard_cells, bool) or not isinstance(dp_scale_guard_cells, int) or dp_scale_guard_cells <= 0:
            raise ValueError("dp_scale_guard_cells must be a positive int")

    total_activation = sum(profile.activation_bytes_estimate or 0 for profile in checked_profiles)
    if target_kind == "activation_budget_bytes":
        saving_target = max(0.0, total_activation - target_value)
    else:
        if not 0 < target_value <= 1:
            raise ValueError("activation_saving_fraction must be in (0, 1]")
        saving_target = target_value * total_activation

    eligible = tuple(profile for profile in checked_profiles if profile.eligible_for_checkpoint)
    selected: set[str] | None
    dp_repair = False
    dp_fallback_reason: str | None = None
    if planner == "greedy":
        selected = _greedy_ids(eligible, saving_target)
    elif planner == "uniform":
        selected = _uniform_ids(eligible, saving_target)
    elif planner == "checkpoint_all":
        selected = {profile.block_id for profile in eligible}
    elif planner == "no_checkpoint":
        selected = set()
    else:
        selected, dp_repair, dp_fallback_reason = _dynamic_ids(
            eligible, saving_target, activation_bucket_bytes, dp_scale_guard_cells
        )

    achieved = sum(
        profile.activation_bytes_estimate or 0
        for profile in eligible
        if selected is not None and profile.block_id in selected
    )
    feasible = achieved >= saving_target
    if not feasible and on_infeasible == "raise":
        raise InfeasibleTargetError(saving_target, sum(profile.activation_bytes_estimate or 0 for profile in eligible))
    if not feasible and on_infeasible == "best_effort":
        selected = {profile.block_id for profile in eligible}
        achieved = sum(profile.activation_bytes_estimate or 0 for profile in eligible)

    decisions = _build_decisions(checked_profiles, selected or set())
    execution_signature = _execution_signature(checked_profiles)
    aliases = _parameter_alias_groups(blocks)
    model_fingerprint = _model_fingerprint(blocks, execution_signature, aliases)
    profile_fingerprint = _profile_fingerprint(checked_profiles, decisions)

    assumptions = list(_BASE_ASSUMPTIONS)
    if planner == "greedy":
        assumptions.append("Greedy selection uses activation-by-recompute-cost density and is not guaranteed cost-optimal.")
    if planner == "dynamic_programming":
        if dp_repair:
            assumptions.append("Dynamic programming required deterministic greedy-order top-up repair to satisfy the exact byte constraint.")
        if dp_fallback_reason:
            assumptions.append("Dynamic programming fell back to greedy selection because exact bytes were feasible but bucketed search was infeasible.")
    if not feasible:
        assumptions.append("This best-effort plan is infeasible for the requested target and checkpoints every eligible block.")

    payload = {
        "planner": planner,
        "target_kind": target_kind,
        "target_value": target_value,
        "decisions": decisions,
        "profile_fingerprint": profile_fingerprint,
        "model_fingerprint": model_fingerprint,
    }
    return CheckpointPlan(
        plan_id=_canonical_hash(payload)[:24],
        plan_format_version=PLAN_FORMAT_VERSION,
        created_at=datetime.now(timezone.utc).isoformat(),
        planner_name=planner,
        planner_version=PLANNER_VERSION,
        target_kind=target_kind,
        target_value=target_value,
        activation_bucket_bytes=activation_bucket_bytes if planner == "dynamic_programming" else None,
        dp_repair_applied=dp_repair,
        dp_fallback_reason=dp_fallback_reason,
        decisions=decisions,
        feasible=feasible,
        predicted_activation_bytes_before=total_activation,
        predicted_activation_bytes_after=total_activation - achieved,
        predicted_recompute_time_upper_bound_ms=sum(
            profile.recompute_time_upper_bound_ms_mean
            for profile in checked_profiles
            if selected is not None and profile.block_id in selected
        ),
        parameter_alias_groups=aliases,
        execution_signature=execution_signature,
        profile_fingerprint=profile_fingerprint,
        model_fingerprint=model_fingerprint,
        use_reentrant=False,
        preserve_rng_state=True,
        assumptions=tuple(assumptions),
    )


def compute_profile_fingerprint(
    profiles: Sequence[BlockProfile], decisions: Sequence[CheckpointDecision]
) -> str:
    """Public provenance helper for auditing a profile/decision pair."""
    checked_profiles = _validate_profiles(profiles)
    checked_decisions = tuple(decisions)
    expected_ids = tuple(profile.block_id for profile in checked_profiles)
    actual_ids = tuple(decision.block_id for decision in checked_decisions)
    if actual_ids != expected_ids:
        raise ValueError("decisions must contain exactly one entry per profile in matching order")
    return _profile_fingerprint(checked_profiles, checked_decisions)
