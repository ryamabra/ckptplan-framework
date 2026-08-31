"""Focused deterministic tests for the pure plan_checkpoints core."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from ckptplan import (
    BlockProfile,
    CheckpointDecision,
    InfeasibleTargetError,
    PlannerScaleError,
    TimingOnlyProfileError,
    compute_profile_fingerprint,
    declare_blocks,
    plan_checkpoints,
)


def _profiles(values, *, eligible=None, timing_only=False):
    eligible = [True] * len(values) if eligible is None else eligible
    result = []
    for order, (activation, cost) in enumerate(values):
        result.append(
            BlockProfile(
                block_id=f"b{order}",
                order=order,
                device="cpu",
                dtype="torch.float32",
                input_shape_signature="ArgsTuple[Tensor(shape=(2, 4), dtype=torch.float32, device=cpu)], KwargsDict{}",
                output_shape_signature="ArgsTuple[Tensor(shape=(2, 4), dtype=torch.float32, device=cpu)], KwargsDict{}",
                param_count=20,
                trainable_param_count=20,
                timing_only=timing_only,
                activation_bytes_estimate=None if timing_only else activation,
                activation_bytes_method=None if timing_only else "isolated_forward_delta",
                forward_time_ms_mean=1.0,
                forward_time_ms_std=0.1,
                recompute_time_upper_bound_ms_mean=cost,
                recompute_time_upper_bound_ms_std=0.1,
                recompute_time_source="measured_full_recompute_early_stop_disabled",
                num_warmup=0,
                num_trials=2,
                is_stochastic=None,
                is_stateful=False,
                stochastic_submodules=(),
                stateful_submodules=(),
                eligible_for_checkpoint=eligible[order],
                exclusion_reason=None if eligible[order] else "stateful_mutation_in_train_mode",
                warnings=(),
                pytorch_version="2.13.0",
                profiler_version="0.1.0.dev0",
            )
        )
    return tuple(result)


def _blocks(n, *, shared_parameter=False):
    model = torch.nn.Module()
    shared = torch.nn.Parameter(torch.ones(4, 4)) if shared_parameter else None
    modules = []
    for index in range(n):
        module = torch.nn.Linear(4, 4)
        if shared is not None:
            module.weight = shared
        modules.append(module)
        model.add_module(f"b{index}", module)
    return declare_blocks(model, [(f"b{index}", module) for index, module in enumerate(modules)])


def test_checkpoint_plan_schema_and_model_provenance_are_populated() -> None:
    profiles = _profiles([(10, 2), (20, 4), (30, 1)])
    blocks = _blocks(3, shared_parameter=True)
    plan = plan_checkpoints(
        profiles,
        blocks,
        target_kind="activation_saving_fraction",
        target_value=0.5,
        planner="greedy",
    )

    assert dataclasses.is_dataclass(plan)
    assert plan.plan_format_version == "3.1"
    assert plan.feasible is True
    assert plan.predicted_activation_bytes_before == 60
    assert plan.predicted_activation_bytes_after == 30
    assert plan.predicted_recompute_time_upper_bound_ms == 1
    assert plan.use_reentrant is False
    assert plan.preserve_rng_state is True
    assert plan.execution_signature.entry_signature == profiles[0].input_shape_signature
    assert plan.execution_signature.block_order == ("b0", "b1", "b2")
    assert plan.parameter_alias_groups == (("b0.weight", "b1.weight", "b2.weight"),)
    assert plan.profile_fingerprint
    assert plan.model_fingerprint


@pytest.mark.parametrize("planner", ["greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"])
def test_all_planners_copy_eligibility_fields_and_are_deterministic(planner: str) -> None:
    profiles = _profiles([(10, 2), (20, 4), (30, 1)], eligible=[True, False, True])
    blocks = _blocks(3)
    kwargs = dict(
        target_kind="activation_budget_bytes",
        target_value=20,
        planner=planner,
        activation_bucket_bytes=10,
        on_infeasible="best_effort",
    )
    first = plan_checkpoints(profiles, blocks, **kwargs)
    second = plan_checkpoints(profiles, blocks, **kwargs)
    assert first.decisions == second.decisions
    assert [decision.eligible_for_checkpoint for decision in first.decisions] == [True, False, True]
    assert first.decisions[1].exclusion_reason == "stateful_mutation_in_train_mode"
    assert first.decisions[1].checkpointed is False


def test_greedy_uses_density_then_declared_tie_breaks() -> None:
    profiles = _profiles([(10, 5), (20, 4), (15, 3)])
    plan = plan_checkpoints(profiles, _blocks(3), target_kind="activation_budget_bytes", target_value=20, planner="greedy")
    assert [d.block_id for d in plan.decisions if d.checkpointed] == ["b1", "b2"]


def test_dynamic_programming_minimizes_cost_and_uses_exact_real_bytes() -> None:
    profiles = _profiles([(60, 8), (40, 2), (30, 1)])
    plan = plan_checkpoints(
        profiles,
        _blocks(3),
        target_kind="activation_budget_bytes",
        target_value=60,
        planner="dynamic_programming",
        activation_bucket_bytes=10,
    )
    assert [d.block_id for d in plan.decisions if d.checkpointed] == ["b1", "b2"]
    assert plan.predicted_activation_bytes_after == 60
    assert plan.predicted_recompute_time_upper_bound_ms == 3


def test_dynamic_programming_sub_bucket_falls_back_to_greedy() -> None:
    profiles = _profiles([(3, 5), (4, 1), (5, 2)])
    plan = plan_checkpoints(
        profiles,
        _blocks(3),
        target_kind="activation_saving_fraction",
        target_value=0.5,
        planner="dynamic_programming",
        activation_bucket_bytes=10,
    )
    assert plan.feasible is True
    assert plan.dp_fallback_reason == "exact_bytes_feasible_bucketed_infeasible"
    assert plan.dp_repair_applied is False
    assert plan.predicted_activation_bytes_before - plan.predicted_activation_bytes_after >= 6


def test_uniform_positions_and_smallest_k_target() -> None:
    profiles = _profiles([(10, 1), (20, 1), (30, 1), (40, 1), (50, 1), (60, 1), (70, 1), (80, 1)])
    plan = plan_checkpoints(
        profiles,
        _blocks(8),
        target_kind="activation_budget_bytes",
        target_value=210,
        planner="uniform",
    )
    assert [d.block_id for d in plan.decisions if d.checkpointed] == ["b0", "b2", "b4", "b6"]


def test_no_checkpoint_and_checkpoint_all_baselines() -> None:
    profiles = _profiles([(10, 2), (20, 4)])
    blocks = _blocks(2)
    none = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=20, planner="no_checkpoint", on_infeasible="best_effort")
    all_plan = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=20, planner="checkpoint_all")
    assert none.feasible is False
    assert all(decision.checkpointed for decision in none.decisions)
    assert all_plan.feasible is True
    assert all(decision.checkpointed for decision in all_plan.decisions)


def test_infeasible_raise_and_best_effort() -> None:
    profiles = _profiles([(10, 2), (20, 4)])
    blocks = _blocks(2)
    with pytest.raises(InfeasibleTargetError) as excinfo:
        plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=-100, planner="greedy")
    assert excinfo.value.max_achievable_bytes == 30
    best = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=-100, planner="greedy", on_infeasible="best_effort")
    assert best.feasible is False
    assert all(decision.checkpointed for decision in best.decisions)


def test_dp_scale_guard_and_timing_only_rejection() -> None:
    profiles = _profiles([(100, 1), (100, 1)])
    blocks = _blocks(2)
    with pytest.raises(PlannerScaleError):
        plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="dynamic_programming", activation_bucket_bytes=1, dp_scale_guard_cells=1)
    with pytest.raises(TimingOnlyProfileError):
        plan_checkpoints(_profiles([(10, 1)], timing_only=True), _blocks(1), target_kind="activation_budget_bytes", target_value=0, planner="greedy")


def test_profile_fingerprint_requires_matching_decisions() -> None:
    profiles = _profiles([(10, 1), (20, 1)])
    plan = plan_checkpoints(profiles, _blocks(2), target_kind="activation_budget_bytes", target_value=20, planner="checkpoint_all")
    with pytest.raises(ValueError, match="exactly one entry"):
        compute_profile_fingerprint(profiles, plan.decisions[:1])


def test_profile_fingerprint_includes_decisions_but_excludes_variance_metadata() -> None:
    profiles = _profiles([(10, 2), (20, 4)])
    blocks = _blocks(2)
    plan = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=20, planner="greedy")
    changed = dataclasses.replace(profiles[0], forward_time_ms_std=99.0, num_trials=999)
    assert compute_profile_fingerprint((changed, profiles[1]), plan.decisions) == plan.profile_fingerprint
    edited = dataclasses.replace(plan.decisions[0], checkpointed=not plan.decisions[0].checkpointed)
    assert compute_profile_fingerprint(profiles, (edited, plan.decisions[1])) != plan.profile_fingerprint


def test_profile_block_correspondence_rejects_length_ids_order_device_and_dtype() -> None:
    profiles = _profiles([(10, 2), (20, 4)])
    blocks = _blocks(2)
    with pytest.raises(ValueError, match="same length"):
        plan_checkpoints(profiles[:1], blocks, target_kind="activation_budget_bytes", target_value=0, planner="greedy")
    with pytest.raises(ValueError, match="mismatch"):
        plan_checkpoints((dataclasses.replace(profiles[0], block_id="wrong"), profiles[1]), blocks, target_kind="activation_budget_bytes", target_value=0, planner="greedy")
    with pytest.raises(ValueError, match="dtype"):
        plan_checkpoints((dataclasses.replace(profiles[0], dtype="torch.float64"), profiles[1]), blocks, target_kind="activation_budget_bytes", target_value=0, planner="greedy")
