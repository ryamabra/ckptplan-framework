"""Regression tests for ``run_benchmark``'s correctness path.

These cover a bug where the gradient comparison and the correctness verdict
lived entirely inside the ``except torch.cuda.OutOfMemoryError`` handler, so a
run that completed normally reported ``correctness_max_abs_grad_diff=None`` and
``correctness_passed=None``, and a verdict could only ever be produced by first
hitting an OOM. Everything here runs on CPU; the OOM paths are exercised by
injecting ``torch.cuda.OutOfMemoryError``, which is importable without CUDA.
"""

import pytest
import torch

import ckptplan.api as api
from ckptplan import declare_blocks, plan_checkpoints, run_benchmark
from ckptplan._execution import compute_execution_signature
from ckptplan.types import BlockProfile


def _cpu_plan(planner: str = "checkpoint_all"):
    """Build a CPU model, blocks, inputs and a plan.

    CPU profiling is timing-only and ``plan_checkpoints`` rejects timing-only
    profiles, so activation-bearing profiles are synthesized here exactly as
    ``tests/unit/test_application.py`` does.
    """
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Linear(4, 2))
    blocks = declare_blocks(model, [("a", model[0]), ("b", model[1])])
    inputs = (torch.ones(2, 4),)
    signature = compute_execution_signature(blocks, inputs, None)

    profiles = []
    for order, block in enumerate(blocks):
        _block_id, input_signature, output_signature = signature.block_signatures[order]
        params = sum(p.numel() for p in block.module.parameters())
        profiles.append(
            BlockProfile(
                block_id=block.block_id, order=order, device="cpu", dtype="torch.float32",
                input_shape_signature=input_signature, output_shape_signature=output_signature,
                param_count=params, trainable_param_count=params,
                timing_only=False, activation_bytes_estimate=10,
                activation_bytes_method="isolated_forward_delta",
                forward_time_ms_mean=1, forward_time_ms_std=0,
                recompute_time_upper_bound_ms_mean=1, recompute_time_upper_bound_ms_std=0,
                recompute_time_source="measured_full_recompute_early_stop_disabled",
                num_warmup=1, num_trials=1, is_stochastic=False, is_stateful=False,
                stochastic_submodules=(), stateful_submodules=(),
                eligible_for_checkpoint=True, exclusion_reason=None, warnings=(),
                pytorch_version="test", profiler_version="test",
            )
        )
    if planner == "no_checkpoint":
        # A no-checkpoint plan saves nothing, so it can only meet a budget equal
        # to the full activation total; a saving fraction would be infeasible.
        plan = plan_checkpoints(
            profiles, blocks, target_kind="activation_budget_bytes",
            target_value=sum(p.activation_bytes_estimate for p in profiles), planner=planner,
        )
    else:
        plan = plan_checkpoints(
            profiles, blocks, target_kind="activation_saving_fraction",
            target_value=0.5, planner=planner,
        )
    return model, blocks, inputs, plan


def _target(out):
    return out.float().square().mean()


# --- the reported bug ------------------------------------------------------------


def test_correctness_verdict_is_reached_without_an_oom() -> None:
    """The verdict must not require passing through the OOM handler."""
    _model, blocks, inputs, plan = _cpu_plan()
    result = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    assert result.oom is False
    assert result.correctness_checked is True
    assert result.correctness_max_abs_output_diff is not None
    assert result.correctness_max_abs_grad_diff is not None, "grad diff must be computed on the normal path"
    assert result.correctness_passed is not None, "a verdict must be reached without an OOM"
    assert result.correctness_passed is True
    assert result.correctness_reference == "no_checkpoint_plan"


def test_checkpointed_gradients_match_the_reference_on_cpu() -> None:
    _model, blocks, inputs, plan = _cpu_plan()
    result = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    # Recomputation replays identical ops in identical order for this model.
    assert result.correctness_max_abs_grad_diff == pytest.approx(0.0, abs=1e-6)
    assert result.correctness_max_abs_output_diff == pytest.approx(0.0, abs=1e-6)


# --- the comparison must not be vacuous ------------------------------------------


def test_container_and_reference_share_parameter_objects() -> None:
    """Why the comparison needs a snapshot rather than two live .grad reads.

    ``apply_plan`` preserves parameter identity, so the checkpointed container
    and the no-checkpoint reference expose the *same* tensor objects. Reading
    ``a.grad`` and ``b.grad`` simultaneously therefore compares each parameter
    with itself and yields 0.0 no matter what the gradients are.
    """
    import dataclasses

    _model, blocks, inputs, plan = _cpu_plan()
    reference_plan = dataclasses.replace(
        plan, plan_id=plan.plan_id + "-reference", planner_name="no_checkpoint",
        decisions=tuple(dataclasses.replace(d, checkpointed=False) for d in plan.decisions),
    )
    container = api.apply_plan(blocks, plan, inputs, None)
    reference = api.apply_plan(blocks, reference_plan, inputs, None)
    pairs = list(zip(container.parameters(), reference.parameters()))
    assert pairs
    assert all(a is b for a, b in pairs)


def test_gradient_comparison_detects_a_real_difference() -> None:
    """A genuinely different backward must produce a non-zero diff and a failure.

    Guards against a comparison that reports 0.0 unconditionally.
    """
    _model, blocks, inputs, plan = _cpu_plan()
    calls = {"n": 0}

    def diverging_target(out):
        calls["n"] += 1
        # First call is the checkpointed pass, second is the reference pass.
        return out.float().square().mean() * (1.0 if calls["n"] == 1 else 3.0)

    result = run_benchmark(
        blocks, plan, inputs, None, diverging_target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    assert result.correctness_max_abs_grad_diff is not None
    assert result.correctness_max_abs_grad_diff > 0.0
    assert result.correctness_passed is False


# --- OOM paths -------------------------------------------------------------------


def test_oom_building_the_reference_does_not_crash_and_claims_no_verdict(monkeypatch) -> None:
    """The old handler used ``reference`` before it was bound and raised."""
    _model, blocks, inputs, plan = _cpu_plan()
    real_apply = api.apply_plan
    calls = {"n": 0}

    def flaky_apply(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:  # the reference apply_plan
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(api, "apply_plan", flaky_apply)
    result = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    assert result.correctness_checked is True
    # "Could not complete" is never reported as "failed".
    assert result.correctness_passed is None
    assert result.correctness_max_abs_grad_diff is None
    assert "correctness check did not complete" in (result.error_message or "")
    # The benchmark itself ran fine and is not marked OOM.
    assert result.oom is False
    assert result.measured_trials == 2


def test_oom_during_the_correctness_backward_claims_no_verdict() -> None:
    _model, blocks, inputs, plan = _cpu_plan()
    calls = {"n": 0}

    def flaky_target(out):
        calls["n"] += 1
        if calls["n"] == 1:  # the checkpointed correctness pass
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        return out.float().square().mean()

    result = run_benchmark(
        blocks, plan, inputs, None, flaky_target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    assert result.correctness_passed is None
    assert result.correctness_max_abs_grad_diff is None
    # The forward-output comparison completed before the failure and is kept.
    assert result.correctness_max_abs_output_diff is not None
    assert "correctness check did not complete" in (result.error_message or "")


def test_correctness_oom_does_not_mask_a_benchmark_oom() -> None:
    _model, blocks, inputs, plan = _cpu_plan()

    def always_oom(out):
        raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")

    result = run_benchmark(
        blocks, plan, inputs, None, always_oom,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    assert result.oom is True
    assert result.correctness_passed is None
    message = result.error_message or ""
    assert "correctness check did not complete" in message
    assert "simulated CUDA OOM" in message


# --- opting out ------------------------------------------------------------------


def test_check_correctness_false_leaves_every_field_unset() -> None:
    _model, blocks, inputs, plan = _cpu_plan()
    result = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=False,
    )
    assert result.correctness_checked is False
    assert result.correctness_reference == "none"
    assert result.correctness_passed is None
    assert result.correctness_max_abs_grad_diff is None
    assert result.correctness_max_abs_output_diff is None


def test_a_no_checkpoint_plan_is_not_compared_against_itself() -> None:
    _model, blocks, inputs, plan = _cpu_plan(planner="no_checkpoint")
    result = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=2, check_correctness=True,
    )
    assert result.correctness_checked is False
    assert result.correctness_passed is None


# --- the correctness phase must not pollute the timing phase ---------------------


def test_timing_phase_records_trials_and_leaves_grads_clean() -> None:
    model, blocks, inputs, plan = _cpu_plan()
    result = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=3, check_correctness=True,
    )
    assert result.measured_trials == 3
    assert len(result.step_latency_ms) == 3
    assert result.step_latency_ms_mean > 0.0
    # Gradients from the final measured trial remain; the correctness snapshot
    # must not have leaked an extra accumulation into them.
    reference_only = run_benchmark(
        blocks, plan, inputs, None, _target,
        device="cpu", num_warmup=1, num_trials=3, check_correctness=False,
    )
    assert reference_only.measured_trials == 3
    assert all(p.grad is not None for p in model.parameters())
