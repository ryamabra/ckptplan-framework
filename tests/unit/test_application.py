import dataclasses

import pytest
import torch

from ckptplan import (
    CheckpointedSequential,
    PlanIncompatibleError,
    UnsupportedPlanVersionError,
    apply_plan,
    declare_blocks,
    plan_checkpoints,
    validate_plan,
)
from ckptplan._execution import compute_execution_signature


def _plan():
    model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Linear(3, 1))
    blocks = declare_blocks(model, [("a", model[0]), ("b", model[1])])
    signature = compute_execution_signature(blocks, (torch.ones(2, 3),), None)
    profiles = []
    from ckptplan.types import BlockProfile
    for order, block in enumerate(blocks):
        _block_id, input_signature, output_signature = signature.block_signatures[order]
        profiles.append(BlockProfile(
            block_id=block.block_id, order=order, device="cpu", dtype="torch.float32",
            input_shape_signature=input_signature, output_shape_signature=output_signature,
            param_count=sum(p.numel() for p in block.module.parameters()), trainable_param_count=sum(p.numel() for p in block.module.parameters()),
            timing_only=False, activation_bytes_estimate=10, activation_bytes_method="isolated_forward_delta",
            forward_time_ms_mean=1, forward_time_ms_std=0, recompute_time_upper_bound_ms_mean=1, recompute_time_upper_bound_ms_std=0,
            recompute_time_source="measured_full_recompute_early_stop_disabled", num_warmup=1, num_trials=1,
            is_stochastic=False, is_stateful=False, stochastic_submodules=(), stateful_submodules=(),
            eligible_for_checkpoint=True, exclusion_reason=None, warnings=(), pytorch_version="test", profiler_version="test",
        ))
    return model, blocks, plan_checkpoints(profiles, blocks, target_kind="activation_saving_fraction", target_value=0.5, planner="checkpoint_all")


def test_apply_plan_runs_and_preserves_parameter_identity():
    _model, blocks, plan = _plan()
    container = apply_plan(blocks, plan, (torch.ones(2, 3),))
    assert isinstance(container, CheckpointedSequential)
    assert list(container.parameters())[0] is next(blocks[0].module.parameters())
    assert container(torch.ones(2, 3)).shape == (2, 1)


def test_runtime_shape_mismatch_is_rejected_before_execution():
    _model, blocks, plan = _plan()
    container = apply_plan(blocks, plan, (torch.ones(2, 3),))
    with pytest.raises(PlanIncompatibleError, match="entry boundary"):
        container(torch.ones(4, 3))


def test_validation_rejects_fingerprint_and_version_mismatch():
    _model, blocks, plan = _plan()
    altered = dataclasses.replace(plan, model_fingerprint="bad")
    with pytest.raises(PlanIncompatibleError, match="fingerprint"):
        validate_plan(altered, blocks, (torch.ones(2, 3),))
    altered = dataclasses.replace(plan, plan_format_version="99.0")
    with pytest.raises(UnsupportedPlanVersionError):
        validate_plan(altered, blocks, (torch.ones(2, 3),))
