"""Core data schemas for ckptplan. See MVP_SPEC.md Sec 5.

This slice implements the declaration, profiling, and pure-planning schemas.
Application and benchmark schemas remain deferred.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Literal, Optional

import torch

BlockId = str
ExclusionReason = Literal[
    "stateful_mutation_in_train_mode",
    "shared_mutable_buffer",
    "no_differentiable_output",
]


@dataclasses.dataclass(frozen=True)
class CheckpointableBlock:
    """One declared checkpointable block. See MVP_SPEC.md Sec 3, Sec 5.

    block_id: user-supplied, non-empty, unique within a declare_blocks() call.
    order: 0-based execution index, assigned from list position by declare_blocks.
    module: the actual submodule instance, shared by reference with the
        caller's model -- never copied.

    frozen=True prevents reassigning these three fields (e.g. block.module = x
    raises FrozenInstanceError); it does not and cannot make the referenced
    module itself immutable -- module's own parameters/buffers remain
    ordinarily mutable, exactly as they are on the caller's model.
    """

    block_id: BlockId
    order: int
    module: torch.nn.Module


@dataclasses.dataclass(frozen=True)
class BlockProfile:
    """Measured cost and compatibility metadata for one declared block."""

    block_id: BlockId
    order: int
    device: str
    dtype: str
    input_shape_signature: str
    output_shape_signature: str
    param_count: int
    trainable_param_count: int

    timing_only: bool
    activation_bytes_estimate: Optional[int]
    activation_bytes_method: Optional[Literal["isolated_forward_delta"]]

    forward_time_ms_mean: float
    forward_time_ms_std: float
    recompute_time_upper_bound_ms_mean: float
    recompute_time_upper_bound_ms_std: float
    recompute_time_source: Literal["measured_full_recompute_early_stop_disabled"]

    num_warmup: int
    num_trials: int

    is_stochastic: Optional[bool]
    is_stateful: Optional[bool]
    stochastic_submodules: tuple[str, ...]
    stateful_submodules: tuple[str, ...]

    eligible_for_checkpoint: bool
    exclusion_reason: Optional[ExclusionReason]

    warnings: tuple[str, ...]

    pytorch_version: str
    profiler_version: str


@dataclasses.dataclass(frozen=True)
class CheckpointDecision:
    block_id: BlockId
    checkpointed: bool
    eligible_for_checkpoint: bool
    exclusion_reason: Optional[ExclusionReason]


@dataclasses.dataclass(frozen=True)
class ExecutionSignature:
    entry_signature: str
    block_signatures: tuple[tuple[BlockId, str, str], ...]
    block_order: tuple[BlockId, ...]


@dataclasses.dataclass(frozen=True)
class CheckpointPlan:
    plan_id: str
    plan_format_version: str
    created_at: str

    planner_name: Literal[
        "greedy", "dynamic_programming", "uniform", "checkpoint_all", "no_checkpoint"
    ]
    planner_version: str

    target_kind: Literal["activation_budget_bytes", "activation_saving_fraction"]
    target_value: float
    activation_bucket_bytes: Optional[int]
    dp_repair_applied: bool
    dp_fallback_reason: Optional[Literal["exact_bytes_feasible_bucketed_infeasible"]]

    decisions: tuple[CheckpointDecision, ...]
    feasible: bool

    predicted_activation_bytes_before: int
    predicted_activation_bytes_after: int
    predicted_recompute_time_upper_bound_ms: float

    parameter_alias_groups: tuple[tuple[str, ...], ...]
    execution_signature: ExecutionSignature
    profile_fingerprint: str
    model_fingerprint: str

    use_reentrant: bool
    preserve_rng_state: bool
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
    peak_allocated_bytes: int
    peak_reserved_bytes: int
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
    environment: dict[str, Any]
