"""ckptplan.

Profiles checkpointable PyTorch model blocks and selects a gradient-checkpointing
plan for a user-specified activation-memory budget, minimizing estimated
recomputation overhead. See ``MVP_SPEC.md`` (accepted design, Revision 3.4) and
``ARCHITECTURE.md`` for the full v0.1 design.

Plan application is provided through ``validate_plan`` and ``apply_plan``.
"""

from ckptplan.api import apply_plan, declare_blocks, plan_checkpoints, profile_blocks, run_benchmark, validate_plan, CheckpointedSequential
from ckptplan.errors import (
    BlockDeclarationError,
    InfeasibleTargetError,
    NoDifferentiableOutputError,
    PlannerScaleError,
    TimingOnlyProfileError,
    UnsupportedBoundaryError,
    PlanIncompatibleError,
    UnsupportedPlanVersionError,
)
from ckptplan.planning.planner import compute_profile_fingerprint
from ckptplan.reporting import ConfigComparison, compare_results
from ckptplan.types import (
    BlockProfile,
    CheckpointDecision,
    CheckpointPlan,
    CheckpointableBlock,
    ExecutionSignature,
    BenchmarkResult,
    ExclusionReason,
)

__version__ = "0.1.0rc1"

__all__ = [
    "__version__",
    "declare_blocks",
    "profile_blocks",
    "plan_checkpoints",
    "validate_plan",
    "apply_plan",
    "CheckpointedSequential",
    "run_benchmark",
    "compute_profile_fingerprint",
    "compare_results",
    "ConfigComparison",
    "CheckpointableBlock",
    "BlockProfile",
    "CheckpointDecision",
    "CheckpointPlan",
    "ExecutionSignature",
    "BenchmarkResult",
    "ExclusionReason",
    "BlockDeclarationError",
    "UnsupportedBoundaryError",
    "NoDifferentiableOutputError",
    "InfeasibleTargetError",
    "PlannerScaleError",
    "TimingOnlyProfileError",
    "PlanIncompatibleError",
    "UnsupportedPlanVersionError",
]
