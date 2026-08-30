"""Typed exceptions raised by ckptplan's public API. See MVP_SPEC.md Sec 4.

This slice implements declaration, profiling, and pure-planning errors.
Application errors are included for the plan validation/application stage.
"""

from __future__ import annotations


class BlockDeclarationError(Exception):
    """Raised by ``declare_blocks()`` when a declared block list violates one
    of the invariants in MVP_SPEC.md Sec 3: non-empty/unique block_id, module
    reachability, duplicate module instance, overlapping subtree, or shared registered-buffer
    identity across blocks.
    """


class UnsupportedBoundaryError(Exception):
    """Raised when a block boundary violates MVP_SPEC.md Sec 9.2."""


class NoDifferentiableOutputError(Exception):
    """Raised when every profiled block lacks differentiable floating output."""

    def __init__(
        self,
        block_id: str | None = None,
        *,
        block_ids: tuple[str, ...] = (),
    ) -> None:
        ids = block_ids or ((block_id,) if block_id is not None else ())
        self.block_id = block_id
        self.block_ids = ids
        joined = ", ".join(repr(item) for item in ids) or "<unknown>"
        super().__init__(
            "no declared block produced a differentiable floating tensor output; "
            f"affected block_ids: {joined}"
        )


class InfeasibleTargetError(Exception):
    """Raised when the requested activation-saving target cannot be met."""

    def __init__(self, required_savings_bytes: float, max_achievable_bytes: float) -> None:
        self.required_savings_bytes = required_savings_bytes
        self.max_achievable_bytes = max_achievable_bytes
        super().__init__(
            f"activation target requires {required_savings_bytes} bytes of savings, "
            f"but eligible blocks can save at most {max_achievable_bytes} bytes"
        )


class PlannerScaleError(Exception):
    """Raised when dynamic-programming discretization exceeds its scale guard."""


class TimingOnlyProfileError(Exception):
    """Raised when activation-based planning receives CPU timing-only profiles."""


class PlanIncompatibleError(Exception):
    """Raised when a plan cannot be safely applied to the supplied blocks."""


class UnsupportedPlanVersionError(Exception):
    """Raised when a serialized or in-memory plan uses an unknown format."""
