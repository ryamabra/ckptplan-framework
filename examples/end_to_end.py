#!/usr/bin/env python3
"""End-to-end ckptplan walkthrough: declare, profile, plan, apply, benchmark.

Run it directly::

    python examples/end_to_end.py

The script adapts to the device it finds. On CUDA it runs the whole pipeline.
On CPU it runs as far as the library legitimately allows and then stops at
``plan_checkpoints``, printing exactly why -- see the "CPU is timing-only"
section of the README. That stop is a deliberate guard, not a failure.
"""

from __future__ import annotations

import torch

from ckptplan import (
    TimingOnlyProfileError,
    apply_plan,
    declare_blocks,
    plan_checkpoints,
    profile_blocks,
    run_benchmark,
)

LAYERS = 4
HIDDEN = 64
HEADS = 4
SEQ_LEN = 32
BATCH = 2


def build_model(device: str) -> torch.nn.ModuleList:
    """A small transformer stack; each layer becomes one checkpointable block."""
    torch.manual_seed(0)
    return torch.nn.ModuleList(
        [
            torch.nn.TransformerEncoderLayer(
                d_model=HIDDEN,
                nhead=HEADS,
                dim_feedforward=4 * HIDDEN,
                # Dropout must be off: a stochastic block is excluded from
                # checkpointing, because recomputation would not reproduce the
                # forward pass.
                dropout=0.0,
                batch_first=True,
                device=device,
                dtype=torch.float32,
            )
            for _ in range(LAYERS)
        ]
    )


def loss_of(output: torch.Tensor) -> torch.Tensor:
    """Scalar target for the backward pass."""
    return output.float().square().mean()


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  (torch {torch.__version__})\n")

    model = build_model(device)
    example_inputs = (torch.randn(BATCH, SEQ_LEN, HIDDEN, device=device),)

    # 1. Declare the checkpointable blocks, in execution order.
    #    Blocks must be disjoint module subtrees and are given stable ids.
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])
    print(f"1. declare_blocks -> {len(blocks)} blocks: {[b.block_id for b in blocks]}\n")

    # 2. Profile them. On CUDA this measures isolated activation bytes and
    #    recomputation time. On CPU only timings are available.
    profiles = profile_blocks(
        blocks, example_inputs, device=device, dtype=torch.float32,
        num_warmup=1, num_trials=3,
    )
    first = profiles[0]
    print("2. profile_blocks:")
    print(f"     timing_only               = {first.timing_only}")
    print(f"     activation_bytes_estimate = {first.activation_bytes_estimate}")
    print(f"     activation_bytes_method   = {first.activation_bytes_method}")
    print(f"     forward_time_ms_mean      = {first.forward_time_ms_mean:.4f}")
    print(f"     eligible_for_checkpoint   = {first.eligible_for_checkpoint}\n")

    # 3. Plan. This needs real activation bytes, so it rejects CPU profiles.
    activation_total = sum(p.activation_bytes_estimate or 0 for p in profiles)
    try:
        plan = plan_checkpoints(
            profiles,
            blocks,
            target_kind="activation_budget_bytes",
            target_value=activation_total // 2,   # keep at most half the activations
            planner="dynamic_programming",
        )
    except TimingOnlyProfileError as exc:
        print("3. plan_checkpoints -> TimingOnlyProfileError (expected on CPU)")
        print(f"     {exc}\n")
        print("Activation-based planning is CUDA-only in v0.1: CPU profiles carry no")
        print("activation-byte measurements, and the planner refuses to optimize a")
        print("memory budget it cannot measure. Steps 4 and 5 need a plan, so they")
        print("are skipped here. Re-run on a CUDA device for the full pipeline.")
        return 0

    checkpointed = [d.block_id for d in plan.decisions if d.checkpointed]
    print("3. plan_checkpoints:")
    print(f"     planner                          = {plan.planner_name}")
    print(f"     checkpointed blocks              = {checkpointed}")
    print(f"     predicted_activation_bytes_after = {plan.predicted_activation_bytes_after:,}")
    print(f"     predicted_recompute_ms (upper)   = {plan.predicted_recompute_time_upper_bound_ms:.4f}")
    print(f"     feasible                         = {plan.feasible}\n")

    # 4. Apply it. The returned container reuses the original module instances
    #    and preserves parameter identity, so optimizers keep working.
    container = apply_plan(blocks, plan, example_inputs, None)
    print(f"4. apply_plan -> {type(container).__name__}")
    print(f"     output shape = {tuple(container(*example_inputs).shape)}\n")

    # 5. Benchmark, including a gradient-level correctness check against an
    #    equivalent no-checkpoint plan.
    result = run_benchmark(
        blocks, plan, example_inputs, None, loss_of,
        device=device, dtype=torch.float32,
        num_warmup=2, num_trials=5,
        check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5,
    )
    print("5. run_benchmark:")
    print(f"     peak_allocated_bytes      = {result.peak_allocated_bytes:,}")
    print(f"     step_latency_ms_mean      = {result.step_latency_ms_mean:.4f}")
    print(f"     correctness_passed        = {result.correctness_passed}")
    print(f"     max_abs_output_diff       = {result.correctness_max_abs_output_diff}")
    print(f"     max_abs_grad_diff         = {result.correctness_max_abs_grad_diff}")
    print(f"     oom                       = {result.oom}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
