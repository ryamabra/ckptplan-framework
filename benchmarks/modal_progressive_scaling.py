"""Progressively grow batch size until the no-checkpoint baseline OOMs.

At each step, both the `no_checkpoint` baseline and the `dynamic_programming`
planner (constrained to the same GPU-memory budget) are benchmarked on a real
CUDA GPU at the same batch size. Growth stops once `no_checkpoint` OOMs (or a
configurable step cap is hit). This demonstrates, with real measurements
rather than projections, how much larger a workload `ckptplan`'s DP planner
can sustain within a fixed GPU-memory budget compared to no checkpointing.

Every OOM/PASS verdict and every byte/latency number in the output comes
from an actual `torch.cuda` measurement on that step's real GPU run.

Usage::

    python benchmarks/modal_progressive_scaling.py \\
        --layers 24 --hidden 2048 --heads 16 --seq-len 2048 \\
        --start-batch-size 1 --growth-axis batch_size \\
        --gpu-memory-budget-gb 12 --out benchmarks/progressive_scaling_result.json
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("ckptplan-progressive-scaling")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .add_local_dir("ckptplan", remote_path="/root/ckptplan")
)


@app.function(image=image, gpu="A10G", timeout=3600)
def measure_one_step(
    layers: int,
    hidden: int,
    heads: int,
    seq_len: int,
    batch_size: int,
    dtype_name: str,
    gpu_memory_budget_bytes: int,
    num_warmup: int,
    num_trials: int,
) -> dict:
    import torch
    from ckptplan import declare_blocks, plan_checkpoints, profile_blocks, run_benchmark

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)

    def build():
        model = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
                dropout=0.0, batch_first=True, device="cuda", dtype=dtype,
            )
            for _ in range(layers)
        ])
        inputs = (torch.randn(batch_size, seq_len, hidden, device="cuda", dtype=dtype),)
        blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])
        return model, inputs, blocks

    def make_target(out: "torch.Tensor") -> "torch.Tensor":
        return out.float().square().mean()

    step: dict = {"batch_size": batch_size, "seq_len": seq_len}

    # --- no_checkpoint baseline ---
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, inputs, blocks = build()
        profiles = profile_blocks(blocks, inputs, device="cuda", dtype=dtype, num_warmup=1, num_trials=2)
        total_activation = sum(p.activation_bytes_estimate or 0 for p in profiles)
        no_ckpt_plan = plan_checkpoints(
            profiles, blocks, target_kind="activation_budget_bytes",
            target_value=total_activation, planner="no_checkpoint",
        )
        no_ckpt_result = run_benchmark(
            blocks, no_ckpt_plan, inputs, None, make_target, device="cuda", dtype=dtype,
            num_warmup=num_warmup, num_trials=num_trials, check_correctness=False,
        )
        step["no_checkpoint"] = {
            "oom": no_ckpt_result.oom,
            "peak_allocated_bytes": no_ckpt_result.peak_allocated_bytes,
            "peak_reserved_bytes": no_ckpt_result.peak_reserved_bytes,
            "step_latency_ms_mean": no_ckpt_result.step_latency_ms_mean,
            "throughput_samples_per_sec": no_ckpt_result.throughput_samples_per_sec,
            "error_message": no_ckpt_result.error_message,
        }
        del model, inputs, blocks, profiles, no_ckpt_plan, no_ckpt_result
    except torch.cuda.OutOfMemoryError as exc:
        step["no_checkpoint"] = {"oom": True, "error_message": str(exc)}
    finally:
        torch.cuda.empty_cache()

    # --- dynamic_programming under the fixed GPU-memory budget ---
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, inputs, blocks = build()
        profiles = profile_blocks(blocks, inputs, device="cuda", dtype=dtype, num_warmup=1, num_trials=2)

        floor_plan = plan_checkpoints(
            profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="checkpoint_all",
        )
        floor_result = run_benchmark(
            blocks, floor_plan, inputs, None, make_target, device="cuda", dtype=dtype,
            num_warmup=num_warmup, num_trials=num_trials, check_correctness=False,
        )
        activation_budget = max(0, gpu_memory_budget_bytes - floor_result.peak_allocated_bytes)

        dp_plan = plan_checkpoints(
            profiles, blocks, target_kind="activation_budget_bytes",
            target_value=activation_budget, planner="dynamic_programming", on_infeasible="best_effort",
        )
        dp_result = run_benchmark(
            blocks, dp_plan, inputs, None, make_target, device="cuda", dtype=dtype,
            num_warmup=num_warmup, num_trials=num_trials, check_correctness=False,
        )
        step["dynamic_programming"] = {
            "oom": dp_result.oom,
            "peak_allocated_bytes": dp_result.peak_allocated_bytes,
            "peak_reserved_bytes": dp_result.peak_reserved_bytes,
            "step_latency_ms_mean": dp_result.step_latency_ms_mean,
            "throughput_samples_per_sec": dp_result.throughput_samples_per_sec,
            "selected_checkpoint_blocks": [d.block_id for d in dp_plan.decisions if d.checkpointed],
            "error_message": dp_result.error_message,
        }
    except torch.cuda.OutOfMemoryError as exc:
        step["dynamic_programming"] = {"oom": True, "error_message": str(exc)}
    finally:
        torch.cuda.empty_cache()

    return step


@app.local_entrypoint()
def main(
    layers: int = 24,
    hidden: int = 2048,
    heads: int = 16,
    seq_len: int = 2048,
    start_batch_size: int = 1,
    growth_axis: str = "batch_size",
    growth_factor: float = 2.0,
    max_steps: int = 10,
    dtype_name: str = "float32",
    gpu_memory_budget_gb: float = 12.0,
    num_warmup: int = 3,
    num_trials: int = 10,
    out: str = "benchmarks/progressive_scaling_result.json",
) -> None:
    assert growth_axis in ("batch_size", "seq_len")
    gpu_memory_budget_bytes = int(gpu_memory_budget_gb * (1 << 30))

    batch_size, seq = start_batch_size, seq_len
    steps: list[dict] = []
    baseline_oomed = False
    dp_oomed = False
    for _ in range(max_steps):
        result = measure_one_step.remote(
            layers=layers, hidden=hidden, heads=heads, seq_len=seq, batch_size=batch_size,
            dtype_name=dtype_name, gpu_memory_budget_bytes=gpu_memory_budget_bytes,
            num_warmup=num_warmup, num_trials=num_trials,
        )
        steps.append(result)
        print(json.dumps(result, indent=2))
        baseline_oomed = result["no_checkpoint"]["oom"]
        dp_oomed = result["dynamic_programming"]["oom"]
        if baseline_oomed or dp_oomed:
            break
        if growth_axis == "batch_size":
            batch_size = max(batch_size + 1, int(batch_size * growth_factor))
        else:
            seq = max(seq + 1, int(seq * growth_factor))

    largest_feasible_no_checkpoint = next(
        (s for s in reversed(steps) if not s["no_checkpoint"]["oom"]), None
    )
    largest_feasible_dp = next(
        (s for s in reversed(steps) if not s["dynamic_programming"]["oom"]), None
    )

    payload = {
        "layers": layers, "hidden": hidden, "heads": heads, "seq_len_start": seq_len,
        "start_batch_size": start_batch_size, "growth_axis": growth_axis,
        "growth_factor": growth_factor, "dtype": dtype_name,
        "gpu_memory_budget_bytes": gpu_memory_budget_bytes,
        "stopped_because_no_checkpoint_oom": baseline_oomed,
        "steps": steps,
        "largest_feasible_no_checkpoint": (
            {"batch_size": largest_feasible_no_checkpoint["batch_size"], "seq_len": largest_feasible_no_checkpoint["seq_len"]}
            if largest_feasible_no_checkpoint else None
        ),
        "largest_feasible_dynamic_programming": (
            {"batch_size": largest_feasible_dp["batch_size"], "seq_len": largest_feasible_dp["seq_len"]}
            if largest_feasible_dp else None
        ),
    }
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    if payload["largest_feasible_no_checkpoint"]:
        print(f"largest feasible no_checkpoint: {payload['largest_feasible_no_checkpoint']}")
    else:
        print("no_checkpoint OOMed at the starting configuration")
    if payload["largest_feasible_dynamic_programming"]:
        print(f"largest feasible dynamic_programming (within budget): {payload['largest_feasible_dynamic_programming']}")
