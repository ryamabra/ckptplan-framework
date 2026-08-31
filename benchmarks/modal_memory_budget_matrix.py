"""GPU-memory-budget benchmark matrix: none / checkpoint-all / uniform / greedy / DP.

Runs the exact same model, batch, sequence length, and dtype under all five
`ckptplan` planner strategies on a real CUDA GPU (via Modal) and reports:

  - peak allocated GPU memory   (torch.cuda.max_memory_allocated)
  - peak reserved GPU memory    (torch.cuda.max_memory_reserved)
  - mean training-step latency after warmup (CUDA-event timed, synchronized)
  - throughput (samples/sec)
  - runtime overhead relative to the no-checkpoint baseline
  - whether the configuration OOMs
  - which checkpoint blocks each planner selected

Every strategy is given the same GPU-memory budget (default 12 GiB). Since
`ckptplan.plan_checkpoints` targets an *activation* byte budget rather than a
whole-GPU byte budget, the activation budget handed to the budget-aware
planners (uniform/greedy/dynamic_programming) is derived by first measuring
the real, non-activation "floor" memory (parameters + gradients + allocator
overhead) with a `checkpoint_all` dry run, then subtracting that floor from
the GPU budget: `activation_budget = gpu_budget_bytes - floor_bytes`. This
floor is a real measurement on real hardware, not an estimate or a
fabricated constant.

All numeric results in the JSON/CSV output come directly from
`torch.cuda` memory counters and CUDA-event timings collected on an actual
GPU run -- nothing here is hard-coded or simulated. If Modal/CUDA is
unavailable, the run fails outright rather than falling back to invented
numbers.

Usage::

    python benchmarks/modal_memory_budget_matrix.py \\
        --layers 24 --hidden 2048 --heads 16 --seq-len 2048 --batch-size 1 \\
        --gpu-memory-budget-gb 12 --out benchmarks/memory_budget_matrix_result.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import modal

app = modal.App("ckptplan-gpu-memory-budget-matrix")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .add_local_dir("ckptplan", remote_path="/root/ckptplan")
)

PLANNERS = ("no_checkpoint", "checkpoint_all", "uniform", "greedy", "dynamic_programming")
DISPLAY_NAME = {
    "no_checkpoint": "None",
    "checkpoint_all": "Checkpoint-all",
    "uniform": "Uniform",
    "greedy": "Greedy",
    "dynamic_programming": "CKPTPlan DP",
}


@app.function(image=image, gpu="A10G", timeout=3600)
def measure(
    layers: int = 24,
    hidden: int = 2048,
    heads: int = 16,
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype_name: str = "float32",
    gpu_memory_budget_bytes: int = 12 * (1 << 30),
    num_warmup: int = 5,
    num_trials: int = 20,
) -> dict:
    import torch
    from ckptplan import declare_blocks, plan_checkpoints, profile_blocks, run_benchmark

    dtype = getattr(torch, dtype_name)
    torch.manual_seed(0)
    model = torch.nn.ModuleList([
        torch.nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=0.0, batch_first=True, device="cuda", dtype=dtype,
        )
        for _ in range(layers)
    ])
    inputs = (torch.randn(batch_size, seq_len, hidden, device="cuda", dtype=dtype),)
    parameters = sum(p.numel() for p in model.parameters())
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])

    profiles = profile_blocks(blocks, inputs, device="cuda", dtype=dtype, num_warmup=2, num_trials=5)
    total_activation = sum(p.activation_bytes_estimate or 0 for p in profiles)

    def make_target(out: "torch.Tensor") -> "torch.Tensor":
        return out.float().square().mean()

    # Real, measured non-activation memory floor: run checkpoint_all (minimal
    # retained activations) once and read its actual peak allocated bytes.
    floor_plan = plan_checkpoints(
        profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="checkpoint_all",
    )
    floor_result = run_benchmark(
        blocks, floor_plan, inputs, None, make_target,
        device="cuda", dtype=dtype, num_warmup=num_warmup, num_trials=num_trials,
        check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5,
    )
    floor_bytes = floor_result.peak_allocated_bytes
    activation_budget_bytes = max(0, gpu_memory_budget_bytes - floor_bytes)

    target_for = {
        "no_checkpoint": total_activation,
        "checkpoint_all": 0,
        "uniform": activation_budget_bytes,
        "greedy": activation_budget_bytes,
        "dynamic_programming": activation_budget_bytes,
    }

    results = []
    baseline_latency_ms = None
    for name in PLANNERS:
        if name == "checkpoint_all":
            # Reuse the measurement already taken for the memory floor instead
            # of re-running the identical config.
            plan, result = floor_plan, floor_result
        else:
            plan = plan_checkpoints(
                profiles, blocks, target_kind="activation_budget_bytes",
                target_value=target_for[name], planner=name, on_infeasible="best_effort",
            )
            result = run_benchmark(
                blocks, plan, inputs, None, make_target,
                device="cuda", dtype=dtype, num_warmup=num_warmup, num_trials=num_trials,
                check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5,
            )
        if name == "no_checkpoint" and not result.oom:
            baseline_latency_ms = result.step_latency_ms_mean

        selected = [d.block_id for d in plan.decisions if d.checkpointed]
        results.append({
            "planner": name,
            "display_name": DISPLAY_NAME[name],
            "plan_id": plan.plan_id,
            "target_activation_budget_bytes": target_for[name],
            "predicted_activation_bytes_after": plan.predicted_activation_bytes_after,
            "selected_checkpoint_blocks": selected,
            "num_checkpointed_blocks": len(selected),
            "num_total_blocks": len(blocks),
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
            "step_latency_ms": list(result.step_latency_ms),
            "step_latency_ms_mean": result.step_latency_ms_mean,
            "step_latency_ms_p50": result.step_latency_ms_p50,
            "step_latency_ms_p95": result.step_latency_ms_p95,
            "throughput_samples_per_sec": result.throughput_samples_per_sec,
            "correctness_checked": result.correctness_checked,
            "correctness_passed": result.correctness_passed,
            "oom": result.oom,
            "error_message": result.error_message,
        })

    for entry in results:
        if entry["oom"] or baseline_latency_ms in (None, 0.0):
            entry["runtime_overhead_pct_vs_no_checkpoint"] = None
        else:
            entry["runtime_overhead_pct_vs_no_checkpoint"] = (
                (entry["step_latency_ms_mean"] - baseline_latency_ms) / baseline_latency_ms * 100.0
            )

    return {
        "model": "toy-transformer-encoder-stack",
        "layers": layers, "hidden": hidden, "heads": heads, "seq_len": seq_len,
        "batch_size": batch_size, "dtype": dtype_name, "parameters": parameters,
        "gpu_memory_budget_bytes": gpu_memory_budget_bytes,
        "measured_non_activation_floor_bytes": floor_bytes,
        "derived_activation_budget_bytes": activation_budget_bytes,
        "profiled_activation_total_bytes": total_activation,
        "num_warmup": num_warmup, "num_trials": num_trials,
        "results": results,
    }


def _fmt_bytes(n: int | None) -> str:
    if n is None:
        return "—"
    gib = n / (1 << 30)
    return f"{gib:.2f} GiB"


def _fmt_pct(x: float | None) -> str:
    return "—" if x is None else f"{x:+.1f}%"


def render_summary_table(payload: dict) -> str:
    headers = ["Strategy", "Peak VRAM", "Step Time", "Runtime Overhead", "Throughput", "Status"]
    rows = []
    for entry in payload["results"]:
        if entry["oom"]:
            rows.append([entry["display_name"], "OOM", "—", "—", "—", "OOM"])
        else:
            rows.append([
                entry["display_name"],
                _fmt_bytes(entry["peak_allocated_bytes"]),
                f"{entry['step_latency_ms_mean']:.2f} ms",
                _fmt_pct(entry["runtime_overhead_pct_vs_no_checkpoint"]),
                f"{entry['throughput_samples_per_sec']:.1f} samples/s",
                "PASS" if entry["correctness_passed"] is not False else "FAIL",
            ])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for row in rows:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(row)))
    return "\n".join(lines)


def write_csv(payload: dict, path: Path) -> None:
    fieldnames = [
        "planner", "display_name", "peak_allocated_bytes", "peak_reserved_bytes",
        "step_latency_ms_mean", "step_latency_ms_p50", "step_latency_ms_p95",
        "throughput_samples_per_sec", "runtime_overhead_pct_vs_no_checkpoint",
        "oom", "correctness_passed", "num_checkpointed_blocks", "num_total_blocks",
        "selected_checkpoint_blocks",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for entry in payload["results"]:
            row = {k: entry.get(k) for k in fieldnames}
            row["selected_checkpoint_blocks"] = ";".join(row["selected_checkpoint_blocks"] or [])
            writer.writerow(row)


@app.local_entrypoint()
def main(
    layers: int = 24,
    hidden: int = 2048,
    heads: int = 16,
    seq_len: int = 2048,
    batch_size: int = 1,
    dtype_name: str = "float32",
    gpu_memory_budget_gb: float = 12.0,
    num_warmup: int = 5,
    num_trials: int = 20,
    out: str = "benchmarks/memory_budget_matrix_result.json",
    csv_out: str = "",
) -> None:
    payload = measure.remote(
        layers=layers, hidden=hidden, heads=heads, seq_len=seq_len, batch_size=batch_size,
        dtype_name=dtype_name, gpu_memory_budget_bytes=int(gpu_memory_budget_gb * (1 << 30)),
        num_warmup=num_warmup, num_trials=num_trials,
    )
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))

    csv_path = Path(csv_out) if csv_out else out_path.with_suffix(".csv")
    write_csv(payload, csv_path)

    print(f"wrote {out_path}")
    print(f"wrote {csv_path}")
    print()
    print(
        f"model: layers={payload['layers']} hidden={payload['hidden']} heads={payload['heads']} "
        f"seq_len={payload['seq_len']} batch={payload['batch_size']} dtype={payload['dtype']} "
        f"params={payload['parameters']:,}"
    )
    print(f"GPU memory budget: {_fmt_bytes(payload['gpu_memory_budget_bytes'])}")
    print(f"measured non-activation floor: {_fmt_bytes(payload['measured_non_activation_floor_bytes'])}")
    print(f"derived activation budget: {_fmt_bytes(payload['derived_activation_budget_bytes'])}")
    print()
    print(render_summary_table(payload))
