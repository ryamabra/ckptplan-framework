"""Five-planner benchmark matrix for the calibrated ~1B model on A10G."""

import json
import modal

app = modal.App("ckptplan-five-planner-matrix")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .add_local_dir("ckptplan", remote_path="/root/ckptplan")
)


@app.function(image=image, gpu="A10G", timeout=3600)
def measure() -> dict:
    import torch
    from ckptplan import declare_blocks, plan_checkpoints, profile_blocks, run_benchmark

    layers, hidden, heads, seq_len, batch = 24, 2048, 16, 2048, 1
    torch.manual_seed(0)
    model = torch.nn.ModuleList([
        torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=4 * hidden, dropout=0.0, batch_first=True, device="cuda", dtype=torch.float32)
        for _ in range(layers)
    ])
    inputs = (torch.randn(batch, seq_len, hidden, device="cuda"),)
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])
    profiles = profile_blocks(blocks, inputs, device="cuda", dtype=torch.float32, num_warmup=2, num_trials=5)
    total = sum(p.activation_bytes_estimate or 0 for p in profiles)
    target = total * 0.5
    configs = [
        ("no_checkpoint", "no_checkpoint", total),
        ("checkpoint_all", "checkpoint_all", 0),
        ("uniform", "uniform", target),
        ("greedy", "greedy", target),
        ("dynamic_programming", "dynamic_programming", target),
    ]
    results = []
    for name, planner, budget in configs:
        plan = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=budget, planner=planner)
        result = run_benchmark(blocks, plan, inputs, None, lambda out: out.float().square().mean(), device="cuda", dtype=torch.float32, num_warmup=5, num_trials=20, check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5)
        results.append({"planner": name, "plan_id": plan.plan_id, "target_budget_bytes": budget, "predicted_activation_bytes_after": plan.predicted_activation_bytes_after, "predicted_recompute_time_upper_bound_ms": plan.predicted_recompute_time_upper_bound_ms, "peak_allocated_bytes": result.peak_allocated_bytes, "peak_reserved_bytes": result.peak_reserved_bytes, "step_latency_ms": list(result.step_latency_ms), "latency_ms_mean": result.step_latency_ms_mean, "latency_ms_p50": result.step_latency_ms_p50, "latency_ms_p95": result.step_latency_ms_p95, "throughput_samples_per_sec": result.throughput_samples_per_sec, "correctness_checked": result.correctness_checked, "max_output_diff": result.correctness_max_abs_output_diff, "max_grad_diff": result.correctness_max_abs_grad_diff, "correctness_passed": result.correctness_passed, "oom": result.oom, "error_message": result.error_message})
    return {"model": "toy-transformer", "parameters": sum(p.numel() for p in model.parameters()), "layers": layers, "hidden": hidden, "heads": heads, "seq_len": seq_len, "batch_size": batch, "profile_activation_total": total, "target_fraction": 0.5, "results": results}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(measure.remote(), indent=2))
