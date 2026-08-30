"""Bounded toy-transformer calibration/benchmark run on Modal A10G."""

import json
import modal

app = modal.App("ckptplan-toy-transformer-calibration")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0")
    .add_local_dir("ckptplan", remote_path="/root/ckptplan")
)


@app.function(image=image, gpu="A10G", timeout=1800)
def measure(n_layers: int = 2, batch_size: int = 1) -> dict:
    import torch
    from ckptplan import declare_blocks, plan_checkpoints, profile_blocks, run_benchmark

    torch.manual_seed(0)
    model = torch.nn.ModuleList([
        torch.nn.TransformerEncoderLayer(
            d_model=1024, nhead=16, dim_feedforward=4096,
            dropout=0.0, batch_first=True, device="cuda", dtype=torch.float32,
        ) for _ in range(n_layers)
    ])
    inputs = (torch.randn(batch_size, 1024, 1024, device="cuda"),)
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])
    profiles = profile_blocks(blocks, inputs, device="cuda", dtype=torch.float32, num_warmup=1, num_trials=3)
    total_activation = sum(p.activation_bytes_estimate or 0 for p in profiles)
    no_checkpoint = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=total_activation, planner="no_checkpoint")
    checkpoint_all = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="checkpoint_all")
    results = []
    for plan in (no_checkpoint, checkpoint_all):
        result = run_benchmark(
            blocks, plan, inputs, None, lambda out: out.float().square().mean(),
            device="cuda", dtype=torch.float32, num_warmup=5, num_trials=20,
            check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5,
        )
        results.append({
            "config_name": result.config_name, "plan_id": result.plan_id,
            "peak_allocated_bytes": result.peak_allocated_bytes,
            "peak_reserved_bytes": result.peak_reserved_bytes,
            "latency_ms_mean": result.step_latency_ms_mean,
            "throughput_samples_per_sec": result.throughput_samples_per_sec,
            "correctness_passed": result.correctness_passed, "oom": result.oom,
            "error_message": result.error_message,
        })
    return {"n_layers": n_layers, "batch_size": batch_size, "hidden": 1024, "heads": 16, "seq_len": 1024, "dtype": "float32", "profiles": [p.activation_bytes_estimate for p in profiles], "results": results}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(measure.remote(), indent=2))
