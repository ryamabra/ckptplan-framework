"""Bounded ~1B-parameter transformer comparison on a Modal A10G."""

import json
import modal

app = modal.App("ckptplan-billion-parameter-verification")
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
        torch.nn.TransformerEncoderLayer(
            d_model=hidden, nhead=heads, dim_feedforward=4 * hidden,
            dropout=0.0, batch_first=True, device="cuda", dtype=torch.float32,
        ) for _ in range(layers)
    ])
    inputs = (torch.randn(batch, seq_len, hidden, device="cuda"),)
    parameters = sum(p.numel() for p in model.parameters())
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])
    profiles = profile_blocks(blocks, inputs, device="cuda", dtype=torch.float32, num_warmup=1, num_trials=2)
    total = sum(p.activation_bytes_estimate or 0 for p in profiles)
    plans = [
        plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=total, planner="no_checkpoint"),
        plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="checkpoint_all"),
    ]
    results = []
    for plan in plans:
        result = run_benchmark(blocks, plan, inputs, None, lambda out: out.float().square().mean(), device="cuda", dtype=torch.float32, num_warmup=2, num_trials=3, check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5)
        results.append({"plan": plan.planner_name, "parameters": parameters, "peak_allocated_bytes": result.peak_allocated_bytes, "peak_reserved_bytes": result.peak_reserved_bytes, "latency_ms_mean": result.step_latency_ms_mean, "correctness_passed": result.correctness_passed, "oom": result.oom, "error_message": result.error_message})
    return {"layers": layers, "hidden": hidden, "heads": heads, "seq_len": seq_len, "batch_size": batch, "parameters": parameters, "results": results}


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(measure.remote(), indent=2))
