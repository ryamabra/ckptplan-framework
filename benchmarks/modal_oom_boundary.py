"""Find a bounded OOM boundary for the ~1B model on A10G."""
import json, modal
app = modal.App("ckptplan-oom-boundary")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.13.0").add_local_dir("ckptplan", remote_path="/root/ckptplan")

@app.function(image=image, gpu="A10G", timeout=3600)
def measure(planner_name: str = "checkpoint_all"):
    import torch
    from ckptplan import declare_blocks, profile_blocks, plan_checkpoints, run_benchmark
    layers, hidden, heads, seq_len, batch = 24, 2048, 16, 4096, 4
    torch.manual_seed(0)
    model = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=heads, dim_feedforward=8192, dropout=0.0, batch_first=True, device="cuda", dtype=torch.float32) for _ in range(layers)])
    x = (torch.randn(batch, seq_len, hidden, device="cuda"),)
    blocks = declare_blocks(model, [(f"layer{i}", m) for i, m in enumerate(model)])
    profiles = profile_blocks(blocks, x, device="cuda", dtype=torch.float32, num_warmup=1, num_trials=1)
    total = sum(p.activation_bytes_estimate or 0 for p in profiles)
    plans = [plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=total if planner_name == "no_checkpoint" else 0, planner=planner_name)]
    results = []
    for plan in plans:
        r = run_benchmark(blocks, plan, x, None, lambda out: out.float().square().mean(), device="cuda", dtype=torch.float32, num_warmup=1, num_trials=2, check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5)
        results.append({"plan": plan.planner_name, "peak_allocated_bytes": r.peak_allocated_bytes, "peak_reserved_bytes": r.peak_reserved_bytes, "latency_ms_mean": r.step_latency_ms_mean, "correctness_passed": r.correctness_passed, "oom": r.oom, "error_message": r.error_message})
    return {"layers": layers, "hidden": hidden, "seq_len": seq_len, "batch_size": batch, "parameters": sum(p.numel() for p in model.parameters()), "profiled_activation_total": total, "results": results}

@app.local_entrypoint()
def main(planner_name: str = "checkpoint_all"):
    print(json.dumps(measure.remote(planner_name), indent=2))
