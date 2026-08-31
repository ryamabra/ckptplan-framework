"""Correctness reference for the 1.2B architecture at a fitting A10G size."""
import json, modal
app = modal.App("ckptplan-boundary-correctness")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.13.0").add_local_dir("ckptplan", remote_path="/root/ckptplan")

@app.function(image=image, gpu="A10G", timeout=3600)
def measure():
    import torch
    from ckptplan import declare_blocks, profile_blocks, plan_checkpoints, run_benchmark
    torch.manual_seed(0); layers, hidden, seq_len, batch = 24, 2048, 512, 1
    model = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=16, dim_feedforward=8192, dropout=0.0, batch_first=True, device="cuda", dtype=torch.float32) for _ in range(layers)])
    inputs = (torch.randn(batch, seq_len, hidden, device="cuda"),)
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])
    profiles = profile_blocks(blocks, inputs, device="cuda", dtype=torch.float32, num_warmup=1, num_trials=2)
    total = sum(p.activation_bytes_estimate or 0 for p in profiles)
    plan = plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="checkpoint_all")
    result = run_benchmark(blocks, plan, inputs, None, lambda out: out.float().square().mean(), device="cuda", dtype=torch.float32, num_warmup=2, num_trials=3, check_correctness=True, correctness_rtol=1e-3, correctness_atol=1e-5)
    return {"parameters": sum(p.numel() for p in model.parameters()), "profiled_activation_total": total, "correctness_checked": result.correctness_checked, "correctness_passed": result.correctness_passed, "max_output_diff": result.correctness_max_abs_output_diff, "max_grad_diff": result.correctness_max_abs_grad_diff, "oom": result.oom, "peak_allocated_bytes": result.peak_allocated_bytes}

@app.local_entrypoint()
def main():
    print(json.dumps(measure.remote(), indent=2))
