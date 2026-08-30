"""Focused A10G memory diagnosis runner."""
import json, modal
app = modal.App("ckptplan-memory-diagnosis")
image = modal.Image.debian_slim(python_version="3.12").pip_install("torch==2.13.0").add_local_dir("ckptplan", remote_path="/root/ckptplan")

@app.function(image=image, gpu="A10G", timeout=3600)
def measure():
    import torch
    from ckptplan import declare_blocks, profile_blocks, plan_checkpoints, apply_plan
    torch.manual_seed(0); layers, hidden, seq, batch = 24, 2048, 2048, 1
    model = torch.nn.ModuleList([torch.nn.TransformerEncoderLayer(d_model=hidden, nhead=16, dim_feedforward=8192, dropout=0.0, batch_first=True, device="cuda", dtype=torch.float32) for _ in range(layers)])
    x = (torch.randn(batch, seq, hidden, device="cuda"),); blocks = declare_blocks(model, [(f"layer{i}", m) for i, m in enumerate(model)])
    profiles = profile_blocks(blocks, x, device="cuda", dtype=torch.float32, num_warmup=1, num_trials=2); total = sum(p.activation_bytes_estimate or 0 for p in profiles)
    plans = [plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=total, planner="no_checkpoint"), plan_checkpoints(profiles, blocks, target_kind="activation_budget_bytes", target_value=0, planner="checkpoint_all")]
    result = {"parameters": sum(p.numel() for p in model.parameters()), "parameter_bytes": sum(p.numel()*p.element_size() for p in model.parameters()), "profiled_activation_bytes": total, "results": []}
    for plan in plans:
        c = apply_plan(blocks, plan, x); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); c.zero_grad(set_to_none=True); y = c(*x); torch.cuda.synchronize(); f = torch.cuda.max_memory_allocated(); y.square().mean().backward(); torch.cuda.synchronize(); b = torch.cuda.max_memory_allocated()
        result["results"].append({"plan": plan.planner_name, "checkpointed_blocks": sum(d.checkpointed for d in plan.decisions), "forward_peak_allocated": f, "backward_peak_allocated": b})
    return result

@app.local_entrypoint()
def main():
    print(json.dumps(measure.remote(), indent=2))
