"""Reproducible CUDA smoke test for Modal's A10G runner.

Run with: ``modal run benchmarks/modal_cuda.py``
"""

import modal

app = modal.App("ckptplan-cuda-verification")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.13.0", "pytest")
    .add_local_dir("ckptplan", remote_path="/root/ckptplan")
)


@app.function(image=image, gpu="A10G", timeout=900)
def verify() -> None:
    import torch

    from ckptplan import declare_blocks, profile_blocks

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
        torch.nn.Linear(64, 64), torch.nn.ReLU(),
    ).cuda()
    inputs = (torch.randn(8, 64, device="cuda"),)
    blocks = declare_blocks(model, [("block0", model[0]), ("block1", model[2])])
    profiles = profile_blocks(
        blocks, inputs, device="cuda", dtype=torch.float32,
        num_warmup=1, num_trials=3,
    )
    assert all(not profile.timing_only for profile in profiles)
    assert all(profile.activation_bytes_estimate is not None for profile in profiles)
    assert all(profile.activation_bytes_estimate >= 0 for profile in profiles)
    print({
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "activation_bytes": [p.activation_bytes_estimate for p in profiles],
        "forward_ms": [p.forward_time_ms_mean for p in profiles],
        "recompute_ms": [p.recompute_time_upper_bound_ms_mean for p in profiles],
        "status": "PASS",
    })


@app.local_entrypoint()
def main() -> None:
    verify.remote()
