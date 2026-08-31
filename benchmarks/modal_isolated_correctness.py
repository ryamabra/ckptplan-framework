#!/usr/bin/env python3
"""Per-config process isolation for OOM-boundary correctness.

Problem
-------
``run_benchmark(check_correctness=True)`` builds the no-checkpoint reference in
the *same* process as the checkpointed run, so both gradient sets are resident
at once. At ~1.2B parameters that exhausts the A10G before gradients can be
compared, which is why ``benchmarks/boundary_correctness_result.json`` records
``correctness_passed: null`` with a ``max_grad_diff`` of ``null``. This module
runs each configuration in its own Modal container so there is zero memory
contention between the reference and the configuration under test.

How the reference is compared without co-residency
--------------------------------------------------
The reference container writes its gradients to a Modal Volume, one file per
parameter. Each candidate container then reloads them **one parameter at a
time** onto the CPU and compares against its own gradient for that parameter.
Peak extra memory is therefore two copies of the single largest parameter (tens
of MB), never two copies of the whole model, and the reference gradients are
never co-resident in GPU memory with the candidate's model. That is precisely
the contention this design removes.

What crosses the Modal boundary
-------------------------------
**Not gradient tensors.** At 1.2B float32 parameters a full gradient set is
~4.8 GB; returning one per configuration would be impractical. Each container
returns only per-parameter scalars: ``max_abs_diff``, the reference's
``reference_max_abs`` for scale, and an exact ``torch.allclose`` verdict at the
declared tolerance. That is a few hundred small dicts. This is sound because
the comparison itself is performed **in-container against the real reference
tensors** streamed from the volume -- the scalars are the *result* of an exact
elementwise comparison, not a lossy proxy for one. Nothing is inferred from
summary statistics.

Cross-container initialization determinism
------------------------------------------
The whole comparison is meaningless unless every container initializes
bit-identical weights. Two things are done about it, and one limit is stated
plainly.

1. **The CUDA RNG is removed from the initialization path.** Every layer is
   constructed on the **CPU** under a seeded global RNG and only then moved to
   the GPU, so initialization never depends on CUDA kernel launch
   configuration, device architecture, or driver version. The device transfer
   is a bit copy and involves no arithmetic. Layers are built, fingerprinted,
   and moved one at a time, so peak host memory is one layer rather than the
   whole model.

2. **Determinism is checked at runtime, not assumed.** Every container returns
   ``init_fingerprint``, a SHA-256 over each parameter's name, shape, dtype,
   element count, and 256 exactly-represented (``float.hex()``) elements
   sampled at fixed strides. The local entrypoint compares all fingerprints and
   marks the whole run ``correctness_valid: false`` if any differ, rather than
   reporting a gradient comparison against an unproven reference.

**Stated limit:** bit-identical initialization across separate A10G containers
has *not* been proven, because that requires running Modal. What has been
proven, by ``tests/unit/test_isolated_correctness.py``, is that this
CPU-initialization path is bit-identical across separate OS **processes** on one
machine at a fixed torch version -- which is the same code path the containers
run, since the GPU is not involved in initialization. Combined with the pinned
image (Python 3.12, ``torch==2.13.0``), that is a strong argument but not a
proof. The fingerprint check in step 2 is what converts it into a checked
precondition at run time.

Cost
----
See the reply accompanying this script; briefly, the default two-configuration
run is 2 A10G containers of roughly 8 minutes each.

Not run
-------
This module has never been executed against Modal. Every number it would
produce is pending.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

try:  # Modal is a deployment dependency, not needed to import or unit-test this module.
    import modal
except ModuleNotFoundError:  # pragma: no cover - exercised implicitly by CPU test runs.
    modal = None  # type: ignore[assignment]

TORCH_PIN = "torch==2.13.0"
PYTHON_PIN = "3.12"
GRADIENT_MOUNT = "/gradients"
GRADIENT_VOLUME_NAME = "ckptplan-isolated-gradients"
REFERENCE_PLANNER = "no_checkpoint"
FINGERPRINT_SAMPLES = 256
FINGERPRINT_VERSION = "ckptplan-init-v1"


# --- specification ---------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """Everything that determines the model, its weights, and its inputs.

    Every field feeds the initialization fingerprint, so two containers agree on
    weights only if they agree on the whole spec.
    """

    layers: int = 24
    hidden: int = 2048
    heads: int = 16
    dim_feedforward: int = 8192
    seq_len: int = 512
    batch: int = 1
    seed: int = 0
    dtype: str = "float32"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ModelSpec":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in payload.items() if k in fields})

    def canonical(self) -> str:
        """Stable string form used in the fingerprint and the volume path."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    def key(self) -> str:
        """Short deterministic identifier, shared by every container in a run."""
        return hashlib.sha256(self.canonical().encode()).hexdigest()[:16]


# --- deterministic construction --------------------------------------------------


def _slug(name: str) -> str:
    """Filesystem-safe form of a parameter name."""
    return name.replace(".", "__")


def _update_fingerprint(hasher: "hashlib._Hash", prefix: str, module: Any, samples: int) -> None:
    """Fold one module's parameters and buffers into ``hasher``.

    Records name, shape, dtype and element count, plus ``samples`` elements at
    fixed strides written as ``float.hex()`` so the comparison is exact rather
    than subject to decimal rounding.
    """
    import torch

    named = sorted(
        list(module.named_parameters()) + list(module.named_buffers()),
        key=lambda item: item[0],
    )
    for name, tensor in named:
        flat = tensor.detach().reshape(-1)
        count = flat.numel()
        hasher.update(
            f"|{prefix}.{name}|{tuple(tensor.shape)}|{tensor.dtype}|{count}|".encode()
        )
        if count:
            taken = min(count, samples)
            index = torch.linspace(0, count - 1, taken).round().long()
            values = ",".join(float(v).hex() for v in flat[index].tolist())
            hasher.update(values.encode())


def build_model_and_fingerprint(
    spec: ModelSpec,
    device: str = "cpu",
    samples_per_tensor: int = FINGERPRINT_SAMPLES,
) -> tuple[Any, str]:
    """Build the model deterministically and return it with its fingerprint.

    Each layer is constructed on the CPU under the seeded global RNG,
    fingerprinted, and only then moved to ``device``. Keeping construction off
    the GPU removes the CUDA RNG -- the part most likely to vary with device
    architecture, driver, or kernel launch configuration -- from the
    initialization path entirely, and building one layer at a time bounds peak
    host memory to a single layer.
    """
    import torch

    dtype = getattr(torch, spec.dtype)
    torch.manual_seed(spec.seed)

    hasher = hashlib.sha256()
    hasher.update(f"{FINGERPRINT_VERSION}|{spec.canonical()}|samples={samples_per_tensor}".encode())

    layers = []
    for index in range(spec.layers):
        layer = torch.nn.TransformerEncoderLayer(
            d_model=spec.hidden,
            nhead=spec.heads,
            dim_feedforward=spec.dim_feedforward,
            dropout=0.0,
            batch_first=True,
            device="cpu",
            dtype=dtype,
        )
        _update_fingerprint(hasher, f"layer{index}", layer, samples_per_tensor)
        layers.append(layer.to(device))
    return torch.nn.ModuleList(layers), hasher.hexdigest()


def make_inputs(spec: ModelSpec, device: str = "cpu") -> tuple[Any, ...]:
    """Build the deterministic example input.

    Uses its own CPU generator rather than the global RNG, so the input does not
    depend on how much RNG the model construction happened to consume.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(spec.seed + 1)
    tensor = torch.randn(
        spec.batch,
        spec.seq_len,
        spec.hidden,
        generator=generator,
        dtype=getattr(torch, spec.dtype),
        device="cpu",
    )
    return (tensor.to(device),)


def make_target(output: Any) -> Any:
    """Scalar loss, matching the other benchmark scripts."""
    return output.float().square().mean()


# --- gradient exchange -----------------------------------------------------------


def save_gradients(model: Any, directory: str | Path) -> list[str]:
    """Write one file per parameter gradient and an ordering index.

    Per-parameter files are what let a candidate container stream the reference
    back one tensor at a time instead of materializing all ~4.8 GB at once.
    """
    import torch

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        torch.save(param.grad.detach().to("cpu"), path / f"{_slug(name)}.pt")
        names.append(name)
    (path / "index.json").write_text(json.dumps(names))
    return names


def compare_gradients(
    directory: str | Path,
    model: Any,
    rtol: float,
    atol: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Stream saved reference gradients and compare against ``model``'s.

    Returns per-parameter entries and the list of names that could not be
    compared. Only one reference tensor is resident at a time.
    """
    import torch

    path = Path(directory)
    names: list[str] = json.loads((path / "index.json").read_text())
    current = dict(model.named_parameters())

    entries: list[dict[str, Any]] = []
    missing: list[str] = []
    for name in names:
        param = current.get(name)
        if param is None or param.grad is None:
            missing.append(name)
            continue
        reference = torch.load(path / f"{_slug(name)}.pt", map_location="cpu", weights_only=True)
        candidate = param.grad.detach().to("cpu")
        if reference.shape != candidate.shape:
            missing.append(name)
            continue
        if reference.numel():
            diff = float((candidate - reference).abs().max().item())
            reference_max = float(reference.abs().max().item())
            close = bool(torch.allclose(candidate, reference, rtol=rtol, atol=atol))
        else:
            diff, reference_max, close = 0.0, 0.0, True
        entries.append(
            {
                "name": name,
                "max_abs_diff": diff,
                "reference_max_abs": reference_max,
                "allclose": close,
            }
        )
        del reference, candidate
    return entries, missing


def summarize_gradient_comparison(
    entries: Iterable[dict[str, Any]],
    missing: Iterable[str],
) -> dict[str, Any]:
    """Reduce per-parameter entries to an overall verdict.

    ``passed`` is ``None`` -- not ``False`` -- when nothing could be compared,
    so "no evidence" is never reported as "failed".
    """
    entries = list(entries)
    missing = list(missing)
    if not entries:
        return {
            "compared_parameters": 0,
            "missing_parameters": missing,
            "max_abs_grad_diff": None,
            "worst_parameter": None,
            "failed_parameters": [],
            "passed": None,
        }
    worst = max(entries, key=lambda entry: entry["max_abs_diff"])
    failed = [entry["name"] for entry in entries if not entry["allclose"]]
    return {
        "compared_parameters": len(entries),
        "missing_parameters": missing,
        "max_abs_grad_diff": worst["max_abs_diff"],
        "worst_parameter": worst["name"],
        "failed_parameters": failed,
        "passed": not failed and not missing,
    }


def target_budget(planner_name: str, activation_total: int, fraction: float) -> int:
    """Activation budget for a planner name.

    ``no_checkpoint`` gets the full total (nothing needs checkpointing),
    ``checkpoint_all`` gets zero, and the budget planners get a fraction.
    """
    if planner_name == REFERENCE_PLANNER:
        return int(activation_total)
    if planner_name == "checkpoint_all":
        return 0
    return int(activation_total * fraction)


# --- Modal -----------------------------------------------------------------------


class _OfflineApp:
    """Stand-in used when Modal is absent so this module still imports.

    The decorators become the identity, which keeps every function above
    importable and unit-testable on a CPU-only machine without Modal installed.
    """

    def function(self, *args: Any, **kwargs: Any):
        def decorate(func):
            return func

        return decorate

    def local_entrypoint(self, *args: Any, **kwargs: Any):
        def decorate(func):
            return func

        return decorate


if modal is not None:
    image = (
        modal.Image.debian_slim(python_version=PYTHON_PIN)
        .pip_install(TORCH_PIN)
        # Required before torch.use_deterministic_algorithms for cuBLAS GEMMs.
        .env({"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        .add_local_dir("ckptplan", remote_path="/root/ckptplan")
    )
    volume = modal.Volume.from_name(GRADIENT_VOLUME_NAME, create_if_missing=True)
    app = modal.App("ckptplan-isolated-correctness")
    _FUNCTION_KWARGS: dict[str, Any] = {
        "image": image,
        "gpu": "A10G",
        # 30 min: ~3.75x the expected ~8 min/container, and it is the timeout,
        # not the estimate, that caps worst-case spend if a container hangs.
        "timeout": 1800,
        "volumes": {GRADIENT_MOUNT: volume},
        # Host memory: one layer at a time during init, plus one streamed
        # reference gradient at a time during comparison.
        "memory": 16384,
    }
else:  # pragma: no cover - import-time fallback for CPU-only environments.
    image = None
    volume = None
    app = _OfflineApp()
    _FUNCTION_KWARGS = {}


@app.function(**_FUNCTION_KWARGS)
def run_isolated_config(
    spec_dict: dict[str, Any],
    planner_name: str,
    is_reference: bool,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    target_fraction: float = 0.5,
    num_warmup: int = 2,
    num_trials: int = 5,
) -> dict[str, Any]:
    """Run exactly one configuration in a fresh container.

    The reference container saves its gradients to the shared volume; every
    other container streams them back and compares. Nothing else in this
    process ever holds a second model.

    Two known properties of this flow, both verified by reading the code:

    * **The init-fingerprint check happens after this function returns**, in
      ``reconcile`` on the local entrypoint. A container never checks its
      fingerprint against the reference's, so a mismatch is detected only once
      every container has finished its profiling, benchmark, and gradient work.
      The run is correctly invalidated, but no compute is saved.
    * **Profiling and planning happen here, per container.** ``no_checkpoint``
      and ``checkpoint_all`` select blocks without consulting any measured
      estimate, so their plans are identical across containers by construction.
      ``greedy`` and ``dynamic_programming`` rank blocks by
      ``recompute_time_upper_bound_ms_mean`` and ``uniform`` depends on measured
      activation bytes and the derived budget, so those three may select
      different blocks in different containers. ``checkpointed_block_ids``
      records what each container actually ran.
    """
    import torch

    from ckptplan import (
        apply_plan,
        declare_blocks,
        plan_checkpoints,
        profile_blocks,
        run_benchmark,
    )

    # Reduce within-config nondeterminism so any gradient difference is
    # attributable to checkpointing rather than to atomics ordering. warn_only
    # avoids hard-failing on ops that have no deterministic kernel.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    spec = ModelSpec.from_dict(spec_dict)
    model, fingerprint = build_model_and_fingerprint(spec, "cuda")
    inputs = make_inputs(spec, "cuda")
    blocks = declare_blocks(model, [(f"layer{i}", layer) for i, layer in enumerate(model)])

    profiles = profile_blocks(
        blocks, inputs, device="cuda", dtype=getattr(torch, spec.dtype),
        num_warmup=1, num_trials=2,
    )
    activation_total = sum(profile.activation_bytes_estimate or 0 for profile in profiles)
    plan = plan_checkpoints(
        profiles,
        blocks,
        target_kind="activation_budget_bytes",
        target_value=target_budget(planner_name, activation_total, target_fraction),
        planner=planner_name,
    )

    # check_correctness=False on purpose: run_benchmark's own correctness path
    # co-locates the reference, which is the contention this script removes.
    benchmark = run_benchmark(
        blocks, plan, inputs, None, make_target,
        device="cuda", dtype=getattr(torch, spec.dtype),
        num_warmup=num_warmup, num_trials=num_trials, check_correctness=False,
    )

    record: dict[str, Any] = {
        "planner": planner_name,
        "is_reference": is_reference,
        "plan_id": plan.plan_id,
        "init_fingerprint": fingerprint,
        "checkpointed_block_ids": [
            decision.block_id for decision in plan.decisions if decision.checkpoint
        ],
        "profiled_activation_total": activation_total,
        "predicted_activation_bytes_after": plan.predicted_activation_bytes_after,
        "predicted_recompute_time_upper_bound_ms": plan.predicted_recompute_time_upper_bound_ms,
        "peak_allocated_bytes": benchmark.peak_allocated_bytes,
        "peak_reserved_bytes": benchmark.peak_reserved_bytes,
        "step_latency_ms": list(benchmark.step_latency_ms),
        "latency_ms_mean": benchmark.step_latency_ms_mean,
        "latency_ms_p50": benchmark.step_latency_ms_p50,
        "latency_ms_p95": benchmark.step_latency_ms_p95,
        "throughput_samples_per_sec": benchmark.throughput_samples_per_sec,
        "batch_size": spec.batch,
        "oom": benchmark.oom,
        "error_message": benchmark.error_message,
        "device_name": benchmark.device_name,
        "gradient_comparison": None,
    }
    if benchmark.oom:
        return record

    # One dedicated deterministic step supplies the gradients that are compared.
    directory = Path(GRADIENT_MOUNT) / spec.key() / "reference"
    try:
        container = apply_plan(blocks, plan)
        container.zero_grad(set_to_none=True)
        make_target(container(*inputs)).backward()
    except torch.cuda.OutOfMemoryError as exc:
        record["oom"] = True
        record["error_message"] = f"correctness step OOM: {exc}"
        return record

    if is_reference:
        saved = save_gradients(model, directory)
        volume.commit()
        record["gradient_comparison"] = {"saved_parameters": len(saved), "role": "reference"}
        return record

    volume.reload()
    if not (directory / "index.json").exists():
        record["gradient_comparison"] = {
            "role": "candidate",
            "error": "reference gradients are not present on the volume",
            "passed": None,
        }
        return record

    entries, missing = compare_gradients(directory, model, rtol=rtol, atol=atol)
    summary = summarize_gradient_comparison(entries, missing)
    summary.update({"role": "candidate", "rtol": rtol, "atol": atol, "per_parameter": entries})
    record["gradient_comparison"] = summary
    return record


@app.local_entrypoint()
def main(
    planners: str = "checkpoint_all",
    layers: int = 24,
    hidden: int = 2048,
    heads: int = 16,
    dim_feedforward: int = 8192,
    seq_len: int = 512,
    batch: int = 1,
    seed: int = 0,
    rtol: float = 1e-3,
    atol: float = 1e-5,
    target_fraction: float = 0.5,
    output: str = "",
) -> None:
    """Fan out one container per configuration, then reconcile the results.

    ``planners`` is a comma-separated list of candidate planners; the
    ``no_checkpoint`` reference is always run first and separately, because the
    candidates need its saved gradients.
    """
    spec = ModelSpec(
        layers=layers, hidden=hidden, heads=heads, dim_feedforward=dim_feedforward,
        seq_len=seq_len, batch=batch, seed=seed,
    )
    candidates = [name.strip() for name in planners.split(",") if name.strip()]
    candidates = [name for name in candidates if name != REFERENCE_PLANNER]

    reference = run_isolated_config.remote(
        spec.to_dict(), REFERENCE_PLANNER, True, rtol, atol, target_fraction
    )
    handles = [
        run_isolated_config.spawn(spec.to_dict(), name, False, rtol, atol, target_fraction)
        for name in candidates
    ]
    results = [reference] + [handle.get() for handle in handles]

    record = json.dumps(reconcile(spec, results), indent=2)
    print(record)
    if output:
        Path(output).write_text(record)


def reconcile(spec: ModelSpec, results: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble the final record and gate it on cross-container init agreement.

    If any container's ``init_fingerprint`` differs, the models were not
    identical and every gradient comparison in the run is meaningless, so
    ``correctness_valid`` is false and no verdict is claimed.
    """
    fingerprints = {result["planner"]: result.get("init_fingerprint") for result in results}
    # Every container must report a fingerprint and they must all be the same;
    # a missing fingerprint is treated as disagreement, never as agreement.
    match = bool(results) and len(set(fingerprints.values())) == 1 and None not in fingerprints.values()

    reference = next((r for r in results if r.get("is_reference")), None)
    comparisons = [
        r for r in results
        if not r.get("is_reference") and isinstance(r.get("gradient_comparison"), dict)
    ]
    verdicts = [r["gradient_comparison"].get("passed") for r in comparisons]

    if not match:
        verdict: Optional[bool] = None
        interpretation = (
            "Initialization fingerprints differ across containers, so the models were "
            "not identical and no gradient comparison from this run is meaningful."
        )
    elif not comparisons or any(v is None for v in verdicts):
        verdict = None
        interpretation = (
            "Initialization matched across containers, but at least one configuration "
            "produced no comparable gradients; no correctness verdict is claimed."
        )
    else:
        verdict = all(verdicts)
        interpretation = (
            "Per-config process isolation removed the memory contention that blocked the "
            "co-located reference; gradients were compared against the isolated "
            "no-checkpoint reference streamed one parameter at a time."
        )

    return {
        "spec": spec.to_dict(),
        "spec_key": spec.key(),
        "container_count": len(results),
        "init_fingerprints": fingerprints,
        "init_fingerprint_match": match,
        "reference_planner": REFERENCE_PLANNER,
        "reference_oom": bool(reference["oom"]) if reference else None,
        "correctness_valid": bool(match),
        "correctness_passed": verdict,
        "results": results,
        "interpretation": interpretation,
    }


if __name__ == "__main__":  # pragma: no cover - Modal drives this via its entrypoint.
    print(json.dumps(ModelSpec().to_dict(), indent=2))
