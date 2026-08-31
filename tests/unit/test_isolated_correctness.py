"""Unit tests for ``benchmarks/modal_isolated_correctness.py``.

Everything here runs on CPU without Modal installed. The cross-process
determinism test is the load-bearing one: it exercises the exact
CPU-initialization path the containers use, in two separate OS processes.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_MODULE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "modal_isolated_correctness.py"


def _load_module(name: str = "modal_isolated_correctness"):
    spec = importlib.util.spec_from_file_location(name, _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


iso = _load_module()

TINY = iso.ModelSpec(layers=2, hidden=8, heads=2, dim_feedforward=16, seq_len=4, batch=1, seed=0)


# --- import surface --------------------------------------------------------------


def test_module_imports_without_modal_installed() -> None:
    """Modal is a deployment dependency; the module must still import for tests.

    The offline stub is asserted only when Modal is genuinely absent, so this
    does not fail in an environment that happens to have Modal installed.
    """
    assert callable(iso.run_isolated_config)
    assert callable(iso.main)
    assert callable(iso.reconcile)
    if iso.modal is None:
        assert isinstance(iso.app, iso._OfflineApp)


# --- spec ------------------------------------------------------------------------


def test_spec_round_trips_and_keys_are_stable() -> None:
    restored = iso.ModelSpec.from_dict(TINY.to_dict())
    assert restored == TINY
    assert restored.key() == TINY.key()
    assert len(TINY.key()) == 16


def test_spec_key_changes_with_any_field() -> None:
    from dataclasses import replace

    assert replace(TINY, seed=1).key() != TINY.key()
    assert replace(TINY, hidden=16).key() != TINY.key()


def test_spec_from_dict_ignores_unknown_keys() -> None:
    payload = dict(TINY.to_dict(), extra="ignored")
    assert iso.ModelSpec.from_dict(payload) == TINY


# --- deterministic initialization ------------------------------------------------


def test_fingerprint_is_reproducible_in_process() -> None:
    _, first = iso.build_model_and_fingerprint(TINY)
    _, second = iso.build_model_and_fingerprint(TINY)
    assert first == second
    assert len(first) == 64


def test_fingerprint_changes_with_seed_and_shape() -> None:
    from dataclasses import replace

    _, base = iso.build_model_and_fingerprint(TINY)
    _, reseeded = iso.build_model_and_fingerprint(replace(TINY, seed=1))
    _, reshaped = iso.build_model_and_fingerprint(replace(TINY, dim_feedforward=32))
    assert base != reseeded
    assert base != reshaped


def test_fingerprint_detects_a_single_perturbed_weight() -> None:
    model, base = iso.build_model_and_fingerprint(TINY)
    with torch.no_grad():
        first = next(iter(model.parameters()))
        first.view(-1)[0] += 1e-3
    hasher_input = iso.hashlib.sha256()
    hasher_input.update(
        f"{iso.FINGERPRINT_VERSION}|{TINY.canonical()}|samples={iso.FINGERPRINT_SAMPLES}".encode()
    )
    for index, layer in enumerate(model):
        iso._update_fingerprint(hasher_input, f"layer{index}", layer, iso.FINGERPRINT_SAMPLES)
    assert hasher_input.hexdigest() != base


_SUBPROCESS_SOURCE = """
import importlib.util, sys, json
spec = importlib.util.spec_from_file_location("iso", {path!r})
module = importlib.util.module_from_spec(spec)
sys.modules["iso"] = module
spec.loader.exec_module(module)
spec_obj = module.ModelSpec(**json.loads({payload!r}))
_, fingerprint = module.build_model_and_fingerprint(spec_obj)
print(fingerprint)
"""


def _fingerprint_in_subprocess() -> str:
    source = _SUBPROCESS_SOURCE.format(path=str(_MODULE_PATH), payload=json.dumps(TINY.to_dict()))
    completed = subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip().splitlines()[-1]


@pytest.mark.slow
def test_initialization_is_bit_identical_across_separate_processes() -> None:
    """The determinism claim the whole comparison rests on.

    Two independent OS processes build the model through the same CPU-only
    initialization path the Modal containers use, and must agree exactly. This
    does not prove agreement across separate A10G *containers* -- that needs a
    real Modal run -- but the GPU is not involved in initialization, so this is
    the same code path.
    """
    first = _fingerprint_in_subprocess()
    second = _fingerprint_in_subprocess()
    _, in_process = iso.build_model_and_fingerprint(TINY)
    assert first == second == in_process


def test_model_is_built_on_cpu_and_has_expected_shape() -> None:
    model, _ = iso.build_model_and_fingerprint(TINY)
    assert len(model) == TINY.layers
    assert all(param.device.type == "cpu" for param in model.parameters())


# --- inputs ----------------------------------------------------------------------


def test_inputs_are_deterministic_and_correctly_shaped() -> None:
    first = iso.make_inputs(TINY)
    second = iso.make_inputs(TINY)
    assert torch.equal(first[0], second[0])
    assert tuple(first[0].shape) == (TINY.batch, TINY.seq_len, TINY.hidden)


def test_inputs_do_not_depend_on_global_rng_state() -> None:
    torch.manual_seed(1234)
    first = iso.make_inputs(TINY)
    torch.manual_seed(4321)
    torch.randn(17)
    second = iso.make_inputs(TINY)
    assert torch.equal(first[0], second[0])


def test_make_target_reduces_to_a_scalar() -> None:
    target = iso.make_target(torch.arange(6.0).reshape(2, 3))
    assert target.ndim == 0


# --- gradient exchange -----------------------------------------------------------


def _model_with_gradients(scale: float = 1.0) -> torch.nn.Module:
    torch.manual_seed(0)
    model = torch.nn.ModuleList([torch.nn.Linear(4, 3), torch.nn.Linear(3, 2)])
    x = torch.ones(1, 4)
    out = model[1](model[0](x))
    (out.square().mean() * scale).backward()
    return model


def test_gradients_round_trip_and_identical_grads_compare_equal(tmp_path: Path) -> None:
    model = _model_with_gradients()
    names = iso.save_gradients(model, tmp_path)
    assert names == [name for name, p in model.named_parameters() if p.grad is not None]
    assert (tmp_path / "index.json").exists()

    entries, missing = iso.compare_gradients(tmp_path, model, rtol=1e-3, atol=1e-5)
    assert missing == []
    assert len(entries) == len(names)
    assert all(entry["max_abs_diff"] == 0.0 for entry in entries)
    assert all(entry["allclose"] for entry in entries)

    summary = iso.summarize_gradient_comparison(entries, missing)
    assert summary["passed"] is True
    assert summary["max_abs_grad_diff"] == 0.0


def test_differing_grads_are_detected_with_exact_max_abs_diff(tmp_path: Path) -> None:
    reference = _model_with_gradients()
    iso.save_gradients(reference, tmp_path)

    candidate = _model_with_gradients()
    with torch.no_grad():
        first = next(iter(candidate.parameters()))
        first.grad.view(-1)[0] += 0.25

    entries, missing = iso.compare_gradients(tmp_path, candidate, rtol=1e-3, atol=1e-5)
    assert missing == []
    worst = max(entries, key=lambda entry: entry["max_abs_diff"])
    assert worst["max_abs_diff"] == pytest.approx(0.25)
    assert worst["allclose"] is False

    summary = iso.summarize_gradient_comparison(entries, missing)
    assert summary["passed"] is False
    assert summary["failed_parameters"] == [worst["name"]]
    assert summary["worst_parameter"] == worst["name"]


def test_tiny_differences_within_tolerance_still_pass(tmp_path: Path) -> None:
    reference = _model_with_gradients()
    iso.save_gradients(reference, tmp_path)
    candidate = _model_with_gradients()
    with torch.no_grad():
        next(iter(candidate.parameters())).grad.view(-1)[0] += 1e-9
    entries, _ = iso.compare_gradients(tmp_path, candidate, rtol=1e-3, atol=1e-5)
    assert all(entry["allclose"] for entry in entries)


def test_missing_parameter_is_reported_not_silently_skipped(tmp_path: Path) -> None:
    reference = _model_with_gradients()
    iso.save_gradients(reference, tmp_path)
    candidate = _model_with_gradients()
    # Simulate a candidate that produced no gradient for one parameter.
    next(iter(candidate.parameters())).grad = None

    entries, missing = iso.compare_gradients(tmp_path, candidate, rtol=1e-3, atol=1e-5)
    assert len(missing) == 1
    summary = iso.summarize_gradient_comparison(entries, missing)
    assert summary["passed"] is False
    assert summary["missing_parameters"] == missing


def test_no_comparable_parameters_yields_none_not_false() -> None:
    summary = iso.summarize_gradient_comparison([], [])
    assert summary["passed"] is None
    assert summary["max_abs_grad_diff"] is None
    assert summary["compared_parameters"] == 0


def test_returned_entries_are_scalars_only(tmp_path: Path) -> None:
    """Nothing tensor-shaped may cross the Modal boundary."""
    model = _model_with_gradients()
    iso.save_gradients(model, tmp_path)
    entries, _ = iso.compare_gradients(tmp_path, model, rtol=1e-3, atol=1e-5)
    for entry in entries:
        assert isinstance(entry["name"], str)
        assert isinstance(entry["max_abs_diff"], float)
        assert isinstance(entry["reference_max_abs"], float)
        assert isinstance(entry["allclose"], bool)
    # The whole payload must survive JSON serialization.
    assert json.loads(json.dumps(entries)) == entries


# --- budgets ---------------------------------------------------------------------


def test_target_budget_per_planner() -> None:
    assert iso.target_budget("no_checkpoint", 1000, 0.5) == 1000
    assert iso.target_budget("checkpoint_all", 1000, 0.5) == 0
    assert iso.target_budget("greedy", 1000, 0.5) == 500
    assert iso.target_budget("dynamic_programming", 1000, 0.25) == 250


# --- reconciliation --------------------------------------------------------------


def _result(planner: str, fingerprint: str, *, is_reference: bool = False, passed=True) -> dict:
    comparison = None if is_reference else {"passed": passed, "role": "candidate"}
    return {
        "planner": planner,
        "is_reference": is_reference,
        "init_fingerprint": fingerprint,
        "oom": False,
        "gradient_comparison": comparison,
    }


def test_reconcile_passes_when_fingerprints_match_and_grads_agree() -> None:
    results = [_result("no_checkpoint", "abc", is_reference=True), _result("checkpoint_all", "abc")]
    record = iso.reconcile(TINY, results)
    assert record["init_fingerprint_match"] is True
    assert record["correctness_valid"] is True
    assert record["correctness_passed"] is True
    assert record["container_count"] == 2


def test_reconcile_invalidates_the_run_on_fingerprint_mismatch() -> None:
    results = [_result("no_checkpoint", "abc", is_reference=True), _result("checkpoint_all", "xyz")]
    record = iso.reconcile(TINY, results)
    assert record["init_fingerprint_match"] is False
    assert record["correctness_valid"] is False
    # No verdict is claimed against an unproven reference.
    assert record["correctness_passed"] is None
    assert "not identical" in record["interpretation"]


def test_reconcile_treats_a_missing_fingerprint_as_disagreement() -> None:
    results = [_result("no_checkpoint", "abc", is_reference=True), _result("checkpoint_all", None)]
    record = iso.reconcile(TINY, results)
    assert record["init_fingerprint_match"] is False
    assert record["correctness_passed"] is None


def test_reconcile_reports_failure_when_grads_disagree() -> None:
    results = [
        _result("no_checkpoint", "abc", is_reference=True),
        _result("checkpoint_all", "abc", passed=False),
    ]
    record = iso.reconcile(TINY, results)
    assert record["correctness_valid"] is True
    assert record["correctness_passed"] is False


def test_reconcile_claims_nothing_without_comparisons() -> None:
    results = [_result("no_checkpoint", "abc", is_reference=True)]
    record = iso.reconcile(TINY, results)
    assert record["correctness_passed"] is None
    assert "no correctness verdict is claimed" in record["interpretation"]


def test_reconcile_output_is_json_serializable() -> None:
    results = [_result("no_checkpoint", "abc", is_reference=True), _result("checkpoint_all", "abc")]
    record = iso.reconcile(TINY, results)
    assert json.loads(json.dumps(record))["spec"]["layers"] == TINY.layers
