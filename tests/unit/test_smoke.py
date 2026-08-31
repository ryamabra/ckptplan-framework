"""Repository-foundation smoke test.

Proves (1) the ``ckptplan`` package imports, and (2) the tiny deterministic
fixture model (MVP_SPEC.md's CPU correctness fixture) produces identical output
across independently constructed model+batch pairs built from the same seed.

Does not exercise any ckptplan API (declare_blocks, profiling, planning,
application, benchmarking) -- none of that exists yet in this slice. See
STATE.md for what is and is not implemented.
"""

from __future__ import annotations

import importlib.metadata

import torch

import ckptplan
from tests.conftest import BATCH_SIZE, BLOCK_WIDTH, build_tiny_batch, build_tiny_sequential_model


def test_package_imports_and_reports_a_version() -> None:
    assert isinstance(ckptplan.__version__, str)
    assert ckptplan.__version__
    # Catches the two version declarations (ckptplan/__init__.py and
    # pyproject.toml) drifting apart -- reviewer-flagged gap, fixed here.
    assert ckptplan.__version__ == importlib.metadata.version("ckptplan")


def test_fixture_model_is_deterministic_across_independent_builds() -> None:
    model_a = build_tiny_sequential_model()
    batch_a = build_tiny_batch()
    output_a = model_a(batch_a)

    model_b = build_tiny_sequential_model()
    batch_b = build_tiny_batch()
    output_b = model_b(batch_b)

    # Independently constructed model+batch pairs, from the same deterministic
    # factories, must be bit-identical -- not just close.
    assert torch.equal(output_a, output_b)


def test_fixture_via_pytest_injection_has_expected_shape_and_dtype(
    tiny_sequential_model: torch.nn.Module, tiny_batch: torch.Tensor
) -> None:
    assert tiny_batch.shape == (BATCH_SIZE, BLOCK_WIDTH)
    assert tiny_batch.dtype == torch.float32
    assert tiny_batch.device.type == "cpu"

    output = tiny_sequential_model(tiny_batch)
    assert output.shape == tiny_batch.shape
    assert output.dtype == torch.float32
    assert output.device.type == "cpu"


def test_end_to_end_example_runs() -> None:
    """The README's example must stay runnable; it is documentation that executes."""
    import subprocess
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parents[2] / "examples" / "end_to_end.py"
    assert script.exists()
    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=300
    )
    assert completed.returncode == 0, completed.stderr
    assert "1. declare_blocks" in completed.stdout
    assert "2. profile_blocks" in completed.stdout
    assert "3. plan_checkpoints" in completed.stdout
