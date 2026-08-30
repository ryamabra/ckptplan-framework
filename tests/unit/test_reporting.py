"""Unit tests for ``ckptplan.reporting.compare_results``."""

import pytest

from ckptplan import ConfigComparison, compare_results
from ckptplan.types import BenchmarkResult


def make_result(
    config_name: str,
    *,
    peak_allocated_bytes: int = 1000,
    peak_reserved_bytes: int = 2000,
    step_latency_ms: tuple[float, ...] = (10.0, 12.0, 14.0),
    step_latency_ms_mean: float = 12.0,
    throughput_samples_per_sec: float = 100.0,
    oom: bool = False,
) -> BenchmarkResult:
    """Build a synthetic ``BenchmarkResult`` with only the fields under test varied."""
    return BenchmarkResult(
        config_name=config_name,
        plan_id=f"plan-{config_name}",
        device_name="cpu",
        pytorch_version="2.13.0",
        python_version="3.12.0",
        dtype="torch.float32",
        batch_shape="(1, 8, 16)",
        warmup_trials=1,
        measured_trials=len(step_latency_ms),
        peak_allocated_bytes=peak_allocated_bytes,
        peak_reserved_bytes=peak_reserved_bytes,
        step_latency_ms=step_latency_ms,
        step_latency_ms_mean=step_latency_ms_mean,
        step_latency_ms_p50=step_latency_ms_mean,
        step_latency_ms_p95=step_latency_ms_mean,
        throughput_samples_per_sec=throughput_samples_per_sec,
        correctness_checked=False,
        correctness_reference="none",
        correctness_max_abs_output_diff=None,
        correctness_max_abs_grad_diff=None,
        correctness_passed=None,
        oom=oom,
        error_message=None,
        environment={},
    )


def test_reference_config_compares_to_itself_as_zero() -> None:
    reference = make_result("no_checkpoint")
    (comparison,) = compare_results([reference], "no_checkpoint")
    assert comparison.is_reference is True
    assert comparison.oom is False
    assert comparison.peak_allocated_reduction_pct == 0.0
    assert comparison.peak_reserved_reduction_pct == 0.0
    assert comparison.latency_overhead_pct == 0.0
    assert comparison.throughput_change_pct == 0.0


def test_reductions_and_overheads_against_reference() -> None:
    reference = make_result("no_checkpoint")
    checkpointed = make_result(
        "checkpoint_all",
        peak_allocated_bytes=600,
        peak_reserved_bytes=1500,
        step_latency_ms=(14.0, 15.0, 16.0),
        step_latency_ms_mean=15.0,
        throughput_samples_per_sec=80.0,
    )
    results = compare_results([reference, checkpointed], "no_checkpoint")
    assert [c.config_name for c in results] == ["no_checkpoint", "checkpoint_all"]

    checked = results[1]
    assert checked.is_reference is False
    assert checked.peak_allocated_reduction_pct == pytest.approx(40.0)
    assert checked.peak_reserved_reduction_pct == pytest.approx(25.0)
    assert checked.latency_overhead_pct == pytest.approx(25.0)
    assert checked.throughput_change_pct == pytest.approx(-20.0)
    assert checked.latency_ms_std == pytest.approx(1.0)


def test_prediction_gap_uses_supplied_predictions() -> None:
    reference = make_result("no_checkpoint")
    checkpointed = make_result("checkpoint_all", peak_allocated_bytes=800)
    results = compare_results(
        [reference, checkpointed],
        "no_checkpoint",
        predicted_activation_bytes={"checkpoint_all": 1000},
    )
    # The additive isolated estimate legitimately exceeds the measured peak;
    # the gap is reported, not corrected.
    assert results[1].prediction_gap_pct == pytest.approx(25.0)
    # A config with no supplied prediction reports no gap.
    assert results[0].prediction_gap_pct is None


def test_prediction_gap_is_none_when_measured_peak_is_zero() -> None:
    reference = make_result("no_checkpoint", peak_allocated_bytes=0)
    results = compare_results(
        [reference], "no_checkpoint", predicted_activation_bytes={"no_checkpoint": 500}
    )
    assert results[0].prediction_gap_pct is None


def test_oom_config_reports_no_ratios() -> None:
    reference = make_result("no_checkpoint")
    oomed = make_result(
        "greedy",
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
        step_latency_ms=(),
        step_latency_ms_mean=0.0,
        throughput_samples_per_sec=0.0,
        oom=True,
    )
    results = compare_results(
        [reference, oomed], "no_checkpoint", predicted_activation_bytes={"greedy": 500}
    )
    failed = results[1]
    assert failed.oom is True
    assert failed.peak_allocated_reduction_pct is None
    assert failed.peak_reserved_reduction_pct is None
    assert failed.latency_overhead_pct is None
    assert failed.throughput_change_pct is None
    assert failed.latency_ms_std is None
    assert failed.prediction_gap_pct is None
    # The non-OOM reference is still reported normally.
    assert results[0].peak_allocated_reduction_pct == 0.0


def test_oomed_reference_makes_all_ratios_none() -> None:
    reference = make_result(
        "no_checkpoint",
        peak_allocated_bytes=0,
        peak_reserved_bytes=0,
        step_latency_ms=(),
        step_latency_ms_mean=0.0,
        throughput_samples_per_sec=0.0,
        oom=True,
    )
    other = make_result("checkpoint_all")
    results = compare_results([reference, other], "no_checkpoint")
    for comparison in results:
        assert comparison.peak_allocated_reduction_pct is None
        assert comparison.latency_overhead_pct is None
    # Non-relative fields survive an unusable reference.
    assert results[1].latency_ms_std == pytest.approx(2.0)


def test_zero_latency_reference_yields_none_rather_than_dividing_by_zero() -> None:
    reference = make_result(
        "no_checkpoint",
        step_latency_ms=(0.0, 0.0),
        step_latency_ms_mean=0.0,
        throughput_samples_per_sec=0.0,
    )
    other = make_result("checkpoint_all", peak_allocated_bytes=500)
    results = compare_results([reference, other], "no_checkpoint")
    assert results[1].latency_overhead_pct is None
    assert results[1].throughput_change_pct is None
    # Memory ratios are unaffected by the zero-latency reference.
    assert results[1].peak_allocated_reduction_pct == pytest.approx(50.0)
    assert results[0].latency_ms_std == pytest.approx(0.0)


def test_single_trial_has_no_latency_std() -> None:
    reference = make_result("no_checkpoint", step_latency_ms=(12.0,))
    (comparison,) = compare_results([reference], "no_checkpoint")
    assert comparison.latency_ms_std is None


def test_missing_reference_raises() -> None:
    with pytest.raises(ValueError, match="not found in results"):
        compare_results([make_result("checkpoint_all")], "no_checkpoint")


def test_duplicate_config_names_raise() -> None:
    with pytest.raises(ValueError, match="duplicate config_name"):
        compare_results(
            [make_result("no_checkpoint"), make_result("no_checkpoint")],
            "no_checkpoint",
        )


def test_result_is_a_tuple_of_comparisons_in_input_order() -> None:
    names = ["uniform", "no_checkpoint", "greedy"]
    results = compare_results([make_result(n) for n in names], "no_checkpoint")
    assert isinstance(results, tuple)
    assert all(isinstance(c, ConfigComparison) for c in results)
    assert [c.config_name for c in results] == names
