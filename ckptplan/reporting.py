"""Benchmark comparison reporting.

Post-hoc comparison of :class:`~ckptplan.types.BenchmarkResult` objects against a
reference configuration (typically ``"no_checkpoint"``). This module is purely
descriptive: it reports measured differences and the planner's prediction gap,
and never alters or second-guesses planner behaviour.

Per MVP_SPEC.md §12.4/§12.5 every quantity here is *reported, not gated*. In
particular the prediction gap is expected to be non-zero: the profiler's
additive per-block isolated activation estimate legitimately exceeds the
measured whole-model peak reduction, and that gap is documented and expected.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Iterable, Mapping, Optional

from ckptplan.types import BenchmarkResult

__all__ = ["ConfigComparison", "compare_results"]


@dataclass(frozen=True)
class ConfigComparison:
    """Comparison of one benchmark configuration against the reference.

    Every ratio field is ``None`` when it cannot be computed (the configuration
    or the reference OOMed, the predicted value was not supplied, or the
    reference denominator is zero).
    """

    config_name: str
    is_reference: bool
    oom: bool
    peak_allocated_reduction_pct: Optional[float]
    peak_reserved_reduction_pct: Optional[float]
    latency_overhead_pct: Optional[float]
    throughput_change_pct: Optional[float]
    latency_ms_std: Optional[float]
    prediction_gap_pct: Optional[float]


def _pct_reduction(reference: float, value: float) -> Optional[float]:
    """Percent decrease from ``reference`` to ``value`` (positive = smaller)."""
    if reference == 0:
        return None
    return (reference - value) / reference * 100.0


def _pct_change(reference: float, value: float) -> Optional[float]:
    """Percent increase from ``reference`` to ``value`` (positive = larger)."""
    if reference == 0:
        return None
    return (value - reference) / reference * 100.0


def _latency_std(result: BenchmarkResult) -> Optional[float]:
    """Sample standard deviation of the measured per-trial latencies."""
    trials = result.step_latency_ms
    if len(trials) < 2:
        return None
    return statistics.stdev(trials)


def compare_results(
    results: Iterable[BenchmarkResult],
    reference_config_name: str,
    predicted_activation_bytes: Optional[Mapping[str, Optional[int]]] = None,
) -> tuple[ConfigComparison, ...]:
    """Compare benchmark results against a reference configuration.

    ``predicted_activation_bytes`` maps ``config_name`` to that configuration's
    ``CheckpointPlan.predicted_activation_bytes_after``. Predictions are passed
    as an explicit mapping rather than by taking ``CheckpointPlan`` objects
    alongside the results: ``BenchmarkResult`` does not carry the prediction, and
    a plain mapping keeps this module decoupled from the planner types so that
    results reloaded from a saved JSON benchmark run (which stores the predicted
    scalar, not the plan) can be reported without reconstructing plans. Configs
    absent from the mapping simply get ``prediction_gap_pct=None``.

    Args:
        results: Benchmark results, one per configuration.
        reference_config_name: ``config_name`` of the baseline configuration.
        predicted_activation_bytes: Optional per-config predicted activation
            bytes after checkpointing.

    Returns:
        One :class:`ConfigComparison` per input result, in input order.

    Raises:
        ValueError: If the reference config is not present in ``results``, or if
            any ``config_name`` appears more than once.
    """
    ordered = tuple(results)
    by_name: dict[str, BenchmarkResult] = {}
    for result in ordered:
        if result.config_name in by_name:
            raise ValueError(f"duplicate config_name in results: {result.config_name!r}")
        by_name[result.config_name] = result

    reference = by_name.get(reference_config_name)
    if reference is None:
        raise ValueError(
            f"reference config {reference_config_name!r} not found in results; "
            f"have {sorted(by_name)}"
        )

    predictions: Mapping[str, Optional[int]] = predicted_activation_bytes or {}
    # An OOMed reference has no valid peak/latency, so no comparison is
    # meaningful against it; all relative fields degrade to None.
    reference_usable = not reference.oom

    comparisons = []
    for result in ordered:
        usable = reference_usable and not result.oom
        if usable:
            peak_allocated = _pct_reduction(
                reference.peak_allocated_bytes, result.peak_allocated_bytes
            )
            peak_reserved = _pct_reduction(
                reference.peak_reserved_bytes, result.peak_reserved_bytes
            )
            latency = _pct_change(
                reference.step_latency_ms_mean, result.step_latency_ms_mean
            )
            throughput = _pct_change(
                reference.throughput_samples_per_sec, result.throughput_samples_per_sec
            )
        else:
            peak_allocated = peak_reserved = latency = throughput = None

        predicted = predictions.get(result.config_name)
        if result.oom or predicted is None or result.peak_allocated_bytes == 0:
            prediction_gap = None
        else:
            prediction_gap = (
                abs(predicted - result.peak_allocated_bytes)
                / result.peak_allocated_bytes
                * 100.0
            )

        comparisons.append(
            ConfigComparison(
                config_name=result.config_name,
                is_reference=result.config_name == reference_config_name,
                oom=result.oom,
                peak_allocated_reduction_pct=peak_allocated,
                peak_reserved_reduction_pct=peak_reserved,
                latency_overhead_pct=latency,
                throughput_change_pct=throughput,
                latency_ms_std=None if result.oom else _latency_std(result),
                prediction_gap_pct=prediction_gap,
            )
        )
    return tuple(comparisons)
