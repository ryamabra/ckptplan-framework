#!/usr/bin/env python3
"""Print a comparison table for a saved benchmark result JSON.

This is a purely local, CPU-only script. It deliberately does **not** run inside
a Modal function: saved runs can be re-reported for free as many times as
wanted, and the Modal functions stay pure measurement jobs.

Usage::

    python benchmarks/report.py benchmarks/matrix_a10g_result.json
    python benchmarks/report.py run.json --reference checkpoint_all
    python benchmarks/report.py run.json --predicted preds.json
    python benchmarks/report.py run.json --predicted no_checkpoint=4433117184 checkpoint_all=0

Three saved-result shapes are recognized (see ``load_configs``):

1. ``{"results": [{"planner": ..., ...}, ...]}`` -- the expanded schema
   ``benchmarks/modal_matrix.py`` writes today.
2. ``{"results": {"<config>": {...}, ...}}`` -- the older matrix schema in
   ``benchmarks/matrix_a10g_result.json``, which predates ``step_latency_ms``
   and several other fields.
3. No ``"results"`` key at all -- config dicts sitting at the top level beside
   scalar metadata, as in ``benchmarks/oom_boundary_a10g.json``.

Two distinct blanks
-------------------
``n/r`` means **not recorded**: the saved artifact never contained the input
this cell needs. ``—`` means **not applicable**: the input exists but the value
is undefined for this row (the configuration OOMed, the reference OOMed, a
reference denominator is zero, or fewer than two latency trials exist). Both
used to render as one em dash, which conflated a gap in the artifact with a
genuinely undefined quantity.

Derived throughput
------------------
``ckptplan.run_benchmark`` computes ``throughput_samples_per_sec`` as
``samples * 1000 / step_latency_ms_mean`` (``ckptplan/api.py``), a pure
algebraic function of batch size and mean latency rather than an independent
measurement. When an artifact records batch size and mean latency but not
throughput, this script reproduces that exact formula and marks the cell with
``~`` for derived. Because batch size is constant across configurations in
these runs, a derived ``throughput_change_pct`` is a monotone restatement of
``latency_overhead_pct`` and carries no independent information -- it is shown
for schema parity with real runs, not as separate evidence.

Prediction gap
--------------
``prediction_gap_pct`` is **necessarily 100% for ``checkpoint_all``**, whose
``predicted_activation_bytes_after`` is 0 by construction (every block is
checkpointed, so the planner predicts no retained block activations), making
the gap ``|0 - peak| / peak``. More generally the gap is large whenever the
profiler's additive per-block isolated activation estimate is compared against
a measured whole-model peak dominated by parameters, gradients, and allocator
reuse. Per MVP_SPEC.md §12.5 the gap is a **reported diagnostic, not a release
gate**; it is surfaced here, not corrected.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ckptplan import compare_results
from ckptplan.types import BenchmarkResult

NOT_APPLICABLE = "—"
NOT_RECORDED = "n/r"
DERIVED_MARK = "~"

# Backwards-compatible alias: the em dash is now specifically "not applicable".
EM_DASH = NOT_APPLICABLE

COLUMNS = (
    "config",
    "peak_alloc",
    "mem_reduction_pct",
    "latency_mean",
    "latency_std",
    "latency_overhead_pct",
    "throughput_change_pct",
    "prediction_gap_pct",
    "correctness",
    "oom",
)

_LATENCY_KEYS = ("step_latency_ms", "step_latency_ms_mean", "latency_ms_mean")


@dataclass(frozen=True)
class LoadedConfig:
    """One configuration rebuilt from a saved result JSON.

    ``source_keys`` is the raw key set of the saved dict. It lets the renderer
    tell "this field was never recorded" apart from "this field was recorded as
    zero", which is what separates ``n/r`` from ``—`` in the output.
    """

    result: BenchmarkResult
    predicted_activation_bytes_after: Optional[int]
    source_keys: frozenset[str]
    throughput_is_derived: bool = False

    @property
    def config_name(self) -> str:
        return self.result.config_name

    def has_any(self, *keys: str) -> bool:
        """True if the saved dict recorded at least one of ``keys``."""
        return any(key in self.source_keys for key in keys)

    @property
    def has_latency(self) -> bool:
        return self.has_any(*_LATENCY_KEYS)

    @property
    def has_throughput(self) -> bool:
        return self.has_any("throughput_samples_per_sec") or self.throughput_is_derived


def _looks_like_config(value: Any) -> bool:
    """Heuristic for a top-level dict that is a per-config result, not metadata."""
    return isinstance(value, dict) and ("oom" in value or "peak_allocated_bytes" in value)


def _first(entry: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present key's value, else ``default``."""
    for key in keys:
        if key in entry and entry[key] is not None:
            return entry[key]
    return default


def build_result(
    config_name: str,
    entry: dict[str, Any],
    batch_size: Optional[int] = None,
) -> LoadedConfig:
    """Rebuild a :class:`BenchmarkResult` from one saved per-config dict.

    Missing fields get neutral defaults; nothing is invented. When
    ``throughput_samples_per_sec`` is absent but ``batch_size`` and a non-zero
    mean latency are available, throughput is derived with ``run_benchmark``'s
    own formula and flagged via ``LoadedConfig.throughput_is_derived``.
    """
    trials = tuple(float(x) for x in _first(entry, "step_latency_ms", default=()) or ())
    mean = _first(entry, "step_latency_ms_mean", "latency_ms_mean", default=None)
    if mean is None:
        mean = sum(trials) / len(trials) if trials else 0.0
    mean = float(mean)

    throughput = _first(entry, "throughput_samples_per_sec")
    derived = False
    if throughput is None and batch_size and mean:
        # Mirrors ckptplan/api.py: samples * 1000 / step_latency_ms_mean.
        throughput = batch_size * 1000.0 / mean
        derived = True

    predicted = _first(entry, "predicted_activation_bytes_after", "predicted_activation_after")

    result = BenchmarkResult(
        config_name=config_name,
        plan_id=str(_first(entry, "plan_id", default="") or ""),
        device_name=str(_first(entry, "device_name", default="unknown") or "unknown"),
        pytorch_version=str(_first(entry, "pytorch_version", default="unknown") or "unknown"),
        python_version=str(_first(entry, "python_version", default="unknown") or "unknown"),
        dtype=str(_first(entry, "dtype", default="unknown") or "unknown"),
        batch_shape=str(_first(entry, "batch_shape", default="unknown") or "unknown"),
        warmup_trials=int(_first(entry, "warmup_trials", default=0) or 0),
        measured_trials=int(_first(entry, "measured_trials", default=len(trials)) or 0),
        peak_allocated_bytes=int(_first(entry, "peak_allocated_bytes", default=0) or 0),
        peak_reserved_bytes=int(_first(entry, "peak_reserved_bytes", default=0) or 0),
        step_latency_ms=trials,
        step_latency_ms_mean=mean,
        step_latency_ms_p50=float(_first(entry, "latency_ms_p50", "step_latency_ms_p50", default=mean)),
        step_latency_ms_p95=float(_first(entry, "latency_ms_p95", "step_latency_ms_p95", default=mean)),
        throughput_samples_per_sec=float(throughput or 0.0),
        correctness_checked=bool(_first(entry, "correctness_checked", default=False)),
        correctness_reference="none",
        correctness_max_abs_output_diff=_first(entry, "correctness_max_abs_output_diff", "max_output_diff"),
        correctness_max_abs_grad_diff=_first(entry, "correctness_max_abs_grad_diff", "max_grad_diff"),
        correctness_passed=entry.get("correctness_passed"),
        oom=bool(_first(entry, "oom", default=False)),
        error_message=entry.get("error_message"),
        environment={},
    )
    return LoadedConfig(
        result=result,
        predicted_activation_bytes_after=None if predicted is None else int(predicted),
        source_keys=frozenset(entry),
        throughput_is_derived=derived,
    )


def load_configs(payload: dict[str, Any]) -> tuple[LoadedConfig, ...]:
    """Extract per-config results from any of the three recognized shapes.

    Raises:
        ValueError: If no per-config results can be located in ``payload``.
    """
    results = payload.get("results")
    payload_batch = payload.get("batch_size")

    def batch_for(entry: Mapping[str, Any]) -> Optional[int]:
        value = _first(entry, "batch_size", default=payload_batch)
        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    if isinstance(results, list):
        loaded = [
            build_result(
                str(_first(entry, "planner", "config_name", default=f"config{index}")),
                entry,
                batch_for(entry),
            )
            for index, entry in enumerate(results)
            if isinstance(entry, dict)
        ]
    elif isinstance(results, dict):
        loaded = [
            build_result(str(name), entry, batch_for(entry))
            for name, entry in results.items()
            if isinstance(entry, dict)
        ]
    else:
        loaded = [
            build_result(str(name), value, batch_for(value))
            for name, value in payload.items()
            if _looks_like_config(value)
        ]

    if not loaded:
        raise ValueError(
            "no per-config benchmark results found; expected a 'results' list, a "
            "'results' object keyed by config name, or top-level config objects"
        )
    return tuple(loaded)


# --- predictions -----------------------------------------------------------------


def parse_predicted_specs(specs: Sequence[str]) -> dict[str, int]:
    """Parse ``--predicted`` specs into a ``config_name -> bytes`` mapping.

    Each spec is either ``name=bytes`` or a path to a JSON object mapping config
    names to byte counts. Later specs override earlier ones.

    Raises:
        ValueError: On an unparseable spec, a non-integer or negative byte
            count, an unreadable JSON file, or a JSON payload that is not a
            flat object of name to number.
    """
    predicted: dict[str, int] = {}
    for spec in specs:
        if "=" in spec:
            name, _, raw = spec.partition("=")
            name = name.strip()
            if not name:
                raise ValueError(f"--predicted spec {spec!r} has an empty config name")
            predicted[name] = _coerce_bytes(name, raw.strip())
            continue

        path = Path(spec)
        try:
            payload = json.loads(path.read_text())
        except OSError as exc:
            raise ValueError(f"--predicted file {spec!r} cannot be read: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(f"--predicted file {spec!r} is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(
                f"--predicted file {spec!r} must contain a JSON object mapping "
                "config names to byte counts"
            )
        for name, raw in payload.items():
            predicted[str(name)] = _coerce_bytes(str(name), raw)
    return predicted


def _coerce_bytes(name: str, raw: Any) -> int:
    """Coerce a predicted-bytes value, rejecting non-integers and negatives."""
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"predicted value for {name!r} must be an integer byte count, got {raw!r}"
        ) from exc
    if value < 0:
        raise ValueError(f"predicted value for {name!r} must be non-negative, got {value}")
    return value


def resolve_predictions(
    configs: Sequence[LoadedConfig],
    overrides: Optional[Mapping[str, int]] = None,
) -> tuple[dict[str, int], tuple[str, ...]]:
    """Merge artifact-embedded predictions with CLI overrides.

    Returns the merged mapping and the tuple of override names that match no
    configuration in the artifact (reported as warnings, not errors, so one
    predictions file can be reused across artifacts with differing config sets).
    """
    merged = {
        config.config_name: config.predicted_activation_bytes_after
        for config in configs
        if config.predicted_activation_bytes_after is not None
    }
    known = {config.config_name for config in configs}
    unknown: list[str] = []
    for name, value in (overrides or {}).items():
        if name not in known:
            unknown.append(name)
            continue
        merged[name] = value
    return merged, tuple(unknown)


# --- rendering -------------------------------------------------------------------


def _cell(value: Optional[Any], recorded: bool, fmt: str = "{:+.2f}", suffix: str = "") -> str:
    """Render one cell, distinguishing not-recorded from not-applicable."""
    if not recorded:
        return NOT_RECORDED
    if value is None:
        return NOT_APPLICABLE
    return fmt.format(value) + suffix


def _correctness_cell(config: LoadedConfig) -> str:
    if "correctness_passed" not in config.source_keys:
        return NOT_RECORDED
    passed = config.result.correctness_passed
    if passed is None:
        return NOT_APPLICABLE
    return "pass" if passed else "FAIL"


def build_rows(
    configs: Sequence[LoadedConfig],
    reference_config_name: str,
    predicted: Optional[Mapping[str, int]] = None,
) -> list[list[str]]:
    """Render the comparison as a list of string rows (header excluded)."""
    if predicted is None:
        predicted, _ = resolve_predictions(configs)

    comparisons = compare_results(
        [config.result for config in configs],
        reference_config_name,
        predicted_activation_bytes=dict(predicted),
    )

    by_name = {config.config_name: config for config in configs}
    reference = by_name[reference_config_name]

    rows: list[list[str]] = []
    for config, comparison in zip(configs, comparisons):
        result = config.result
        has_peak = config.has_any("peak_allocated_bytes")
        has_latency = config.has_latency
        # A ratio is "recorded" only if both sides of it were recorded.
        peak_pair = has_peak and reference.has_any("peak_allocated_bytes")
        latency_pair = has_latency and reference.has_latency
        throughput_pair = config.has_throughput and reference.has_throughput
        throughput_suffix = (
            DERIVED_MARK
            if config.throughput_is_derived or reference.throughput_is_derived
            else ""
        )
        rows.append(
            [
                result.config_name + (" *" if comparison.is_reference else ""),
                _cell(None if result.oom else result.peak_allocated_bytes, has_peak, "{:,}"),
                _cell(comparison.peak_allocated_reduction_pct, peak_pair),
                _cell(
                    None if result.oom else result.step_latency_ms_mean,
                    has_latency,
                    "{:.2f}",
                ),
                _cell(comparison.latency_ms_std, config.has_any("step_latency_ms"), "{:.2f}"),
                _cell(comparison.latency_overhead_pct, latency_pair),
                _cell(comparison.throughput_change_pct, throughput_pair, suffix=throughput_suffix),
                _cell(comparison.prediction_gap_pct, config.config_name in predicted),
                _correctness_cell(config),
                _cell(
                    "yes" if result.oom else "no",
                    "oom" in config.source_keys,
                    "{}",
                ),
            ]
        )
    return rows


def format_table(rows: Sequence[Sequence[str]], headers: Sequence[str] = COLUMNS) -> str:
    """Format rows as an aligned fixed-width table."""
    widths = [
        max(len(str(headers[i])), *(len(row[i]) for row in rows)) if rows else len(str(headers[i]))
        for i in range(len(headers))
    ]

    def render(cells: Sequence[str]) -> str:
        parts = [str(cells[0]).ljust(widths[0])]
        parts += [str(cells[i]).rjust(widths[i]) for i in range(1, len(headers))]
        return "  ".join(parts).rstrip()

    lines = [render(headers), "  ".join("-" * width for width in widths)]
    lines += [render(row) for row in rows]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print a comparison table for a saved benchmark result JSON."
    )
    parser.add_argument("json_path", type=Path, help="path to a saved benchmark result JSON")
    parser.add_argument(
        "--reference",
        default="no_checkpoint",
        help="config name to compare against (default: no_checkpoint)",
    )
    parser.add_argument(
        "--predicted",
        action="extend",
        nargs="+",
        default=[],
        metavar="SPEC",
        help=(
            "predicted_activation_bytes_after values, as name=bytes pairs and/or "
            "paths to a JSON object mapping config names to byte counts. May be "
            "repeated; later values win, and both override values embedded in the "
            "artifact."
        ),
    )
    args = parser.parse_args(argv)

    try:
        payload = json.loads(args.json_path.read_text())
    except OSError as exc:
        print(f"error: cannot read {args.json_path}: {exc}", file=sys.stderr)
        return 2
    except json.JSONDecodeError as exc:
        print(f"error: {args.json_path} is not valid JSON: {exc}", file=sys.stderr)
        return 2

    try:
        overrides = parse_predicted_specs(args.predicted)
        configs = load_configs(payload)
        predicted, unknown = resolve_predictions(configs, overrides)
        rows = build_rows(configs, args.reference, predicted)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for name in unknown:
        print(
            f"warning: --predicted names {name!r}, which is not a config in this artifact",
            file=sys.stderr,
        )

    print(f"{args.json_path}  (reference: {args.reference} *)")
    model = payload.get("model")
    if model:
        params = payload.get("parameters")
        suffix = f"  params: {params:,}" if isinstance(params, int) else ""
        print(f"model: {model}{suffix}")
    print()
    print(format_table(rows))
    print()
    print(f"{NOT_RECORDED} = not recorded in this artifact; {NOT_APPLICABLE} = not applicable.")
    print(f"{DERIVED_MARK} = throughput derived as batch_size * 1000 / mean latency, not measured.")
    print("peak_alloc in bytes; latency in ms.")
    print("prediction_gap_pct is a reported diagnostic (MVP_SPEC.md 12.5), not a gate;")
    print("it is 100% for checkpoint_all by construction (predicted-after is 0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
