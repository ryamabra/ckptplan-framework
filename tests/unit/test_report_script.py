"""Unit tests for the local ``benchmarks/report.py`` reporting script.

The script lives outside the installed package (it is a standalone local tool,
not part of the ``ckptplan`` public API), so it is loaded by path.
"""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_REPORT_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "report.py"


def _load_report_module():
    spec = importlib.util.spec_from_file_location("benchmarks_report", _REPORT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


report = _load_report_module()

COLUMN = {name: i for i, name in enumerate(report.COLUMNS)}


EXPANDED_PAYLOAD = {
    "model": "toy-transformer",
    "batch_size": 1,
    "results": [
        {
            "planner": "no_checkpoint",
            "plan_id": "plan-a",
            "peak_allocated_bytes": 1000,
            "peak_reserved_bytes": 2000,
            "step_latency_ms": [10.0, 12.0, 14.0],
            "latency_ms_mean": 12.0,
            "latency_ms_p50": 12.0,
            "latency_ms_p95": 14.0,
            "throughput_samples_per_sec": 100.0,
            "predicted_activation_bytes_after": 1500,
            "correctness_passed": None,
            "oom": False,
            "error_message": None,
        },
        {
            "planner": "checkpoint_all",
            "plan_id": "plan-b",
            "peak_allocated_bytes": 600,
            "peak_reserved_bytes": 1500,
            "step_latency_ms": [14.0, 15.0, 16.0],
            "latency_ms_mean": 15.0,
            "latency_ms_p50": 15.0,
            "latency_ms_p95": 16.0,
            "throughput_samples_per_sec": 80.0,
            "predicted_activation_bytes_after": 0,
            "correctness_passed": True,
            "oom": False,
            "error_message": None,
        },
    ],
}

# Mirrors benchmarks/matrix_a10g_result.json: "results" keyed by planner name,
# with no step_latency_ms, peak_reserved_bytes, throughput, or predictions.
OLD_MATRIX_PAYLOAD = {
    "model": "toy-transformer",
    "parameters": 1208598528,
    "batch_size": 1,
    "results": {
        "no_checkpoint": {
            "peak_allocated_bytes": 9896961536,
            "latency_ms_mean": 1630.259375,
            "correctness_passed": None,
            "oom": False,
        },
        "checkpoint_all": {
            "peak_allocated_bytes": 9887221760,
            "latency_ms_mean": 2192.90546875,
            "correctness_passed": True,
            "oom": False,
        },
    },
}

# Mirrors benchmarks/oom_boundary_a10g.json: no "results" key at all; config
# dicts sit at the top level beside scalar and string metadata.
TOP_LEVEL_PAYLOAD = {
    "modal_no_checkpoint_run": "ap-abc",
    "parameters": 1208598528,
    "seq_len": 4096,
    "no_checkpoint": {"oom": True, "peak_allocated_bytes": 0},
    "checkpoint_all": {
        "oom": False,
        "peak_allocated_bytes": 12505498624,
        "peak_reserved_bytes": 13182697472,
        "correctness_passed": None,
    },
    "interpretation": "No-checkpoint OOMs while checkpoint-all completes.",
}


# --- shape detection and rebuild -------------------------------------------------


def test_expanded_list_schema_rebuilds_every_field() -> None:
    configs = report.load_configs(EXPANDED_PAYLOAD)
    assert [c.config_name for c in configs] == ["no_checkpoint", "checkpoint_all"]

    reference = configs[0].result
    assert reference.plan_id == "plan-a"
    assert reference.peak_allocated_bytes == 1000
    assert reference.peak_reserved_bytes == 2000
    assert reference.step_latency_ms == (10.0, 12.0, 14.0)
    assert reference.step_latency_ms_mean == pytest.approx(12.0)
    assert reference.step_latency_ms_p95 == pytest.approx(14.0)
    assert reference.throughput_samples_per_sec == pytest.approx(100.0)
    assert reference.measured_trials == 3
    assert reference.oom is False
    assert configs[0].predicted_activation_bytes_after == 1500
    # A predicted value of 0 is preserved, not coerced to None.
    assert configs[1].predicted_activation_bytes_after == 0


def test_old_matrix_schema_does_not_crash_and_defaults_missing_fields() -> None:
    configs = report.load_configs(OLD_MATRIX_PAYLOAD)
    assert [c.config_name for c in configs] == ["no_checkpoint", "checkpoint_all"]

    reference = configs[0].result
    assert reference.peak_allocated_bytes == 9896961536
    assert reference.step_latency_ms_mean == pytest.approx(1630.259375)
    # Fields the old schema predates default neutrally rather than raising.
    assert reference.step_latency_ms == ()
    assert reference.measured_trials == 0
    assert reference.peak_reserved_bytes == 0
    assert configs[0].predicted_activation_bytes_after is None
    assert configs[1].result.correctness_passed is True


def test_top_level_schema_finds_configs_and_ignores_metadata() -> None:
    configs = report.load_configs(TOP_LEVEL_PAYLOAD)
    assert [c.config_name for c in configs] == ["no_checkpoint", "checkpoint_all"]
    assert configs[0].result.oom is True
    assert configs[1].result.peak_reserved_bytes == 13182697472
    assert "interpretation" not in {c.config_name for c in configs}
    assert "parameters" not in {c.config_name for c in configs}


def test_source_keys_distinguish_absent_from_zero() -> None:
    old = report.load_configs(OLD_MATRIX_PAYLOAD)[0]
    assert old.has_any("latency_ms_mean") is True
    assert old.has_any("step_latency_ms") is False
    assert old.has_any("peak_reserved_bytes") is False

    expanded = report.load_configs(EXPANDED_PAYLOAD)[0]
    assert expanded.has_any("step_latency_ms") is True


def test_mean_is_derived_from_trials_when_no_mean_recorded() -> None:
    loaded = report.build_result("x", {"step_latency_ms": [10.0, 20.0], "oom": False})
    assert loaded.result.step_latency_ms_mean == pytest.approx(15.0)
    assert loaded.result.measured_trials == 2


def test_empty_config_dict_yields_neutral_defaults() -> None:
    loaded = report.build_result("x", {"oom": False})
    result = loaded.result
    assert result.peak_allocated_bytes == 0
    assert result.step_latency_ms == ()
    assert result.step_latency_ms_mean == pytest.approx(0.0)
    assert result.correctness_passed is None
    assert loaded.predicted_activation_bytes_after is None


def test_payload_without_any_configs_raises() -> None:
    with pytest.raises(ValueError, match="no per-config benchmark results"):
        report.load_configs({"model": "toy", "interpretation": "nothing here"})


# --- derived throughput ----------------------------------------------------------


def test_throughput_is_derived_with_run_benchmark_formula() -> None:
    configs = report.load_configs(OLD_MATRIX_PAYLOAD)
    reference = configs[0]
    assert reference.throughput_is_derived is True
    # ckptplan/api.py: samples * 1000 / step_latency_ms_mean, batch_size == 1.
    assert reference.result.throughput_samples_per_sec == pytest.approx(1000.0 / 1630.259375)
    assert configs[1].result.throughput_samples_per_sec == pytest.approx(1000.0 / 2192.90546875)


def test_recorded_throughput_is_not_overwritten_by_derivation() -> None:
    configs = report.load_configs(EXPANDED_PAYLOAD)
    assert configs[0].throughput_is_derived is False
    assert configs[0].result.throughput_samples_per_sec == pytest.approx(100.0)


def test_throughput_not_derived_without_batch_size() -> None:
    loaded = report.build_result("x", {"latency_ms_mean": 10.0, "oom": False})
    assert loaded.throughput_is_derived is False
    assert loaded.has_throughput is False


def test_throughput_not_derived_from_zero_latency() -> None:
    loaded = report.build_result("x", {"latency_ms_mean": 0.0, "oom": False}, batch_size=4)
    assert loaded.throughput_is_derived is False


def test_derived_throughput_cells_are_marked() -> None:
    rows = report.build_rows(report.load_configs(OLD_MATRIX_PAYLOAD), "no_checkpoint")
    cell = rows[1][COLUMN["throughput_change_pct"]]
    assert cell.endswith(report.DERIVED_MARK)
    assert cell == "-25.66" + report.DERIVED_MARK
    # Measured throughput carries no derived mark.
    measured = report.build_rows(report.load_configs(EXPANDED_PAYLOAD), "no_checkpoint")
    assert not measured[1][COLUMN["throughput_change_pct"]].endswith(report.DERIVED_MARK)


# --- not-recorded vs not-applicable ----------------------------------------------


def test_not_recorded_and_not_applicable_are_distinct_markers() -> None:
    assert report.NOT_RECORDED != report.NOT_APPLICABLE

    rows = report.build_rows(report.load_configs(TOP_LEVEL_PAYLOAD), "no_checkpoint")
    # Peak was recorded for the OOMed reference but is meaningless: not applicable.
    assert rows[0][COLUMN["peak_alloc"]] == report.NOT_APPLICABLE
    # Reduction is undefined because the reference OOMed: not applicable.
    assert rows[0][COLUMN["mem_reduction_pct"]] == report.NOT_APPLICABLE
    # Latency was never saved in this artifact at all: not recorded.
    assert rows[0][COLUMN["latency_mean"]] == report.NOT_RECORDED
    assert rows[0][COLUMN["latency_overhead_pct"]] == report.NOT_RECORDED
    # correctness_passed absent entirely vs present-but-null.
    assert rows[0][COLUMN["correctness"]] == report.NOT_RECORDED
    assert rows[1][COLUMN["correctness"]] == report.NOT_APPLICABLE


def test_old_schema_missing_trial_tuple_is_not_recorded_not_undefined() -> None:
    rows = report.build_rows(report.load_configs(OLD_MATRIX_PAYLOAD), "no_checkpoint")
    assert rows[0][COLUMN["latency_std"]] == report.NOT_RECORDED
    assert rows[1][COLUMN["latency_mean"]] == "2192.91"
    assert rows[1][COLUMN["latency_overhead_pct"]] == "+34.51"


def test_single_trial_std_is_not_applicable_not_missing() -> None:
    payload = {
        "results": [
            {"planner": "no_checkpoint", "peak_allocated_bytes": 10, "step_latency_ms": [5.0], "oom": False},
        ]
    }
    rows = report.build_rows(report.load_configs(payload), "no_checkpoint")
    assert rows[0][COLUMN["latency_std"]] == report.NOT_APPLICABLE


def test_missing_prediction_is_not_recorded() -> None:
    rows = report.build_rows(report.load_configs(OLD_MATRIX_PAYLOAD), "no_checkpoint")
    assert rows[0][COLUMN["prediction_gap_pct"]] == report.NOT_RECORDED


def test_rows_never_render_the_literal_none() -> None:
    for payload in (EXPANDED_PAYLOAD, OLD_MATRIX_PAYLOAD, TOP_LEVEL_PAYLOAD):
        rows = report.build_rows(report.load_configs(payload), "no_checkpoint")
        assert "None" not in [cell for row in rows for cell in row]


# --- predictions -----------------------------------------------------------------


def test_parse_inline_pairs() -> None:
    assert report.parse_predicted_specs(["a=10", "b=0"]) == {"a": 10, "b": 0}


def test_parse_json_file(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"a": 1, "b": 2}))
    assert report.parse_predicted_specs([str(path)]) == {"a": 1, "b": 2}


def test_later_specs_override_earlier(tmp_path: Path) -> None:
    path = tmp_path / "preds.json"
    path.write_text(json.dumps({"a": 1}))
    assert report.parse_predicted_specs([str(path), "a=99"]) == {"a": 99}
    assert report.parse_predicted_specs(["a=99", str(path)]) == {"a": 1}


def test_parse_rejects_bad_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="integer byte count"):
        report.parse_predicted_specs(["a=notanumber"])
    with pytest.raises(ValueError, match="non-negative"):
        report.parse_predicted_specs(["a=-5"])
    with pytest.raises(ValueError, match="empty config name"):
        report.parse_predicted_specs(["=5"])
    with pytest.raises(ValueError, match="cannot be read"):
        report.parse_predicted_specs([str(tmp_path / "missing.json")])

    bad = tmp_path / "bad.json"
    bad.write_text("[1, 2]")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        report.parse_predicted_specs([str(bad)])


def test_resolve_predictions_overrides_artifact_values() -> None:
    configs = report.load_configs(EXPANDED_PAYLOAD)
    merged, unknown = report.resolve_predictions(configs)
    assert merged == {"no_checkpoint": 1500, "checkpoint_all": 0}
    assert unknown == ()

    merged, unknown = report.resolve_predictions(configs, {"no_checkpoint": 7})
    assert merged["no_checkpoint"] == 7
    assert merged["checkpoint_all"] == 0
    assert unknown == ()


def test_resolve_predictions_reports_unknown_names() -> None:
    configs = report.load_configs(EXPANDED_PAYLOAD)
    merged, unknown = report.resolve_predictions(configs, {"typo": 5})
    assert unknown == ("typo",)
    assert "typo" not in merged


def test_supplied_predictions_populate_the_gap_column() -> None:
    configs = report.load_configs(OLD_MATRIX_PAYLOAD)
    predicted, _ = report.resolve_predictions(
        configs, {"no_checkpoint": 4433117184, "checkpoint_all": 0}
    )
    rows = report.build_rows(configs, "no_checkpoint", predicted)
    assert rows[0][COLUMN["prediction_gap_pct"]] == "+55.21"
    # checkpoint_all predicts zero retained activations by construction.
    assert rows[1][COLUMN["prediction_gap_pct"]] == "+100.00"


# --- rendering -------------------------------------------------------------------


def test_reference_row_is_marked() -> None:
    rows = report.build_rows(report.load_configs(EXPANDED_PAYLOAD), "no_checkpoint")
    assert rows[0][0].endswith(" *")
    assert not rows[1][0].endswith(" *")


def test_format_table_is_aligned_and_has_header_rule() -> None:
    rows = report.build_rows(report.load_configs(EXPANDED_PAYLOAD), "no_checkpoint")
    lines = report.format_table(rows).splitlines()
    assert lines[0].startswith("config")
    assert set(lines[1]) <= {"-", " "}
    assert len(lines) == len(rows) + 2


# --- CLI -------------------------------------------------------------------------


def test_main_reports_real_artifacts(capsys: pytest.CaptureFixture[str]) -> None:
    for name in ("matrix_a10g_result.json", "oom_boundary_a10g.json"):
        path = _REPORT_PATH.parent / name
        assert report.main([str(path)]) == 0
        out = capsys.readouterr().out
        assert "no_checkpoint *" in out
        assert "checkpoint_all" in out
        assert "None" not in out


def test_main_accepts_inline_predictions(capsys: pytest.CaptureFixture[str]) -> None:
    path = _REPORT_PATH.parent / "matrix_a10g_result.json"
    assert report.main([str(path), "--predicted", "no_checkpoint=4433117184", "checkpoint_all=0"]) == 0
    out = capsys.readouterr().out
    assert "+55.21" in out
    assert "+100.00" in out


def test_main_accepts_predictions_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    preds = tmp_path / "preds.json"
    preds.write_text(json.dumps({"no_checkpoint": 4433117184, "checkpoint_all": 0}))
    path = _REPORT_PATH.parent / "matrix_a10g_result.json"
    assert report.main([str(path), "--predicted", str(preds)]) == 0
    assert "+55.21" in capsys.readouterr().out


def test_main_warns_but_succeeds_on_unknown_predicted_name(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(EXPANDED_PAYLOAD))
    assert report.main([str(path), "--predicted", "typo=5"]) == 0
    assert "not a config in this artifact" in capsys.readouterr().err


def test_main_rejects_invalid_predicted_value(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(EXPANDED_PAYLOAD))
    assert report.main([str(path), "--predicted", "no_checkpoint=abc"]) == 2
    assert "integer byte count" in capsys.readouterr().err


def test_main_honours_reference_flag(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(EXPANDED_PAYLOAD))
    assert report.main([str(path), "--reference", "checkpoint_all"]) == 0
    assert "checkpoint_all *" in capsys.readouterr().out


def test_main_rejects_missing_reference(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "run.json"
    path.write_text(json.dumps(EXPANDED_PAYLOAD))
    assert report.main([str(path), "--reference", "nope"]) == 2
    assert "not found in results" in capsys.readouterr().err


def test_main_rejects_unreadable_and_invalid_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert report.main([str(tmp_path / "missing.json")]) == 2
    assert "cannot read" in capsys.readouterr().err

    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert report.main([str(bad)]) == 2
    assert "not valid JSON" in capsys.readouterr().err


def test_legend_explains_both_blank_markers(capsys: pytest.CaptureFixture[str]) -> None:
    path = _REPORT_PATH.parent / "oom_boundary_a10g.json"
    assert report.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "not recorded" in out
    assert "not applicable" in out
    assert "derived" in out
