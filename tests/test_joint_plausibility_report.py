from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_joint_plausibility_report import parse_args
from src.analysis.forecast_joint_plausibility_report import (
    build_head_interpretation_rows,
    generate_joint_plausibility_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def diagnostic_dir(tmp_path: Path) -> Path:
    diag_dir = tmp_path / "distribution_diagnosis"
    diag_dir.mkdir()

    _write_csv(
        diag_dir / "joint_plausibility_summary.csv",
        [
            {
                "split": "overall",
                "num_eval_rows": 100,
                "rare_threshold": 0.0001,
                "overall_rare_combo_rate_oracle": 0.0003,
                "overall_rare_combo_rate_predicted": 0.0002,
                "overall_rare_combo_rate_mixed_D_pred": 0.00025,
                "overall_rare_combo_rate_mixed_LOS_pred": 0.00022,
            }
        ],
        [
            "split",
            "num_eval_rows",
            "rare_threshold",
            "overall_rare_combo_rate_oracle",
            "overall_rare_combo_rate_predicted",
            "overall_rare_combo_rate_mixed_D_pred",
            "overall_rare_combo_rate_mixed_LOS_pred",
        ],
    )
    _write_csv(
        diag_dir / "per_head_conditional_distribution.csv",
        [
            {
                "split": "overall",
                "target_name": "SERVICES_D",
                "los_bin": 0,
                "d_value": 0,
                "oracle_count": 10,
                "oracle_prob": 0.1,
                "predicted_count": 2,
                "predicted_prob": 0.02,
                "mixed_D_pred_count": 9,
                "mixed_D_pred_prob": 0.09,
                "mixed_LOS_pred_count": 3,
                "mixed_LOS_pred_prob": 0.03,
                "train_oracle_probability": 0.001,
                "train_oracle_is_rare": "False",
                "cramers_v_oracle": 0.33,
                "cramers_v_predicted": 0.49,
                "delta_cramers_v": 0.16,
                "js_divergence_P_D_given_LOS": 0.05,
                "rare_combo_rate_oracle": 0.0,
                "rare_combo_rate_predicted": 0.0,
                "rare_combo_rate_mixed_D_pred": 0.0,
                "rare_combo_rate_mixed_LOS_pred": 0.0,
                "positive_rate_drift_by_los": 0.09,
            },
            {
                "split": "overall",
                "target_name": "SUB1_D",
                "los_bin": 1,
                "d_value": 1,
                "oracle_count": 5,
                "oracle_prob": 0.05,
                "predicted_count": 20,
                "predicted_prob": 0.20,
                "mixed_D_pred_count": 8,
                "mixed_D_pred_prob": 0.08,
                "mixed_LOS_pred_count": 7,
                "mixed_LOS_pred_prob": 0.07,
                "train_oracle_probability": 0.00005,
                "train_oracle_is_rare": "True",
                "cramers_v_oracle": 0.10,
                "cramers_v_predicted": 0.19,
                "delta_cramers_v": 0.09,
                "js_divergence_P_D_given_LOS": 0.013,
                "rare_combo_rate_oracle": 0.0001,
                "rare_combo_rate_predicted": 0.0011,
                "rare_combo_rate_mixed_D_pred": 0.0012,
                "rare_combo_rate_mixed_LOS_pred": 0.0008,
                "positive_rate_drift_by_los": 0.09,
            },
            {
                "split": "overall",
                "target_name": "FREQ_ATND_SELF_HELP_D",
                "los_bin": 2,
                "d_value": 2,
                "oracle_count": 7,
                "oracle_prob": 0.07,
                "predicted_count": 9,
                "predicted_prob": 0.09,
                "mixed_D_pred_count": 7,
                "mixed_D_pred_prob": 0.07,
                "mixed_LOS_pred_count": 8,
                "mixed_LOS_pred_prob": 0.08,
                "train_oracle_probability": 0.002,
                "train_oracle_is_rare": "False",
                "cramers_v_oracle": 0.10,
                "cramers_v_predicted": 0.19,
                "delta_cramers_v": 0.09,
                "js_divergence_P_D_given_LOS": 0.021,
                "rare_combo_rate_oracle": 0.0,
                "rare_combo_rate_predicted": 0.0,
                "rare_combo_rate_mixed_D_pred": 0.0,
                "rare_combo_rate_mixed_LOS_pred": 0.0,
                "positive_rate_drift_by_los": 0.09,
            },
        ],
        [
            "split",
            "target_name",
            "los_bin",
            "d_value",
            "oracle_count",
            "oracle_prob",
            "predicted_count",
            "predicted_prob",
            "mixed_D_pred_count",
            "mixed_D_pred_prob",
            "mixed_LOS_pred_count",
            "mixed_LOS_pred_prob",
            "train_oracle_probability",
            "train_oracle_is_rare",
            "cramers_v_oracle",
            "cramers_v_predicted",
            "delta_cramers_v",
            "js_divergence_P_D_given_LOS",
            "rare_combo_rate_oracle",
            "rare_combo_rate_predicted",
            "rare_combo_rate_mixed_D_pred",
            "rare_combo_rate_mixed_LOS_pred",
            "positive_rate_drift_by_los",
        ],
    )
    _write_csv(
        diag_dir / "confidence_vs_joint_mismatch.csv",
        [
            {
                "split": "train",
                "row_idx": 1,
                "caseid": 101,
                "target_name": "SERVICES_D",
                "y": 1,
                "oracle_d": 1,
                "predicted_d": 2,
                "oracle_los_bin": 1,
                "predicted_los_bin": 2,
                "rare_oracle": "False",
                "rare_predicted": "False",
                "rare_mixed_D_pred": "False",
                "rare_mixed_LOS_pred": "False",
                "joint_mismatch_predicted": "True",
                "discharge_confidence": 0.97,
                "los_confidence": 0.82,
            },
            {
                "split": "train",
                "row_idx": 2,
                "caseid": 102,
                "target_name": "SUB1_D",
                "y": 0,
                "oracle_d": 1,
                "predicted_d": 1,
                "oracle_los_bin": 1,
                "predicted_los_bin": 2,
                "rare_oracle": "False",
                "rare_predicted": "True",
                "rare_mixed_D_pred": "False",
                "rare_mixed_LOS_pred": "False",
                "joint_mismatch_predicted": "True",
                "discharge_confidence": 0.91,
                "los_confidence": 0.55,
            },
            {
                "split": "valid",
                "row_idx": 3,
                "caseid": 103,
                "target_name": "FREQ_ATND_SELF_HELP_D",
                "y": 0,
                "oracle_d": 2,
                "predicted_d": 2,
                "oracle_los_bin": 2,
                "predicted_los_bin": 2,
                "rare_oracle": "False",
                "rare_predicted": "False",
                "rare_mixed_D_pred": "False",
                "rare_mixed_LOS_pred": "False",
                "joint_mismatch_predicted": "False",
                "discharge_confidence": 0.99,
                "los_confidence": 0.88,
            },
        ],
        [
            "split",
            "row_idx",
            "caseid",
            "target_name",
            "y",
            "oracle_d",
            "predicted_d",
            "oracle_los_bin",
            "predicted_los_bin",
            "rare_oracle",
            "rare_predicted",
            "rare_mixed_D_pred",
            "rare_mixed_LOS_pred",
            "joint_mismatch_predicted",
            "discharge_confidence",
            "los_confidence",
        ],
    )
    return diag_dir


def test_build_head_interpretation_rows_ranks_by_joint_shift() -> None:
    rows = build_head_interpretation_rows(
        [
            {
                "split": "overall",
                "target_name": "SERVICES_D",
                "cramers_v_oracle": 0.33,
                "cramers_v_predicted": 0.49,
                "delta_cramers_v": 0.16,
                "js_divergence_P_D_given_LOS": 0.05,
                "rare_combo_rate_oracle": 0.0,
                "rare_combo_rate_predicted": 0.0,
                "rare_combo_rate_mixed_D_pred": 0.0,
                "rare_combo_rate_mixed_LOS_pred": 0.0,
                "positive_rate_drift_by_los": 0.09,
            },
            {
                "split": "overall",
                "target_name": "SUB1_D",
                "cramers_v_oracle": 0.10,
                "cramers_v_predicted": 0.19,
                "delta_cramers_v": 0.09,
                "js_divergence_P_D_given_LOS": 0.013,
                "rare_combo_rate_oracle": 0.0,
                "rare_combo_rate_predicted": 0.0011,
                "rare_combo_rate_mixed_D_pred": 0.0012,
                "rare_combo_rate_mixed_LOS_pred": 0.0008,
                "positive_rate_drift_by_los": 0.09,
            },
        ],
        top_k=2,
    )

    assert rows[0]["target_name"] == "SERVICES_D"
    assert rows[0]["dominant_issue"] == "strong_conditional_shift"
    assert rows[1]["dominant_issue"] in {"moderate_conditional_shift", "rare_combo_pressure"}


def test_generate_joint_plausibility_report_writes_artifacts(diagnostic_dir: Path) -> None:
    payload = generate_joint_plausibility_report(
        diagnostic_dir,
        top_k=3,
        split="overall",
        heads=["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
        discharge_confidence_min=0.9,
        los_confidence_min=0.5,
        limit=50,
        script_path="scripts/analyze_joint_plausibility_report.py",
    )

    artifacts = payload["artifacts"]
    assert Path(artifacts["head_report_csv"]).exists()
    assert Path(artifacts["head_report_md"]).exists()
    assert Path(artifacts["high_confidence_mismatches_csv"]).exists()
    assert Path(artifacts["focused_heads_summary_csv"]).exists()
    assert Path(artifacts["focused_heads_prob_deltas_csv"]).exists()
    assert Path(artifacts["focused_heads_mismatches_csv"]).exists()
    assert Path(artifacts["extraction_command_sh"]).exists()

    summary = json.loads(Path(artifacts["summary_json"]).read_text(encoding="utf-8"))
    assert summary["num_high_confidence_mismatches"] == 2
    assert "SERVICES_D" in Path(artifacts["head_report_md"]).read_text(encoding="utf-8")
    assert "analyze_joint_plausibility_report.py" in Path(artifacts["extraction_command_sh"]).read_text(encoding="utf-8")


def test_cli_accepts_joint_plausibility_report_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_joint_plausibility_report.py",
            "--diagnostic-dir",
            "/tmp/distribution_diagnosis",
            "--heads",
            "SERVICES_D,SUB1_D",
            "--discharge-confidence-min",
            "0.95",
            "--los-confidence-min",
            "0.70",
            "--limit",
            "25",
        ],
    )

    args = parse_args()

    assert args.diagnostic_dir == "/tmp/distribution_diagnosis"
    assert args.heads == "SERVICES_D,SUB1_D"
    assert args.discharge_confidence_min == pytest.approx(0.95)
    assert args.los_confidence_min == pytest.approx(0.70)
    assert args.limit == 25
