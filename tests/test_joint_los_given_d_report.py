from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.analyze_joint_los_given_d_report import parse_args
from src.analysis.joint_los_given_d_report import (
    build_head_interpretation_rows,
    generate_joint_los_given_d_report,
)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


@pytest.fixture
def diagnostic_dir(tmp_path: Path) -> Path:
    diag_dir = tmp_path / "joint_stats_test"
    diag_dir.mkdir()

    (diag_dir / "joint_stats_summary.json").write_text(
        json.dumps(
            {
                "split": "test",
                "los_bin_mode": "coarse6",
                "num_rows_eval": 100,
                "mean_js_los_given_d": 0.07,
                "max_js_los_given_d": 0.15,
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        diag_dir / "joint_stats_per_head.csv",
        [
            {
                "head_name": "SERVICES_D",
                "js_d_given_los": 0.23,
                "js_los_given_d": 0.15,
                "num_d_classes": 8,
                "eval_unique_pred_classes": 8,
            },
            {
                "head_name": "SUB1_D",
                "js_d_given_los": 0.19,
                "js_los_given_d": 0.10,
                "num_d_classes": 20,
                "eval_unique_pred_classes": 18,
            },
            {
                "head_name": "EMPLOY_D",
                "js_d_given_los": 0.15,
                "js_los_given_d": 0.04,
                "num_d_classes": 5,
                "eval_unique_pred_classes": 3,
            },
        ],
        [
            "head_name",
            "js_d_given_los",
            "js_los_given_d",
            "num_d_classes",
            "eval_unique_pred_classes",
        ],
    )

    _write_csv(
        diag_dir / "per_head_conditional_los_given_d.csv",
        [
            {
                "head_name": "SERVICES_D",
                "d_value": 0,
                "los_bin": 0,
                "train_count": 50,
                "train_prob": 0.50,
                "eval_count": 20,
                "eval_prob": 0.20,
                "train_d_count": 100,
                "eval_d_count": 40,
                "js_los_given_d_for_d_value": 0.12,
                "js_los_given_d_head": 0.15,
                "los_bin_mode": "coarse6",
            },
            {
                "head_name": "SERVICES_D",
                "d_value": 0,
                "los_bin": 1,
                "train_count": 20,
                "train_prob": 0.20,
                "eval_count": 30,
                "eval_prob": 0.30,
                "train_d_count": 100,
                "eval_d_count": 40,
                "js_los_given_d_for_d_value": 0.12,
                "js_los_given_d_head": 0.15,
                "los_bin_mode": "coarse6",
            },
            {
                "head_name": "SUB1_D",
                "d_value": 1,
                "los_bin": 2,
                "train_count": 10,
                "train_prob": 0.10,
                "eval_count": 35,
                "eval_prob": 0.35,
                "train_d_count": 80,
                "eval_d_count": 25,
                "js_los_given_d_for_d_value": 0.09,
                "js_los_given_d_head": 0.10,
                "los_bin_mode": "coarse6",
            },
        ],
        [
            "head_name",
            "d_value",
            "los_bin",
            "train_count",
            "train_prob",
            "eval_count",
            "eval_prob",
            "train_d_count",
            "eval_d_count",
            "js_los_given_d_for_d_value",
            "js_los_given_d_head",
            "los_bin_mode",
        ],
    )
    return diag_dir


def test_build_head_interpretation_rows_ranks_by_los_given_d_shift() -> None:
    rows = build_head_interpretation_rows(
        [
            {
                "head_name": "SERVICES_D",
                "js_d_given_los": 0.23,
                "js_los_given_d": 0.15,
                "num_d_classes": 8,
                "eval_unique_pred_classes": 8,
                "pred_class_coverage_ratio": 1.0,
                "pred_class_coverage_gap": 0.0,
            },
            {
                "head_name": "EMPLOY_D",
                "js_d_given_los": 0.15,
                "js_los_given_d": 0.04,
                "num_d_classes": 5,
                "eval_unique_pred_classes": 3,
                "pred_class_coverage_ratio": 0.6,
                "pred_class_coverage_gap": 0.4,
            },
        ],
        top_k=2,
    )

    assert rows[0]["head_name"] == "SERVICES_D"
    assert rows[0]["dominant_issue"] == "strong_conditional_shift"
    assert rows[1]["dominant_issue"] in {"coverage_gap", "mild_shift"}


def test_generate_joint_los_given_d_report_writes_artifacts(diagnostic_dir: Path) -> None:
    payload = generate_joint_los_given_d_report(
        diagnostic_dir,
        top_k=3,
        heads=["SERVICES_D", "SUB1_D"],
        limit=50,
        script_path="scripts/analyze_joint_los_given_d_report.py",
    )

    artifacts = payload["artifacts"]
    assert Path(artifacts["head_report_csv"]).exists()
    assert Path(artifacts["head_report_md"]).exists()
    assert Path(artifacts["focused_heads_summary_csv"]).exists()
    assert Path(artifacts["focused_heads_prob_deltas_csv"]).exists()
    assert Path(artifacts["extraction_command_sh"]).exists()

    summary = json.loads(Path(artifacts["summary_json"]).read_text(encoding="utf-8"))
    assert summary["num_focused_heads"] == 2
    assert summary["source_summary"]["mean_js_los_given_d"] == pytest.approx(0.07)
    assert "SERVICES_D" in Path(artifacts["head_report_md"]).read_text(encoding="utf-8")
    assert "analyze_joint_los_given_d_report.py" in Path(artifacts["extraction_command_sh"]).read_text(encoding="utf-8")


def test_cli_accepts_joint_los_given_d_report_args(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_joint_los_given_d_report.py",
            "--diagnostic-dir",
            "/tmp/joint_stats_test",
            "--heads",
            "SERVICES_D,SUB1_D",
            "--top-k",
            "12",
            "--limit",
            "25",
        ],
    )

    args = parse_args()

    assert args.diagnostic_dir == "/tmp/joint_stats_test"
    assert args.heads == "SERVICES_D,SUB1_D"
    assert args.top_k == 12
    assert args.limit == 25
