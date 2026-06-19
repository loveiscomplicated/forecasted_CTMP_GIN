from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def _write_fake_fold(
    fold_dir: Path,
    *,
    test_auc: float,
    pred_los: list[int],
    y_true: list[int],
    y_pred: list[int],
    y_score: list[float],
) -> None:
    (fold_dir / "joint_predictor").mkdir(parents=True, exist_ok=True)
    (fold_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (fold_dir / "fold_result.json").write_text(
        json.dumps(
            {
                "best_epoch": 3,
                "best_valid_metric": test_auc - 0.01,
                "test_loss": 0.4,
                "test_acc": 0.8,
                "test_precision": 0.79,
                "test_recall": 0.73,
                "test_f1": 0.76,
                "test_auc": test_auc,
                "fold": 0,
                "run_dir": str(fold_dir),
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "row_idx": [0, 1, 2, 3, 4, 5],
            "y_true": y_true,
            "y_pred": y_pred,
            "y_score": y_score,
        }
    ).to_csv(fold_dir / "diagnostic_predictions.csv", index=False)
    pd.DataFrame(
        {
            "row_idx": [0, 1, 2, 3, 4, 5],
            "true_los": [8, 9, 16, 24, 30, 36],
            "pred_los": pred_los,
            "true_SERVICES_D": [1, 1, 1, 1, 1, 1],
            "pred_SERVICES_D": [1, 1, 1, 1, 1, 1],
            "true_SUB1_D": [0, 0, 1, 1, 2, 2],
            "pred_SUB1_D": [0, 0, 1, 1, 2, 2],
            "true_FREQ_ATND_SELF_HELP_D": [0, 0, 1, 1, 1, 1],
            "pred_FREQ_ATND_SELF_HELP_D": [0, 0, 1, 1, 1, 1],
        }
    ).to_csv(fold_dir / "joint_predictor" / "test_predictions.csv", index=False)


def test_cli_writes_expected_outputs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    target_dir = runs_root / "target" / "folds" / "fold_0"
    baseline_dir = runs_root / "baseline" / "folds" / "fold_0"
    _write_fake_fold(
        target_dir,
        test_auc=0.8877,
        pred_los=[5, 5, 6, 4, 5, 8],
        y_true=[0, 1, 0, 1, 1, 0],
        y_pred=[0, 1, 0, 1, 0, 0],
        y_score=[0.1, 0.8, 0.2, 0.7, 0.4, 0.3],
    )
    _write_fake_fold(
        baseline_dir,
        test_auc=0.8860,
        pred_los=[5, 5, 5, 4, 5, 5],
        y_true=[0, 1, 0, 1, 1, 0],
        y_pred=[0, 1, 0, 1, 1, 0],
        y_score=[0.1, 0.8, 0.2, 0.7, 0.6, 0.3],
    )
    out_dir = tmp_path / "reports"
    cmd = [
        sys.executable,
        "scripts/diagnose_los_breakdown_downstream.py",
        "--runs-root",
        str(runs_root),
        "--target-run-dir",
        str(target_dir),
        "--target-name",
        "id26_9bin_breakdown",
        "--baseline-run-dir",
        str(baseline_dir),
        "--baseline-run-name",
        "id26_coarse6",
        "--out-dir",
        str(out_dir),
        "--strict",
        "false",
    ]
    subprocess.run(cmd, cwd=Path(__file__).resolve().parents[1], check=True)

    expected = [
        "run_metrics_comparison.csv",
        "los_bin_outcome_metrics.csv",
        "los_confusion_matrix_counts.csv",
        "los_confusion_matrix_row_pct.csv",
        "los_population_contamination.csv",
        "middle_to_long_flow_summary.csv",
        "long_stay_recall_comparison.csv",
        "head_level_drift_decomposition.csv",
        "run_level_drift_auc_summary.csv",
        "diagnostic_report.md",
        "manifest.json",
    ]
    for name in expected:
        assert (out_dir / name).exists(), name

    summary_df = pd.read_csv(out_dir / "run_level_drift_auc_summary.csv")
    assert set(summary_df["run_name"]) == {"id26_9bin_breakdown", "id26_coarse6"}
    target_row = summary_df.loc[summary_df["run_name"] == "id26_9bin_breakdown"].iloc[0]
    baseline_row = summary_df.loc[summary_df["run_name"] == "id26_coarse6"].iloc[0]
    assert target_row["downstream_test_auc"] != baseline_row["downstream_test_auc"]
    assert target_row["long_stay_macro_recall"] != baseline_row["long_stay_macro_recall"]
