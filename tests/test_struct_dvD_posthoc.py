from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.compare_struct_dvD_posthoc import (
    ComparisonRun,
    DiagnosticWarningLog,
    JointPredictionSource,
    OutcomeSource,
    add_metric_deltas,
    build_comparison_table,
    build_posthoc,
    classify_run_decision,
    compute_los_routing,
    compute_outcome_diagnostics,
    discover_fold0_run_dir,
    load_run_metrics,
    main,
    parse_args,
    robust_candidate_run_names,
    summarize_drift,
)


def _prediction_frame(*, structured: bool = False) -> pd.DataFrame:
    true_los = np.array([8, 9, 16, 24, 30, 32, 34, 36, 11, 18, 25, 37])
    pred_los = np.array([5, 5, 3, 4, 5, 5, 5, 5, 2, 5, 5, 5])
    true_head = np.array([0, 0, 1, 1, 2, 2, 2, 2, 0, 1, 1, 2])
    if structured:
        pred_head = np.array([2, 2, 1, 1, 2, 2, 1, 1, 0, 1, 1, 1])
    else:
        pred_head = true_head.copy()
    return pd.DataFrame(
        {
            "row_idx": np.arange(true_los.size),
            "true_los": true_los,
            "pred_los": pred_los,
            "true_SUB1_D": true_head,
            "pred_SUB1_D": pred_head,
            "true_FREQ_ATND_SELF_HELP_D": true_head,
            "pred_FREQ_ATND_SELF_HELP_D": pred_head,
            "true_FREQ1_D": true_head,
            "pred_FREQ1_D": true_head,
            "true_FREQ2_D": true_head,
            "pred_FREQ2_D": pred_head,
            "true_EMPLOY_D": true_head,
            "pred_EMPLOY_D": true_head,
            "true_DETNLF_D": true_head,
            "pred_DETNLF_D": true_head,
        }
    )


def _outcome_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_idx": np.arange(12),
            "y_true": [0, 1, 0, 1, 1, 0, 1, 1, 0, 1, 0, 1],
            "y_pred": [0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0],
            "y_score": [0.05, 0.9, 0.1, 0.8, 0.7, 0.2, 0.45, 0.3, 0.1, 0.75, 0.2, 0.4],
        }
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _write_fold(
    fold_dir: Path,
    *,
    run_name: str,
    structured: bool,
    valid_auc: float = 0.88,
    test_auc: float | None = 0.887,
    struct_enabled: bool = False,
) -> None:
    jp = fold_dir / "joint_predictor"
    jp.mkdir(parents=True, exist_ok=True)
    pred = _prediction_frame(structured=structured)
    pred.to_csv(jp / "test_predictions.csv", index=False)
    pred.to_csv(jp / "val_predictions.csv", index=False)
    _outcome_frame().to_csv(fold_dir / "diagnostic_predictions.csv", index=False)
    result = {
        "best_epoch": 2,
        "best_valid_metric": valid_auc,
        "best_valid_metrics": {
            "valid_auc": valid_auc,
            "valid_acc": 0.8,
            "valid_precision": 0.79,
            "valid_recall": 0.74,
            "valid_f1": 0.76,
            "valid_loss": 0.41,
        },
        "fold": 0,
        "status": "completed",
    }
    if test_auc is not None:
        result.update(
            {
                "test_auc": test_auc,
                "test_acc": 0.8,
                "test_precision": 0.79,
                "test_recall": 0.74,
                "test_f1": 0.76,
                "test_loss": 0.41,
            }
        )
    (fold_dir / "fold_result.json").write_text(json.dumps(result), encoding="utf-8")
    cfg = {
        "joint_predictor": {"joint_heads": "new_dvD_top6" if "top6" in run_name else "new_robust_top3"},
        "joint_struct_loss": {
            "enabled": struct_enabled,
            "lambda_struct": 0.03 if "robust" in run_name else 0.01,
            "loss_type": "soft_js_d",
            "risk_head_set": "new_dvD_top6" if "top6" in run_name else "new_robust_top3",
            "stopgrad_los": True,
        },
    }
    (jp / "config.final.yaml").write_text(json.dumps(cfg), encoding="utf-8")
    if struct_enabled:
        _write_jsonl(
            jp / "metrics.jsonl",
            [
                {
                    "epoch": 1,
                    "train_loss": 8.0,
                    "valid_loss": 7.0,
                    "train_struct_loss": 0.02,
                    "valid_struct_loss": 0.03,
                    "train_struct_SUB1_D": 0.02,
                    "valid_struct_SUB1_D": 0.03,
                    "lambda_struct": cfg["joint_struct_loss"]["lambda_struct"],
                },
                {
                    "epoch": 2,
                    "train_loss": 7.5,
                    "valid_loss": 6.8,
                    "train_struct_loss": 0.01,
                    "valid_struct_loss": 0.02,
                    "train_struct_SUB1_D": 0.01,
                    "valid_struct_SUB1_D": 0.02,
                    "lambda_struct": cfg["joint_struct_loss"]["lambda_struct"],
                },
            ],
        )


def test_robust_run_auto_discovery_by_name_substring(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    fold = runs_root / "20260615_struct_robust_top3_lambda003" / "folds" / "fold_0"
    fold.mkdir(parents=True)
    resolved = discover_fold0_run_dir(runs_root, "struct_robust_top3_lambda003", warnings=DiagnosticWarningLog())
    assert resolved == fold


def test_missing_robust_run_failure_lists_candidates(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    (runs_root / "candidate_robust_top3_lambda001" / "folds" / "fold_0").mkdir(parents=True)
    warnings = DiagnosticWarningLog()
    with pytest.raises(SystemExit):
        discover_fold0_run_dir(runs_root, "struct_robust_top3_lambda003", warnings=warnings)
    assert "candidate_robust_top3_lambda001" in robust_candidate_run_names(runs_root)
    assert warnings.warnings


def test_run_metric_merge_with_missing_test_metrics(tmp_path: Path) -> None:
    fold = tmp_path / "runs" / "baseline" / "folds" / "fold_0"
    _write_fold(fold, run_name="baseline", structured=False, valid_auc=0.88, test_auc=None)
    run = ComparisonRun("baseline_id26", "baseline_coarse6", fold)
    metrics = add_metric_deltas(pd.DataFrame([load_run_metrics(run)]))
    assert metrics.loc[0, "valid_auc"] == 0.88
    assert pd.isna(metrics.loc[0, "test_auc"])


def test_true_breakdown9_bin_mapping_and_predicted_representative_limit(tmp_path: Path) -> None:
    fold = tmp_path / "runs" / "r" / "folds" / "fold_0"
    _write_fold(fold, run_name="r", structured=False)
    run = ComparisonRun("r", "target", fold)
    joint = JointPredictionSource(_prediction_frame(), "fixture", None)
    outcome = OutcomeSource(_outcome_frame(), "fixture", None, "ok")
    true_df, pred_df = compute_outcome_diagnostics(
        run,
        "test",
        joint,
        outcome,
        ["breakdown9_true"],
        DiagnosticWarningLog(),
    )
    labels = set(true_df["los_bin_label"].tolist())
    assert {"29-31", "32-33", "34-35", "36-37"}.issubset(labels)
    assert set(pred_df["status"]) == {"representative_limited"}


def test_middle_to_long_flow_calculation(tmp_path: Path) -> None:
    fold = tmp_path / "runs" / "r" / "folds" / "fold_0"
    _write_fold(fold, run_name="r", structured=False)
    run = ComparisonRun("r", "target", fold)
    joint = JointPredictionSource(_prediction_frame(), "fixture", None)
    _confusion, flow = compute_los_routing(run, "test", joint, ["coarse6"], DiagnosticWarningLog())
    row = flow[flow["true_bin_label"].eq("8-14")].iloc[0]
    assert row["support"] == 3
    assert row["total_to_long_count"] == 2


def test_decision_classification_struct_down_dv_unchanged_and_dv_down_auc_unchanged() -> None:
    row = pd.Series(
        {
            "run_name": "struct",
            "struct_loss_delta": -0.01,
            "mean_risk_dV_D": 0.02,
            "test_auc": 0.8878,
            "late_long_recall_mean": np.nan,
        }
    )
    assert "AUC remains plateau-level" in classify_run_decision(row)

    comparison = pd.DataFrame(
        [
            {"run_name": "baseline_id26", "valid_auc": 0.88, "test_auc": 0.8877, "struct_loss_delta": np.nan},
            {"run_name": "struct", "valid_auc": 0.881, "test_auc": 0.8878, "struct_loss_delta": -0.01},
        ]
    )
    drift_summary = pd.DataFrame(
        [
            {"run_name": "baseline_id26", "split": "test", "los_scheme": "coarse6", "mean_dV_D_risk_heads": 0.02},
            {
                "run_name": "struct",
                "split": "test",
                "los_scheme": "coarse6",
                "mean_dV_D_risk_heads": 0.01,
                "delta_risk_mean_dV_D_vs_baseline": -0.01,
            },
        ]
    )
    table = build_comparison_table(comparison, drift_summary, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    assert table.loc[table["run_name"].eq("struct"), "interpretation"].iloc[0]


def test_cli_writes_reports_without_training_or_checkpoints(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    baseline = runs_root / "baseline_id26" / "folds" / "fold_0"
    top6 = runs_root / "struct_dvD_top6_lambda001" / "folds" / "fold_0"
    robust = runs_root / "struct_robust_top3_lambda003" / "folds" / "fold_0"
    _write_fold(baseline, run_name="baseline", structured=False, valid_auc=0.884, test_auc=0.886)
    _write_fold(top6, run_name="top6", structured=True, valid_auc=0.885, test_auc=0.8879, struct_enabled=True)
    _write_fold(robust, run_name="robust", structured=True, valid_auc=0.884, test_auc=0.8873, struct_enabled=True)
    out_dir = tmp_path / "reports" / "posthoc"

    exit_code = main(
        [
            "--runs-root",
            str(runs_root),
            "--baseline-run-dir",
            str(baseline),
            "--top6-run-dir",
            str(top6),
            "--robust-run-name-contains",
            "struct_robust_top3_lambda003",
            "--splits",
            "valid",
            "test",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0
    assert (out_dir / "decision_report.md").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "run_metrics_comparison.csv").exists()
    assert not (baseline / "checkpoints").exists()

    args = parse_args(
        [
            "--runs-root",
            str(runs_root),
            "--baseline-run-dir",
            str(baseline),
            "--top6-run-dir",
            str(top6),
            "--robust-run-dir",
            str(robust),
            "--out-dir",
            str(out_dir / "second"),
        ]
    )
    tables, _manifest, _warnings = build_posthoc(args)
    assert not tables["drift_summary"].empty
