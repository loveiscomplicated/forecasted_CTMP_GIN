from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.build_joint_dv_auc_table import main


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prediction_frame(*, structured: bool) -> pd.DataFrame:
    rng = np.random.default_rng(123 if structured else 456)
    pred_los = np.tile(np.arange(3), 80)
    true_los = pred_los.copy()
    true_a = rng.integers(0, 2, size=pred_los.size)
    pred_a = (pred_los > 0).astype(int) if structured else true_a.copy()
    true_b = pred_los % 2
    pred_b = true_b.copy()
    return pd.DataFrame(
        {
            "row_idx": np.arange(pred_los.size),
            "true_A_D": true_a,
            "pred_A_D": pred_a,
            "true_B_D": true_b,
            "pred_B_D": pred_b,
            "true_los": true_los,
            "pred_los": pred_los,
        }
    )


def _write_downstream_run(path: Path, *, valid_auc: float, test_auc: float) -> None:
    fold = path / "folds" / "fold_0"
    fold.mkdir(parents=True)
    rows = [
        {"epoch": 1, "valid_auc": valid_auc - 0.01, "valid_acc": 0.70, "valid_f1": 0.60},
        {"epoch": 2, "valid_auc": valid_auc, "valid_acc": 0.71, "valid_f1": 0.61},
    ]
    with (fold / "metrics.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    (fold / "best.txt").write_text("best_epoch: 2\nvalid_auc: %.6f\n" % valid_auc, encoding="utf-8")
    (fold / "fallback_ablation_eval.json").write_text(
        json.dumps(
            {
                "baseline_metrics": {
                    "test_auc": test_auc,
                    "test_acc": 0.80,
                    "test_f1": 0.76,
                    "test_precision": 0.79,
                    "test_recall": 0.74,
                    "test_loss": 0.41,
                }
            }
        ),
        encoding="utf-8",
    )


def test_cli_writes_run_and_head_tables_without_implicit_run_state(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    run_a = runs_root / "predictor_a"
    run_b = runs_root / "predictor_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    _prediction_frame(structured=True).to_csv(run_a / "test_predictions.csv", index=False)
    _prediction_frame(structured=False).to_csv(run_b / "test_predictions.csv", index=False)

    downstream_a = runs_root / "downstream_a"
    downstream_b = runs_root / "downstream_b"
    _write_downstream_run(downstream_a, valid_auc=0.82, test_auc=0.84)
    _write_downstream_run(downstream_b, valid_auc=0.83, test_auc=0.86)

    run_map = tmp_path / "run_map.csv"
    _write_csv(
        run_map,
        [
            {
                "id": 1,
                "predictor_run_name": "predictor_a",
                "predictor_run_dir": str(run_a),
                "direction": "D_TO_LOS",
                "heads_mode": "all",
                "los_mode": "coarse",
                "detach": "F",
                "lambda_joint": 0.0,
                "seed": 1,
                "fold": 0,
            },
            {
                "id": 2,
                "predictor_run_name": "predictor_b",
                "predictor_run_dir": str(run_b),
                "direction": "D_TO_LOS",
                "heads_mode": "all",
                "los_mode": "coarse",
                "detach": "F",
                "lambda_joint": 0.3,
                "seed": 1,
                "fold": 0,
            },
        ],
        [
            "id",
            "predictor_run_name",
            "predictor_run_dir",
            "direction",
            "heads_mode",
            "los_mode",
            "detach",
            "lambda_joint",
            "seed",
            "fold",
        ],
    )
    downstream_map = tmp_path / "downstream_map.csv"
    _write_csv(
        downstream_map,
        [
            {"id": 1, "predictor_run_name": "predictor_a", "downstream_run_dir": str(downstream_a), "fold": 0},
            {"id": 2, "predictor_run_name": "predictor_b", "downstream_run_dir": str(downstream_b), "fold": 0},
        ],
        ["id", "predictor_run_name", "downstream_run_dir", "fold"],
    )

    out_dir = tmp_path / "out"
    exit_code = main(
        [
            "--runs-root",
            str(runs_root),
            "--run-map-csv",
            str(run_map),
            "--run-ids",
            "1",
            "2",
            "--downstream-map",
            str(downstream_map),
            "--split",
            "test",
            "--los-bins",
            "coarse6",
            "--out-dir",
            str(out_dir),
        ]
    )
    assert exit_code == 0

    run_level = pd.read_csv(out_dir / "run_level_dv_auc.csv")
    head_level = pd.read_csv(out_dir / "head_level_dv_auc.csv")
    assert set(run_level["id"]) == {1, 2}
    assert set(head_level["id"]) == {1, 2}
    assert (out_dir / "run_level_dv_auc.md").exists()
    assert (out_dir / "manifest.json").exists()

    by_id = run_level.set_index("id")
    assert by_id.loc[1, "predictor_run_name"] == "predictor_a"
    assert by_id.loc[2, "predictor_run_name"] == "predictor_b"
    assert by_id.loc[1, "mean_dV_D"] > by_id.loc[2, "mean_dV_D"]
    assert by_id.loc[1, "downstream_test_auc"] == 0.84
    assert by_id.loc[2, "downstream_test_auc"] == 0.86
    assert by_id.loc[1, "downstream_status"] == "fold0_only"
