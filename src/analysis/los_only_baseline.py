#!/usr/bin/env python3
"""
LOS-only baseline for REASONb prediction.

Purpose
-------
This script evaluates how predictive LOS alone is for the binary treatment
completion target REASONb. It is intended as a sanity check for possible
LOS-related shortcut / tautology concerns.

Input
-----
A CSV file containing at least:
    - LOS column
    - either REASONb column, or REASON column from which REASONb can be derived

Output
------
Two CSV files:
    - los_only_per_fold.csv
    - los_only_summary.csv

Example
-------
python src/analysis/los_only_baseline.py \
    --csv_path src/data/raw/TEDS_Discharge.csv \
    --los_col LOS \
    --target_col REASONb \
    --out_dir runs/los_only_baseline

If REASONb does not exist but REASON exists:
python src/analysis/los_only_baseline.py \
    --csv_path src/data/raw/TEDS_Discharge.csv \
    --los_col LOS \
    --reason_col REASON \
    --positive_reason_code 1 \
    --out_dir runs/los_only_baseline


"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate LOS-only baseline for REASONb prediction."
    )

    parser.add_argument(
        "--csv_path",
        type=str,
        required=True,
        help="Path to input CSV file.",
    )
    parser.add_argument(
        "--los_col",
        type=str,
        default="LOS",
        help="Name of LOS column.",
    )
    parser.add_argument(
        "--target_col",
        type=str,
        default="REASONb",
        help="Name of binary target column. Used if present.",
    )
    parser.add_argument(
        "--reason_col",
        type=str,
        default="REASON",
        help="Original REASON column used to derive REASONb if target_col is absent.",
    )
    parser.add_argument(
        "--positive_reason_code",
        type=int,
        default=1,
        help=(
            "REASON code treated as treatment completion when deriving REASONb. "
            "Use the code consistent with your preprocessing pipeline."
        ),
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of CV folds.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Random seeds for repeated CV.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="runs/los_only_baseline",
        help="Output directory.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=1000,
        help="Max iterations for logistic regression.",
    )
    parser.add_argument(
        "--sample_n",
        type=int,
        default=None,
        help="Optional row subsampling for quick debugging.",
    )

    return parser.parse_args()


def load_data(
    csv_path: Path,
    los_col: str,
    target_col: str,
    reason_col: str,
    positive_reason_code: int,
    sample_n: int | None = None,
) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    if sample_n is not None:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

    if los_col not in df.columns:
        raise ValueError(
            f"LOS column '{los_col}' not found. Available columns include: "
            f"{list(df.columns[:20])}"
        )

    if target_col in df.columns:
        df["_target"] = df[target_col].astype(int)
    elif reason_col in df.columns:
        df["_target"] = (df[reason_col].astype(int) == positive_reason_code).astype(int)
    else:
        raise ValueError(
            f"Neither target_col='{target_col}' nor reason_col='{reason_col}' "
            "exists in the input CSV."
        )

    # Keep only the required columns.
    out = df[[los_col, "_target"]].copy()

    # Treat LOS as categorical exactly as provided by TEDS-D.
    # Do not interpret it as a continuous numeric duration.
    out[los_col] = out[los_col].astype("category")

    # Drop missing target rows if any.
    out = out.dropna(subset=[los_col, "_target"]).reset_index(drop=True)

    unique_targets = sorted(out["_target"].unique().tolist())
    if unique_targets != [0, 1]:
        raise ValueError(
            f"Target must be binary with values [0, 1], got {unique_targets}."
        )

    return out


def build_los_logistic_pipeline(los_col: str, max_iter: int) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "los_onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                [los_col],
            )
        ],
        remainder="drop",
    )

    clf = LogisticRegression(
        solver="lbfgs",
        max_iter=max_iter,
        class_weight=None,
        n_jobs=None,
    )

    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob),
        "loss": log_loss(y_true, y_prob, labels=[0, 1]),
    }


def run_repeated_cv(
    df: pd.DataFrame,
    los_col: str,
    seeds: list[int],
    n_splits: int,
    max_iter: int,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    X = df[[los_col]]
    y = df["_target"].to_numpy()

    for seed in seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=seed,
        )

        for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
            X_train = X.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            model = build_los_logistic_pipeline(los_col=los_col, max_iter=max_iter)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_prob)

            # Majority-class dummy baseline for reference.
            dummy = DummyClassifier(strategy="prior")
            dummy.fit(X_train, y_train)
            dummy_prob = dummy.predict_proba(X_test)[:, 1]
            dummy_metrics = compute_metrics(y_test, dummy_prob)

            record: dict[str, Any] = {
                "seed": seed,
                "fold": fold,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "train_pos_rate": float(y_train.mean()),
                "test_pos_rate": float(y_test.mean()),
            }

            for k, v in metrics.items():
                record[f"los_only_{k}"] = float(v)

            for k, v in dummy_metrics.items():
                record[f"dummy_{k}"] = float(v)

            records.append(record)

            print(
                f"[seed={seed} fold={fold}] "
                f"LOS-only AUC={metrics['auc']:.6f}, "
                f"F1={metrics['f1']:.6f}, "
                f"Dummy AUC={dummy_metrics['auc']:.6f}"
            )

    return pd.DataFrame(records)


def summarize(per_fold: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        col
        for col in per_fold.columns
        if col.startswith("los_only_") or col.startswith("dummy_")
    ]

    rows: list[dict[str, Any]] = []
    for col in metric_cols:
        rows.append(
            {
                "metric": col,
                "mean": per_fold[col].mean(),
                "std": per_fold[col].std(ddof=1),
                "min": per_fold[col].min(),
                "max": per_fold[col].max(),
            }
        )

    return pd.DataFrame(rows)


def save_metadata(
    out_dir: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
) -> None:
    metadata = {
        "csv_path": args.csv_path,
        "los_col": args.los_col,
        "target_col": args.target_col,
        "reason_col": args.reason_col,
        "positive_reason_code": args.positive_reason_code,
        "n_rows_used": int(len(df)),
        "target_positive_rate": float(df["_target"].mean()),
        "los_n_categories": int(df[args.los_col].nunique()),
        "los_categories": [str(x) for x in sorted(df[args.los_col].unique())],
        "n_splits": args.n_splits,
        "seeds": args.seeds,
    }

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_data(
        csv_path=Path(args.csv_path),
        los_col=args.los_col,
        target_col=args.target_col,
        reason_col=args.reason_col,
        positive_reason_code=args.positive_reason_code,
        sample_n=args.sample_n,
    )

    print(f"Loaded rows: {len(df):,}")
    print(f"Positive rate: {df['_target'].mean():.4f}")
    print(f"LOS categories: {df[args.los_col].nunique()}")

    per_fold = run_repeated_cv(
        df=df,
        los_col=args.los_col,
        seeds=args.seeds,
        n_splits=args.n_splits,
        max_iter=args.max_iter,
    )
    summary = summarize(per_fold)

    per_fold_path = out_dir / "los_only_per_fold.csv"
    summary_path = out_dir / "los_only_summary.csv"

    per_fold.to_csv(per_fold_path, index=False)
    summary.to_csv(summary_path, index=False)
    save_metadata(out_dir, args, df)

    print("\nSaved:")
    print(f"  {per_fold_path}")
    print(f"  {summary_path}")
    print(f"  {out_dir / 'metadata.json'}")

    auc_row = summary.loc[summary["metric"] == "los_only_auc"].iloc[0]
    print("\nLOS-only AUC: " f"{auc_row['mean']:.6f} ± {auc_row['std']:.6f}")


if __name__ == "__main__":
    main()
