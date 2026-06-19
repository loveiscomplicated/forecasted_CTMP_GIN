"""
extract_gated_fusion_kfold.py

Export sample-level GatedFusion weights from k-fold CV runs and derive
paper-ready summaries for fold variability, label-wise behavior, and LOS trends.

Primary output:
  - sample-level CSV with seed/fold/sample metadata and predictions

Derived outputs:
  - seed x fold summary
  - overall stream summary
  - label summary
  - per-LOS summary
  - tidy plot CSVs
  - legacy LOS flat CSV for los_group_detection.py compatibility
  - optional PNG figures

단일 run:

  python src/analysis/extract_gated_fusion_kfold.py \
    --run_name "(final)20260413-071956__ctmp_gin__bs=1024__lr=6.10e-
  04__seed=1__cv=5__test=0.15" \
    --all_matching_seeds \
    --device mps

  여러 seed run을 한 번에:

  python src/analysis/extract_gated_fusion_kfold.py \
    --run_names \
    "(final)20260413-071956__ctmp_gin__bs=1024__lr=6.10e-04__seed=1__cv=5__test=0.15"
  \
    "(final)20260413-021503__ctmp_gin__bs=1024__lr=6.10e-04__seed=2__cv=5__test=0.15"
  \
    "(final)20260413-021541__ctmp_gin__bs=1024__lr=6.10e-04__seed=3__cv=5__test=0.15"
  \
    --device mps

  fold 0만 테스트:

  python src/analysis/extract_gated_fusion_kfold.py \
    --run_name "(final)20260413-071956__ctmp_gin__bs=1024__lr=6.10e-
  04__seed=1__cv=5__test=0.15" \
    --fold 0 \
    --device cpu \
    --output_dir /private/tmp/gated_fusion_check \
    --prefix smoke \
    --no-save_figures

  주요 옵션:

  - --fold all|0|1|2|3|4
  - --output_dir src/analysis/gated_fusion_exports
  - --prefix gated_fusion
  - --all_matching_seeds
  - --save_figures / --no-save_figures
  - --figure_format png
  - --batch_size 1024
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_processing.data_utils import make_binary
from src.data_processing.splits import holdout_test_split_stratified, kfold_stratified
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.factory import build_edge, build_model
from src.utils.device_set import device_set
from src.utils.seed_set import set_seed

STREAM_COLS = ["w_admission", "w_discharge", "w_merged"]
STREAM_LABELS = {
    "w_admission": "Admission",
    "w_discharge": "Discharge",
    "w_merged": "Merged / Cross-temporal",
}
DOMINANT_LABELS = ["admission", "discharge", "merged"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Extract sample-level GatedFusion weights from k-fold CV models"
    )
    p.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Single run directory name under runs/protected/k_fold_CV/",
    )
    p.add_argument(
        "--run_names",
        nargs="+",
        default=None,
        help="One or more run directory names under runs/protected/k_fold_CV/",
    )
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Folds to use: integer 0-4 or 'all'",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=str(_THIS_DIR / "gated_fusion_exports"),
        help="Directory to save all exported CSVs and figures",
    )
    p.add_argument(
        "--prefix",
        type=str,
        default="gated_fusion",
        help="Filename prefix for generated artifacts",
    )
    p.add_argument(
        "--figure_format",
        type=str,
        default="png",
        help="Figure extension when --save_figures is enabled",
    )
    p.add_argument(
        "--save_figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to save fold/LOS figure files",
    )
    p.add_argument(
        "--all_matching_seeds",
        action="store_true",
        help=(
            "When --run_name is provided, automatically find sibling run directories "
            "with the same experiment signature but different __seed=K__ values."
        ),
    )
    args = p.parse_args()
    if not args.run_name and not args.run_names:
        p.error("Provide either --run_name or --run_names.")
    return args


def load_yaml(path):
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_fold_arg(fold_str: str, n_folds: int) -> list[int]:
    if fold_str.strip().lower() == "all":
        return list(range(n_folds))
    fold_id = int(fold_str.strip())
    if not (0 <= fold_id < n_folds):
        raise ValueError(f"fold {fold_id} out of range [0, {n_folds})")
    return [fold_id]


def _resolve_run_names(args) -> list[str]:
    if args.run_names:
        return list(dict.fromkeys(args.run_names))
    return [args.run_name]


def _normalize_run_name_for_seed_match(run_name: str) -> str:
    parts = run_name.split("__")
    if len(parts) < 2:
        raise ValueError(
            f"Run name does not match expected format with '__' separators: {run_name}"
        )

    normalized_parts = parts[1:]
    for idx, part in enumerate(normalized_parts):
        if re.fullmatch(r"seed=\d+", part):
            normalized_parts[idx] = "seed=*"
            break
    else:
        raise ValueError(
            f"Run name does not contain a '__seed=<int>__' segment: {run_name}"
        )

    return "__".join(normalized_parts)


def _extract_seed_from_run_name(run_name: str) -> int:
    match = re.search(r"__seed=(\d+)(?:__|$)", run_name)
    if not match:
        raise ValueError(f"Unable to parse seed from run name: {run_name}")
    return int(match.group(1))


def _resolve_all_matching_seed_runs(base_run_name: str, runs_base: Path) -> list[str]:
    target_signature = _normalize_run_name_for_seed_match(base_run_name)
    matched_run_names: list[str] = []
    for path in runs_base.iterdir():
        if not path.is_dir():
            continue
        try:
            candidate_signature = _normalize_run_name_for_seed_match(path.name)
        except ValueError:
            continue
        if candidate_signature == target_signature:
            matched_run_names.append(path.name)
    if not matched_run_names:
        raise FileNotFoundError(
            f"No matching run directories found for seed group of: {base_run_name}"
        )

    return sorted(
        dict.fromkeys(matched_run_names),
        key=lambda name: (_extract_seed_from_run_name(name), name),
    )


def _load_fold_model(fold_dir: str, device: torch.device, best_or_last: str = "best"):
    if best_or_last == "best":
        ckpt_path = os.path.join(fold_dir, "checkpoints", "best.pt")
    else:
        ckpt_path = os.path.join(fold_dir, "checkpoints", "last.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    cfg["model"]["params"]["device"] = str(device)

    model = build_model(
        model_name=cfg["model"]["name"], **cfg["model"].get("params", {})
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, cfg


def _get_model_name(cfg: dict) -> str:
    return cfg["model"]["name"]


def _get_remove_los(model_name: str) -> bool:
    return model_name not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]


def _build_dataset(data_root: str, cfg: dict) -> TEDSTensorDataset:
    model_name = _get_model_name(cfg)
    return TEDSTensorDataset(
        root=data_root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
        remove_los=_get_remove_los(model_name),
        do_preprocess=cfg["train"].get("do_preprocess", True),
    )


def _safe_num_workers(cfg: dict) -> int:
    requested = int(cfg["train"].get("num_workers", 0))
    if requested > 0:
        print(
            f"  [Info] Overriding num_workers={requested} -> 0 for stable export execution."
        )
    return 0


def _get_num_nodes(dataset: TEDSTensorDataset, model_name: str) -> int:
    if model_name == "gin":
        return len(dataset.col_info[0])
    return len(dataset.col_info[2])


def _get_train_df_for_fold(
    dataset: TEDSTensorDataset,
    labels: np.ndarray,
    fold_cfg: dict,
    target_fold_id: int,
) -> pd.DataFrame:
    seed = fold_cfg["train"]["seed"]
    test_ratio = fold_cfg["train"]["test_ratio"]
    n_folds = fold_cfg["train"]["n_folds"]

    trainval_idx, _ = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )
    for fold, train_idx, _ in kfold_stratified(
        trainval_idx=trainval_idx, labels=labels, n_folds=n_folds, seed=seed
    ):
        if fold == target_fold_id:
            return dataset.processed_df.iloc[train_idx]

    raise ValueError(f"fold {target_fold_id} not found in {n_folds}-fold split")


def _prepare_test_data(
    dataset: TEDSTensorDataset,
    cfg: dict,
    batch_size: int,
) -> dict[str, object]:
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    seed = cfg["train"]["seed"]
    test_ratio = cfg["train"]["test_ratio"]
    _, test_idx = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )

    loader = DataLoader(
        Subset(dataset, test_idx.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=_safe_num_workers(cfg),
        drop_last=False,
    )

    xs, ys, lss = [], [], []
    for x, y, los in loader:
        xs.append(x)
        ys.append(y)
        lss.append(los)

    all_x = torch.cat(xs, dim=0)
    all_y = torch.cat(ys, dim=0)
    all_los = torch.cat(lss, dim=0)
    metadata = _load_test_metadata(cfg, test_idx)

    return {
        "labels": labels,
        "test_idx": test_idx,
        "all_x": all_x,
        "all_y": all_y,
        "all_los": all_los,
        "metadata": metadata,
    }


def _load_test_metadata(cfg: dict, test_idx: np.ndarray) -> pd.DataFrame:
    raw_path = _PROJECT_ROOT / "src" / "data" / "raw" / "TEDS_Discharge.csv"
    raw_df = pd.read_csv(raw_path)
    if cfg["train"].get("binary", True):
        raw_df = make_binary(raw_df)
    else:
        raw_df["REASONb"] = raw_df["REASON"]

    meta = raw_df.iloc[test_idx].copy().reset_index()
    meta = meta.rename(
        columns={
            "index": "sample_idx",
            "CASEID": "sample_id",
            "LOS": "los",
            "REASONb": "label",
        }
    )
    meta["sample_id"] = meta["sample_id"].astype(str)
    meta["sample_idx"] = meta["sample_idx"].astype(int)
    meta["los"] = meta["los"].astype(int)
    meta["label"] = meta["label"].astype(int)
    return meta[["sample_idx", "sample_id", "los", "label"]]


def _build_edge_index(
    dataset: TEDSTensorDataset,
    labels: np.ndarray,
    fold_cfg: dict,
    fold_id: int,
    fold_dir: str,
    device: torch.device,
    data_root: str,
    num_nodes: int,
) -> torch.Tensor:
    edge_index_path = os.path.join(fold_dir, "edge_index.pt")
    if os.path.exists(edge_index_path):
        print(f"  Loading saved edge_index from {edge_index_path}")
        return torch.load(edge_index_path, map_location=device)

    print("  edge_index.pt not found, recomputing...")
    train_df = _get_train_df_for_fold(dataset, labels, fold_cfg, fold_id)
    edge_cfg = fold_cfg.get("edge", {})
    set_seed(fold_cfg["train"]["seed"])
    built = build_edge(
        model_name=_get_model_name(fold_cfg),
        root=data_root,
        seed=fold_cfg["train"]["seed"],
        train_df=train_df,
        num_nodes=num_nodes,
        batch_size=fold_cfg["train"]["batch_size"],
        **edge_cfg,
    )
    return (built[0] if isinstance(built, tuple) else built).to(device)


def extract_fold_predictions(
    model: torch.nn.Module,
    all_x: torch.Tensor,
    all_y: torch.Tensor,
    all_los: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> pd.DataFrame:
    if not hasattr(model, "gated_fusion") or model.gated_fusion is None:
        raise AttributeError("model.gated_fusion not found or None.")

    all_rows = []
    n_samples = all_x.size(0)

    model.eval()
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            x_b = all_x[start:end].to(device)
            los_b = all_los[start:end].to(device)
            y_b = all_y[start:end].cpu().numpy().astype(int)

            logits, _, w, _, _, _, _ = model(
                x_b, los_b, edge_index, return_internals=True
            )
            logits = logits.squeeze(-1).detach().cpu()
            probs = torch.sigmoid(logits).numpy()
            preds = (probs >= 0.5).astype(int)
            w_np = w.detach().cpu().numpy()

            batch_df = pd.DataFrame(
                {
                    "w_admission": w_np[:, 0],
                    "w_discharge": w_np[:, 1],
                    "w_merged": w_np[:, 2],
                    "y_true": y_b,
                    "y_pred": preds,
                    "y_prob": probs,
                }
            )
            all_rows.append(batch_df)

    df = pd.concat(all_rows, ignore_index=True)
    dominant = df[STREAM_COLS].to_numpy().argmax(axis=1)
    df["dominant_stream"] = [DOMINANT_LABELS[i] for i in dominant]
    return df


def _sanity_check(
    df_fold: pd.DataFrame,
    fold_id: int,
    run_dir: Path,
) -> None:
    targets = df_fold["y_true"].to_numpy().astype(int)
    scores = df_fold["y_prob"].to_numpy()
    preds = df_fold["y_pred"].to_numpy().astype(int)

    auc = roc_auc_score(targets, scores)
    acc = accuracy_score(targets, preds)

    summary_path = run_dir / "cv_summary.json"
    stored_auc, stored_acc = None, None
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for fr in summary.get("fold_results", []):
            if fr.get("fold") == fold_id:
                stored_auc = fr.get("test_auc")
                stored_acc = fr.get("test_acc")
                break

    print(
        f"  [Sanity] scores: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}"
    )
    print(
        f"  [Sanity] targets: pos_rate={targets.mean():.4f} ({targets.sum()}/{len(targets)})"
    )
    print(f"  [Sanity] preds:   pos_rate={preds.mean():.4f}")
    print(f"  [Sanity] recomputed  — AUC={auc:.6f}, ACC={acc:.6f}")
    if stored_auc is not None and stored_acc is not None:
        auc_delta = abs(auc - stored_auc)
        acc_delta = abs(acc - stored_acc)
        flag = " *** MISMATCH ***" if auc_delta > 0.005 else ""
        print(f"  [Sanity] stored      — AUC={stored_auc:.6f}, ACC={stored_acc:.6f}")
        print(
            f"  [Sanity] delta       — ΔAUC={auc_delta:.6f}, ΔACC={acc_delta:.6f}{flag}"
        )


def _dominant_from_means(df: pd.DataFrame) -> pd.Series:
    arr = df[STREAM_COLS].to_numpy()
    return pd.Series([DOMINANT_LABELS[i] for i in arr.argmax(axis=1)], index=df.index)


def _group_summary(
    df: pd.DataFrame,
    group_cols: list[str],
    summary_level: str,
) -> pd.DataFrame:
    grouped = (
        df.groupby(group_cols, dropna=False)[STREAM_COLS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in grouped.columns
    ]

    rename_map = {
        "w_admission_mean": "w_admission_mean",
        "w_admission_std": "w_admission_std",
        "w_admission_count": "n",
        "w_discharge_mean": "w_discharge_mean",
        "w_discharge_std": "w_discharge_std",
        "w_merged_mean": "w_merged_mean",
        "w_merged_std": "w_merged_std",
    }
    grouped = grouped.rename(columns=rename_map)
    for col in ["w_discharge_count", "w_merged_count"]:
        if col in grouped.columns:
            grouped = grouped.drop(columns=[col])

    grouped["summary_level"] = summary_level
    grouped["dominant_stream"] = _dominant_from_means(
        grouped.rename(
            columns={
                "w_admission_mean": "w_admission",
                "w_discharge_mean": "w_discharge",
                "w_merged_mean": "w_merged",
            }
        )
    )
    return grouped


def build_fold_summary(df_samples: pd.DataFrame) -> pd.DataFrame:
    summary = _group_summary(df_samples, ["seed", "fold"], "per_fold")
    return summary.sort_values(["seed", "fold"]).reset_index(drop=True)


def build_stream_overall_summary(df_samples: pd.DataFrame) -> pd.DataFrame:
    fold_means = df_samples.groupby(["seed", "fold"])[STREAM_COLS].mean().reset_index()
    rows = []
    for col in STREAM_COLS:
        rows.append(
            {
                "stream": STREAM_LABELS[col],
                "mean_weight": df_samples[col].mean(),
                "std_across_samples": df_samples[col].std(ddof=1),
                "std_across_folds": fold_means[col].std(ddof=1),
            }
        )
    return pd.DataFrame(rows)


def build_label_summary(df_samples: pd.DataFrame) -> pd.DataFrame:
    per_fold = _group_summary(df_samples, ["seed", "fold", "label"], "per_fold")
    pooled = _group_summary(df_samples, ["label"], "pooled_all_samples")
    pooled["seed"] = np.nan
    pooled["fold"] = np.nan

    per_fold_means = (
        per_fold.groupby("label", dropna=False)[
            [
                "w_admission_mean",
                "w_admission_std",
                "w_discharge_mean",
                "w_discharge_std",
                "w_merged_mean",
                "w_merged_std",
                "n",
            ]
        ]
        .mean()
        .reset_index()
    )
    per_fold_means["summary_level"] = "mean_across_folds"
    per_fold_means["seed"] = np.nan
    per_fold_means["fold"] = np.nan
    per_fold_means["dominant_stream"] = _dominant_from_means(
        per_fold_means.rename(
            columns={
                "w_admission_mean": "w_admission",
                "w_discharge_mean": "w_discharge",
                "w_merged_mean": "w_merged",
            }
        )
    )

    cols = [
        "summary_level",
        "seed",
        "fold",
        "label",
        "n",
        "w_admission_mean",
        "w_admission_std",
        "w_discharge_mean",
        "w_discharge_std",
        "w_merged_mean",
        "w_merged_std",
        "dominant_stream",
    ]
    return pd.concat(
        [per_fold[cols], pooled[cols], per_fold_means[cols]], ignore_index=True
    )


def build_los_summary(df_samples: pd.DataFrame) -> pd.DataFrame:
    per_fold = _group_summary(df_samples, ["seed", "fold", "los"], "per_fold")
    pooled = _group_summary(df_samples, ["los"], "pooled_all_samples")
    pooled["seed"] = np.nan
    pooled["fold"] = np.nan

    per_fold_means = (
        per_fold.groupby("los", dropna=False)[
            [
                "w_admission_mean",
                "w_admission_std",
                "w_discharge_mean",
                "w_discharge_std",
                "w_merged_mean",
                "w_merged_std",
                "n",
            ]
        ]
        .mean()
        .reset_index()
    )
    per_fold_means["summary_level"] = "mean_across_folds"
    per_fold_means["seed"] = np.nan
    per_fold_means["fold"] = np.nan
    per_fold_means["dominant_stream"] = _dominant_from_means(
        per_fold_means.rename(
            columns={
                "w_admission_mean": "w_admission",
                "w_discharge_mean": "w_discharge",
                "w_merged_mean": "w_merged",
            }
        )
    )

    cols = [
        "summary_level",
        "seed",
        "fold",
        "los",
        "n",
        "w_admission_mean",
        "w_admission_std",
        "w_discharge_mean",
        "w_discharge_std",
        "w_merged_mean",
        "w_merged_std",
        "dominant_stream",
    ]
    return pd.concat(
        [per_fold[cols], pooled[cols], per_fold_means[cols]], ignore_index=True
    ).sort_values(["summary_level", "los", "seed", "fold"], na_position="last")


def build_plot_fold_mean(df_samples: pd.DataFrame) -> pd.DataFrame:
    summary = df_samples.groupby(["seed", "fold"])[STREAM_COLS].mean().reset_index()
    return summary.melt(
        id_vars=["seed", "fold"],
        value_vars=STREAM_COLS,
        var_name="stream",
        value_name="mean_weight",
    )


def build_plot_los_mean(df_samples: pd.DataFrame) -> pd.DataFrame:
    per_fold = (
        df_samples.groupby(["seed", "fold", "los"])[STREAM_COLS].mean().reset_index()
    )
    mean_df = per_fold.groupby("los")[STREAM_COLS].mean().reset_index()
    std_df = per_fold.groupby("los")[STREAM_COLS].std(ddof=1).reset_index()
    n_df = df_samples.groupby("los").size().reset_index(name="n_total")

    rows = []
    for col in STREAM_COLS:
        tmp = mean_df[["los", col]].rename(columns={col: "mean_weight"})
        tmp["std_across_folds"] = std_df[col]
        tmp["stream"] = col
        tmp = tmp.merge(n_df, on="los", how="left")
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True).sort_values(["los", "stream"])


def build_legacy_los_csv(df_samples: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df_samples.groupby("los")[STREAM_COLS]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in agg.columns
    ]
    agg = agg.rename(
        columns={
            "los": "LOS",
            "w_admission_mean": "w_ad_mean",
            "w_admission_std": "w_ad_std",
            "w_admission_count": "n_samples",
            "w_discharge_mean": "w_dis_mean",
            "w_discharge_std": "w_dis_std",
            "w_merged_mean": "w_merged_mean",
            "w_merged_std": "w_merged_std",
        }
    )
    for col in ["w_discharge_count", "w_merged_count"]:
        if col in agg.columns:
            agg = agg.drop(columns=[col])
    return agg.sort_values("LOS").reset_index(drop=True)


def _print_stability_metrics(df_fold: pd.DataFrame) -> None:
    print("\n=== Fold Stability Metrics ===")
    for weight in ["w_admission_mean", "w_discharge_mean", "w_merged_mean"]:
        values = df_fold[weight].to_numpy()
        mean = np.mean(values)
        std = np.std(values)
        cv = (std / mean) * 100 if mean != 0 else np.nan
        min_val = np.min(values)
        max_val = np.max(values)
        stability = (
            "안정적 (< 20%)"
            if cv < 20
            else "약간 불안정 (20-30%)" if cv < 30 else "불안정 (30%+)"
        )
        print(
            f"{weight:18s}: CV={cv:5.1f}% [{stability}]  "
            f"Range=[{min_val:.4f}, {max_val:.4f}]  Δ={max_val - min_val:.4f}"
        )


def save_figures(
    df_plot_fold: pd.DataFrame,
    df_plot_los: pd.DataFrame,
    output_dir: Path,
    prefix: str,
    figure_format: str,
) -> None:
    import matplotlib.pyplot as plt

    fold_pivot = df_plot_fold.pivot_table(
        index=["seed", "fold"], columns="stream", values="mean_weight"
    ).reset_index()
    fold_pivot["label"] = fold_pivot.apply(
        lambda r: f"seed={int(r['seed'])} fold={int(r['fold'])}", axis=1
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fold_pivot))
    width = 0.25
    for i, col in enumerate(STREAM_COLS):
        ax.bar(
            x + (i - 1) * width, fold_pivot[col], width=width, label=STREAM_LABELS[col]
        )
    ax.set_xticks(x)
    ax.set_xticklabels(fold_pivot["label"], rotation=45, ha="right")
    ax.set_ylabel("Mean gate weight")
    ax.set_title("Fold-wise mean GatedFusion weights")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_fold_mean_bar.{figure_format}", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    for col in STREAM_COLS:
        tmp = df_plot_los[df_plot_los["stream"] == col].sort_values("los")
        ax.errorbar(
            tmp["los"],
            tmp["mean_weight"],
            yerr=tmp["std_across_folds"].fillna(0.0),
            marker="o",
            capsize=3,
            label=STREAM_LABELS[col],
        )
    ax.set_xlabel("LOS")
    ax.set_ylabel("Mean gate weight")
    ax.set_title("Per-LOS GatedFusion weights")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_los_mean_line.{figure_format}", dpi=200)
    plt.close(fig)


def process_run(
    run_name: str,
    args,
    runs_base: Path,
    device: torch.device,
    data_root: str,
) -> pd.DataFrame:
    run_dir = runs_base / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    fold_0_cfg_path = run_dir / "folds" / "fold_0" / "config.final.yaml"
    fold_0_cfg = load_yaml(str(fold_0_cfg_path))
    n_folds = fold_0_cfg["train"].get("n_folds", 5)
    folds_to_run = _parse_fold_arg(args.fold, n_folds)
    batch_size = args.batch_size or fold_0_cfg["train"]["batch_size"]

    dataset = _build_dataset(data_root, fold_0_cfg)
    model_name = _get_model_name(fold_0_cfg)
    num_nodes = _get_num_nodes(dataset, model_name)
    test_bundle = _prepare_test_data(dataset, fold_0_cfg, batch_size)
    seed = int(fold_0_cfg["train"]["seed"])

    print(f"\n=== Run: {run_name} (seed={seed}) ===")
    print(f"Test set size: {len(test_bundle['test_idx'])}")
    print(
        f"  all_x: {tuple(test_bundle['all_x'].shape)}, "
        f"all_los range: [{int(test_bundle['all_los'].min())}, {int(test_bundle['all_los'].max())}]"
    )

    fold_frames: list[pd.DataFrame] = []
    for fold_id in folds_to_run:
        fold_dir = str(run_dir / "folds" / f"fold_{fold_id}")
        fold_cfg = load_yaml(
            str(run_dir / "folds" / f"fold_{fold_id}" / "config.final.yaml")
        )
        print(f"\n=== Fold {fold_id}: building edge_index ===")
        edge_index = _build_edge_index(
            dataset=dataset,
            labels=test_bundle["labels"],
            fold_cfg=fold_cfg,
            fold_id=fold_id,
            fold_dir=fold_dir,
            device=device,
            data_root=data_root,
            num_nodes=num_nodes,
        )
        print(f"  edge_index shape: {tuple(edge_index.shape)}")

        print(f"=== Fold {fold_id}: extracting GatedFusion weights ===")
        model, _ = _load_fold_model(fold_dir, device)
        pred_df = extract_fold_predictions(
            model=model,
            all_x=test_bundle["all_x"],
            all_y=test_bundle["all_y"],
            all_los=test_bundle["all_los"],
            edge_index=edge_index,
            device=device,
            batch_size=batch_size,
        )

        df_fold = pd.concat(
            [
                test_bundle["metadata"].reset_index(drop=True),
                pred_df.reset_index(drop=True),
            ],
            axis=1,
        )
        df_fold["seed"] = seed
        df_fold["fold"] = fold_id
        cols = [
            "seed",
            "fold",
            "sample_idx",
            "sample_id",
            "label",
            "los",
            "w_admission",
            "w_discharge",
            "w_merged",
            "dominant_stream",
            "y_true",
            "y_pred",
            "y_prob",
        ]
        df_fold = df_fold[cols]

        print(
            "  Mean weights — "
            f"w_ad={df_fold['w_admission'].mean():.3f}, "
            f"w_dis={df_fold['w_discharge'].mean():.3f}, "
            f"w_merged={df_fold['w_merged'].mean():.3f}"
        )
        _sanity_check(df_fold, fold_id, run_dir)
        fold_frames.append(df_fold)

    return pd.concat(fold_frames, ignore_index=True)


def main():
    args = parse_args()
    runs_base = _PROJECT_ROOT / "runs" / "protected" / "k_fold_CV"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_names = _resolve_run_names(args)
    if args.all_matching_seeds:
        if args.run_names:
            raise ValueError(
                "--all_matching_seeds cannot be used together with --run_names. "
                "Provide a single --run_name instead."
            )
        run_names = _resolve_all_matching_seed_runs(args.run_name, runs_base)
        print(f"Auto-discovered seed-matched runs ({len(run_names)}):")
        for run_name in run_names:
            print(f"  - {run_name}")

    device = device_set(args.device)
    data_root = str(_PROJECT_ROOT / "src" / "data")

    sample_frames: list[pd.DataFrame] = []
    for run_name in run_names:
        sample_frames.append(process_run(run_name, args, runs_base, device, data_root))

    df_samples = pd.concat(sample_frames, ignore_index=True)
    df_samples = df_samples.sort_values(["seed", "fold", "sample_idx"]).reset_index(
        drop=True
    )

    df_fold_summary = build_fold_summary(df_samples)
    df_stream_summary = build_stream_overall_summary(df_samples)
    df_label_summary = build_label_summary(df_samples)
    df_los_summary = build_los_summary(df_samples)
    df_plot_fold = build_plot_fold_mean(df_samples)
    df_plot_los = build_plot_los_mean(df_samples)
    df_legacy_los = build_legacy_los_csv(df_samples)

    samples_path = output_dir / f"{args.prefix}_samples.csv"
    fold_summary_path = output_dir / f"{args.prefix}_fold_summary.csv"
    stream_summary_path = output_dir / f"{args.prefix}_stream_overall_summary.csv"
    label_summary_path = output_dir / f"{args.prefix}_label_summary.csv"
    los_summary_path = output_dir / f"{args.prefix}_los_summary.csv"
    plot_fold_path = output_dir / f"{args.prefix}_plot_fold_mean.csv"
    plot_los_path = output_dir / f"{args.prefix}_plot_los_mean.csv"
    legacy_path = output_dir / f"{args.prefix}_legacy_los.csv"
    legacy_default_path = _THIS_DIR / "gated_fusion_w_los_kfold.csv"

    df_samples.to_csv(samples_path, index=False)
    df_fold_summary.to_csv(fold_summary_path, index=False)
    df_stream_summary.to_csv(stream_summary_path, index=False)
    df_label_summary.to_csv(label_summary_path, index=False)
    df_los_summary.to_csv(los_summary_path, index=False)
    df_plot_fold.to_csv(plot_fold_path, index=False)
    df_plot_los.to_csv(plot_los_path, index=False)
    df_legacy_los.to_csv(legacy_path, index=False)
    df_legacy_los.to_csv(legacy_default_path, index=False)

    print("\n=== Fold Stability Analysis ===")
    print(df_fold_summary.to_string(index=False))
    _print_stability_metrics(df_fold_summary)

    print("\n=== Stream Overall Summary ===")
    print(df_stream_summary.to_string(index=False))

    if args.save_figures:
        save_figures(
            df_plot_fold=df_plot_fold,
            df_plot_los=df_plot_los,
            output_dir=output_dir,
            prefix=args.prefix,
            figure_format=args.figure_format,
        )

    print(f"\nSaved sample-level CSV: {samples_path}")
    print(f"Saved fold summary: {fold_summary_path}")
    print(f"Saved label summary: {label_summary_path}")
    print(f"Saved LOS summary: {los_summary_path}")
    print(f"Saved legacy LOS CSV: {legacy_default_path}")


if __name__ == "__main__":
    main()
