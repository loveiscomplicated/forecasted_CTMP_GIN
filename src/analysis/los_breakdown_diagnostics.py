from __future__ import annotations

import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis.joint_drift_decomposition import LOS_PRED_BIN_COL, LOS_TRUE_BIN_COL, decompose_head_drift


PROJECT_ROOT = Path(__file__).resolve().parents[2]

COARSE6_LABELS = {
    0: "1",
    1: "2-7",
    2: "8-14",
    3: "15-21",
    4: "22-28",
    5: "29-37",
}
BREAKDOWN9_LABELS = {
    0: "1",
    1: "2-7",
    2: "8-14",
    3: "15-21",
    4: "22-28",
    5: "29-31",
    6: "32-33",
    7: "34-35",
    8: "36-37",
}
COARSE_REPRESENTATIVES = {1: 0, 4: 1, 11: 2, 18: 3, 25: 4, 33: 5}
BREAKDOWN_LONG_BINS = (5, 6, 7, 8)
MIDDLE_BINS = (2, 3, 4)
LONG_BIN_LABEL_TO_COLUMN = {
    5: ("to_29_31_count", "to_29_31_pct"),
    6: ("to_32_33_count", "to_32_33_pct"),
    7: ("to_34_35_count", "to_34_35_pct"),
    8: ("to_36_37_count", "to_36_37_pct"),
}
DEFAULT_D_HEADS = (
    "SERVICES_D",
    "EMPLOY_D",
    "LIVARAG_D",
    "ARRESTS_D",
    "DETNLF_D",
    "SUB1_D",
    "SUB2_D",
    "SUB3_D",
    "FREQ1_D",
    "FREQ2_D",
    "FREQ3_D",
    "FREQ_ATND_SELF_HELP_D",
)


@dataclass
class WarningCollector:
    strict: bool = False
    warnings: list[str] = field(default_factory=list)
    missing_artifacts: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if self.strict:
            raise RuntimeError(message)
        self.warnings.append(message)
        print(f"[warning] {message}", file=sys.stderr)

    def missing(self, path: Path | str, context: str) -> None:
        item = f"{context}: missing {path}"
        self.missing_artifacts.append(str(path))
        self.warn(item)


@dataclass
class RunSpec:
    run_name: str
    run_dir: Path
    run_type: str
    predictor_run_name: str | None = None
    predictor_run_dir: Path | None = None
    los_scheme: str | None = None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _series_to_int_array(values: pd.Series, column: str) -> np.ndarray:
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"{column} contains non-numeric or missing values.")
    arr = parsed.to_numpy(dtype=np.float64)
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded):
        raise ValueError(f"{column} contains non-integer values.")
    return rounded.astype(np.int64)


def map_los_to_coarse6(los_raw: int) -> int:
    value = int(los_raw)
    if value == 0:
        return 0
    if value in COARSE_REPRESENTATIVES:
        return COARSE_REPRESENTATIVES[value]
    if value == 1:
        return 0
    if 2 <= value <= 7:
        return 1
    if 8 <= value <= 14:
        return 2
    if 15 <= value <= 21:
        return 3
    if 22 <= value <= 28:
        return 4
    if 29 <= value <= 37:
        return 5
    raise ValueError(f"LOS value {los_raw} cannot be mapped to coarse6.")


def map_los_to_breakdown9(los_raw: int) -> int:
    value = int(los_raw)
    if value == 0:
        return 0
    if value == 1:
        return 0
    if 2 <= value <= 7:
        return 1
    if 8 <= value <= 14:
        return 2
    if 15 <= value <= 21:
        return 3
    if 22 <= value <= 28:
        return 4
    if 29 <= value <= 31:
        return 5
    if 32 <= value <= 33:
        return 6
    if 34 <= value <= 35:
        return 7
    if 36 <= value <= 37:
        return 8
    raise ValueError(f"LOS value {los_raw} cannot be mapped to breakdown9.")


def los_scheme_labels(scheme: str) -> dict[int, str]:
    if scheme == "coarse6":
        return COARSE6_LABELS
    if scheme == "breakdown9":
        return BREAKDOWN9_LABELS
    raise ValueError(f"Unsupported LOS scheme: {scheme}")


def _collapse_breakdown9_to_coarse6(values: np.ndarray) -> np.ndarray:
    mapping = np.asarray([0, 1, 2, 3, 4, 5, 5, 5, 5], dtype=np.int64)
    if values.size == 0:
        return values.astype(np.int64)
    if values.min() < 0 or values.max() > 8:
        raise ValueError(f"Cannot collapse non-breakdown9 values with min={values.min()}, max={values.max()}.")
    return mapping[values]


def normalize_los_values(values: np.ndarray, target_scheme: str, context: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.int64)
    if arr.size == 0:
        return arr.astype(np.int64)
    unique = set(np.unique(arr).tolist())
    min_value = int(arr.min())
    max_value = int(arr.max())
    if target_scheme == "coarse6":
        if min_value >= 0 and max_value <= 5:
            return arr.astype(np.int64)
        if min_value >= 0 and max_value <= 8:
            return _collapse_breakdown9_to_coarse6(arr)
        return np.asarray([map_los_to_coarse6(int(value)) for value in arr], dtype=np.int64)
    if target_scheme == "breakdown9":
        if min_value >= 0 and max_value <= 8:
            return arr.astype(np.int64)
        if min_value >= 0 and max_value <= 5 and unique.issubset(set(range(6))):
            raise ValueError(f"{context}: coarse6 class ids cannot be expanded losslessly to breakdown9.")
        return np.asarray([map_los_to_breakdown9(int(value)) for value in arr], dtype=np.int64)
    raise ValueError(f"{context}: unsupported target LOS scheme {target_scheme!r}.")


def infer_los_scheme(pred_values: np.ndarray, explicit_scheme: str | None = None) -> str:
    if explicit_scheme in {"coarse6", "breakdown9"}:
        return explicit_scheme
    values = np.asarray(pred_values, dtype=np.int64)
    if values.size == 0:
        return "coarse6"
    min_value = int(values.min())
    max_value = int(values.max())
    if min_value >= 0 and max_value <= 5:
        return "coarse6"
    if min_value >= 0 and max_value <= 8:
        return "breakdown9"
    if min_value >= 1 and max_value <= 37:
        if set(np.unique(values).tolist()).issubset(set(COARSE_REPRESENTATIVES)):
            return "coarse6"
        return "breakdown9"
    return "coarse6"


def normalize_joint_prediction_df(
    df: pd.DataFrame,
    *,
    target_scheme: str,
    context: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    if "true_los" not in df.columns or "pred_los" not in df.columns:
        raise ValueError(f"{context}: expected true_los and pred_los columns.")
    true_raw = _series_to_int_array(df["true_los"], "true_los")
    pred_raw = _series_to_int_array(df["pred_los"], "pred_los")
    true_bins = normalize_los_values(true_raw, target_scheme, context)
    pred_bins = normalize_los_values(pred_raw, target_scheme, context)
    normalized = df.copy()
    normalized[LOS_TRUE_BIN_COL] = true_bins
    normalized[LOS_PRED_BIN_COL] = pred_bins
    heads = [
        col[len("true_") :]
        for col in normalized.columns
        if col.startswith("true_")
        and col not in {"true_los", LOS_TRUE_BIN_COL}
        and f"pred_{col[len('true_') :]}" in normalized.columns
    ]
    info = {
        "target_scheme": target_scheme,
        "n_los_bins": len(los_scheme_labels(target_scheme)),
        "unique_true_los": sorted(np.unique(true_raw).astype(int).tolist()),
        "unique_pred_los": sorted(np.unique(pred_raw).astype(int).tolist()),
        "unique_true_bins": sorted(np.unique(true_bins).astype(int).tolist()),
        "unique_pred_bins": sorted(np.unique(pred_bins).astype(int).tolist()),
        "rows_used": int(len(normalized)),
        "invalid_or_unmapped_rows": 0,
    }
    return normalized, heads, info


def find_joint_prediction_file(run_dir: Path, split: str, warnings: WarningCollector) -> Path | None:
    alias = {"valid": "val", "test": "test", "train": "train"}.get(split, split)
    candidates = [
        run_dir / f"{alias}_predictions.csv",
        run_dir / "joint_predictor" / f"{alias}_predictions.csv",
        run_dir / f"{split}_predictions.csv",
        run_dir / "joint_predictor" / f"{split}_predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    for name in (f"{alias}_predictions.csv", f"{split}_predictions.csv"):
        matches = sorted(run_dir.rglob(name))
        if matches:
            return matches[0]
    warnings.missing(run_dir / f"{alias}_predictions.csv", f"joint predictions ({split})")
    return None


def discover_joint_predictions(
    *,
    downstream_run_dir: Path,
    predictor_run_dir: Path | None,
    split: str,
    warnings: WarningCollector,
) -> pd.DataFrame:
    search_dirs = [downstream_run_dir]
    if predictor_run_dir is not None and predictor_run_dir != downstream_run_dir:
        search_dirs.append(predictor_run_dir)
    for directory in search_dirs:
        path = find_joint_prediction_file(directory, split, warnings)
        if path is not None:
            return pd.read_csv(path)
    raise RuntimeError(f"No joint prediction file found for split={split}.")


def build_los_confusion_tables(
    df: pd.DataFrame,
    *,
    run_name: str,
    scheme: str,
    true_col: str = LOS_TRUE_BIN_COL,
    pred_col: str = LOS_PRED_BIN_COL,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels = los_scheme_labels(scheme)
    bins = list(labels)
    counts = pd.crosstab(df[true_col], df[pred_col], dropna=False).reindex(index=bins, columns=bins, fill_value=0)
    counts = counts.astype(int)
    count_rows: list[dict[str, Any]] = []
    row_pct_rows: list[dict[str, Any]] = []
    for true_bin in bins:
        row_total = int(counts.loc[true_bin].sum())
        for pred_bin in bins:
            count = int(counts.loc[true_bin, pred_bin])
            count_rows.append(
                {
                    "run_name": run_name,
                    "los_scheme": scheme,
                    "true_bin": true_bin,
                    "true_bin_label": labels[true_bin],
                    "pred_bin": pred_bin,
                    "pred_bin_label": labels[pred_bin],
                    "count": count,
                }
            )
            row_pct_rows.append(
                {
                    "run_name": run_name,
                    "los_scheme": scheme,
                    "true_bin": true_bin,
                    "true_bin_label": labels[true_bin],
                    "pred_bin": pred_bin,
                    "pred_bin_label": labels[pred_bin],
                    "row_pct": float(count / row_total) if row_total else np.nan,
                }
            )
    return pd.DataFrame(count_rows), pd.DataFrame(row_pct_rows)


def build_population_contamination(
    counts_df: pd.DataFrame,
    *,
    run_name: str,
    scheme: str,
) -> pd.DataFrame:
    if counts_df.empty:
        return counts_df.copy()
    pred_totals = counts_df.groupby("pred_bin")["count"].sum()
    true_totals = counts_df.groupby("true_bin")["count"].sum()
    rows: list[dict[str, Any]] = []
    for row in counts_df.to_dict(orient="records"):
        pred_total = int(pred_totals.get(row["pred_bin"], 0))
        true_total = int(true_totals.get(row["true_bin"], 0))
        rows.append(
            {
                "run_name": run_name,
                "los_scheme": scheme,
                "pred_bin": row["pred_bin"],
                "pred_bin_label": row["pred_bin_label"],
                "true_bin": row["true_bin"],
                "true_bin_label": row["true_bin_label"],
                "count": int(row["count"]),
                "share_within_pred_bin": float(row["count"] / pred_total) if pred_total else np.nan,
                "share_within_true_bin": float(row["count"] / true_total) if true_total else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_middle_to_long_flow_summary(
    counts_df: pd.DataFrame,
    *,
    run_name: str,
    scheme: str,
) -> pd.DataFrame:
    if counts_df.empty:
        return pd.DataFrame()
    labels = los_scheme_labels(scheme)
    long_bins = BREAKDOWN_LONG_BINS if scheme == "breakdown9" else (5,)
    rows: list[dict[str, Any]] = []
    for true_bin in MIDDLE_BINS:
        group = counts_df[counts_df["true_bin"] == true_bin]
        support = int(group["count"].sum())
        correct_count = int(group.loc[group["pred_bin"] == true_bin, "count"].sum())
        row: dict[str, Any] = {
            "run_name": run_name,
            "true_bin": true_bin,
            "true_bin_label": labels[true_bin],
            "support": support,
            "correct_count": correct_count,
            "accuracy": float(correct_count / support) if support else np.nan,
        }
        total_to_long = 0
        if scheme == "breakdown9":
            for long_bin in BREAKDOWN_LONG_BINS:
                count = int(group.loc[group["pred_bin"] == long_bin, "count"].sum())
                pct = float(count / support) if support else np.nan
                count_col, pct_col = LONG_BIN_LABEL_TO_COLUMN[long_bin]
                row[count_col] = count
                row[pct_col] = pct
                total_to_long += count
        else:
            count = int(group.loc[group["pred_bin"] == 5, "count"].sum())
            total_to_long = count
            row["to_29_31_count"] = np.nan
            row["to_29_31_pct"] = np.nan
            row["to_32_33_count"] = np.nan
            row["to_32_33_pct"] = np.nan
            row["to_34_35_count"] = np.nan
            row["to_34_35_pct"] = np.nan
            row["to_36_37_count"] = np.nan
            row["to_36_37_pct"] = np.nan
        row["total_to_long_count"] = total_to_long
        row["total_to_long_pct"] = float(total_to_long / support) if support else np.nan
        if support:
            pred_counts = group.groupby("pred_bin")["count"].sum()
            top_pred_bin = int(pred_counts.idxmax())
            row["top_pred_bin"] = top_pred_bin
            row["top_pred_pct"] = float(pred_counts.max() / support)
        else:
            row["top_pred_bin"] = np.nan
            row["top_pred_pct"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y_true.size == 0 or np.unique(y_true).size < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score))
    except ValueError:
        return np.nan


def binary_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

    if y_true.size == 0:
        return {
            "support": 0,
            "positive_count": 0,
            "negative_count": 0,
            "positive_rate": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "recall": np.nan,
            "f1": np.nan,
            "auc": np.nan,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    conf = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = conf.ravel()
    return {
        "support": int(y_true.size),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "positive_rate": float(np.mean(y_true == 1)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _safe_auc(y_true, y_score),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def normalize_outcome_prediction_df(df: pd.DataFrame, context: str) -> pd.DataFrame:
    cols = {col.lower(): col for col in df.columns}
    true_col = next((cols[name] for name in ("y_true", "true_label", "label", "reasonb") if name in cols), None)
    score_col = next((cols[name] for name in ("y_prob", "prob", "pred_prob", "score", "y_score") if name in cols), None)
    pred_col = next((cols[name] for name in ("y_pred", "pred_label", "prediction") if name in cols), None)
    row_col = next((cols[name] for name in ("row_idx", "index") if name in cols), None)
    if true_col is None or score_col is None:
        raise ValueError(f"{context}: missing outcome prediction columns.")
    out = pd.DataFrame(
        {
            "row_idx": pd.to_numeric(df[row_col], errors="coerce").astype("Int64") if row_col is not None else pd.Series(range(len(df)), dtype="Int64"),
            "y_true": pd.to_numeric(df[true_col], errors="coerce").astype("Int64"),
            "y_score": pd.to_numeric(df[score_col], errors="coerce"),
        }
    )
    if pred_col is not None:
        out["y_pred"] = pd.to_numeric(df[pred_col], errors="coerce").astype("Int64")
    else:
        out["y_pred"] = (out["y_score"] >= 0.5).astype("Int64")
    if out.isna().any().any():
        raise ValueError(f"{context}: outcome prediction frame contains invalid rows.")
    return out.astype({"row_idx": int, "y_true": int, "y_pred": int})


def find_outcome_prediction_file(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "diagnostic_predictions.csv",
        run_dir / "outer_test_predictions.csv",
        run_dir / "test_predictions.csv",
        run_dir / "predictions.csv",
        run_dir / "outer_test_predictions.parquet",
        run_dir / "test_predictions.parquet",
    ]
    for path in candidates:
        if path.exists():
            return path
    matches = sorted(
        path
        for path in run_dir.rglob("*predictions*")
        if path.suffix in {".csv", ".parquet"} and "joint_predictor" not in path.parts
    )
    return matches[0] if matches else None


def _load_outcome_prediction_artifact(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _resolve_best_checkpoint(fold_dir: Path) -> Path:
    direct = fold_dir / "checkpoints" / "best.pt"
    if direct.exists():
        return direct
    best_epoch = None
    best_text = fold_dir / "best.txt"
    if best_text.exists():
        match = re.search(r"best_epoch:\s*([0-9]+)", best_text.read_text(encoding="utf-8", errors="ignore"))
        if match:
            best_epoch = int(match.group(1))
    if best_epoch is None:
        payload = _read_json(fold_dir / "fold_result.json") or {}
        best_epoch = _parse_int(payload.get("best_epoch"))
    if best_epoch is not None:
        candidate = fold_dir / "checkpoints" / f"epoch_{best_epoch}.pt"
        if candidate.exists():
            return candidate
    checkpoints = sorted((fold_dir / "checkpoints").glob("epoch_*.pt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"No checkpoint found under {fold_dir / 'checkpoints'}.")


def _safe_device(requested: str | None):
    import torch
    from src.utils.device_set import device_set

    device = device_set(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if device.type == "mps" and not torch.mps.is_available():
        return torch.device("cpu")
    return device


def _build_eval_loader(
    *,
    fold_dir: Path,
    dataset_root: Path,
    split: str,
    device: str | None,
):
    import copy
    import torch
    from torch.utils.data import DataLoader, Dataset

    from src.analysis.forecasted_gnn_performance_analysis import _build_dataset

    class _CachedSplitDataset(Dataset):
        def __init__(self, base_dataset, split_payload: dict[str, Any]) -> None:
            self.base_dataset = base_dataset
            self.x = split_payload["x"].long()
            los = split_payload["los"]
            self.los = los.float() if los.ndim == 2 else los.long()
            self.indices = torch.as_tensor(split_payload["indices"], dtype=torch.long)
            self.soft_discharge_cache = split_payload.get("soft_discharge")

        def __len__(self) -> int:
            return int(self.indices.numel())

        def __getitem__(self, index: int):
            row_idx = int(self.indices[index].item())
            _, y, los_raw = self.base_dataset[row_idx]
            if self.soft_discharge_cache is None:
                return self.x[index], y, self.los[index], {"row_idx": row_idx, "los_raw": int(los_raw)}
            soft_discharge: dict[str, dict[str, Any]] = {}
            soft_discharge_mask: dict[str, Any] = {}
            for head_name, head_payload in self.soft_discharge_cache["heads"].items():
                soft_discharge[head_name] = {
                    "probs": head_payload["probs"][index],
                    "target_col_idx": head_payload["target_col_idx"],
                    "class_to_embedding_idx": head_payload["class_to_embedding_idx"],
                    "num_classes": head_payload["num_classes"],
                }
                soft_discharge_mask[head_name] = head_payload["mask"][index]
            forecast_meta = {
                "soft_discharge": soft_discharge,
                "soft_discharge_mask": soft_discharge_mask,
                "metadata": copy.deepcopy(self.soft_discharge_cache.get("metadata", {})),
                "row_idx": row_idx,
                "los_raw": int(los_raw),
            }
            return self.x[index], y, self.los[index], forecast_meta

    cfg = _read_yaml(fold_dir / "config.final.yaml")
    split_payload = _read_json(fold_dir / "joint_forecast_pipeline_splits.json")
    if split_payload is None:
        raise FileNotFoundError(f"Missing joint_forecast_pipeline_splits.json in {fold_dir}.")
    device_obj = _safe_device(device)
    dataset = _build_dataset(cfg, dataset_root)
    split_key = "outer_test" if split == "test" else split
    cache_name = {"outer_test": "outer_test_joint.pt", "gnn_val": "gnn_val_joint.pt"}.get(split_key, f"{split_key}_joint.pt")
    cache_path = fold_dir / "cached_predictions" / cache_name
    payload = torch.load(cache_path, map_location="cpu", weights_only=False)
    expected = np.asarray(split_payload[f"{split_key}_idx"], dtype=np.int64)
    actual = torch.as_tensor(payload["indices"], dtype=torch.long).cpu().numpy()
    if not np.array_equal(expected, actual):
        raise ValueError(f"Cached indices do not match {split_key}_idx for {fold_dir}.")
    dataset_obj = _CachedSplitDataset(dataset, payload)
    loader = DataLoader(
        dataset_obj,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=True,
    )
    return cfg, dataset, loader, device_obj


def run_downstream_inference(
    *,
    fold_dir: Path,
    dataset_root: Path,
    split: str,
    device: str | None,
) -> pd.DataFrame:
    import torch
    import torch.nn as nn

    from src.models.factory import build_model
    from src.models.forecast_inputs import ensure_model_forecast_defaults
    from src.trainers.base import _move_soft_discharge_to_device, _unpack_batch

    cfg, _dataset, loader, device_obj = _build_eval_loader(
        fold_dir=fold_dir,
        dataset_root=dataset_root,
        split=split,
        device=device,
    )
    ensure_model_forecast_defaults(cfg)
    cfg["model"]["params"]["device"] = str(device_obj)
    model = build_model(model_name=cfg["model"]["name"], **cfg["model"].get("params", {})).to(device_obj)
    checkpoint = torch.load(_resolve_best_checkpoint(fold_dir), map_location=device_obj, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()
    edge_index = torch.load(fold_dir / "edge_index.pt", map_location=device_obj, weights_only=False)
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, int(cfg["train"]["batch_size"]))
    criterion = nn.BCEWithLogitsLoss()
    decision_threshold = float(cfg["train"].get("decision_threshold", 0.5))

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            x_batch, y_batch, los_batch, forecast_meta = _unpack_batch(batch)
            x_batch = x_batch.to(device_obj, non_blocking=True)
            y_batch = y_batch.to(device_obj, non_blocking=True)
            los_batch = los_batch.to(device_obj, non_blocking=True)
            soft_discharge = None
            if forecast_meta is not None:
                soft_discharge = _move_soft_discharge_to_device(forecast_meta.get("soft_discharge"), device_obj)
            logits = model(x_batch, los_batch, edge_index, device=device_obj, soft_discharge=soft_discharge)
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, y_batch.float())
            probs = torch.sigmoid(logits)
            preds = (probs >= decision_threshold).long()
            for idx, y_true, y_pred, y_score, los_raw in zip(
                forecast_meta["row_idx"],
                y_batch.detach().cpu().numpy().astype(np.int64).tolist(),
                preds.detach().cpu().numpy().astype(np.int64).tolist(),
                probs.detach().cpu().numpy().astype(np.float64).tolist(),
                forecast_meta["los_raw"],
            ):
                rows.append(
                    {
                        "row_idx": int(idx),
                        "y_true": int(y_true),
                        "y_pred": int(y_pred),
                        "y_score": float(y_score),
                        "true_los_raw": int(los_raw),
                        "loss": float(loss.detach().cpu()),
                    }
                )
    return pd.DataFrame(rows)


def load_or_compute_outcome_predictions(
    *,
    fold_dir: Path,
    dataset_root: Path,
    split: str,
    warnings: WarningCollector,
    device: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    files_read: list[str] = []
    artifact = find_outcome_prediction_file(fold_dir)
    if artifact is not None:
        files_read.append(str(artifact))
        return normalize_outcome_prediction_df(_load_outcome_prediction_artifact(artifact), str(artifact)), files_read
    warnings.warn(f"{fold_dir}: downstream prediction artifact not found; replaying read-only inference from cached inputs.")
    df = run_downstream_inference(fold_dir=fold_dir, dataset_root=dataset_root, split=split, device=device)
    files_read.extend(
        [
            str(fold_dir / "config.final.yaml"),
            str(fold_dir / "joint_forecast_pipeline_splits.json"),
            str(fold_dir / "edge_index.pt"),
            str(_resolve_best_checkpoint(fold_dir)),
            str(fold_dir / "cached_predictions" / ("outer_test_joint.pt" if split == "test" else f"{split}_joint.pt")),
        ]
    )
    return df, files_read


def merge_outcome_with_joint_predictions(
    outcome_df: pd.DataFrame,
    joint_df: pd.DataFrame,
    *,
    target_scheme: str,
    context: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    normalized_joint, _, los_info = normalize_joint_prediction_df(joint_df, target_scheme=target_scheme, context=context)
    merged = outcome_df.merge(
        normalized_joint[["row_idx", "true_los", "pred_los", LOS_TRUE_BIN_COL, LOS_PRED_BIN_COL]],
        on="row_idx",
        how="inner",
    )
    if merged.empty:
        raise ValueError(f"{context}: no matching rows between downstream predictions and joint predictions.")
    los_info = dict(los_info)
    los_info["rows_used"] = int(len(merged))
    los_info["invalid_or_unmapped_rows"] = int(len(outcome_df) - len(merged))
    return merged, los_info


def compute_outcome_metrics_by_los_bin(
    df: pd.DataFrame,
    *,
    run_name: str,
    split: str,
    scheme: str,
) -> pd.DataFrame:
    labels = los_scheme_labels(scheme)
    rows: list[dict[str, Any]] = []
    for basis, col in (("true_los_bin", LOS_TRUE_BIN_COL), ("pred_los_bin", LOS_PRED_BIN_COL)):
        for los_bin, label in labels.items():
            mask = df[col] == los_bin
            metrics = binary_metrics_from_arrays(
                df.loc[mask, "y_true"].to_numpy(dtype=np.int64),
                df.loc[mask, "y_pred"].to_numpy(dtype=np.int64),
                df.loc[mask, "y_score"].to_numpy(dtype=np.float64),
            )
            row = {
                "run_name": run_name,
                "split": split,
                "los_basis": basis,
                "los_scheme": scheme,
                "los_bin": los_bin,
                "los_bin_label": label,
            }
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def build_long_stay_recall_comparison(
    metrics_df: pd.DataFrame,
    *,
    baseline_run_name: str | None,
) -> pd.DataFrame:
    if metrics_df.empty:
        return metrics_df.copy()
    keep = metrics_df["los_basis"].eq("true_los_bin")
    long_labels = {"29-37", "29-31", "32-33", "34-35", "36-37"}
    keep &= metrics_df["los_bin_label"].isin(long_labels)
    out = metrics_df.loc[keep, ["run_name", "los_scheme", "los_bin_label", "support", "positive_count", "recall", "f1", "auc"]].copy()
    out = out.rename(columns={"los_bin_label": "bin_label"})
    out["delta_vs_baseline"] = np.nan
    if baseline_run_name is not None:
        baseline = out[out["run_name"] == baseline_run_name]
        if not baseline.empty:
            baseline_recall = _parse_float(baseline["recall"].mean())
            if baseline_recall is not None:
                out.loc[out["run_name"] != baseline_run_name, "delta_vs_baseline"] = (
                    out.loc[out["run_name"] != baseline_run_name, "recall"] - baseline_recall
                )
    return out


def compute_drift_decomposition(
    joint_df: pd.DataFrame,
    *,
    run_name: str,
    run_type: str,
    native_scheme: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for eval_scheme in sorted({"coarse6", native_scheme}):
        normalized, heads, _ = normalize_joint_prediction_df(
            joint_df,
            target_scheme=eval_scheme,
            context=f"{run_name}:{eval_scheme}",
        )
        n_los_bins = len(los_scheme_labels(eval_scheme))
        for head in heads:
            row = decompose_head_drift(normalized, head, n_los_bins)
            row["run_name"] = run_name
            row["run_type"] = run_type
            row["los_scheme_eval"] = eval_scheme
            row["native_los_scheme"] = native_scheme
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_run_level(
    *,
    run_name: str,
    run_type: str,
    los_scheme: str,
    downstream_metrics: dict[str, Any],
    merged_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    outcome_metrics_df: pd.DataFrame,
    middle_flow_df: pd.DataFrame,
) -> dict[str, Any]:
    native_drift = drift_df[drift_df["los_scheme_eval"] == los_scheme].copy()
    native_drift = native_drift.sort_values("dV_D", ascending=False)
    summary: dict[str, Any] = {
        "run_name": run_name,
        "run_type": run_type,
        "los_scheme": los_scheme,
        "n_prediction_rows": int(len(merged_df)),
        "n_los_bins": len(los_scheme_labels(los_scheme)),
    }
    for key, value in downstream_metrics.items():
        summary[key] = value
    for col in ("dV_D", "abs_dV_D", "js_D", "dV_LOS"):
        values = pd.to_numeric(native_drift[col], errors="coerce").dropna()
        if col == "dV_D":
            summary["mean_dV_D"] = float(values.mean()) if not values.empty else np.nan
            summary["median_dV_D"] = float(values.median()) if not values.empty else np.nan
            summary["max_dV_D"] = float(values.max()) if not values.empty else np.nan
        elif col == "abs_dV_D":
            summary["mean_abs_dV_D"] = float(values.mean()) if not values.empty else np.nan
        elif col == "js_D":
            summary["mean_js_D"] = float(values.mean()) if not values.empty else np.nan
            summary["max_js_D"] = float(values.max()) if not values.empty else np.nan
        elif col == "dV_LOS":
            summary["mean_dV_LOS"] = float(values.mean()) if not values.empty else np.nan
            summary["max_dV_LOS"] = float(values.max()) if not values.empty else np.nan
    top_dv = native_drift.head(3).reset_index(drop=True)
    for idx in range(3):
        if idx < len(top_dv):
            summary[f"top{idx + 1}_dV_D_head"] = top_dv.loc[idx, "head"]
            summary[f"top{idx + 1}_dV_D"] = float(top_dv.loc[idx, "dV_D"])
        else:
            summary[f"top{idx + 1}_dV_D_head"] = None
            summary[f"top{idx + 1}_dV_D"] = np.nan
    for head, prefix in (
        ("SERVICES_D", "services_d"),
        ("FREQ_ATND_SELF_HELP_D", "freq_atnd"),
        ("SUB1_D", "sub1"),
    ):
        row = native_drift[native_drift["head"] == head]
        if row.empty:
            summary[f"{prefix}_acc"] = np.nan
            summary[f"{prefix}_dV_LOS"] = np.nan
            summary[f"{prefix}_dV_D"] = np.nan
            continue
        record = row.iloc[0]
        summary[f"{prefix}_acc"] = float(record["acc"])
        if prefix == "services_d":
            summary[f"{prefix}_dV_LOS"] = float(record["dV_LOS"])
        summary[f"{prefix}_dV_D"] = float(record["dV_D"])
    long_mask = outcome_metrics_df["los_basis"].eq("true_los_bin") & outcome_metrics_df["los_bin_label"].isin(
        {"29-37", "29-31", "32-33", "34-35", "36-37"}
    )
    long_rows = outcome_metrics_df.loc[long_mask]
    if not long_rows.empty:
        summary["long_stay_macro_recall"] = float(pd.to_numeric(long_rows["recall"], errors="coerce").mean())
        worst = long_rows.sort_values("recall", ascending=True).iloc[0]
        summary["worst_long_stay_bin"] = worst["los_bin_label"]
        summary["worst_long_stay_recall"] = float(worst["recall"])
    else:
        summary["long_stay_macro_recall"] = np.nan
        summary["worst_long_stay_bin"] = None
        summary["worst_long_stay_recall"] = np.nan
    summary["middle_to_long_flow_pct"] = float(pd.to_numeric(middle_flow_df["total_to_long_pct"], errors="coerce").mean()) if not middle_flow_df.empty else np.nan
    return summary


def load_fold_metrics(fold_dir: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "run_dir": str(fold_dir),
        "fold": _parse_int(fold_dir.name.replace("fold_", "")),
        "status": "missing",
    }
    fold_result = _read_json(fold_dir / "fold_result.json")
    if fold_result:
        metrics["status"] = str(fold_result.get("status", "completed"))
        metrics["best_epoch"] = _parse_int(fold_result.get("best_epoch"))
        metrics["best_valid_metric"] = _parse_float(fold_result.get("best_valid_metric"))
        nested_valid = fold_result.get("best_valid_metrics") or {}
        if isinstance(nested_valid, dict):
            for key in ("auc", "acc", "f1", "precision", "recall", "loss"):
                value = _parse_float(nested_valid.get(f"valid_{key}"))
                if value is not None:
                    metrics[f"valid_{key}"] = value
        for key in ("auc", "acc", "f1", "precision", "recall", "loss"):
            value = _parse_float(fold_result.get(f"test_{key}"))
            if value is not None:
                metrics[f"test_{key}"] = value
    metrics_jsonl = fold_dir / "metrics.jsonl"
    if metrics_jsonl.exists():
        best_valid: dict[str, Any] | None = None
        with metrics_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                score = _parse_float(row.get("valid_auc"))
                if score is not None and (best_valid is None or score > _parse_float(best_valid.get("valid_auc"))):
                    best_valid = row
                for key in ("auc", "acc", "f1", "precision", "recall", "loss"):
                    value = _parse_float(row.get(f"test_{key}"))
                    if value is not None:
                        metrics[f"test_{key}"] = value
        if best_valid is not None:
            metrics["status"] = metrics.get("status", "fold0_only") or "fold0_only"
            metrics["best_epoch"] = _parse_int(best_valid.get("epoch")) or metrics.get("best_epoch")
            metrics["best_valid_metric"] = _parse_float(best_valid.get("valid_auc")) or metrics.get("best_valid_metric")
            for key in ("auc", "acc", "f1", "precision", "recall", "loss"):
                value = _parse_float(best_valid.get(f"valid_{key}"))
                if value is not None:
                    metrics[f"valid_{key}"] = value
    if metrics.get("fold") == 0 and metrics.get("status") == "missing":
        metrics["status"] = "fold0_only"
    return metrics


def downstream_metric_row(run_name: str, run_type: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_name": run_name,
        "run_type": run_type,
        "best_epoch": metrics.get("best_epoch"),
        "best_valid_metric": metrics.get("best_valid_metric"),
        "valid_auc": metrics.get("valid_auc"),
        "valid_acc": metrics.get("valid_acc"),
        "valid_f1": metrics.get("valid_f1"),
        "valid_precision": metrics.get("valid_precision"),
        "valid_recall": metrics.get("valid_recall"),
        "test_auc": metrics.get("test_auc"),
        "test_acc": metrics.get("test_acc"),
        "test_f1": metrics.get("test_f1"),
        "test_precision": metrics.get("test_precision"),
        "test_recall": metrics.get("test_recall"),
        "test_loss": metrics.get("test_loss"),
        "fold": metrics.get("fold"),
        "run_dir": metrics.get("run_dir"),
        "status": metrics.get("status"),
    }


def simple_markdown_table(df: pd.DataFrame, max_rows: int | None = None) -> str:
    if df.empty:
        return "_No rows._"
    view = df.head(max_rows).copy() if max_rows is not None else df.copy()
    columns = list(view.columns)
    rows = [columns]
    for _, series in view.iterrows():
        formatted: list[str] = []
        for value in series.tolist():
            if isinstance(value, float):
                formatted.append("nan" if math.isnan(value) else f"{value:.6f}")
            else:
                formatted.append("" if value is None else str(value))
        rows.append(formatted)
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(columns))]
    out_lines = []
    header = "| " + " | ".join(str(columns[idx]).ljust(widths[idx]) for idx in range(len(columns))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    out_lines.extend([header, sep])
    for row in rows[1:]:
        out_lines.append("| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(columns))) + " |")
    return "\n".join(out_lines)


def try_git_commit(project_root: Path) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None
