from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.joint_drift_decomposition import (  # noqa: E402
    LOS_PRED_BIN_COL,
    LOS_TRUE_BIN_COL,
    decompose_head_drift,
)
from src.models.discharge_predictor.los_utils import (  # noqa: E402
    LOS_COARSE_BIN_REPRESENTATIVES,
    map_los_array_to_coarse_bins,
)


VALID_SPLITS = {"train", "valid", "test"}
SPLIT_FILE_ALIASES = {"valid": "val", "train": "train", "test": "test"}
METRIC_KEYS = ("auc", "acc", "f1", "precision", "recall", "loss")
RUN_LEVEL_METRIC_COLS = [
    "downstream_valid_auc",
    "downstream_test_auc",
    "downstream_test_acc",
    "downstream_test_f1",
    "downstream_test_precision",
    "downstream_test_recall",
    "downstream_test_loss",
    "downstream_best_epoch",
    "downstream_fold",
    "downstream_status",
]


@dataclass
class WarningCollector:
    strict: bool = False
    warnings: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if self.strict:
            raise RuntimeError(message)
        self.warnings.append(message)
        print(f"[warning] {message}", file=sys.stderr)

    def missing(self, path: Path | str, context: str) -> None:
        msg = f"{context}: missing {path}"
        self.missing_files.append(str(path))
        self.warn(msg)


@dataclass
class PredictorRun:
    id: int | str
    predictor_run_name: str
    predictor_run_dir: Path | None = None
    direction: str | None = None
    heads_mode: str | None = None
    los_mode: str | None = None
    detach: str | None = None
    lambda_joint: float | None = None
    lambda_los: float | None = None
    lambda_entropy: float | None = None
    seed: int | None = None
    fold: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class DownstreamHint:
    downstream_run_dir: Path | None = None
    fold: int | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str | None = None


def _clean_md_cell(cell: str) -> str:
    value = cell.strip()
    if value.startswith("`") and value.endswith("`"):
        value = value[1:-1]
    return value.strip()


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None or not math.isfinite(parsed):
        return None
    return int(parsed)


def _parse_markdown_tables(path: Path) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    section = ""
    while idx < len(lines):
        line = lines[idx]
        if line.startswith("## "):
            section = line.lstrip("#").strip()
        if line.startswith("|") and idx + 1 < len(lines) and set(lines[idx + 1].replace("|", "").strip()) <= {"-", ":", " "}:
            headers = [_clean_md_cell(part) for part in line.strip().strip("|").split("|")]
            idx += 2
            rows: list[dict[str, str]] = []
            while idx < len(lines) and lines[idx].startswith("|"):
                values = [_clean_md_cell(part) for part in lines[idx].strip().strip("|").split("|")]
                if len(values) < len(headers):
                    values.extend([""] * (len(headers) - len(values)))
                rows.append(dict(zip(headers, values[: len(headers)])))
                idx += 1
            tables.append({"section": section, "headers": headers, "rows": rows})
            continue
        idx += 1
    return tables


def parse_registry(path: Path, warnings: WarningCollector) -> tuple[dict[int, PredictorRun], dict[int, DownstreamHint]]:
    if not path.exists():
        warnings.missing(path, "registry")
        return {}, {}

    runs: dict[int, PredictorRun] = {}
    downstream: dict[int, DownstreamHint] = {}
    for table in _parse_markdown_tables(path):
        headers = set(table["headers"])
        section = str(table["section"])
        if {"ID", "Run ID", "Direction", "Joint Heads Mode", "LOS Target Mode"}.issubset(headers):
            for row in table["rows"]:
                run_id = _parse_int(row.get("ID"))
                if run_id is None:
                    continue
                runs[run_id] = PredictorRun(
                    id=run_id,
                    predictor_run_name=row.get("Run ID", ""),
                    direction=row.get("Direction") or None,
                    heads_mode=row.get("Joint Heads Mode") or None,
                    los_mode=row.get("LOS Target Mode") or None,
                    detach=row.get("Detach") or None,
                    lambda_joint=_parse_float(row.get("λ_joint")),
                    lambda_los=_parse_float(row.get("λ_LOS")),
                    lambda_entropy=_parse_float(row.get("λ_entropy")),
                    seed=_parse_int(row.get("Seed")),
                    fold=_parse_int(row.get("Fold")),
                    extra={"registry_section": section, "best_epoch": _parse_int(row.get("Best Epoch"))},
                )
        if {"ID", "Forecast Cache", "Downstream Model", "Test AUC"}.issubset(headers):
            for row in table["rows"]:
                run_id = _parse_int(row.get("ID"))
                if run_id is None:
                    continue
                notes = row.get("Notes", "")
                path_match = re.search(r"from\s+`([^`]+)`", notes)
                downstream_dir = Path(path_match.group(1)) if path_match else None
                metrics = {
                    "downstream_best_epoch": _parse_int(row.get("Best Epoch")),
                    "downstream_valid_auc": _parse_float(row.get("Valid AUC")),
                    "downstream_test_acc": _parse_float(row.get("Test Acc")),
                    "downstream_test_f1": _parse_float(row.get("Test F1")),
                    "downstream_test_precision": _parse_float(row.get("Test Precision")),
                    "downstream_test_recall": _parse_float(row.get("Test Recall")),
                    "downstream_test_auc": _parse_float(row.get("Test AUC")),
                    "downstream_test_loss": _parse_float(row.get("Test Loss")),
                }
                status = "fold0_only" if "Fold 0 only" in notes else ("missing" if "Not run" in notes else None)
                downstream[run_id] = DownstreamHint(
                    downstream_run_dir=downstream_dir,
                    fold=0 if status == "fold0_only" else None,
                    metrics={k: v for k, v in metrics.items() if v is not None},
                    status=status,
                )
    return runs, downstream


def _resolve_path(path_text: str | Path | None, base: Path) -> Path | None:
    if path_text is None or str(path_text).strip() == "":
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = base / path
    return path


def _find_run_dir(runs_root: Path, run_name: str) -> Path | None:
    if not run_name:
        return None
    direct = runs_root / run_name
    if direct.exists():
        return direct
    matches = [path for path in runs_root.rglob(run_name) if path.is_dir()]
    return sorted(matches)[0] if matches else None


def load_run_map_csv(path: Path, runs_root: Path, warnings: WarningCollector) -> dict[int, PredictorRun]:
    if not path.exists():
        warnings.missing(path, "run map")
        return {}
    rows: dict[int, PredictorRun] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_id = _parse_int(row.get("id"))
            if raw_id is None:
                warnings.warn(f"Skipping run-map row without integer id: {row}")
                continue
            run_name = row.get("predictor_run_name", "").strip()
            run_dir = _resolve_path(row.get("predictor_run_dir"), PROJECT_ROOT)
            if run_dir is None and run_name:
                run_dir = _find_run_dir(runs_root, run_name)
            rows[raw_id] = PredictorRun(
                id=raw_id,
                predictor_run_name=run_name,
                predictor_run_dir=run_dir,
                direction=row.get("direction") or None,
                heads_mode=row.get("heads_mode") or None,
                los_mode=row.get("los_mode") or None,
                detach=row.get("detach") or None,
                lambda_joint=_parse_float(row.get("lambda_joint")),
                lambda_los=_parse_float(row.get("lambda_los")),
                lambda_entropy=_parse_float(row.get("lambda_entropy")),
                seed=_parse_int(row.get("seed")),
                fold=_parse_int(row.get("fold")),
            )
    return rows


def load_downstream_map_csv(path: Path, warnings: WarningCollector) -> dict[int, DownstreamHint]:
    if not path.exists():
        warnings.missing(path, "downstream map")
        return {}
    rows: dict[int, DownstreamHint] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_id = _parse_int(row.get("id"))
            if raw_id is None:
                warnings.warn(f"Skipping downstream-map row without integer id: {row}")
                continue
            rows[raw_id] = DownstreamHint(
                downstream_run_dir=_resolve_path(row.get("downstream_run_dir"), PROJECT_ROOT),
                fold=_parse_int(row.get("fold")),
            )
    return rows


def _split_alias(split: str) -> str:
    return SPLIT_FILE_ALIASES.get(split, split)


def locate_prediction_file(run_dir: Path, split: str, warnings: WarningCollector) -> Path | None:
    alias = _split_alias(split)
    candidates = [
        run_dir / f"{alias}_predictions.csv",
        run_dir / f"{split}_predictions.csv",
        run_dir / "predictions.csv",
        run_dir / "joint_predictor" / f"{alias}_predictions.csv",
        run_dir / "joint_predictor" / f"{split}_predictions.csv",
        run_dir / "joint_predictor" / "predictions.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    pattern_names = [f"{alias}_predictions.csv", f"{split}_predictions.csv"]
    if split == "test":
        pattern_names.append("predictions.csv")
    for name in pattern_names:
        matches = sorted(run_dir.rglob(name))
        if matches:
            return matches[0]
    warnings.missing(run_dir / f"{alias}_predictions.csv", f"prediction file for split={split}")
    return None


def _find_los_col(df: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    lower_to_col = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in lower_to_col:
            return lower_to_col[candidate.lower()]
    return None


def _integer_values(values: pd.Series, column: str) -> np.ndarray:
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"LOS column {column!r} contains non-numeric or missing values.")
    arr = parsed.to_numpy(dtype=np.float64)
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded):
        raise ValueError(f"LOS column {column!r} contains non-integer values.")
    return rounded.astype(np.int64)


def _looks_like_coarse(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    unique = set(np.unique(values).tolist())
    return min(unique) >= 0 and max(unique) <= 5


def _looks_like_raw(values: np.ndarray) -> bool:
    if values.size == 0:
        return True
    unique = set(np.unique(values).tolist())
    return min(unique) >= 1 and max(unique) <= 37


def _to_coarse6(values: np.ndarray, *, role: str, warnings: WarningCollector, context: str) -> np.ndarray:
    if _looks_like_coarse(values) and (0 in set(np.unique(values).tolist()) or role == "pred"):
        return values.astype(np.int64)
    if _looks_like_raw(values):
        if set(np.unique(values).tolist()).issubset(set(LOS_COARSE_BIN_REPRESENTATIVES)):
            warnings.warn(f"{context}: {role}_los uses coarse representative raw values; mapping representatives to coarse6.")
        return np.asarray(map_los_array_to_coarse_bins(values), dtype=np.int64)
    raise ValueError(f"{context}: cannot map {role}_los values to coarse6; min={values.min()}, max={values.max()}.")


def _to_raw37_index(values: np.ndarray, *, role: str, context: str) -> np.ndarray:
    if values.size == 0:
        return values.astype(np.int64)
    min_value = int(values.min())
    max_value = int(values.max())
    unique = set(np.unique(values).tolist())
    if min_value >= 1 and max_value <= 37 and 0 not in unique:
        return values.astype(np.int64) - 1
    if min_value >= 0 and max_value <= 36:
        return values.astype(np.int64)
    raise ValueError(f"{context}: cannot map {role}_los values to raw37 index space; min={min_value}, max={max_value}.")


def _infer_los_bins(
    true_values: np.ndarray,
    pred_values: np.ndarray,
    run: PredictorRun,
    requested: str,
    warnings: WarningCollector,
    context: str,
) -> tuple[str, int]:
    if requested != "auto":
        if requested not in {"coarse6", "raw37"}:
            raise ValueError(f"Unsupported --los-bins {requested!r}; expected coarse6/raw37/auto.")
        return requested, 6 if requested == "coarse6" else 37
    los_mode = str(run.los_mode or "").lower()
    if los_mode == "coarse":
        warnings.warn(f"{context}: inferred los-bins=coarse6 from registry LOS mode.")
        return "coarse6", 6
    if los_mode == "raw37":
        warnings.warn(f"{context}: inferred los-bins=raw37 from registry LOS mode.")
        return "raw37", 37
    if _looks_like_coarse(pred_values) and (not _looks_like_raw(pred_values) or int(pred_values.max(initial=0)) <= 5):
        warnings.warn(f"{context}: inferred los-bins=coarse6 from pred_los unique values.")
        return "coarse6", 6
    warnings.warn(f"{context}: inferred los-bins=raw37 by fallback.")
    return "raw37", 37


def normalize_prediction_dataframe(
    df: pd.DataFrame,
    run: PredictorRun,
    los_bins: str,
    warnings: WarningCollector,
    context: str,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    true_los_col = _find_los_col(df, ("true_los", "los_true", "target_los", "true_LOS", "LOS_true"))
    pred_los_col = _find_los_col(df, ("pred_los", "los_pred", "predicted_los", "pred_LOS", "LOS_pred"))
    if true_los_col is None or pred_los_col is None:
        raise ValueError(f"{context}: prediction dataframe must contain true_los and pred_los columns.")

    true_values = _integer_values(df[true_los_col], true_los_col)
    pred_values = _integer_values(df[pred_los_col], pred_los_col)
    resolved_los_bins, n_los_bins = _infer_los_bins(true_values, pred_values, run, los_bins, warnings, context)
    if resolved_los_bins == "coarse6":
        true_bins = _to_coarse6(true_values, role="true", warnings=warnings, context=context)
        pred_bins = _to_coarse6(pred_values, role="pred", warnings=warnings, context=context)
    else:
        true_bins = _to_raw37_index(true_values, role="true", context=context)
        pred_bins = _to_raw37_index(pred_values, role="pred", context=context)

    normalized = df.copy()
    normalized[LOS_TRUE_BIN_COL] = true_bins
    normalized[LOS_PRED_BIN_COL] = pred_bins
    bad_true = sorted(set(np.unique(true_bins).tolist()) - set(range(n_los_bins)))
    bad_pred = sorted(set(np.unique(pred_bins).tolist()) - set(range(n_los_bins)))
    if bad_true or bad_pred:
        raise ValueError(f"{context}: LOS bins outside 0..{n_los_bins - 1}; true_bad={bad_true}, pred_bad={bad_pred}.")

    heads: list[str] = []
    for col in normalized.columns:
        if not col.startswith("true_"):
            continue
        head = col[len("true_") :]
        if head.lower() in {"los", "los_bin"}:
            continue
        if f"pred_{head}" in normalized.columns:
            heads.append(head)
    if not heads:
        raise ValueError(f"{context}: no true_<HEAD>/pred_<HEAD> discharge head pairs found.")

    info = {
        "los_bins": resolved_los_bins,
        "n_los_bins": n_los_bins,
        "true_los_column": true_los_col,
        "pred_los_column": pred_los_col,
        "true_los_unique": sorted(np.unique(true_values).astype(int).tolist()),
        "pred_los_unique": sorted(np.unique(pred_values).astype(int).tolist()),
        "true_los_bin_unique": sorted(np.unique(true_bins).astype(int).tolist()),
        "pred_los_bin_unique": sorted(np.unique(pred_bins).astype(int).tolist()),
        "true_los_min": int(true_values.min()) if true_values.size else None,
        "true_los_max": int(true_values.max()) if true_values.size else None,
        "pred_los_min": int(pred_values.min()) if pred_values.size else None,
        "pred_los_max": int(pred_values.max()) if pred_values.size else None,
        "rows_used": int(len(normalized)),
    }
    print(
        f"[los] {context}: mode={resolved_los_bins}, true_unique={info['true_los_unique']}, "
        f"pred_unique={info['pred_los_unique']}, bins={n_los_bins}, rows={len(normalized)}"
    )
    return normalized, heads, info


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _metric_from_payload(payload: dict[str, Any], split: str, key: str) -> float | None:
    candidates = [
        f"{split}_{key}",
        key if split == "test" else f"{split}_{key}",
        f"{split}_final_{key}",
        f"{split}/{'auc' if key == 'auc' else key}",
    ]
    for candidate in candidates:
        if candidate in payload:
            return _parse_float(payload.get(candidate))
    nested = payload.get(split)
    if isinstance(nested, dict):
        return _parse_float(nested.get(key) or nested.get(f"{split}_{key}"))
    return None


def _best_valid_from_metrics_jsonl(path: Path) -> dict[str, Any]:
    best: dict[str, Any] = {}
    last_test: dict[str, Any] = {}
    if not path.exists():
        return {"valid": best, "test": last_test}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            valid_auc = _parse_float(row.get("valid_auc"))
            if valid_auc is not None and (not best or valid_auc > float(best.get("valid_auc", -np.inf))):
                best = row
            if row.get("split") == "test" or any(str(key).startswith("test_") for key in row):
                last_test = row
    return {"valid": best, "test": last_test}


def _best_epoch_from_text(path: Path) -> int | None:
    if not path.exists():
        return None
    match = re.search(r"best_epoch:\s*([0-9]+)", path.read_text(encoding="utf-8", errors="ignore"))
    return int(match.group(1)) if match else None


def _extract_test_metrics_from_json(path: Path) -> dict[str, float]:
    payload = _read_json(path)
    if not payload:
        return {}
    if "baseline_metrics" in payload and isinstance(payload["baseline_metrics"], dict):
        payload = payload["baseline_metrics"]
    out: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = _metric_from_payload(payload, "test", key)
        if value is not None:
            out[f"downstream_test_{key}"] = value
    return out


def _parse_fold_metrics(fold_dir: Path) -> dict[str, Any] | None:
    if not fold_dir.exists() or not fold_dir.is_dir():
        return None
    metrics: dict[str, Any] = {"downstream_fold": _parse_int(fold_dir.name.replace("fold_", ""))}
    metrics_jsonl = _best_valid_from_metrics_jsonl(fold_dir / "metrics.jsonl")
    best_valid = metrics_jsonl["valid"]
    if best_valid:
        for key in METRIC_KEYS:
            value = _parse_float(best_valid.get(f"valid_{key}"))
            if value is not None:
                metrics[f"downstream_valid_{key}"] = value
        metrics["downstream_best_epoch"] = _parse_int(best_valid.get("epoch"))
    best_epoch = _best_epoch_from_text(fold_dir / "best.txt")
    if best_epoch is not None:
        metrics["downstream_best_epoch"] = best_epoch

    for key in METRIC_KEYS:
        value = _parse_float(metrics_jsonl["test"].get(f"test_{key}")) if metrics_jsonl["test"] else None
        if value is not None:
            metrics[f"downstream_test_{key}"] = value

    for json_name in (
        "diagnostic_summary.json",
        "summary.json",
        "fallback_ablation_eval.json",
        "ablation_top3_admission_fallback.json",
    ):
        metrics.update(_extract_test_metrics_from_json(fold_dir / json_name))

    if len(metrics) > 1:
        return metrics
    return None


def _parse_cv_summary(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not payload:
        return []
    rows: list[dict[str, Any]] = []
    for row in payload.get("fold_results", []) or []:
        if not isinstance(row, dict):
            continue
        parsed: dict[str, Any] = {"downstream_fold": _parse_int(row.get("fold"))}
        metrics = row.get("metrics", row)
        if isinstance(metrics, dict):
            for split in ("valid", "test"):
                for key in METRIC_KEYS:
                    value = _metric_from_payload(metrics, split, key)
                    if value is not None:
                        parsed[f"downstream_{split}_{key}"] = value
            if _parse_int(metrics.get("best_epoch")) is not None:
                parsed["downstream_best_epoch"] = _parse_int(metrics.get("best_epoch"))
        if len(parsed) > 1:
            rows.append(parsed)
    aggregates = payload.get("aggregates")
    if not rows and isinstance(aggregates, dict):
        parsed = {"downstream_fold": None}
        for split in ("valid", "test"):
            for key in METRIC_KEYS:
                for suffix in ("mean", ""):
                    name = f"{split}_{key}_{suffix}".rstrip("_")
                    value = _parse_float(aggregates.get(name))
                    if value is not None:
                        parsed[f"downstream_{split}_{key}"] = value
                        break
        if len(parsed) > 1:
            rows.append(parsed)
    return rows


def _candidate_downstream_dirs(
    run_id: int,
    hint: DownstreamHint | None,
    runs_root: Path,
) -> list[Path]:
    dirs: list[Path] = []
    if hint and hint.downstream_run_dir is not None:
        dirs.append(hint.downstream_run_dir)
    pattern = f"*joint_fresh_id{run_id}*"
    dirs.extend(path for path in sorted(runs_root.glob(pattern)) if path.is_dir())
    seen: set[Path] = set()
    out: list[Path] = []
    for path in dirs:
        resolved = path if path.is_absolute() else PROJECT_ROOT / path
        if resolved not in seen:
            seen.add(resolved)
            out.append(resolved)
    return out


def _aggregate_fold_metrics(fold_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_rows:
        return {}
    out: dict[str, Any] = {}
    for col in RUN_LEVEL_METRIC_COLS:
        if col in {"downstream_status", "downstream_fold"}:
            continue
        values = [_parse_float(row.get(col)) for row in fold_rows]
        values = [value for value in values if value is not None and math.isfinite(value)]
        if values:
            out[col] = float(np.mean(values))
            if len(values) > 1:
                out[f"{col}_std"] = float(np.std(values, ddof=1))
    folds = [row.get("downstream_fold") for row in fold_rows if row.get("downstream_fold") is not None]
    out["downstream_fold"] = ",".join(str(int(fold)) for fold in sorted(set(folds))) if folds else None
    out["downstream_status"] = "fold0_only" if len(fold_rows) == 1 and folds == [0] else "ok"
    return out


def load_downstream_metrics(
    run_id: int,
    runs_root: Path,
    hint: DownstreamHint | None,
    warnings: WarningCollector,
) -> tuple[dict[str, Any], list[str]]:
    dirs = _candidate_downstream_dirs(run_id, hint, runs_root)
    used_dirs: list[str] = []
    fold_rows: list[dict[str, Any]] = []
    for directory in dirs:
        if not directory.exists():
            warnings.missing(directory, f"downstream run for id={run_id}")
            continue
        used_dirs.append(str(directory))
        if (directory / "cv_summary.json").exists():
            fold_rows.extend(_parse_cv_summary(directory / "cv_summary.json"))
        direct_metrics = _parse_fold_metrics(directory)
        if direct_metrics is not None:
            fold_rows.append(direct_metrics)
        folds_root = directory / "folds"
        if folds_root.exists():
            for fold_dir in sorted(path for path in folds_root.glob("fold_*") if path.is_dir()):
                parsed = _parse_fold_metrics(fold_dir)
                if parsed is not None:
                    fold_rows.append(parsed)

    deduped: dict[Any, dict[str, Any]] = {}
    for row in fold_rows:
        key = row.get("downstream_fold")
        if key not in deduped or len(row) > len(deduped[key]):
            deduped[key] = row
    fold_rows = list(deduped.values())
    metrics = _aggregate_fold_metrics(fold_rows)
    if hint and hint.metrics:
        for key, value in hint.metrics.items():
            metrics.setdefault(key, value)
        if not metrics.get("downstream_status") and hint.status:
            metrics["downstream_status"] = hint.status
    if not metrics:
        metrics = {col: np.nan for col in RUN_LEVEL_METRIC_COLS if col != "downstream_status"}
        metrics["downstream_status"] = "missing"
    else:
        for col in RUN_LEVEL_METRIC_COLS:
            metrics.setdefault(col, np.nan if col != "downstream_status" else "unknown")
    return metrics, used_dirs


def _safe_mean(values: pd.Series) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric.mean()) if not numeric.empty else np.nan


def aggregate_run(
    run: PredictorRun,
    prediction_path: Path | None,
    los_info: dict[str, Any],
    head_rows: list[dict[str, Any]],
    downstream_metrics: dict[str, Any],
    downstream_dirs: list[str],
    top_k: int,
    split: str,
) -> dict[str, Any]:
    head_df = pd.DataFrame(head_rows)
    sorted_dv = head_df.sort_values("dV_D", ascending=False, na_position="last")
    sorted_js = head_df.sort_values("js_D", ascending=False, na_position="last")
    row: dict[str, Any] = {
        "id": run.id,
        "predictor_run_name": run.predictor_run_name,
        "predictor_run_dir": str(run.predictor_run_dir) if run.predictor_run_dir else "",
        "prediction_path": str(prediction_path) if prediction_path else "",
        "direction": run.direction,
        "heads_mode": run.heads_mode,
        "los_mode": run.los_mode,
        "detach": run.detach,
        "lambda_joint": run.lambda_joint,
        "lambda_los": run.lambda_los,
        "lambda_entropy": run.lambda_entropy,
        "seed": run.seed,
        "fold": run.fold,
        "split": split,
        "n_rows": int(los_info.get("rows_used", 0)),
        "n_los_bins": int(los_info.get("n_los_bins", 0)),
        "resolved_los_bins": los_info.get("los_bins"),
        "mean_dV_D": _safe_mean(head_df.get("dV_D", pd.Series(dtype=float))),
        "median_dV_D": float(pd.to_numeric(head_df.get("dV_D", pd.Series(dtype=float)), errors="coerce").median()),
        "max_dV_D": float(pd.to_numeric(head_df.get("dV_D", pd.Series(dtype=float)), errors="coerce").max()),
        "mean_abs_dV_D": _safe_mean(head_df.get("abs_dV_D", pd.Series(dtype=float))),
        "mean_js_D": _safe_mean(head_df.get("js_D", pd.Series(dtype=float))),
        "max_js_D": float(pd.to_numeric(head_df.get("js_D", pd.Series(dtype=float)), errors="coerce").max()),
        "mean_dV_LOS": _safe_mean(head_df.get("dV_LOS", pd.Series(dtype=float))),
        "max_dV_LOS": float(pd.to_numeric(head_df.get("dV_LOS", pd.Series(dtype=float)), errors="coerce").max()),
        "downstream_dirs": ";".join(downstream_dirs),
    }
    for idx in range(1, top_k + 1):
        if idx <= len(sorted_dv):
            item = sorted_dv.iloc[idx - 1]
            row[f"top{idx}_dV_D_head"] = item.get("head")
            row[f"top{idx}_dV_D"] = item.get("dV_D")
        else:
            row[f"top{idx}_dV_D_head"] = ""
            row[f"top{idx}_dV_D"] = np.nan
    for idx in range(1, min(3, top_k) + 1):
        if idx <= len(sorted_js):
            item = sorted_js.iloc[idx - 1]
            row[f"top{idx}_js_D_head"] = item.get("head")
            row[f"top{idx}_js_D"] = item.get("js_D")
        else:
            row[f"top{idx}_js_D_head"] = ""
            row[f"top{idx}_js_D"] = np.nan
    row.update(downstream_metrics)
    return row


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.6g}"
    text = str(value)
    return text.replace("|", "\\|")


def _df_to_md_table(df: pd.DataFrame, columns: list[str], max_rows: int | None = None) -> str:
    if max_rows is not None:
        df = df.head(max_rows)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(_format_cell(row.get(col)) for col in columns) + " |")
    if not rows:
        rows.append("| " + " | ".join([""] * len(columns)) + " |")
    return "\n".join([header, sep, *rows])


def _rank(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").rank(method="average")


def _corr_rows(run_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    auc = pd.to_numeric(run_df["downstream_test_auc"], errors="coerce")
    for metric in ("mean_dV_D", "max_dV_D", "mean_js_D", "mean_abs_dV_D"):
        lhs = pd.to_numeric(run_df[metric], errors="coerce")
        valid = lhs.notna() & auc.notna()
        n = int(valid.sum())
        row: dict[str, Any] = {"metric": metric, "n": n, "pearson_vs_test_auc": np.nan, "spearman_vs_test_auc": np.nan}
        if n >= 3:
            row["pearson_vs_test_auc"] = float(np.corrcoef(lhs[valid], auc[valid])[0, 1])
            row["spearman_vs_test_auc"] = float(np.corrcoef(_rank(lhs[valid]), _rank(auc[valid]))[0, 1])
        rows.append(row)
    return rows


def _discordant_pairs(run_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = run_df.dropna(subset=["mean_dV_D", "downstream_test_auc"])
    for _, lower in valid.iterrows():
        for _, higher in valid.iterrows():
            if lower["id"] == higher["id"]:
                continue
            if float(lower["mean_dV_D"]) < float(higher["mean_dV_D"]) and float(lower["downstream_test_auc"]) <= float(higher["downstream_test_auc"]):
                rows.append(
                    {
                        "lower_dv_id": lower["id"],
                        "higher_dv_id": higher["id"],
                        "lower_mean_dV_D": lower["mean_dV_D"],
                        "higher_mean_dV_D": higher["mean_dV_D"],
                        "lower_test_auc": lower["downstream_test_auc"],
                        "higher_test_auc": higher["downstream_test_auc"],
                        "delta_mean_dV_D": float(lower["mean_dV_D"]) - float(higher["mean_dV_D"]),
                        "delta_auc": float(lower["downstream_test_auc"]) - float(higher["downstream_test_auc"]),
                    }
                )
    return pd.DataFrame(rows)


def write_markdown_report(
    path: Path,
    run_df: pd.DataFrame,
    head_df: pd.DataFrame,
    args: argparse.Namespace,
    created_files: list[str],
    warnings: WarningCollector,
) -> None:
    table_cols = [
        "id",
        "predictor_run_name",
        "mean_dV_D",
        "max_dV_D",
        "mean_abs_dV_D",
        "mean_js_D",
        "downstream_test_auc",
        "downstream_test_f1",
        "downstream_test_acc",
        "downstream_status",
    ]
    top_head_cols = ["id", "head", "dV_D", "abs_dV_D", "js_D", "dV_LOS", "acc"]
    corr_df = pd.DataFrame(_corr_rows(run_df)) if not run_df.empty else pd.DataFrame(columns=["metric", "n", "pearson_vs_test_auc", "spearman_vs_test_auc"])
    discordant = _discordant_pairs(run_df) if not run_df.empty else pd.DataFrame()
    top_heads = (
        head_df.sort_values(["id", "dV_D"], ascending=[True, False]).groupby("id").head(int(args.top_k_heads))
        if not head_df.empty
        else pd.DataFrame(columns=top_head_cols)
    )
    included_ids = args.run_ids or (run_df["id"].tolist() if "id" in run_df else [])

    parts = [
        "# Joint dV_D / Downstream AUC Analysis",
        "",
        f"- analysis timestamp: {datetime.now(timezone.utc).isoformat()}",
        f"- included run IDs: {', '.join(map(str, included_ids))}",
        f"- split: {args.split}",
        f"- requested LOS bins: {args.los_bins}",
        f"- created files: {', '.join(created_files)}",
        "",
        "## Sorted by downstream_test_auc descending",
        _df_to_md_table(run_df.sort_values("downstream_test_auc", ascending=False, na_position="last"), table_cols),
        "",
        "## Sorted by mean_dV_D ascending",
        _df_to_md_table(run_df.sort_values("mean_dV_D", ascending=True, na_position="last"), table_cols),
        "",
        "## Sorted by max_dV_D descending",
        _df_to_md_table(run_df.sort_values("max_dV_D", ascending=False, na_position="last"), table_cols),
        "",
        "## Top dV_D Heads per Run",
        _df_to_md_table(top_heads, top_head_cols),
        "",
        "## Correlations",
        _df_to_md_table(corr_df, ["metric", "n", "pearson_vs_test_auc", "spearman_vs_test_auc"]),
        "",
        "## dV_D Improves but AUC Does Not",
        _df_to_md_table(discordant, list(discordant.columns) if not discordant.empty else ["lower_dv_id", "higher_dv_id", "delta_mean_dV_D", "delta_auc"]),
        "",
        "## Interpretation Helper",
        "- dV_D down + AUC up: joint regularization likely useful.",
        "- dV_D down + AUC unchanged: joint target may be wrong or downstream insensitive.",
        "- dV_D unchanged + AUC unchanged: regularization likely ineffective.",
        "- dV_D up + AUC up: dV_D is not the relevant bottleneck for that run.",
        "",
        "## File Paths Used",
        _df_to_md_table(run_df, ["id", "prediction_path", "predictor_run_dir", "downstream_dirs"]),
        "",
        "## Warnings",
        "\n".join(f"- {message}" for message in warnings.warnings) if warnings.warnings else "- none",
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return None


def write_manifest(
    path: Path,
    args: argparse.Namespace,
    created_files: list[str],
    run_df: pd.DataFrame,
    warnings: WarningCollector,
) -> None:
    payload = {
        "command_args": vars(args),
        "git_commit": _git_commit(),
        "created_files": created_files,
        "run_ids": run_df["id"].tolist() if "id" in run_df else [],
        "resolved_run_dirs": dict(zip(run_df["id"].astype(str), run_df["predictor_run_dir"])),
        "downstream_dirs": dict(zip(run_df["id"].astype(str), run_df["downstream_dirs"])),
        "warnings": warnings.warnings,
        "missing_files": warnings.missing_files,
    }
    path.write_text(json.dumps(_json_sanitize(payload), indent=2, sort_keys=True), encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a per-run joint dV_D/downstream AUC matching table.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--run-ids", type=int, nargs="*", default=None)
    parser.add_argument("--run-name-list", type=str, nargs="*", default=None)
    parser.add_argument("--run-map-csv", type=Path, default=None)
    parser.add_argument("--downstream-map", type=Path, default=None)
    parser.add_argument("--split", choices=sorted(VALID_SPLITS), default="test")
    parser.add_argument("--los-bins", choices=("coarse6", "raw37", "auto"), default="auto")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--analysis-name", type=str, default=None)
    parser.add_argument("--top-k-heads", type=int, default=5)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def _select_runs(args: argparse.Namespace, warnings: WarningCollector) -> tuple[list[PredictorRun], dict[int, DownstreamHint]]:
    runs_root = args.runs_root if args.runs_root.is_absolute() else PROJECT_ROOT / args.runs_root
    registry_runs: dict[int, PredictorRun] = {}
    registry_downstream: dict[int, DownstreamHint] = {}
    if args.registry_path is not None:
        registry_path = args.registry_path if args.registry_path.is_absolute() else PROJECT_ROOT / args.registry_path
        registry_runs, registry_downstream = parse_registry(registry_path, warnings)

    if args.run_map_csv is not None:
        run_map_path = args.run_map_csv if args.run_map_csv.is_absolute() else PROJECT_ROOT / args.run_map_csv
        run_map = load_run_map_csv(run_map_path, runs_root, warnings)
    else:
        run_map = registry_runs

    if not run_map and args.run_name_list:
        run_map = {
            idx + 1: PredictorRun(id=idx + 1, predictor_run_name=name, predictor_run_dir=_find_run_dir(runs_root, name))
            for idx, name in enumerate(args.run_name_list)
        }
    if not run_map:
        raise RuntimeError("No predictor runs resolved. Provide --registry-path, --run-map-csv, or --run-name-list.")

    selected_ids = list(dict.fromkeys(args.run_ids)) if args.run_ids else list(run_map.keys())
    selected: list[PredictorRun] = []
    for run_id in selected_ids:
        if run_id not in run_map:
            warnings.warn(f"Requested run id {run_id} is not present in resolved run map.")
            continue
        run = run_map[run_id]
        if run.predictor_run_dir is None:
            run.predictor_run_dir = _find_run_dir(runs_root, run.predictor_run_name)
        if run.predictor_run_dir is None:
            warnings.warn(f"Could not resolve predictor run directory for id={run_id}, name={run.predictor_run_name!r}.")
        selected.append(run)

    downstream = registry_downstream
    if args.downstream_map is not None:
        downstream_path = args.downstream_map if args.downstream_map.is_absolute() else PROJECT_ROOT / args.downstream_map
        csv_downstream = load_downstream_map_csv(downstream_path, warnings)
        downstream = {**registry_downstream, **csv_downstream}
    return selected, downstream


def build_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, WarningCollector, list[str]]:
    warnings = WarningCollector(strict=bool(args.strict))
    runs_root = args.runs_root if args.runs_root.is_absolute() else PROJECT_ROOT / args.runs_root
    selected_runs, downstream_hints = _select_runs(args, warnings)
    head_rows_all: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []

    for run in selected_runs:
        run_id_int = int(run.id) if str(run.id).isdigit() else run.id
        if run.predictor_run_dir is None or not run.predictor_run_dir.exists():
            warnings.warn(f"Skipping id={run.id}: predictor run directory is missing.")
            continue
        prediction_path = locate_prediction_file(run.predictor_run_dir, args.split, warnings)
        if prediction_path is None:
            continue
        try:
            raw_df = pd.read_csv(prediction_path)
            norm_df, heads, los_info = normalize_prediction_dataframe(
                raw_df,
                run,
                args.los_bins,
                warnings,
                context=f"id={run.id} {run.predictor_run_name}",
            )
            head_rows: list[dict[str, Any]] = []
            for head in heads:
                metrics = decompose_head_drift(norm_df, head, int(los_info["n_los_bins"]))
                row = {
                    "id": run.id,
                    "predictor_run_name": run.predictor_run_name,
                    "predictor_run_dir": str(run.predictor_run_dir),
                    "prediction_path": str(prediction_path),
                    "direction": run.direction,
                    "heads_mode": run.heads_mode,
                    "los_mode": run.los_mode,
                    "detach": run.detach,
                    "lambda_joint": run.lambda_joint,
                    "lambda_los": run.lambda_los,
                    "lambda_entropy": run.lambda_entropy,
                    "seed": run.seed,
                    "fold": run.fold,
                    "split": args.split,
                    "n_los_bins": los_info["n_los_bins"],
                    "resolved_los_bins": los_info["los_bins"],
                    **metrics,
                }
                head_rows.append(row)
            downstream_metrics, downstream_dirs = load_downstream_metrics(
                int(run_id_int),
                runs_root,
                downstream_hints.get(int(run_id_int)) if isinstance(run_id_int, int) else None,
                warnings,
            )
            for row in head_rows:
                row.update(downstream_metrics)
            head_rows_all.extend(head_rows)
            run_rows.append(
                aggregate_run(
                    run,
                    prediction_path,
                    los_info,
                    head_rows,
                    downstream_metrics,
                    downstream_dirs,
                    int(args.top_k_heads),
                    args.split,
                )
            )
        except Exception as exc:
            message = f"id={run.id} failed: {exc}"
            if args.strict:
                raise
            warnings.warn(message)

    return pd.DataFrame(run_rows), pd.DataFrame(head_rows_all), warnings, []


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out_dir is None:
        analysis_name = args.analysis_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        args.out_dir = Path("reports") / "joint_dv_auc" / analysis_name
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_df, head_df, warnings, _ = build_tables(args)
    run_csv = out_dir / "run_level_dv_auc.csv"
    head_csv = out_dir / "head_level_dv_auc.csv"
    report_md = out_dir / "run_level_dv_auc.md"
    manifest_json = out_dir / "manifest.json"
    run_df.to_csv(run_csv, index=False)
    head_df.to_csv(head_csv, index=False)
    created_files = [str(run_csv), str(head_csv), str(report_md), str(manifest_json)]
    write_markdown_report(report_md, run_df, head_df, args, created_files, warnings)
    write_manifest(manifest_json, args, created_files, run_df, warnings)

    print(f"[done] wrote {run_csv}")
    print(f"[done] wrote {head_csv}")
    print(f"[done] wrote {report_md}")
    print(f"[done] wrote {manifest_json}")
    if not run_df.empty:
        summary_cols = ["id", "mean_dV_D", "max_dV_D", "mean_js_D", "downstream_test_auc", "downstream_test_f1", "downstream_status"]
        print(run_df[summary_cols].sort_values("id").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
