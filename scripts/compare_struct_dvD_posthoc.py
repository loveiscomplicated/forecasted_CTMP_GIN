from __future__ import annotations

import argparse
import json
import math
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

from src.analysis.joint_drift_decomposition import decompose_head_drift  # noqa: E402
from src.analysis.los_breakdown_diagnostics import (  # noqa: E402
    BREAKDOWN9_LABELS,
    COARSE6_LABELS,
    LOS_PRED_BIN_COL,
    LOS_TRUE_BIN_COL,
    WarningCollector,
    binary_metrics_from_arrays,
    build_los_confusion_tables,
    build_middle_to_long_flow_summary,
    find_joint_prediction_file,
    find_outcome_prediction_file,
    infer_los_scheme,
    load_fold_metrics,
    los_scheme_labels,
    normalize_joint_prediction_df,
    normalize_los_values,
    normalize_outcome_prediction_df,
)
from src.models.discharge_predictor.risk_heads import (  # noqa: E402
    get_named_risk_head_set,
)


RISK_HEADS = (
    "FREQ_ATND_SELF_HELP_D",
    "SUB1_D",
    "FREQ1_D",
    "FREQ2_D",
    "EMPLOY_D",
    "DETNLF_D",
)
REFERENCE_TEST_AUC_9BIN = 0.8877
RUN_METRIC_COLUMNS = (
    "run_name",
    "run_type",
    "run_dir",
    "config_path",
    "fold",
    "best_epoch",
    "best_valid_metric",
    "valid_loss",
    "valid_acc",
    "valid_precision",
    "valid_recall",
    "valid_f1",
    "valid_auc",
    "test_loss",
    "test_acc",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_auc",
    "status",
    "delta_valid_auc_vs_baseline",
    "delta_test_auc_vs_baseline",
    "delta_valid_auc_vs_9bin_reference",
    "delta_test_auc_vs_0_8877_reference",
)


@dataclass
class DiagnosticWarningLog:
    strict: bool = False
    warnings: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    def warn(self, message: str) -> None:
        if self.strict:
            raise RuntimeError(message)
        self.warnings.append(message)
        print(f"[warning] {message}", file=sys.stderr)

    def missing(self, path: Path | str, context: str) -> None:
        text = f"{context}: missing {path}"
        self.missing_files.append(str(path))
        self.warn(text)


@dataclass
class ComparisonRun:
    run_name: str
    run_type: str
    run_dir: Path


@dataclass
class JointPredictionSource:
    frame: pd.DataFrame
    source: str
    path: Path | None
    status: str = "ok"


@dataclass
class OutcomeSource:
    frame: pd.DataFrame | None
    source: str
    path: Path | None
    status: str


def _resolve_path(path: Path | str, *, base: Path = PROJECT_ROOT) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else base / candidate


def _safe_float(value: Any) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return np.nan
    return parsed if math.isfinite(parsed) else np.nan


def _safe_int(value: Any) -> int | None:
    parsed = _safe_float(value)
    return None if math.isnan(parsed) else int(parsed)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, tuple):
        return [_json_sanitize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def _split_alias(split: str) -> str:
    return {"valid": "val", "test": "test", "train": "train"}.get(split, split)


def _diagnostic_split_key(split: str) -> str:
    return {"valid": "gnn_val", "test": "outer_test"}.get(split, split)


def _cache_split_key(split: str) -> str:
    return {"valid": "gnn_val", "test": "outer_test"}.get(split, split)


def _df_to_markdown(df: pd.DataFrame, columns: list[str] | None = None, *, max_rows: int | None = None) -> str:
    if df.empty:
        cols = columns or list(df.columns)
        return "|" + "|".join(cols) + "|\n|" + "|".join("---" for _ in cols) + "|"
    view = df.copy()
    if columns is not None:
        for col in columns:
            if col not in view.columns:
                view[col] = np.nan
        view = view[columns]
    if max_rows is not None:
        view = view.head(max_rows)
    rows = [list(view.columns)]
    for record in view.to_dict(orient="records"):
        rows.append([_format_cell(record.get(col)) for col in view.columns])
    widths = [max(len(str(row[idx])) for row in rows) for idx in range(len(rows[0]))]
    header = "| " + " | ".join(str(rows[0][idx]).ljust(widths[idx]) for idx in range(len(widths))) + " |"
    sep = "| " + " | ".join("-" * widths[idx] for idx in range(len(widths))) + " |"
    body = ["| " + " | ".join(str(row[idx]).ljust(widths[idx]) for idx in range(len(widths))) + " |" for row in rows[1:]]
    return "\n".join([header, sep, *body])


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.6g}"
    if isinstance(value, (np.floating,)):
        parsed = float(value)
        return "nan" if math.isnan(parsed) else f"{parsed:.6g}"
    return str(value).replace("\n", " ")


def _write_md_table(path: Path, title: str, df: pd.DataFrame, columns: list[str] | None = None) -> None:
    path.write_text(f"# {title}\n\n{_df_to_markdown(df, columns)}\n", encoding="utf-8")


def try_git_commit() -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def robust_candidate_run_names(runs_root: Path) -> list[str]:
    candidates: set[str] = set()
    tokens = ("robust", "top3", "lambda003")
    for path in runs_root.rglob("*"):
        if not path.is_dir():
            continue
        name = path.name
        lowered = name.lower()
        if any(token in lowered for token in tokens):
            if path.name == "fold_0" and path.parent.name == "folds":
                candidates.add(path.parent.parent.name)
            else:
                candidates.add(name)
    return sorted(candidates)


def discover_fold0_run_dir(runs_root: Path, name_contains: str, *, warnings: DiagnosticWarningLog) -> Path:
    matches: list[Path] = []
    needle = name_contains.lower()
    for fold_dir in runs_root.rglob("fold_0"):
        if not fold_dir.is_dir() or fold_dir.parent.name != "folds":
            continue
        run_root = fold_dir.parent.parent
        if needle in run_root.name.lower():
            matches.append(fold_dir)
    if not matches:
        candidates = robust_candidate_run_names(runs_root)
        message = (
            f"No fold_0 run found with parent run name containing {name_contains!r}. "
            f"Candidate robust/top3/lambda003 run names: {candidates or 'none'}"
        )
        warnings.warn(message)
        raise SystemExit(message)
    if len(matches) > 1:
        warnings.warn(
            f"Multiple fold_0 runs matched {name_contains!r}; using {matches[0]}. "
            f"All matches: {[str(path) for path in matches]}"
        )
    return sorted(matches)[0]


def resolve_comparison_runs(args: argparse.Namespace, warnings: DiagnosticWarningLog) -> list[ComparisonRun]:
    runs_root = _resolve_path(args.runs_root)
    baseline = _resolve_path(args.baseline_run_dir)
    top6 = _resolve_path(args.top6_run_dir)
    robust = _resolve_path(args.robust_run_dir) if args.robust_run_dir else None
    if robust is None:
        if not args.robust_run_name_contains:
            raise SystemExit("Provide --robust-run-dir or --robust-run-name-contains.")
        robust = discover_fold0_run_dir(runs_root, args.robust_run_name_contains, warnings=warnings)
    runs = [
        ComparisonRun("baseline_id26", "baseline_coarse6", baseline),
        ComparisonRun("struct_top6_l001", "struct_dvD_top6_lambda001", top6),
        ComparisonRun("struct_robust_top3_l003", "struct_robust_top3_lambda003", robust),
    ]
    for run in runs:
        if not run.run_dir.exists():
            warnings.missing(run.run_dir, f"{run.run_name} run dir")
            raise SystemExit(f"Missing required run directory for {run.run_name}: {run.run_dir}")
    return runs


def _extract_overall_metrics(fold_dir: Path, split: str) -> dict[str, float]:
    payload = _read_json(fold_dir / "diagnostics" / "forecasted_gnn_performance" / "overall_metrics.json")
    if not payload:
        return {}
    split_key = _diagnostic_split_key(split)
    overall = (payload.get("splits") or {}).get(split_key, {}).get("overall_metrics", {})
    if not isinstance(overall, dict):
        return {}
    prefix = "valid" if split == "valid" else "test"
    out: dict[str, float] = {}
    for source_key, target_key in (
        ("loss", "loss"),
        ("accuracy", "acc"),
        ("precision", "precision"),
        ("recall", "recall"),
        ("f1", "f1"),
        ("auc", "auc"),
    ):
        if source_key in overall:
            out[f"{prefix}_{target_key}"] = _safe_float(overall.get(source_key))
    return out


def load_run_metrics(run: ComparisonRun) -> dict[str, Any]:
    metrics = load_fold_metrics(run.run_dir)
    for split in ("valid", "test"):
        for key, value in _extract_overall_metrics(run.run_dir, split).items():
            metrics[key] = value
    config_path = run.run_dir / "config.final.yaml"
    if not config_path.exists() and (run.run_dir / "joint_predictor" / "config.final.yaml").exists():
        config_path = run.run_dir / "joint_predictor" / "config.final.yaml"
    row: dict[str, Any] = {
        "run_name": run.run_name,
        "run_type": run.run_type,
        "run_dir": str(run.run_dir),
        "config_path": str(config_path) if config_path.exists() else "",
    }
    for key in RUN_METRIC_COLUMNS:
        if key not in row:
            row[key] = metrics.get(key)
    return row


def add_metric_deltas(run_metrics: pd.DataFrame) -> pd.DataFrame:
    out = run_metrics.copy()
    baseline = out[out["run_name"] == "baseline_id26"]
    baseline_valid = _safe_float(baseline["valid_auc"].iloc[0]) if not baseline.empty else np.nan
    baseline_test = _safe_float(baseline["test_auc"].iloc[0]) if not baseline.empty else np.nan
    out["delta_valid_auc_vs_baseline"] = pd.to_numeric(out["valid_auc"], errors="coerce") - baseline_valid
    out["delta_test_auc_vs_baseline"] = pd.to_numeric(out["test_auc"], errors="coerce") - baseline_test
    out["delta_valid_auc_vs_9bin_reference"] = pd.to_numeric(out["valid_auc"], errors="coerce") - REFERENCE_TEST_AUC_9BIN
    out["delta_test_auc_vs_0_8877_reference"] = pd.to_numeric(out["test_auc"], errors="coerce") - REFERENCE_TEST_AUC_9BIN
    return out[list(RUN_METRIC_COLUMNS)]


def _load_struct_config(run_dir: Path) -> dict[str, Any]:
    cfg = _read_yaml(run_dir / "joint_predictor" / "config.final.yaml")
    if not cfg:
        cfg = _read_yaml(run_dir / "config.final.yaml")
    struct_cfg = dict(cfg.get("joint_struct_loss") or {})
    joint_cfg = dict(cfg.get("joint_predictor") or {})
    if "risk_head_set" in struct_cfg and "resolved_risk_heads" not in struct_cfg:
        try:
            struct_cfg["resolved_risk_heads"] = get_named_risk_head_set(str(struct_cfg["risk_head_set"]))
        except Exception:
            pass
    if "joint_heads" not in struct_cfg and joint_cfg.get("joint_heads") is not None:
        struct_cfg["joint_heads"] = joint_cfg.get("joint_heads")
    return struct_cfg


def _load_checkpoint_struct_metadata(run_dir: Path) -> dict[str, Any]:
    checkpoint_root = run_dir / "joint_predictor" / "checkpoints"
    if not checkpoint_root.exists():
        return {}
    checkpoints = sorted(checkpoint_root.glob("*.pt"))
    if not checkpoints:
        return {}
    try:
        import torch

        payload = torch.load(checkpoints[-1], map_location="cpu", weights_only=False)
    except Exception:
        return {}
    if isinstance(payload, dict):
        metadata = payload.get("joint_struct_loss")
        return metadata if isinstance(metadata, dict) else {}
    return {}


def load_struct_settings(run: ComparisonRun) -> dict[str, Any]:
    cfg = _load_struct_config(run.run_dir)
    checkpoint_meta = _load_checkpoint_struct_metadata(run.run_dir)
    merged = {**cfg, **checkpoint_meta}
    enabled = bool(merged.get("enabled", False)) and _safe_float(merged.get("lambda_struct", 0.0)) > 0.0
    return {
        "run_name": run.run_name,
        "struct_loss_enabled": bool(enabled),
        "lambda_struct": _safe_float(merged.get("lambda_struct", np.nan)),
        "loss_type": merged.get("loss_type"),
        "risk_head_set": merged.get("risk_head_set"),
        "resolved_risk_heads": merged.get("resolved_risk_heads") or [],
        "stopgrad_los": merged.get("stopgrad_los"),
        "use_ema": merged.get("use_ema"),
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def extract_struct_loss_trace(run: ComparisonRun, settings: dict[str, Any]) -> pd.DataFrame:
    rows = read_jsonl(run.run_dir / "joint_predictor" / "metrics.jsonl")
    trace_rows: list[dict[str, Any]] = []
    for row in rows:
        if "train_struct_loss" not in row and "valid_struct_loss" not in row and "valid_loss_struct" not in row:
            continue
        out: dict[str, Any] = {
            "run_name": run.run_name,
            "run_type": run.run_type,
            "epoch": row.get("epoch"),
            "train_loss": row.get("train_loss"),
            "valid_loss": row.get("valid_loss"),
            "train_struct_loss": row.get("train_struct_loss"),
            "valid_struct_loss": row.get("valid_struct_loss", row.get("valid_loss_struct")),
            "lambda_struct": row.get("lambda_struct", settings.get("lambda_struct")),
            "risk_head_set": settings.get("risk_head_set"),
            "resolved_risk_heads": ",".join(map(str, settings.get("resolved_risk_heads") or [])),
            "stopgrad_los": settings.get("stopgrad_los"),
            "loss_type": settings.get("loss_type"),
        }
        for key, value in row.items():
            if key.startswith("train_struct_") and key != "train_struct_loss":
                out[key] = value
            elif key.startswith("valid_struct_") and key != "valid_struct_loss":
                out[key] = value
            elif key.startswith("valid_loss_struct_"):
                head = key[len("valid_loss_struct_") :]
                out[f"valid_struct_{head}"] = value
        trace_rows.append(out)
    if trace_rows:
        return pd.DataFrame(trace_rows)
    return pd.DataFrame(
        [
            {
                "run_name": run.run_name,
                "run_type": run.run_type,
                "status": "missing_struct_loss_logs",
                "struct_loss_enabled": settings.get("struct_loss_enabled"),
                "risk_head_set": settings.get("risk_head_set"),
                "resolved_risk_heads": ",".join(map(str, settings.get("resolved_risk_heads") or [])),
            }
        ]
    )


def _loss_summary_row(run_name: str, metric: str, values: pd.Series, trace: pd.DataFrame) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    row: dict[str, Any] = {"run_name": run_name, "metric": metric}
    if numeric.empty:
        row.update({"status": "missing", "first": np.nan, "last": np.nan, "min": np.nan, "delta_last_first": np.nan, "decreased": np.nan})
        return row
    first_idx = numeric.index[0]
    last_idx = numeric.index[-1]
    min_idx = numeric.idxmin()
    row.update(
        {
            "status": "ok",
            "first_epoch": trace.loc[first_idx, "epoch"] if "epoch" in trace.columns else np.nan,
            "last_epoch": trace.loc[last_idx, "epoch"] if "epoch" in trace.columns else np.nan,
            "min_epoch": trace.loc[min_idx, "epoch"] if "epoch" in trace.columns else np.nan,
            "first": float(numeric.loc[first_idx]),
            "last": float(numeric.loc[last_idx]),
            "min": float(numeric.loc[min_idx]),
            "delta_last_first": float(numeric.loc[last_idx] - numeric.loc[first_idx]),
            "decreased": bool(numeric.loc[last_idx] < numeric.loc[first_idx]),
        }
    )
    return row


def summarize_struct_loss(trace_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if trace_df.empty:
        return pd.DataFrame()
    for run_name, group in trace_df.groupby("run_name", dropna=False):
        if "status" in group.columns and group["status"].eq("missing_struct_loss_logs").all():
            rows.append({"run_name": run_name, "metric": "train_struct_loss", "status": "missing_struct_loss_logs"})
            continue
        for metric in sorted(col for col in group.columns if col.startswith("train_struct") or col.startswith("valid_struct")):
            rows.append(_loss_summary_row(str(run_name), metric, group[metric], group))
        if "train_struct_loss" in group.columns and "train_loss" in group.columns:
            lam = pd.to_numeric(group.get("lambda_struct"), errors="coerce")
            struct = pd.to_numeric(group["train_struct_loss"], errors="coerce")
            total = pd.to_numeric(group["train_loss"], errors="coerce")
            ratio = (lam * struct) / total
            ratio_row = _loss_summary_row(str(run_name), "train_lambda_struct_loss_ratio", ratio, group)
            rows.append(ratio_row)
        if "valid_struct_loss" in group.columns and "valid_loss" in group.columns:
            lam = pd.to_numeric(group.get("lambda_struct"), errors="coerce")
            struct = pd.to_numeric(group["valid_struct_loss"], errors="coerce")
            total = pd.to_numeric(group["valid_loss"], errors="coerce")
            ratio = (lam * struct) / total
            ratio_row = _loss_summary_row(str(run_name), "valid_lambda_struct_loss_ratio", ratio, group)
            rows.append(ratio_row)
    return pd.DataFrame(rows)


def load_joint_cache_prediction_frame(path: Path) -> pd.DataFrame:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows: dict[str, Any] = {"row_idx": payload["row_idx"].cpu().numpy().astype(int)}
    targets = payload.get("targets", {})
    d_targets = targets.get("d", {}) if isinstance(targets, dict) else {}
    d_pred = payload.get("final_d_pred", {})
    for head, values in d_targets.items():
        rows[f"true_{head}"] = values.cpu().numpy().astype(int)
        if head in d_pred:
            rows[f"pred_{head}"] = d_pred[head].cpu().numpy().astype(int)
    rows["true_los"] = targets.get("los_raw").cpu().numpy().astype(int)
    rows["pred_los"] = payload["final_los_pred"].cpu().numpy().astype(int)
    return pd.DataFrame(rows)


def load_joint_predictions(run: ComparisonRun, split: str, warnings: DiagnosticWarningLog) -> JointPredictionSource:
    existing_warnings = WarningCollector(strict=False)
    path = find_joint_prediction_file(run.run_dir, split, existing_warnings)
    if path is not None:
        return JointPredictionSource(pd.read_csv(path), "joint_prediction_csv", path)
    split_key = _cache_split_key(split)
    cache_candidates = [
        run.run_dir / "joint_predictor" / "joint_cache" / f"{split_key}.pt",
        run.run_dir / "joint_predictor" / "joint_cache" / f"{_split_alias(split)}.pt",
    ]
    for cache_path in cache_candidates:
        if cache_path.exists():
            return JointPredictionSource(load_joint_cache_prediction_frame(cache_path), "joint_cache", cache_path)
    warnings.missing(run.run_dir / "joint_predictor" / f"{_split_alias(split)}_predictions.csv", f"{run.run_name} joint predictions {split}")
    return JointPredictionSource(pd.DataFrame(), "missing", None, status="missing_joint_predictions")


def _compute_head_macro_f1(norm_df: pd.DataFrame, head: str) -> float:
    from sklearn.metrics import f1_score

    true_col = f"true_{head}"
    pred_col = f"pred_{head}"
    if true_col not in norm_df.columns or pred_col not in norm_df.columns:
        return np.nan
    y_true = pd.to_numeric(norm_df[true_col], errors="coerce")
    y_pred = pd.to_numeric(norm_df[pred_col], errors="coerce")
    mask = y_true.notna() & y_pred.notna()
    if not mask.any():
        return np.nan
    return float(f1_score(y_true[mask].astype(int), y_pred[mask].astype(int), average="macro", zero_division=0))


def compute_head_drift_for_run(
    run: ComparisonRun,
    split: str,
    joint: JointPredictionSource,
    scheme: str,
    warnings: DiagnosticWarningLog,
) -> pd.DataFrame:
    if joint.frame.empty:
        return pd.DataFrame()
    try:
        normalized, heads, _info = normalize_joint_prediction_df(joint.frame, target_scheme=scheme, context=f"{run.run_name}:{split}:{scheme}")
    except Exception as exc:
        warnings.warn(f"{run.run_name} {split} {scheme}: drift unavailable: {exc}")
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for head in heads:
        row = decompose_head_drift(normalized, head, len(los_scheme_labels(scheme)))
        row.update(
            {
                "run_name": run.run_name,
                "run_type": run.run_type,
                "split": split,
                "los_scheme": scheme,
                "source": joint.source,
                "source_path": str(joint.path) if joint.path else "",
                "support": row.get("n_rows"),
                "macro_f1": _compute_head_macro_f1(normalized, head),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_drift(head_df: pd.DataFrame, risk_heads: list[str]) -> pd.DataFrame:
    if head_df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for (run_name, split, los_scheme), group in head_df.groupby(["run_name", "split", "los_scheme"], dropna=False):
        risk = group[group["head"].isin(risk_heads)]
        sorted_dv = group.sort_values("dV_D", ascending=False, na_position="last").reset_index(drop=True)
        row: dict[str, Any] = {
            "run_name": run_name,
            "split": split,
            "los_scheme": los_scheme,
            "mean_dV_D_all_heads": _safe_float(pd.to_numeric(group["dV_D"], errors="coerce").mean()),
            "max_dV_D_all_heads": _safe_float(pd.to_numeric(group["dV_D"], errors="coerce").max()),
            "mean_abs_dV_D_all_heads": _safe_float(pd.to_numeric(group["abs_dV_D"], errors="coerce").mean()),
            "mean_js_D_all_heads": _safe_float(pd.to_numeric(group["js_D"], errors="coerce").mean()),
            "mean_dV_D_risk_heads": _safe_float(pd.to_numeric(risk["dV_D"], errors="coerce").mean()) if not risk.empty else np.nan,
            "max_dV_D_risk_heads": _safe_float(pd.to_numeric(risk["dV_D"], errors="coerce").max()) if not risk.empty else np.nan,
            "mean_abs_dV_D_risk_heads": _safe_float(pd.to_numeric(risk["abs_dV_D"], errors="coerce").mean()) if not risk.empty else np.nan,
            "mean_js_D_risk_heads": _safe_float(pd.to_numeric(risk["js_D"], errors="coerce").mean()) if not risk.empty else np.nan,
        }
        for idx in range(3):
            if idx < len(sorted_dv):
                row[f"top{idx + 1}_dV_D_head"] = sorted_dv.loc[idx, "head"]
                row[f"top{idx + 1}_dV_D"] = sorted_dv.loc[idx, "dV_D"]
            else:
                row[f"top{idx + 1}_dV_D_head"] = ""
                row[f"top{idx + 1}_dV_D"] = np.nan
        rows.append(row)
    summary = pd.DataFrame(rows)
    for split in summary["split"].dropna().unique().tolist():
        for scheme in summary["los_scheme"].dropna().unique().tolist():
            mask = summary["split"].eq(split) & summary["los_scheme"].eq(scheme)
            baseline = summary[mask & summary["run_name"].eq("baseline_id26")]
            if baseline.empty:
                continue
            base = baseline.iloc[0]
            for metric, delta_col in (
                ("mean_dV_D_all_heads", "delta_mean_dV_D_vs_baseline"),
                ("max_dV_D_all_heads", "delta_max_dV_D_vs_baseline"),
                ("mean_dV_D_risk_heads", "delta_risk_mean_dV_D_vs_baseline"),
                ("mean_js_D_risk_heads", "delta_risk_js_D_vs_baseline"),
            ):
                summary.loc[mask, delta_col] = pd.to_numeric(summary.loc[mask, metric], errors="coerce") - _safe_float(base.get(metric))
    return summary


def risk_head_accuracy_comparison(head_df: pd.DataFrame, risk_heads: list[str]) -> pd.DataFrame:
    if head_df.empty:
        return pd.DataFrame()
    out = head_df[head_df["head"].isin(risk_heads)].copy()
    cols = [
        "run_name",
        "split",
        "los_scheme",
        "head",
        "acc",
        "macro_f1",
        "support",
        "dV_D",
        "js_D",
    ]
    out = out[cols]
    out["delta_acc_vs_baseline"] = np.nan
    out["delta_macro_f1_vs_baseline"] = np.nan
    for (split, scheme, head), group in out.groupby(["split", "los_scheme", "head"], dropna=False):
        base = group[group["run_name"].eq("baseline_id26")]
        if base.empty:
            continue
        base_acc = _safe_float(base["acc"].iloc[0])
        base_f1 = _safe_float(base["macro_f1"].iloc[0])
        mask = out["split"].eq(split) & out["los_scheme"].eq(scheme) & out["head"].eq(head)
        out.loc[mask, "delta_acc_vs_baseline"] = pd.to_numeric(out.loc[mask, "acc"], errors="coerce") - base_acc
        out.loc[mask, "delta_macro_f1_vs_baseline"] = pd.to_numeric(out.loc[mask, "macro_f1"], errors="coerce") - base_f1
    return out


def _prediction_uses_coarse_class_ids(joint_df: pd.DataFrame) -> bool:
    if "pred_los" not in joint_df.columns:
        return False
    values = pd.to_numeric(joint_df["pred_los"], errors="coerce").dropna().astype(int)
    return not values.empty and values.min() >= 0 and values.max() <= 5


def load_outcome_source(run: ComparisonRun, warnings: DiagnosticWarningLog) -> OutcomeSource:
    path = find_outcome_prediction_file(run.run_dir)
    if path is not None:
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        return OutcomeSource(normalize_outcome_prediction_df(frame, str(path)), "per_row_outcome_predictions", path, "ok")
    warnings.missing(run.run_dir / "diagnostic_predictions.csv", f"{run.run_name} per-row downstream outcome predictions")
    return OutcomeSource(None, "missing", None, "missing_outcome_predictions")


def aggregate_los_group_metrics(run: ComparisonRun, split: str) -> pd.DataFrame:
    path = run.run_dir / "diagnostics" / "forecasted_gnn_performance" / "los_group_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    split_key = _diagnostic_split_key(split)
    if "split" in df.columns:
        df = df[df["split"].eq(split_key)]
    if "los_space" in df.columns:
        df = df[df["los_space"].eq("coarse6")]
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        rows.append(
            {
                "run_name": run.run_name,
                "split": split,
                "los_basis": "true_los_bin",
                "los_scheme": "coarse6",
                "los_bin": record.get("los_label"),
                "los_bin_label": record.get("los_label_name"),
                "support": record.get("support"),
                "positive_count": record.get("positive_count"),
                "positive_rate": _safe_float(record.get("positive_count")) / _safe_float(record.get("support")) if _safe_float(record.get("support")) else np.nan,
                "acc": record.get("accuracy"),
                "precision": record.get("precision"),
                "recall": record.get("recall"),
                "f1": record.get("f1"),
                "auc": record.get("auc"),
                "predicted_positive_rate": (_safe_float(record.get("tp")) + _safe_float(record.get("fp"))) / _safe_float(record.get("support")) if _safe_float(record.get("support")) else np.nan,
                "status": "aggregate_los_group_metrics",
                "source": str(path),
            }
        )
    return pd.DataFrame(rows)


def _outcome_metric_rows(
    df: pd.DataFrame,
    labels: dict[int, str],
    col: str,
    *,
    run_name: str,
    split: str,
    basis: str,
    scheme: str,
    status: str,
    source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for los_bin, label in labels.items():
        mask = df[col].eq(los_bin)
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
            "acc": metrics.pop("accuracy"),
            "predicted_positive_rate": float(np.mean(df.loc[mask, "y_pred"] == 1)) if int(mask.sum()) else np.nan,
            "status": status,
            "source": source,
        }
        row.update(metrics)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_outcome_diagnostics(
    run: ComparisonRun,
    split: str,
    joint: JointPredictionSource,
    outcome: OutcomeSource,
    los_schemes: list[str],
    warnings: DiagnosticWarningLog,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    true_frames: list[pd.DataFrame] = []
    pred_frames: list[pd.DataFrame] = []
    aggregate = aggregate_los_group_metrics(run, split)
    if not aggregate.empty:
        true_frames.append(aggregate)
    if outcome.frame is None or joint.frame.empty:
        for scheme in los_schemes:
            normalized_scheme = "breakdown9" if scheme == "breakdown9_true" else scheme
            labels = los_scheme_labels(normalized_scheme)
            for basis, frames in (("true_los_bin", true_frames), ("pred_los_bin", pred_frames)):
                if not (scheme == "coarse6" and not aggregate.empty and basis == "true_los_bin"):
                    frames.append(
                        pd.DataFrame(
                            [
                                {
                                    "run_name": run.run_name,
                                    "split": split,
                                    "los_basis": basis,
                                    "los_scheme": normalized_scheme,
                                    "los_bin": los_bin,
                                    "los_bin_label": label,
                                    "status": outcome.status if outcome.frame is None else "missing_joint_predictions",
                                    "source": outcome.source,
                                }
                                for los_bin, label in labels.items()
                            ]
                        )
                    )
        return pd.concat(true_frames, ignore_index=True) if true_frames else pd.DataFrame(), pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()

    merged = outcome.frame.merge(joint.frame[["row_idx", "true_los", "pred_los"]], on="row_idx", how="inner")
    if merged.empty:
        warnings.warn(f"{run.run_name} {split}: no matching rows between outcome and joint predictions.")
        return pd.DataFrame(), pd.DataFrame()
    for scheme in los_schemes:
        if scheme == "coarse6":
            true_bins = normalize_los_values(pd.to_numeric(merged["true_los"], errors="coerce").to_numpy(dtype=np.int64), "coarse6", f"{run.run_name}:{split}:true")
            pred_bins = normalize_los_values(pd.to_numeric(merged["pred_los"], errors="coerce").to_numpy(dtype=np.int64), "coarse6", f"{run.run_name}:{split}:pred")
            work = merged.copy()
            work[LOS_TRUE_BIN_COL] = true_bins
            work[LOS_PRED_BIN_COL] = pred_bins
            source_text = str(outcome.path) if outcome.path else outcome.source
            true_frames.append(_outcome_metric_rows(work, COARSE6_LABELS, LOS_TRUE_BIN_COL, run_name=run.run_name, split=split, basis="true_los_bin", scheme="coarse6", status="ok", source=source_text))
            pred_frames.append(_outcome_metric_rows(work, COARSE6_LABELS, LOS_PRED_BIN_COL, run_name=run.run_name, split=split, basis="pred_los_bin", scheme="coarse6", status="ok", source=source_text))
        elif scheme == "breakdown9_true":
            true_bins = normalize_los_values(pd.to_numeric(merged["true_los"], errors="coerce").to_numpy(dtype=np.int64), "breakdown9", f"{run.run_name}:{split}:true_breakdown9")
            work = merged.copy()
            work[LOS_TRUE_BIN_COL] = true_bins
            source_text = str(outcome.path) if outcome.path else outcome.source
            true_frames.append(_outcome_metric_rows(work, BREAKDOWN9_LABELS, LOS_TRUE_BIN_COL, run_name=run.run_name, split=split, basis="true_los_bin", scheme="breakdown9", status="ok", source=source_text))
            pred_status = "representative_limited" if _prediction_uses_coarse_class_ids(joint.frame) else "unavailable"
            pred_frames.append(
                pd.DataFrame(
                    [
                        {
                            "run_name": run.run_name,
                            "split": split,
                            "los_basis": "pred_los_bin",
                            "los_scheme": "breakdown9",
                            "los_bin": los_bin,
                            "los_bin_label": label,
                            "status": pred_status,
                            "source": joint.source,
                        }
                        for los_bin, label in BREAKDOWN9_LABELS.items()
                    ]
                )
            )
    true_df = pd.concat(true_frames, ignore_index=True) if true_frames else pd.DataFrame()
    pred_df = pd.concat(pred_frames, ignore_index=True) if pred_frames else pd.DataFrame()
    return true_df, pred_df


def build_long_stay_comparison(true_metrics: pd.DataFrame) -> pd.DataFrame:
    if true_metrics.empty:
        return pd.DataFrame()
    keep = true_metrics["los_basis"].eq("true_los_bin") & true_metrics["los_bin_label"].isin(["29-37", "29-31", "32-33", "34-35", "36-37"])
    rows = true_metrics.loc[keep].copy()
    if rows.empty:
        return rows
    rows = rows.rename(columns={"los_bin_label": "bin_label"})
    baseline = rows[rows["run_name"].eq("baseline_id26")].set_index(["split", "los_scheme", "bin_label"])
    rows["delta_recall_vs_baseline"] = np.nan
    for idx, row in rows.iterrows():
        key = (row["split"], row["los_scheme"], row["bin_label"])
        if row["run_name"] == "baseline_id26" or key not in baseline.index:
            continue
        rows.loc[idx, "delta_recall_vs_baseline"] = _safe_float(row.get("recall")) - _safe_float(baseline.loc[key, "recall"])
    summary_rows: list[dict[str, Any]] = []
    for (run_name, split), group in rows[rows["los_scheme"].eq("breakdown9")].groupby(["run_name", "split"], dropna=False):
        recalls = pd.to_numeric(group[group["bin_label"].isin(["29-31", "32-33", "34-35", "36-37"])]["recall"], errors="coerce")
        late = pd.to_numeric(group[group["bin_label"].isin(["34-35", "36-37"])]["recall"], errors="coerce")
        summary_rows.append(
            {
                "run_name": run_name,
                "split": split,
                "bin_label": "summary",
                "long_stay_recall_mean": float(recalls.mean()) if recalls.notna().any() else np.nan,
                "late_long_stay_recall_mean": float(late.mean()) if late.notna().any() else np.nan,
                "status": "ok" if recalls.notna().any() else "unavailable",
            }
        )
    if summary_rows:
        rows = pd.concat([rows, pd.DataFrame(summary_rows)], ignore_index=True)
    return rows


def compute_los_routing(
    run: ComparisonRun,
    split: str,
    joint: JointPredictionSource,
    los_schemes: list[str],
    warnings: DiagnosticWarningLog,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    count_frames: list[pd.DataFrame] = []
    pct_frames: list[pd.DataFrame] = []
    flow_frames: list[pd.DataFrame] = []
    for scheme in los_schemes:
        if scheme == "breakdown9_true" and _prediction_uses_coarse_class_ids(joint.frame):
            continue
        normalized_scheme = "breakdown9" if scheme == "breakdown9_true" else scheme
        if joint.frame.empty:
            continue
        try:
            normalized, _heads, _info = normalize_joint_prediction_df(joint.frame, target_scheme=normalized_scheme, context=f"{run.run_name}:{split}:{normalized_scheme}:routing")
            counts, pct = build_los_confusion_tables(normalized, run_name=run.run_name, scheme=normalized_scheme)
            flow = build_middle_to_long_flow_summary(counts, run_name=run.run_name, scheme=normalized_scheme)
        except Exception as exc:
            warnings.warn(f"{run.run_name} {split} {normalized_scheme}: LOS routing unavailable: {exc}")
            continue
        for frame in (counts, pct, flow):
            frame["split"] = split
            frame["source"] = joint.source
        count_frames.append(counts)
        pct_frames.append(pct)
        flow_frames.append(flow)
    confusion = pd.concat(count_frames, ignore_index=True) if count_frames else pd.DataFrame()
    if pct_frames:
        pct_df = pd.concat(pct_frames, ignore_index=True)
        if not confusion.empty:
            confusion = confusion.merge(
                pct_df[["run_name", "split", "los_scheme", "true_bin", "pred_bin", "row_pct"]],
                on=["run_name", "split", "los_scheme", "true_bin", "pred_bin"],
                how="left",
            )
    flow_df = pd.concat(flow_frames, ignore_index=True) if flow_frames else pd.DataFrame()
    return confusion, flow_df


def add_run_deltas(df: pd.DataFrame, value_col: str, delta_col: str, keys: list[str]) -> pd.DataFrame:
    if df.empty or value_col not in df.columns:
        return df
    out = df.copy()
    out[delta_col] = np.nan
    baseline = out[out["run_name"].eq("baseline_id26")]
    for _, base in baseline.iterrows():
        mask = pd.Series(True, index=out.index)
        for key in keys:
            mask &= out[key].eq(base[key])
        out.loc[mask, delta_col] = pd.to_numeric(out.loc[mask, value_col], errors="coerce") - _safe_float(base.get(value_col))
    return out


def build_comparison_table(
    run_metrics: pd.DataFrame,
    drift_summary: pd.DataFrame,
    long_stay: pd.DataFrame,
    middle_flow: pd.DataFrame,
    struct_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    test_drift = drift_summary[(drift_summary["split"].eq("test")) & (drift_summary["los_scheme"].eq("coarse6"))] if not drift_summary.empty else pd.DataFrame()
    long_summary = (
        long_stay[(long_stay["bin_label"].eq("summary")) & (long_stay["split"].eq("test"))]
        if not long_stay.empty and {"bin_label", "split", "run_name"}.issubset(long_stay.columns)
        else pd.DataFrame(columns=["run_name", "long_stay_recall_mean", "late_long_stay_recall_mean"])
    )
    flow = (
        middle_flow[middle_flow["split"].eq("test")]
        if not middle_flow.empty and {"split", "run_name", "total_to_long_pct"}.issubset(middle_flow.columns)
        else pd.DataFrame(columns=["run_name", "total_to_long_pct"])
    )
    for _, metric_row in run_metrics.iterrows():
        run_name = metric_row["run_name"]
        drift_row = test_drift[test_drift["run_name"].eq(run_name)]
        long_row = long_summary[long_summary["run_name"].eq(run_name)]
        flow_row = flow[flow["run_name"].eq(run_name)]
        struct_loss = struct_summary[(struct_summary["run_name"].eq(run_name)) & (struct_summary["metric"].eq("train_struct_loss"))] if not struct_summary.empty else pd.DataFrame()
        rows.append(
            {
                "run_name": run_name,
                "valid_auc": metric_row.get("valid_auc"),
                "test_auc": metric_row.get("test_auc"),
                "mean_risk_dV_D": _first_or_nan(drift_row, "mean_dV_D_risk_heads"),
                "max_risk_dV_D": _first_or_nan(drift_row, "max_dV_D_risk_heads"),
                "mean_risk_js_D": _first_or_nan(drift_row, "mean_js_D_risk_heads"),
                "long_stay_recall_mean": long_row["long_stay_recall_mean"].iloc[0] if not long_row.empty else np.nan,
                "late_long_recall_mean": long_row["late_long_stay_recall_mean"].iloc[0] if not long_row.empty else np.nan,
                "middle_to_long_flow": pd.to_numeric(flow_row["total_to_long_pct"], errors="coerce").mean() if not flow_row.empty else np.nan,
                "struct_loss_delta": struct_loss["delta_last_first"].iloc[0] if not struct_loss.empty and "delta_last_first" in struct_loss.columns else np.nan,
            }
        )
    table = pd.DataFrame(rows)
    table["interpretation"] = table.apply(classify_run_decision, axis=1)
    return table


def _first_or_nan(df: pd.DataFrame, column: str) -> Any:
    if df.empty or column not in df.columns:
        return np.nan
    return df[column].iloc[0]


def classify_run_decision(row: pd.Series) -> str:
    if row.get("run_name") == "baseline_id26":
        return "baseline reference"
    struct_delta = _safe_float(row.get("struct_loss_delta"))
    dV = _safe_float(row.get("mean_risk_dV_D"))
    test_auc = _safe_float(row.get("test_auc"))
    late = _safe_float(row.get("late_long_recall_mean"))
    if not math.isnan(struct_delta) and struct_delta >= 0:
        return "struct_loss did not decrease; implementation/config/loss-scale issue"
    if math.isnan(dV):
        return "post-hoc drift unavailable; cannot classify dV_D effect"
    if math.isnan(late):
        if not math.isnan(test_auc) and test_auc <= REFERENCE_TEST_AUC_9BIN + 0.002:
            return "dV_D effect must be judged without long-stay recall; AUC remains plateau-level"
        return "long-stay recall unavailable"
    if not math.isnan(test_auc) and test_auc <= REFERENCE_TEST_AUC_9BIN + 0.002:
        return "AUC remains plateau-level; check whether dV_D drop is downstream-insensitive"
    return "requires manual review"


def render_decision_report(
    comparison: pd.DataFrame,
    drift_summary: pd.DataFrame,
    accuracy: pd.DataFrame,
    long_stay: pd.DataFrame,
    warnings: DiagnosticWarningLog,
) -> str:
    parts = [
        "# Executive Summary",
        "",
        "1. Did structured loss improve downstream AUC beyond the old plateau?",
        _answer_auc_plateau(comparison),
        "",
        "2. Did structured loss reduce post-hoc dV_D/js_D?",
        _answer_drift(drift_summary),
        "",
        "3. Did structured loss improve long-stay recall, especially 34-35 and 36-37?",
        _answer_long_stay(long_stay),
        "",
        "4. Did structured loss improve or harm risk-head D accuracy?",
        _answer_accuracy(accuracy),
        "",
        "5. Failure diagnosis:",
        _answer_failure_mode(comparison, drift_summary, long_stay),
        "",
        "# Comparison Table",
        "",
        _df_to_markdown(
            comparison,
            [
                "run_name",
                "valid_auc",
                "test_auc",
                "mean_risk_dV_D",
                "max_risk_dV_D",
                "mean_risk_js_D",
                "long_stay_recall_mean",
                "late_long_recall_mean",
                "middle_to_long_flow",
                "struct_loss_delta",
                "interpretation",
            ],
        ),
        "",
        "# Diagnostic Decision Rules",
        "",
        "- A. struct_loss did not decrease -> implementation/config/loss-scale issue.",
        "- B. struct_loss decreased but post-hoc dV_D did not decrease -> batch-local soft js_D surrogate is not aligned with hard post-hoc drift.",
        "- C. post-hoc dV_D decreased but AUC did not improve -> dV_D proxy is not sufficient for downstream recovery.",
        "- D. AUC improved but long-stay recall did not -> possible shortcut, not true semantic recovery.",
        "- E. AUC, dV_D, and late long-stay recall all improved -> structured-loss direction is supported.",
        "- F. none improved -> soft js_D surrogate v1 should be considered failed.",
        "",
        "# Recommendation",
        "",
        _recommendation(comparison, long_stay),
        "",
        "# Warnings",
        "",
        "\n".join(f"- {message}" for message in warnings.warnings) if warnings.warnings else "- none",
        "",
    ]
    return "\n".join(parts)


def _answer_auc_plateau(comparison: pd.DataFrame) -> str:
    if comparison.empty:
        return "- unavailable."
    rows = comparison[comparison["run_name"].ne("baseline_id26")]
    if rows.empty:
        return "- unavailable."
    best = pd.to_numeric(rows["test_auc"], errors="coerce").max()
    if math.isnan(best):
        return "- test AUC unavailable."
    return f"- Best structured test AUC is {best:.6f}; plateau reference is {REFERENCE_TEST_AUC_9BIN:.4f}."


def _answer_drift(drift_summary: pd.DataFrame) -> str:
    if drift_summary.empty:
        return "- post-hoc dV_D/js_D unavailable."
    rows = drift_summary[(drift_summary["split"].eq("test")) & (drift_summary["los_scheme"].eq("coarse6")) & drift_summary["run_name"].ne("baseline_id26")]
    if rows.empty:
        return "- post-hoc dV_D/js_D unavailable for structured runs."
    chunks = []
    for _, row in rows.iterrows():
        chunks.append(
            f"{row['run_name']}: delta risk mean dV_D={_safe_float(row.get('delta_risk_mean_dV_D_vs_baseline')):.6g}, "
            f"delta risk js_D={_safe_float(row.get('delta_risk_js_D_vs_baseline')):.6g}"
        )
    return "- " + "; ".join(chunks) + "."


def _answer_long_stay(long_stay: pd.DataFrame) -> str:
    if long_stay.empty:
        return "- unavailable: no per-row downstream outcome diagnostics or compatible aggregate long-stay breakdown."
    summary = long_stay[long_stay.get("bin_label", pd.Series(dtype=str)).eq("summary")]
    if summary.empty or pd.to_numeric(summary.get("late_long_stay_recall_mean"), errors="coerce").dropna().empty:
        return "- unavailable or incomplete for late long-stay bins."
    return "- late long-stay recall summary is available in `long_stay_recall_comparison.csv`."


def _answer_accuracy(accuracy: pd.DataFrame) -> str:
    if accuracy.empty:
        return "- risk-head D accuracy unavailable."
    rows = accuracy[(accuracy["split"].eq("test")) & (accuracy["los_scheme"].eq("coarse6")) & accuracy["run_name"].ne("baseline_id26")]
    if rows.empty:
        return "- risk-head D accuracy unavailable for structured runs."
    degraded = rows[pd.to_numeric(rows["delta_acc_vs_baseline"], errors="coerce") < 0]
    if degraded.empty:
        return "- no risk-head accuracy degradation detected versus baseline in test/coarse6."
    heads = ", ".join(sorted(degraded["head"].unique().tolist()))
    return f"- risk-head accuracy degraded for: {heads}."


def _answer_failure_mode(comparison: pd.DataFrame, drift_summary: pd.DataFrame, long_stay: pd.DataFrame) -> str:
    if comparison.empty:
        return "- unavailable."
    struct = comparison[comparison["run_name"].ne("baseline_id26")]
    if struct.empty:
        return "- unavailable."
    struct_down = pd.to_numeric(struct["struct_loss_delta"], errors="coerce").dropna()
    if not struct_down.empty and (struct_down >= 0).all():
        return "- struct loss did not decrease; treat as implementation/config/loss-scale issue."
    drift_delta = pd.to_numeric(
        drift_summary[(drift_summary["split"].eq("test")) & (drift_summary["los_scheme"].eq("coarse6")) & drift_summary["run_name"].ne("baseline_id26")]["delta_risk_mean_dV_D_vs_baseline"],
        errors="coerce",
    )
    auc_delta = pd.to_numeric(struct["test_auc"], errors="coerce") - REFERENCE_TEST_AUC_9BIN
    if drift_delta.notna().any() and (drift_delta < 0).any() and auc_delta.notna().any() and (auc_delta <= 0.002).all():
        return "- post-hoc dV_D improved but AUC stayed plateau-level; proxy appears downstream-insensitive or insufficient alone."
    if long_stay.empty:
        return "- long-stay evidence is unavailable; current evidence supports checking drift/AUC only."
    return "- mixed evidence; inspect CSV artifacts."


def _recommendation(comparison: pd.DataFrame, long_stay: pd.DataFrame) -> str:
    if comparison.empty:
        return "No recommendation: comparison table unavailable."
    struct = comparison[comparison["run_name"].ne("baseline_id26")]
    if struct.empty:
        return "No recommendation: structured runs unavailable."
    best_auc = pd.to_numeric(struct["test_auc"], errors="coerce").max()
    late_available = not long_stay.empty and pd.to_numeric(long_stay.get("late_long_stay_recall_mean", pd.Series(dtype=float)), errors="coerce").notna().any()
    if math.isnan(best_auc) or best_auc <= REFERENCE_TEST_AUC_9BIN + 0.002:
        if not late_available:
            return "Do not combine with 9-bin yet. Treat v1 as inconclusive on long-stay recovery and prioritize outcome-aware fine-tuning or a joint-generative predictor."
        return "Do not combine with 9-bin unless late long-stay recall improved. Prioritize outcome-aware fine-tuning or joint-generative predictor."
    return "Structured loss may warrant a targeted lambda sweep only if late long-stay recall also improves."


def write_outputs(
    out_dir: Path,
    *,
    run_metrics: pd.DataFrame,
    struct_trace: pd.DataFrame,
    struct_summary: pd.DataFrame,
    head_drift: pd.DataFrame,
    drift_summary: pd.DataFrame,
    true_los_metrics: pd.DataFrame,
    pred_los_metrics: pd.DataFrame,
    long_stay: pd.DataFrame,
    los_confusion: pd.DataFrame,
    middle_flow: pd.DataFrame,
    accuracy: pd.DataFrame,
    comparison: pd.DataFrame,
    manifest: dict[str, Any],
    warnings: DiagnosticWarningLog,
) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    def write_csv(name: str, df: pd.DataFrame) -> None:
        path = out_dir / name
        df.to_csv(path, index=False)
        created.append(path)

    write_csv("run_metrics_comparison.csv", run_metrics)
    _write_md_table(out_dir / "run_metrics_comparison.md", "Run Metrics Comparison", run_metrics)
    created.append(out_dir / "run_metrics_comparison.md")
    write_csv("struct_loss_trace.csv", struct_trace)
    write_csv("struct_loss_summary.csv", struct_summary)
    _write_md_table(out_dir / "struct_loss_summary.md", "Structured Loss Summary", struct_summary)
    created.append(out_dir / "struct_loss_summary.md")
    write_csv("head_drift_posthoc.csv", head_drift)
    risk_head_drift = head_drift[head_drift.get("head", pd.Series(dtype=str)).isin(RISK_HEADS)].copy() if not head_drift.empty else pd.DataFrame()
    write_csv("risk_head_drift_comparison.csv", risk_head_drift)
    write_csv("drift_run_summary.csv", drift_summary)
    _write_md_table(out_dir / "drift_comparison.md", "Drift Comparison", drift_summary)
    created.append(out_dir / "drift_comparison.md")
    write_csv("true_los_bin_outcome_metrics.csv", true_los_metrics)
    write_csv("pred_los_bin_outcome_metrics.csv", pred_los_metrics)
    write_csv("long_stay_recall_comparison.csv", long_stay)
    _write_md_table(out_dir / "los_bin_outcome_metrics.md", "LOS-Bin Outcome Metrics", true_los_metrics)
    created.append(out_dir / "los_bin_outcome_metrics.md")
    write_csv("los_confusion_matrix.csv", los_confusion)
    write_csv("middle_to_long_flow.csv", middle_flow)
    _write_md_table(out_dir / "middle_to_long_flow.md", "Middle-to-Long Flow", middle_flow)
    created.append(out_dir / "middle_to_long_flow.md")
    write_csv("risk_head_accuracy_comparison.csv", accuracy)
    _write_md_table(out_dir / "risk_head_accuracy_comparison.md", "Risk-Head Accuracy Comparison", accuracy)
    created.append(out_dir / "risk_head_accuracy_comparison.md")
    decision = render_decision_report(comparison, drift_summary, accuracy, long_stay, warnings)
    decision_path = out_dir / "decision_report.md"
    decision_path.write_text(decision, encoding="utf-8")
    created.append(decision_path)
    manifest["created_files"] = [str(path) for path in created]
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(_json_sanitize(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    created.append(manifest_path)
    return [str(path) for path in created]


def build_posthoc(args: argparse.Namespace) -> tuple[dict[str, pd.DataFrame], dict[str, Any], DiagnosticWarningLog]:
    warnings = DiagnosticWarningLog(strict=bool(args.strict))
    runs = resolve_comparison_runs(args, warnings)
    risk_heads = sorted(set().union(*(get_named_risk_head_set(name) for name in args.risk_head_sets), set(RISK_HEADS)))
    los_schemes = list(args.los_bin_schemes)

    settings = {run.run_name: load_struct_settings(run) for run in runs}
    run_metrics = add_metric_deltas(pd.DataFrame([load_run_metrics(run) for run in runs]))
    trace_frames = [extract_struct_loss_trace(run, settings[run.run_name]) for run in runs if run.run_name != "baseline_id26"]
    if not trace_frames:
        trace_frames = [pd.DataFrame()]
    struct_trace = pd.concat(trace_frames, ignore_index=True) if trace_frames else pd.DataFrame()
    struct_summary = summarize_struct_loss(struct_trace)

    head_frames: list[pd.DataFrame] = []
    true_los_frames: list[pd.DataFrame] = []
    pred_los_frames: list[pd.DataFrame] = []
    confusion_frames: list[pd.DataFrame] = []
    flow_frames: list[pd.DataFrame] = []
    prediction_sources: dict[str, dict[str, str]] = {}
    outcome_sources: dict[str, str] = {}
    for run in runs:
        outcome = load_outcome_source(run, warnings)
        outcome_sources[run.run_name] = outcome.status
        for split in args.splits:
            joint = load_joint_predictions(run, split, warnings)
            prediction_sources.setdefault(run.run_name, {})[split] = joint.source
            if joint.frame.empty:
                continue
            for scheme in los_schemes:
                if scheme == "breakdown9_true":
                    continue
                head_frames.append(compute_head_drift_for_run(run, split, joint, scheme, warnings))
            true_metrics, pred_metrics = compute_outcome_diagnostics(run, split, joint, outcome, los_schemes, warnings)
            true_los_frames.append(true_metrics)
            pred_los_frames.append(pred_metrics)
            confusion, flow = compute_los_routing(run, split, joint, los_schemes, warnings)
            confusion_frames.append(confusion)
            flow_frames.append(flow)

    head_drift = pd.concat([frame for frame in head_frames if not frame.empty], ignore_index=True) if head_frames else pd.DataFrame()
    drift_summary = summarize_drift(head_drift, risk_heads)
    true_los_metrics = pd.concat([frame for frame in true_los_frames if not frame.empty], ignore_index=True) if true_los_frames else pd.DataFrame()
    pred_los_metrics = pd.concat([frame for frame in pred_los_frames if not frame.empty], ignore_index=True) if pred_los_frames else pd.DataFrame()
    long_stay = build_long_stay_comparison(true_los_metrics)
    los_confusion = pd.concat([frame for frame in confusion_frames if not frame.empty], ignore_index=True) if confusion_frames else pd.DataFrame()
    middle_flow = pd.concat([frame for frame in flow_frames if not frame.empty], ignore_index=True) if flow_frames else pd.DataFrame()
    accuracy = risk_head_accuracy_comparison(head_drift, risk_heads)
    comparison = build_comparison_table(run_metrics, drift_summary, long_stay, middle_flow, struct_summary)

    manifest = {
        "command_args": vars(args),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_git_commit(),
        "resolved_run_dirs": {run.run_name: str(run.run_dir) for run in runs},
        "config_paths": {
            run.run_name: str(run.run_dir / "joint_predictor" / "config.final.yaml")
            if (run.run_dir / "joint_predictor" / "config.final.yaml").exists()
            else str(run.run_dir / "config.final.yaml")
            for run in runs
        },
        "run_names": [run.run_name for run in runs],
        "split_names": list(args.splits),
        "warnings": warnings.warnings,
        "missing_files": warnings.missing_files,
        "struct_loss_settings": settings,
        "prediction_sources": prediction_sources,
        "outcome_sources": outcome_sources,
    }
    tables = {
        "run_metrics": run_metrics,
        "struct_trace": struct_trace,
        "struct_summary": struct_summary,
        "head_drift": head_drift,
        "drift_summary": drift_summary,
        "true_los_metrics": true_los_metrics,
        "pred_los_metrics": pred_los_metrics,
        "long_stay": long_stay,
        "los_confusion": los_confusion,
        "middle_flow": middle_flow,
        "accuracy": accuracy,
        "comparison": comparison,
    }
    return tables, manifest, warnings


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-hoc comparison for structured dV_D Forecasted CTMP-GIN runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--baseline-run-dir", type=Path, required=True)
    parser.add_argument("--top6-run-dir", type=Path, required=True)
    parser.add_argument("--robust-run-dir", type=Path, default=None)
    parser.add_argument("--robust-run-name-contains", default=None)
    parser.add_argument("--splits", nargs="+", choices=("valid", "test"), default=["valid", "test"])
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--risk-head-sets", nargs="+", default=["new_robust_top3", "new_dvD_top6"])
    parser.add_argument("--los-bin-schemes", nargs="+", choices=("coarse6", "breakdown9_true"), default=["coarse6", "breakdown9_true"])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.out_dir = _resolve_path(args.out_dir)
    tables, manifest, warnings = build_posthoc(args)
    created = write_outputs(
        args.out_dir,
        run_metrics=tables["run_metrics"],
        struct_trace=tables["struct_trace"],
        struct_summary=tables["struct_summary"],
        head_drift=tables["head_drift"],
        drift_summary=tables["drift_summary"],
        true_los_metrics=tables["true_los_metrics"],
        pred_los_metrics=tables["pred_los_metrics"],
        long_stay=tables["long_stay"],
        los_confusion=tables["los_confusion"],
        middle_flow=tables["middle_flow"],
        accuracy=tables["accuracy"],
        comparison=tables["comparison"],
        manifest=manifest,
        warnings=warnings,
    )
    print(f"[done] wrote outputs to {args.out_dir}")
    print(_df_to_markdown(tables["comparison"]))
    print("[created]")
    for path in created:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
