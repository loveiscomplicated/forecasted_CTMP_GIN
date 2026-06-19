from __future__ import annotations

import csv
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from src.diagnostics.diagnose_joint_predictor_joint_stats import compute_joint_stats


@dataclass(frozen=True)
class ReportArtifacts:
    head_report_csv: str
    head_report_md: str
    focused_heads_summary_csv: str
    focused_heads_prob_deltas_csv: str
    summary_json: str
    extraction_command_sh: str


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "None", "nan", "NaN"}:
        return float("nan")
    return float(value)


def _int(row: dict[str, str], key: str) -> int:
    return int(row[key])


def _dominant_issue(row: dict[str, Any]) -> tuple[str, str]:
    js = abs(float(row["js_los_given_d"]))
    coverage_gap = float(row["pred_class_coverage_gap"])
    unique_pred = int(row["eval_unique_pred_classes"])
    num_classes = int(row["num_d_classes"])

    if js >= 0.12:
        return (
            "strong_conditional_shift",
            "Predicted P(LOS|D) departs materially from train reference; inspect this head first.",
        )
    if js >= 0.06:
        return (
            "moderate_conditional_shift",
            "Predicted P(LOS|D) is noticeably distorted for this discharge head.",
        )
    if coverage_gap >= 0.25 or unique_pred < num_classes:
        return (
            "coverage_gap",
            "Predicted discharge support is narrower than reference; inspect collapsed D values.",
        )
    return (
        "mild_shift",
        "P(LOS|D) drift is present but not dominant relative to the rest of the heads.",
    )


def build_head_interpretation_rows(
    per_head_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(
        per_head_rows,
        key=lambda row: (
            float(row["js_los_given_d"]),
            float(row["js_d_given_los"]),
            float(row["pred_class_coverage_gap"]),
            str(row["head_name"]),
        ),
        reverse=True,
    )
    report_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:top_k], start=1):
        dominant_issue, interpretation = _dominant_issue(row)
        report_rows.append(
            {
                "rank": rank,
                "head_name": row["head_name"],
                "dominant_issue": dominant_issue,
                "interpretation": interpretation,
                **row,
            }
        )
    return report_rows


def write_head_interpretation_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# LOS Given D Head Interpretation Report", ""]
    for row in rows:
        lines.append(f"## {row['rank']}. {row['head_name']}")
        lines.append(f"- dominant_issue: {row['dominant_issue']}")
        lines.append(f"- interpretation: {row['interpretation']}")
        lines.append(f"- JS(P(LOS|D)): {row['js_los_given_d']:.6f}")
        lines.append(f"- JS(P(D|LOS)): {row['js_d_given_los']:.6f}")
        lines.append(f"- num_d_classes: {row['num_d_classes']}")
        lines.append(f"- eval_unique_pred_classes: {row['eval_unique_pred_classes']}")
        lines.append(f"- pred_class_coverage_ratio: {row['pred_class_coverage_ratio']:.6f}")
        lines.append(f"- pred_class_coverage_gap: {row['pred_class_coverage_gap']:.6f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _load_per_head_rows(path: Path) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    parsed: list[dict[str, Any]] = []
    for row in rows:
        num_d_classes = _int(row, "num_d_classes")
        eval_unique_pred_classes = _int(row, "eval_unique_pred_classes")
        coverage_ratio = (
            float(eval_unique_pred_classes) / float(num_d_classes)
            if num_d_classes > 0
            else 0.0
        )
        parsed.append(
            {
                "head_name": row["head_name"],
                "js_d_given_los": _float(row, "js_d_given_los"),
                "js_los_given_d": _float(row, "js_los_given_d"),
                "num_d_classes": num_d_classes,
                "eval_unique_pred_classes": eval_unique_pred_classes,
                "pred_class_coverage_ratio": coverage_ratio,
                "pred_class_coverage_gap": 1.0 - coverage_ratio,
            }
        )
    return parsed


def build_focused_probability_rows(
    conditional_path: Path,
    *,
    target_names: set[str],
    limit: int,
) -> list[dict[str, Any]]:
    rows = _read_csv_rows(conditional_path)
    focused: list[dict[str, Any]] = []
    for row in rows:
        if row["head_name"] not in target_names:
            continue
        train_prob = _float(row, "train_prob")
        eval_prob = _float(row, "eval_prob")
        focused.append(
            {
                "head_name": row["head_name"],
                "d_value": _int(row, "d_value"),
                "los_bin": _int(row, "los_bin"),
                "train_count": _int(row, "train_count"),
                "train_prob": train_prob,
                "eval_count": _int(row, "eval_count"),
                "eval_prob": eval_prob,
                "train_d_count": _int(row, "train_d_count"),
                "eval_d_count": _int(row, "eval_d_count"),
                "delta_eval_minus_train": eval_prob - train_prob,
                "abs_delta_eval_minus_train": abs(eval_prob - train_prob),
                "js_los_given_d_for_d_value": _float(row, "js_los_given_d_for_d_value"),
                "js_los_given_d_head": _float(row, "js_los_given_d_head"),
                "los_bin_mode": row["los_bin_mode"],
            }
        )
    focused.sort(
        key=lambda row: (
            float(row["js_los_given_d_head"]),
            float(row["js_los_given_d_for_d_value"]),
            float(row["abs_delta_eval_minus_train"]),
            str(row["head_name"]),
        ),
        reverse=True,
    )
    return focused[:limit]


def create_extraction_command(
    script_path: Path,
    diagnostic_dir: Path,
    *,
    target_names: list[str],
    top_k: int,
    limit: int,
) -> str:
    command = [
        "uv",
        "run",
        "python",
        str(script_path),
        "--diagnostic-dir",
        str(diagnostic_dir),
        "--top-k",
        str(top_k),
        "--limit",
        str(limit),
        "--heads",
        ",".join(target_names),
    ]
    return " ".join(shlex.quote(part) for part in command)


def write_extraction_command_script(path: Path, *, command: str) -> None:
    script = f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)


def _ensure_conditional_los_given_d_csv(
    diagnostic_path: Path,
    summary_path: Path,
    conditional_path: Path,
) -> None:
    if conditional_path.exists():
        return

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    train_cache_path = summary_payload.get("train_cache_path")
    eval_cache_path = summary_payload.get("eval_cache_path")
    los_bin_mode = summary_payload.get("los_bin_mode")
    if not train_cache_path or not eval_cache_path:
        raise FileNotFoundError(
            f"Required diagnostic file not found: {conditional_path}. "
            "Could not backfill it because train/eval cache paths are missing from joint_stats_summary.json."
        )

    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    eval_cache = torch.load(eval_cache_path, map_location="cpu", weights_only=False)
    train_cache["_path"] = str(train_cache_path)
    eval_cache["_path"] = str(eval_cache_path)
    regenerated = compute_joint_stats(
        train_cache,
        eval_cache,
        los_bin_mode=str(los_bin_mode) if los_bin_mode is not None else None,
    )
    rows = regenerated.get("los_given_d_rows", [])
    if not rows:
        raise FileNotFoundError(
            f"Required diagnostic file not found: {conditional_path}. "
            "Attempted to backfill it from caches, but no LOS|D rows were produced."
        )
    _write_csv(
        conditional_path,
        rows,
        list(rows[0].keys()),
    )


def generate_joint_los_given_d_report(
    diagnostic_dir: str | Path,
    *,
    top_k: int = 10,
    heads: list[str] | None = None,
    limit: int = 500,
    script_path: str | Path | None = None,
) -> dict[str, Any]:
    diagnostic_path = Path(diagnostic_dir)
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"Diagnostic directory not found: {diagnostic_path}")

    per_head_path = diagnostic_path / "joint_stats_per_head.csv"
    conditional_path = diagnostic_path / "per_head_conditional_los_given_d.csv"
    summary_path = diagnostic_path / "joint_stats_summary.json"
    for required_path in (per_head_path, summary_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required diagnostic file not found: {required_path}")
    _ensure_conditional_los_given_d_csv(diagnostic_path, summary_path, conditional_path)

    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    per_head_rows = _load_per_head_rows(per_head_path)
    head_report_rows = build_head_interpretation_rows(per_head_rows, top_k=top_k)

    default_heads = [row["head_name"] for row in head_report_rows[: min(3, len(head_report_rows))]]
    head_names = heads or default_heads
    head_name_set = set(head_names)

    focused_head_rows = [row for row in head_report_rows if row["head_name"] in head_name_set]
    focused_probability_rows = build_focused_probability_rows(
        conditional_path,
        target_names=head_name_set,
        limit=limit,
    )

    head_report_csv = diagnostic_path / "los_given_d_head_interpretation_report.csv"
    head_report_md = diagnostic_path / "los_given_d_head_interpretation_report.md"
    focused_summary_csv = diagnostic_path / "los_given_d_focused_heads_summary.csv"
    focused_prob_csv = diagnostic_path / "los_given_d_focused_prob_deltas.csv"
    extraction_script = diagnostic_path / "analyze_joint_los_given_d_report.sh"
    output_summary_json = diagnostic_path / "los_given_d_report_summary.json"

    fieldnames = [
        "rank",
        "head_name",
        "dominant_issue",
        "interpretation",
        "js_los_given_d",
        "js_d_given_los",
        "num_d_classes",
        "eval_unique_pred_classes",
        "pred_class_coverage_ratio",
        "pred_class_coverage_gap",
    ]
    _write_csv(head_report_csv, head_report_rows, fieldnames)
    write_head_interpretation_markdown(head_report_md, head_report_rows)
    _write_csv(focused_summary_csv, focused_head_rows, fieldnames)
    _write_csv(
        focused_prob_csv,
        focused_probability_rows,
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
            "delta_eval_minus_train",
            "abs_delta_eval_minus_train",
            "js_los_given_d_for_d_value",
            "js_los_given_d_head",
            "los_bin_mode",
        ],
    )

    command = create_extraction_command(
        Path(script_path) if script_path is not None else Path("scripts/analyze_joint_los_given_d_report.py"),
        diagnostic_path,
        target_names=head_names,
        top_k=top_k,
        limit=limit,
    )
    write_extraction_command_script(extraction_script, command=command)

    payload = {
        "diagnostic_dir": str(diagnostic_path),
        "top_k": top_k,
        "heads": head_names,
        "limit": limit,
        "num_report_heads": len(head_report_rows),
        "num_focused_heads": len(focused_head_rows),
        "num_focused_probability_rows": len(focused_probability_rows),
        "source_summary": {
            "split": summary_payload.get("split"),
            "los_bin_mode": summary_payload.get("los_bin_mode"),
            "num_rows_eval": summary_payload.get("num_rows_eval"),
            "mean_js_los_given_d": summary_payload.get("mean_js_los_given_d"),
            "max_js_los_given_d": summary_payload.get("max_js_los_given_d"),
        },
        "artifacts": ReportArtifacts(
            head_report_csv=str(head_report_csv),
            head_report_md=str(head_report_md),
            focused_heads_summary_csv=str(focused_summary_csv),
            focused_heads_prob_deltas_csv=str(focused_prob_csv),
            summary_json=str(output_summary_json),
            extraction_command_sh=str(extraction_script),
        ).__dict__,
        "analysis_command": command,
    }
    _write_json(output_summary_json, payload)
    return payload
