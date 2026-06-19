from __future__ import annotations

import csv
import json
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReportArtifacts:
    head_report_csv: str
    head_report_md: str
    high_confidence_mismatches_csv: str
    focused_heads_summary_csv: str
    focused_heads_prob_deltas_csv: str
    focused_heads_mismatches_csv: str
    extraction_command_sh: str
    summary_json: str


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


def _bool(row: dict[str, str], key: str) -> bool:
    return row[key] == "True"


def _load_per_head_summary_rows(path: Path, *, split: str) -> list[dict[str, Any]]:
    rows = _read_csv_rows(path)
    seen: set[tuple[str, str]] = set()
    summary_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != split:
            continue
        key = (row["split"], row["target_name"])
        if key in seen:
            continue
        seen.add(key)
        summary_rows.append(
            {
                "split": row["split"],
                "target_name": row["target_name"],
                "cramers_v_oracle": _float(row, "cramers_v_oracle"),
                "cramers_v_predicted": _float(row, "cramers_v_predicted"),
                "delta_cramers_v": _float(row, "delta_cramers_v"),
                "js_divergence_P_D_given_LOS": _float(row, "js_divergence_P_D_given_LOS"),
                "rare_combo_rate_oracle": _float(row, "rare_combo_rate_oracle"),
                "rare_combo_rate_predicted": _float(row, "rare_combo_rate_predicted"),
                "rare_combo_rate_mixed_D_pred": _float(row, "rare_combo_rate_mixed_D_pred"),
                "rare_combo_rate_mixed_LOS_pred": _float(row, "rare_combo_rate_mixed_LOS_pred"),
                "positive_rate_drift_by_los": _float(row, "positive_rate_drift_by_los"),
            }
        )
    return summary_rows


def _problematic_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row["js_divergence_P_D_given_LOS"]),
        float(row["rare_combo_rate_predicted"]),
        abs(float(row["delta_cramers_v"])),
        str(row["target_name"]),
    )


def _dominant_issue(row: dict[str, Any]) -> tuple[str, str]:
    js = abs(float(row["js_divergence_P_D_given_LOS"]))
    delta_cv = abs(float(row["delta_cramers_v"]))
    rare_pred = float(row["rare_combo_rate_predicted"])
    rare_mixed = max(
        float(row["rare_combo_rate_mixed_D_pred"]),
        float(row["rare_combo_rate_mixed_LOS_pred"]),
    )
    los_drift = abs(float(row["positive_rate_drift_by_los"]))

    if js >= 0.03 or delta_cv >= 0.12:
        return (
            "strong_conditional_shift",
            "Predicted P(D|LOS) departs materially from oracle; prioritize this head for ablation.",
        )
    if js >= 0.01 or delta_cv >= 0.05:
        return (
            "moderate_conditional_shift",
            "Predicted P(D|LOS) is noticeably distorted; inspect LOS-bin-specific mode collapse.",
        )
    if rare_pred >= 0.001 or rare_mixed >= 0.001:
        return (
            "rare_combo_pressure",
            "Rare train-oracle (D, LOS) pairs appear often enough to warrant direct sample review.",
        )
    if los_drift >= 0.05:
        return (
            "los_semantic_shift",
            "This head is mostly affected by LOS-bin label drift rather than rare combinations.",
        )
    return (
        "mild_shift",
        "Joint drift is present but not dominant relative to the rest of the heads.",
    )


def build_head_interpretation_rows(
    per_head_rows: list[dict[str, Any]],
    *,
    top_k: int,
) -> list[dict[str, Any]]:
    ranked = sorted(per_head_rows, key=_problematic_sort_key, reverse=True)
    report_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked[:top_k], start=1):
        dominant_issue, interpretation = _dominant_issue(row)
        report_rows.append(
            {
                "rank": rank,
                "target_name": row["target_name"],
                "dominant_issue": dominant_issue,
                "interpretation": interpretation,
                **row,
            }
        )
    return report_rows


def write_head_interpretation_markdown(
    path: Path,
    head_rows: list[dict[str, Any]],
    *,
    split: str,
) -> None:
    lines = ["# Head Interpretation Report", "", f"split={split}", ""]
    for row in head_rows:
        lines.append(f"## {row['rank']}. {row['target_name']}")
        lines.append(f"- dominant_issue: {row['dominant_issue']}")
        lines.append(f"- interpretation: {row['interpretation']}")
        lines.append(f"- JS(P(D|LOS)): {row['js_divergence_P_D_given_LOS']:.6f}")
        lines.append(f"- delta_cramers_v: {row['delta_cramers_v']:.6f}")
        lines.append(f"- cramers_v_oracle: {row['cramers_v_oracle']:.6f}")
        lines.append(f"- cramers_v_predicted: {row['cramers_v_predicted']:.6f}")
        lines.append(f"- rare_combo_rate_predicted: {row['rare_combo_rate_predicted']:.6f}")
        lines.append(f"- rare_combo_rate_mixed_D_pred: {row['rare_combo_rate_mixed_D_pred']:.6f}")
        lines.append(f"- rare_combo_rate_mixed_LOS_pred: {row['rare_combo_rate_mixed_LOS_pred']:.6f}")
        lines.append(f"- positive_rate_drift_by_los: {row['positive_rate_drift_by_los']:.6f}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _collect_focused_mismatches(
    confidence_rows_path: Path,
    *,
    split: str | None,
    target_names: set[str] | None,
    discharge_confidence_min: float,
    los_confidence_min: float,
    limit: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    high_conf_rows: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    with confidence_rows_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if split is not None and row["split"] != split:
                continue
            if target_names is not None and row["target_name"] not in target_names:
                continue
            if not _bool(row, "joint_mismatch_predicted"):
                continue
            dc = _float(row, "discharge_confidence")
            lc = _float(row, "los_confidence")
            enriched_row = {
                **row,
                "confidence_product": dc * lc,
                "los_bin_changed": _int(row, "oracle_los_bin") != _int(row, "predicted_los_bin"),
                "d_value_changed": _int(row, "oracle_d") != _int(row, "predicted_d"),
            }
            all_rows.append(enriched_row)
            if dc >= discharge_confidence_min and lc >= los_confidence_min:
                high_conf_rows.append(enriched_row)

    def _sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
        return (
            float(row["confidence_product"]),
            float(row["discharge_confidence"]),
            float(row["los_confidence"]),
        )

    high_conf_rows.sort(key=_sort_key, reverse=True)
    all_rows.sort(key=_sort_key, reverse=True)
    return high_conf_rows[:limit], all_rows[:limit]


def extract_high_confidence_mismatches(
    confidence_rows_path: Path,
    *,
    output_path: Path,
    split: str | None,
    target_names: set[str] | None,
    discharge_confidence_min: float,
    los_confidence_min: float,
    limit: int,
) -> list[dict[str, Any]]:
    high_conf_rows, _ = _collect_focused_mismatches(
        confidence_rows_path,
        split=split,
        target_names=target_names,
        discharge_confidence_min=discharge_confidence_min,
        los_confidence_min=los_confidence_min,
        limit=limit,
    )
    _write_csv(
        output_path,
        high_conf_rows,
        [
            "split",
            "row_idx",
            "caseid",
            "target_name",
            "y",
            "oracle_d",
            "predicted_d",
            "oracle_los_bin",
            "predicted_los_bin",
            "rare_oracle",
            "rare_predicted",
            "rare_mixed_D_pred",
            "rare_mixed_LOS_pred",
            "joint_mismatch_predicted",
            "discharge_confidence",
            "los_confidence",
            "confidence_product",
            "los_bin_changed",
            "d_value_changed",
        ],
    )
    return high_conf_rows


def build_focused_probability_rows(
    per_head_distribution_path: Path,
    *,
    split: str,
    target_names: set[str],
) -> list[dict[str, Any]]:
    rows = _read_csv_rows(per_head_distribution_path)
    focused: list[dict[str, Any]] = []
    for row in rows:
        if row["split"] != split or row["target_name"] not in target_names:
            continue
        oracle_prob = _float(row, "oracle_prob")
        predicted_prob = _float(row, "predicted_prob")
        mixed_d_prob = _float(row, "mixed_D_pred_prob")
        mixed_los_prob = _float(row, "mixed_LOS_pred_prob")
        focused.append(
            {
                "split": row["split"],
                "target_name": row["target_name"],
                "los_bin": _int(row, "los_bin"),
                "d_value": _int(row, "d_value"),
                "oracle_prob": oracle_prob,
                "predicted_prob": predicted_prob,
                "mixed_D_pred_prob": mixed_d_prob,
                "mixed_LOS_pred_prob": mixed_los_prob,
                "delta_predicted_minus_oracle": predicted_prob - oracle_prob,
                "delta_mixed_D_pred_minus_oracle": mixed_d_prob - oracle_prob,
                "delta_mixed_LOS_pred_minus_oracle": mixed_los_prob - oracle_prob,
                "abs_delta_predicted_minus_oracle": abs(predicted_prob - oracle_prob),
                "train_oracle_probability": _float(row, "train_oracle_probability"),
                "train_oracle_is_rare": _bool(row, "train_oracle_is_rare"),
                "js_divergence_P_D_given_LOS": _float(row, "js_divergence_P_D_given_LOS"),
                "delta_cramers_v": _float(row, "delta_cramers_v"),
            }
        )
    focused.sort(
        key=lambda row: (
            row["target_name"],
            float(row["abs_delta_predicted_minus_oracle"]),
            abs(float(row["delta_mixed_D_pred_minus_oracle"])),
            abs(float(row["delta_mixed_LOS_pred_minus_oracle"])),
        ),
        reverse=True,
    )
    return focused


def build_focused_head_summary(
    head_report_rows: list[dict[str, Any]],
    *,
    target_names: set[str],
) -> list[dict[str, Any]]:
    return [row for row in head_report_rows if row["target_name"] in target_names]


def create_extraction_command(
    script_path: Path,
    diagnostic_dir: Path,
    *,
    split: str | None,
    target_names: list[str],
    discharge_confidence_min: float,
    los_confidence_min: float,
    limit: int,
    top_k: int,
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
        "--discharge-confidence-min",
        str(discharge_confidence_min),
        "--los-confidence-min",
        str(los_confidence_min),
        "--limit",
        str(limit),
        "--heads",
        ",".join(target_names),
    ]
    if split is not None:
        command.extend(["--split", split])
    return " ".join(shlex.quote(part) for part in command)


def write_extraction_command_script(path: Path, *, command: str) -> None:
    script = f"#!/usr/bin/env bash\nset -euo pipefail\n{command}\n"
    path.write_text(script, encoding="utf-8")
    os.chmod(path, 0o755)


def generate_joint_plausibility_report(
    diagnostic_dir: str | Path,
    *,
    top_k: int = 10,
    split: str = "overall",
    heads: list[str] | None = None,
    discharge_confidence_min: float = 0.9,
    los_confidence_min: float = 0.5,
    limit: int = 500,
    script_path: str | Path | None = None,
) -> dict[str, Any]:
    diagnostic_path = Path(diagnostic_dir)
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"Diagnostic directory not found: {diagnostic_path}")

    per_head_path = diagnostic_path / "per_head_conditional_distribution.csv"
    confidence_path = diagnostic_path / "confidence_vs_joint_mismatch.csv"
    summary_path = diagnostic_path / "joint_plausibility_summary.csv"
    for required_path in (per_head_path, confidence_path, summary_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required diagnostic file not found: {required_path}")

    head_names = heads or ["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"]
    head_name_set = set(head_names)

    per_head_summary_rows = _load_per_head_summary_rows(per_head_path, split=split)
    head_report_rows = build_head_interpretation_rows(per_head_summary_rows, top_k=top_k)
    focused_head_rows = build_focused_head_summary(head_report_rows, target_names=head_name_set)
    focused_probability_rows = build_focused_probability_rows(
        per_head_path,
        split=split,
        target_names=head_name_set,
    )
    focused_mismatch_rows, all_focused_mismatch_rows = _collect_focused_mismatches(
        confidence_path,
        split=None if split == "overall" else split,
        target_names=head_name_set,
        discharge_confidence_min=discharge_confidence_min,
        los_confidence_min=los_confidence_min,
        limit=limit,
    )

    head_report_csv = diagnostic_path / "head_interpretation_report.csv"
    head_report_md = diagnostic_path / "head_interpretation_report.md"
    high_conf_mismatch_csv = diagnostic_path / "high_confidence_mismatches.csv"
    focused_summary_csv = diagnostic_path / "focused_heads_summary.csv"
    focused_prob_csv = diagnostic_path / "focused_heads_prob_deltas.csv"
    focused_mismatch_csv = diagnostic_path / "focused_heads_mismatches.csv"
    extraction_script = diagnostic_path / "extract_high_confidence_mismatches.sh"
    summary_json = diagnostic_path / "head_interpretation_summary.json"

    _write_csv(
        head_report_csv,
        head_report_rows,
        [
            "rank",
            "target_name",
            "dominant_issue",
            "interpretation",
            "split",
            "cramers_v_oracle",
            "cramers_v_predicted",
            "delta_cramers_v",
            "js_divergence_P_D_given_LOS",
            "rare_combo_rate_oracle",
            "rare_combo_rate_predicted",
            "rare_combo_rate_mixed_D_pred",
            "rare_combo_rate_mixed_LOS_pred",
            "positive_rate_drift_by_los",
        ],
    )
    write_head_interpretation_markdown(head_report_md, head_report_rows, split=split)
    _write_csv(
        focused_summary_csv,
        focused_head_rows,
        [
            "rank",
            "target_name",
            "dominant_issue",
            "interpretation",
            "split",
            "cramers_v_oracle",
            "cramers_v_predicted",
            "delta_cramers_v",
            "js_divergence_P_D_given_LOS",
            "rare_combo_rate_oracle",
            "rare_combo_rate_predicted",
            "rare_combo_rate_mixed_D_pred",
            "rare_combo_rate_mixed_LOS_pred",
            "positive_rate_drift_by_los",
        ],
    )
    _write_csv(
        focused_prob_csv,
        focused_probability_rows,
        [
            "split",
            "target_name",
            "los_bin",
            "d_value",
            "oracle_prob",
            "predicted_prob",
            "mixed_D_pred_prob",
            "mixed_LOS_pred_prob",
            "delta_predicted_minus_oracle",
            "delta_mixed_D_pred_minus_oracle",
            "delta_mixed_LOS_pred_minus_oracle",
            "abs_delta_predicted_minus_oracle",
            "train_oracle_probability",
            "train_oracle_is_rare",
            "js_divergence_P_D_given_LOS",
            "delta_cramers_v",
        ],
    )
    _write_csv(
        high_conf_mismatch_csv,
        focused_mismatch_rows,
        [
            "split",
            "row_idx",
            "caseid",
            "target_name",
            "y",
            "oracle_d",
            "predicted_d",
            "oracle_los_bin",
            "predicted_los_bin",
            "rare_oracle",
            "rare_predicted",
            "rare_mixed_D_pred",
            "rare_mixed_LOS_pred",
            "joint_mismatch_predicted",
            "discharge_confidence",
            "los_confidence",
            "confidence_product",
            "los_bin_changed",
            "d_value_changed",
        ],
    )
    _write_csv(
        focused_mismatch_csv,
        all_focused_mismatch_rows,
        [
            "split",
            "row_idx",
            "caseid",
            "target_name",
            "y",
            "oracle_d",
            "predicted_d",
            "oracle_los_bin",
            "predicted_los_bin",
            "rare_oracle",
            "rare_predicted",
            "rare_mixed_D_pred",
            "rare_mixed_LOS_pred",
            "joint_mismatch_predicted",
            "discharge_confidence",
            "los_confidence",
            "confidence_product",
            "los_bin_changed",
            "d_value_changed",
        ],
    )

    command = create_extraction_command(
        Path(script_path) if script_path is not None else Path("scripts/analyze_joint_plausibility_report.py"),
        diagnostic_path,
        split=None if split == "overall" else split,
        target_names=head_names,
        discharge_confidence_min=discharge_confidence_min,
        los_confidence_min=los_confidence_min,
        limit=limit,
        top_k=top_k,
    )
    write_extraction_command_script(extraction_script, command=command)

    summary_payload = {
        "diagnostic_dir": str(diagnostic_path),
        "split": split,
        "top_k": top_k,
        "heads": head_names,
        "discharge_confidence_min": discharge_confidence_min,
        "los_confidence_min": los_confidence_min,
        "limit": limit,
        "num_report_heads": len(head_report_rows),
        "num_focused_heads": len(focused_head_rows),
        "num_high_confidence_mismatches": len(focused_mismatch_rows),
        "num_focused_mismatches": len(all_focused_mismatch_rows),
        "artifacts": ReportArtifacts(
            head_report_csv=str(head_report_csv),
            head_report_md=str(head_report_md),
            high_confidence_mismatches_csv=str(high_conf_mismatch_csv),
            focused_heads_summary_csv=str(focused_summary_csv),
            focused_heads_prob_deltas_csv=str(focused_prob_csv),
            focused_heads_mismatches_csv=str(focused_mismatch_csv),
            extraction_command_sh=str(extraction_script),
            summary_json=str(summary_json),
        ).__dict__,
        "high_confidence_extraction_command": command,
    }
    _write_json(summary_json, summary_payload)
    return summary_payload
