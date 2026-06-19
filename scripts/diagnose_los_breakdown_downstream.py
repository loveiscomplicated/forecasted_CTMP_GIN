from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.los_breakdown_diagnostics import (  # noqa: E402
    RunSpec,
    WarningCollector,
    build_long_stay_recall_comparison,
    build_los_confusion_tables,
    build_middle_to_long_flow_summary,
    build_population_contamination,
    compute_drift_decomposition,
    compute_outcome_metrics_by_los_bin,
    discover_joint_predictions,
    downstream_metric_row,
    load_fold_metrics,
    load_or_compute_outcome_predictions,
    los_scheme_labels,
    merge_outcome_with_joint_predictions,
    simple_markdown_table,
    summarize_run_level,
    try_git_commit,
)


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _resolve_path(path_text: str | Path | None) -> Path | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _find_run_dir(runs_root: Path, pattern: str) -> Path | None:
    matches = sorted(path for path in runs_root.glob(pattern) if path.is_dir())
    return matches[0] if matches else None


def _resolve_predictor_dir(runs_root: Path, run_name: str | None) -> Path | None:
    if not run_name:
        return None
    direct = runs_root / run_name
    if direct.exists():
        return direct
    matches = sorted(path for path in runs_root.rglob(run_name) if path.is_dir())
    return matches[0] if matches else None


def _resolve_additional_run_specs(runs_root: Path, include_flag: bool) -> list[RunSpec]:
    if not include_flag:
        return []
    specs: list[RunSpec] = []
    for run_id in (9, 22, 38):
        root = _find_run_dir(runs_root, f"*joint_fresh_id{run_id}*")
        if root is None:
            continue
        fold_dir = root / "folds" / "fold_0"
        if not fold_dir.exists():
            continue
        specs.append(
            RunSpec(
                run_name=f"id{run_id}",
                run_dir=fold_dir,
                run_type=f"id{run_id}",
                predictor_run_name=None,
                predictor_run_dir=None,
                los_scheme="coarse6",
            )
        )
    return specs


def _correlation_rows(summary_df: pd.DataFrame) -> pd.DataFrame:
    try:
        from scipy.stats import pearsonr, spearmanr
    except Exception:
        return pd.DataFrame()
    usable = summary_df.dropna(subset=["downstream_test_auc"]).copy()
    if len(usable) < 3:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for col in ("mean_dV_D", "max_dV_D", "mean_js_D", "mean_abs_dV_D"):
        pair = usable.dropna(subset=[col])
        if len(pair) < 3:
            continue
        pearson = pearsonr(pair[col], pair["downstream_test_auc"])
        spearman = spearmanr(pair[col], pair["downstream_test_auc"])
        rows.append(
            {
                "metric": col,
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
                "spearman_rho": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "n": int(len(pair)),
            }
        )
    return pd.DataFrame(rows)


def _prepare_run_spec(
    *,
    run_name: str,
    run_dir: Path,
    run_type: str,
    predictor_run_name: str | None,
    predictor_run_dir: Path | None,
    explicit_scheme: str | None,
) -> RunSpec:
    return RunSpec(
        run_name=run_name,
        run_dir=run_dir,
        run_type=run_type,
        predictor_run_name=predictor_run_name,
        predictor_run_dir=predictor_run_dir,
        los_scheme=explicit_scheme,
    )


def _analyze_run(
    spec: RunSpec,
    *,
    dataset_root: Path,
    split: str,
    warnings: WarningCollector,
    device: str | None,
) -> dict[str, Any]:
    files_read: list[str] = []
    downstream_metrics = load_fold_metrics(spec.run_dir)
    joint_df = discover_joint_predictions(
        downstream_run_dir=spec.run_dir,
        predictor_run_dir=spec.predictor_run_dir,
        split=split,
        warnings=warnings,
    )
    joint_source = spec.run_dir / "joint_predictor" / f"{'val' if split == 'valid' else split}_predictions.csv"
    if joint_source.exists():
        files_read.append(str(joint_source))
    outcome_df, outcome_files = load_or_compute_outcome_predictions(
        fold_dir=spec.run_dir,
        dataset_root=dataset_root,
        split=split,
        warnings=warnings,
        device=device,
    )
    files_read.extend(outcome_files)
    scheme = spec.los_scheme
    if scheme is None:
        pred_values = pd.to_numeric(joint_df["pred_los"], errors="coerce").dropna().to_numpy(dtype=np.int64)
        scheme = "coarse6" if (pred_values.size == 0 or int(pred_values.max()) <= 5) else "breakdown9"
    merged_df, los_info = merge_outcome_with_joint_predictions(
        outcome_df,
        joint_df,
        target_scheme=scheme,
        context=spec.run_name,
    )
    confusion_counts, confusion_row_pct = build_los_confusion_tables(
        merged_df,
        run_name=spec.run_name,
        scheme=scheme,
    )
    contamination = build_population_contamination(confusion_counts, run_name=spec.run_name, scheme=scheme)
    middle_flow = build_middle_to_long_flow_summary(confusion_counts, run_name=spec.run_name, scheme=scheme)
    outcome_metrics = compute_outcome_metrics_by_los_bin(
        merged_df,
        run_name=spec.run_name,
        split=split,
        scheme=scheme,
    )
    drift_df = compute_drift_decomposition(
        joint_df,
        run_name=spec.run_name,
        run_type=spec.run_type,
        native_scheme=scheme,
    )
    run_summary = summarize_run_level(
        run_name=spec.run_name,
        run_type=spec.run_type,
        los_scheme=scheme,
        downstream_metrics={
            "downstream_test_auc": downstream_metrics.get("test_auc"),
            "downstream_test_acc": downstream_metrics.get("test_acc"),
            "downstream_test_f1": downstream_metrics.get("test_f1"),
            "downstream_test_precision": downstream_metrics.get("test_precision"),
            "downstream_test_recall": downstream_metrics.get("test_recall"),
            "downstream_test_loss": downstream_metrics.get("test_loss"),
            "best_epoch": downstream_metrics.get("best_epoch"),
        },
        merged_df=merged_df,
        drift_df=drift_df,
        outcome_metrics_df=outcome_metrics,
        middle_flow_df=middle_flow,
    )
    return {
        "spec": spec,
        "downstream_metrics_row": downstream_metric_row(spec.run_name, spec.run_type, downstream_metrics),
        "merged_df": merged_df,
        "los_info": los_info,
        "confusion_counts": confusion_counts,
        "confusion_row_pct": confusion_row_pct,
        "contamination": contamination,
        "middle_flow": middle_flow,
        "outcome_metrics": outcome_metrics,
        "drift_df": drift_df,
        "run_summary": run_summary,
        "files_read": files_read,
    }


def _render_report(
    *,
    target_name: str,
    baseline_name: str | None,
    run_metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    long_stay_df: pd.DataFrame,
    middle_flow_df: pd.DataFrame,
    drift_df: pd.DataFrame,
    counts_df: pd.DataFrame,
    artifacts: list[dict[str, Any]],
) -> str:
    target_row = summary_df.loc[summary_df["run_name"] == target_name].iloc[0]
    baseline_row = summary_df.loc[summary_df["run_name"] == baseline_name].iloc[0] if baseline_name and baseline_name in set(summary_df["run_name"]) else None
    auc_delta = (
        float(target_row["downstream_test_auc"] - baseline_row["downstream_test_auc"])
        if baseline_row is not None and pd.notna(target_row["downstream_test_auc"]) and pd.notna(baseline_row["downstream_test_auc"])
        else np.nan
    )
    target_long = _long_recall_value(long_stay_df, target_name)
    baseline_long = _long_recall_value(long_stay_df, baseline_name) if baseline_name else np.nan
    long_delta = target_long - baseline_long if pd.notna(target_long) and pd.notna(baseline_long) else np.nan
    target_mid = _middle_flow_value(middle_flow_df, target_name)
    baseline_mid = _middle_flow_value(middle_flow_df, baseline_name) if baseline_name else np.nan
    mid_delta = target_mid - baseline_mid if pd.notna(target_mid) and pd.notna(baseline_mid) else np.nan

    target_drift = _coarse6_compare_mean_dv(drift_df, target_name)
    baseline_drift = _coarse6_compare_mean_dv(drift_df, baseline_name) if baseline_name else np.nan
    drift_delta = target_drift - baseline_drift if pd.notna(target_drift) and pd.notna(baseline_drift) else np.nan

    if pd.notna(auc_delta) and auc_delta <= 0.002:
        auc_sentence = f"No. The new test AUC is {target_row['downstream_test_auc']:.4f}, only a marginal increase over the previous {baseline_row['downstream_test_auc']:.4f} baseline."
    else:
        auc_sentence = f"The new test AUC is {target_row['downstream_test_auc']:.4f} versus {baseline_row['downstream_test_auc']:.4f}."

    if pd.isna(long_delta):
        long_sentence = "Long-stay recall could not be compared because one side is missing subgroup predictions."
    elif long_delta > 0:
        long_sentence = f"Long-stay recall improved by {long_delta:.4f}."
    else:
        long_sentence = f"Long-stay recall did not improve; delta={long_delta:.4f}."

    if pd.isna(mid_delta):
        mid_sentence = "Middle-to-long misassignment could not be compared directly."
    elif mid_delta < 0:
        mid_sentence = f"Middle-to-long flow decreased by {-mid_delta:.4f}, indicating less long-stay sink behavior."
    else:
        mid_sentence = f"Middle-to-long flow remained high or worsened; delta={mid_delta:.4f}."

    if pd.isna(drift_delta):
        drift_sentence = "Direct coarse6-comparable dV_D comparison was unavailable."
    elif drift_delta < 0:
        drift_sentence = f"dV_D decreased on the coarse6-comparable view by {-drift_delta:.6f}."
    else:
        drift_sentence = f"dV_D did not decrease on the coarse6-comparable view; delta={drift_delta:.6f}."

    if pd.notna(auc_delta) and auc_delta <= 0.002 and (pd.isna(long_delta) or long_delta <= 0) and (pd.isna(drift_delta) or drift_delta >= 0):
        conclusion = "9-bin does not materially address the failure mode."
    elif pd.notna(auc_delta) and auc_delta <= 0.002 and pd.notna(long_delta) and long_delta > 0:
        conclusion = "9-bin partially fixes LOS representation, but D-side/joint assignment mismatch remains."
    elif pd.notna(drift_delta) and drift_delta < 0 and pd.notna(auc_delta) and auc_delta <= 0.002:
        conclusion = "dV_D reduction alone is not sufficient, or the downstream task is insensitive to this drift axis."
    else:
        conclusion = "9-bin is better interpreted as a diagnostic probe than a main solution unless longer-stay behavior improves consistently."

    report_lines = [
        "# LOS Breakdown Downstream Diagnostic",
        "",
        "## Executive summary",
        f"- {auc_sentence}",
        f"- {long_sentence}",
        f"- {mid_sentence}",
        f"- {drift_sentence}",
        f"- Conclusion: {conclusion}",
        "",
        "## Run paths and artifacts used",
    ]
    for artifact in artifacts:
        report_lines.append(f"- `{artifact['spec'].run_name}`: `{artifact['spec'].run_dir}`")
    report_lines.extend(
        [
            "",
            "## Overall downstream metric comparison",
            simple_markdown_table(run_metrics_df.sort_values("test_auc", ascending=False)),
            "",
            "## LOS confusion matrix summary",
            simple_markdown_table(counts_df.groupby(["run_name", "true_bin_label", "pred_bin_label"], as_index=False)["count"].sum().head(24)),
            "",
            "## Middle-to-long LOS misassignment",
            simple_markdown_table(middle_flow_df),
            "",
            "## True LOS-bin outcome metrics",
            simple_markdown_table(
                artifacts[0]["outcome_metrics"].query("los_basis == 'true_los_bin'")[["run_name", "los_bin_label", "support", "recall", "f1", "auc"]]
            ),
            "",
            "## Predicted LOS-bin outcome metrics",
            simple_markdown_table(
                artifacts[0]["outcome_metrics"].query("los_basis == 'pred_los_bin'")[["run_name", "los_bin_label", "support", "recall", "f1", "auc"]]
            ),
            "",
            "## Long-stay recall comparison",
            simple_markdown_table(long_stay_df),
            "",
            "## dV_LOS / dV_D decomposition",
            simple_markdown_table(
                drift_df.sort_values(["run_name", "los_scheme_eval", "dV_D"], ascending=[True, True, False])[
                    ["run_name", "los_scheme_eval", "head", "acc", "dV_LOS", "dV_D", "js_D"]
                ].head(24)
            ),
            "",
            "## Direct coarse6 ID26 vs 9-bin ID26 conclusion",
            f"- Did 9-bin improve overall AUC enough? {auc_sentence}",
            f"- Did 9-bin improve long-stay recall? {long_sentence}",
            f"- Did 9-bin reduce middle-to-long misassignment? {mid_sentence}",
            f"- Did dV_D decrease? {drift_sentence}",
            f"- Is 9-bin a main solution or only a diagnostic improvement? {conclusion}",
            "",
            "## Recommended next step",
            "- Use the 9-bin run as a diagnostic reference, but prioritize fixes that directly reduce D-side structured mismatch and long-stay sink routing together.",
        ]
    )
    correlation_df = _correlation_rows(summary_df)
    if not correlation_df.empty:
        report_lines.extend(["", "## Correlations", simple_markdown_table(correlation_df)])
    return "\n".join(report_lines) + "\n"


def _long_recall_value(long_df: pd.DataFrame, run_name: str | None) -> float:
    if run_name is None or long_df.empty:
        return np.nan
    subset = long_df[long_df["run_name"] == run_name]
    if subset.empty:
        return np.nan
    return float(pd.to_numeric(subset["recall"], errors="coerce").mean())


def _middle_flow_value(flow_df: pd.DataFrame, run_name: str | None) -> float:
    if run_name is None or flow_df.empty:
        return np.nan
    subset = flow_df[flow_df["run_name"] == run_name]
    if subset.empty:
        return np.nan
    return float(pd.to_numeric(subset["total_to_long_pct"], errors="coerce").mean())


def _coarse6_compare_mean_dv(drift_df: pd.DataFrame, run_name: str | None) -> float:
    if run_name is None or drift_df.empty:
        return np.nan
    subset = drift_df[(drift_df["run_name"] == run_name) & (drift_df["los_scheme_eval"] == "coarse6")]
    if subset.empty:
        return np.nan
    return float(pd.to_numeric(subset["dV_D"], errors="coerce").mean())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only LOS breakdown downstream diagnostic.")
    parser.add_argument("--runs-root", type=Path, default=PROJECT_ROOT / "runs")
    parser.add_argument("--registry-path", type=Path, default=None)
    parser.add_argument("--target-run-dir", type=Path, required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--baseline-run-dir", type=Path, default=None)
    parser.add_argument("--baseline-run-name", default=None)
    parser.add_argument("--predictor-run-name", default=None)
    parser.add_argument("--baseline-predictor-run-name", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--strict", type=_bool_arg, default=False)
    parser.add_argument("--include-id9-id22-id38", type=_bool_arg, default=False)
    parser.add_argument("--top-k-heads", type=int, default=5)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_ROOT / "src" / "data")
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warnings = WarningCollector(strict=bool(args.strict))
    runs_root = _resolve_path(args.runs_root) or args.runs_root
    out_dir = _resolve_path(args.out_dir) or args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    target_run_dir = _resolve_path(args.target_run_dir)
    baseline_run_dir = _resolve_path(args.baseline_run_dir) if args.baseline_run_dir else None
    if target_run_dir is None or not target_run_dir.exists():
        raise FileNotFoundError(f"Missing target run dir: {args.target_run_dir}")
    if baseline_run_dir is not None and not baseline_run_dir.exists():
        warnings.missing(baseline_run_dir, "baseline run dir")
        baseline_run_dir = None

    target_predictor_dir = _resolve_predictor_dir(runs_root, args.predictor_run_name)
    baseline_predictor_dir = _resolve_predictor_dir(runs_root, args.baseline_predictor_run_name)
    specs = [
        _prepare_run_spec(
            run_name=args.target_name,
            run_dir=target_run_dir,
            run_type="breakdown9_target",
            predictor_run_name=args.predictor_run_name,
            predictor_run_dir=target_predictor_dir,
            explicit_scheme="breakdown9",
        )
    ]
    if baseline_run_dir is not None and args.baseline_run_name:
        specs.append(
            _prepare_run_spec(
                run_name=args.baseline_run_name,
                run_dir=baseline_run_dir,
                run_type="coarse6_baseline",
                predictor_run_name=args.baseline_predictor_run_name,
                predictor_run_dir=baseline_predictor_dir,
                explicit_scheme="coarse6",
            )
        )
    specs.extend(_resolve_additional_run_specs(runs_root, bool(args.include_id9_id22_id38)))

    artifacts = [
        _analyze_run(
            spec,
            dataset_root=_resolve_path(args.dataset_root) or args.dataset_root,
            split=args.split,
            warnings=warnings,
            device=args.device,
        )
        for spec in specs
    ]

    run_metrics_df = pd.DataFrame([item["downstream_metrics_row"] for item in artifacts])
    outcome_metrics_df = pd.concat([item["outcome_metrics"] for item in artifacts], ignore_index=True)
    counts_df = pd.concat([item["confusion_counts"] for item in artifacts], ignore_index=True)
    row_pct_df = pd.concat([item["confusion_row_pct"] for item in artifacts], ignore_index=True)
    contamination_df = pd.concat([item["contamination"] for item in artifacts], ignore_index=True)
    middle_flow_df = pd.concat([item["middle_flow"] for item in artifacts], ignore_index=True)
    drift_df = pd.concat([item["drift_df"] for item in artifacts], ignore_index=True)
    summary_df = pd.DataFrame([item["run_summary"] for item in artifacts])
    long_stay_df = build_long_stay_recall_comparison(outcome_metrics_df, baseline_run_name=args.baseline_run_name)

    files_written = [
        out_dir / "run_metrics_comparison.csv",
        out_dir / "los_bin_outcome_metrics.csv",
        out_dir / "los_confusion_matrix_counts.csv",
        out_dir / "los_confusion_matrix_row_pct.csv",
        out_dir / "los_population_contamination.csv",
        out_dir / "middle_to_long_flow_summary.csv",
        out_dir / "long_stay_recall_comparison.csv",
        out_dir / "head_level_drift_decomposition.csv",
        out_dir / "run_level_drift_auc_summary.csv",
        out_dir / "diagnostic_report.md",
        out_dir / "manifest.json",
    ]
    run_metrics_df.to_csv(files_written[0], index=False)
    outcome_metrics_df.to_csv(files_written[1], index=False)
    counts_df.to_csv(files_written[2], index=False)
    row_pct_df.to_csv(files_written[3], index=False)
    contamination_df.to_csv(files_written[4], index=False)
    middle_flow_df.to_csv(files_written[5], index=False)
    long_stay_df.to_csv(files_written[6], index=False)
    drift_df.to_csv(files_written[7], index=False)
    summary_df.to_csv(files_written[8], index=False)

    report_text = _render_report(
        target_name=args.target_name,
        baseline_name=args.baseline_run_name,
        run_metrics_df=run_metrics_df,
        summary_df=summary_df,
        long_stay_df=long_stay_df,
        middle_flow_df=middle_flow_df,
        drift_df=drift_df,
        counts_df=counts_df,
        artifacts=artifacts,
    )
    files_written[9].write_text(report_text, encoding="utf-8")

    manifest = {
        "command_args": vars(args),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": try_git_commit(PROJECT_ROOT),
        "resolved_target_run_dir": str(target_run_dir),
        "resolved_baseline_run_dir": str(baseline_run_dir) if baseline_run_dir is not None else None,
        "resolved_predictor_run_dirs": {
            spec.run_name: str(spec.predictor_run_dir) if spec.predictor_run_dir is not None else None for spec in specs
        },
        "files_read": sorted({path for item in artifacts for path in item["files_read"]}),
        "files_written": [str(path) for path in files_written],
        "missing_artifacts": warnings.missing_artifacts,
        "warnings": warnings.warnings,
        "inferred_los_schemes": {item["spec"].run_name: item["los_info"]["target_scheme"] for item in artifacts},
        "row_counts": {item["spec"].run_name: item["los_info"]["rows_used"] for item in artifacts},
    }
    files_written[10].write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    compact = summary_df[["run_name", "downstream_test_auc", "long_stay_macro_recall", "middle_to_long_flow_pct", "top1_dV_D_head", "top1_dV_D"]].copy()
    print(compact.to_string(index=False))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
