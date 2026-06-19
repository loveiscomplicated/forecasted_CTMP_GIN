from __future__ import annotations

import numpy as np
import pandas as pd

from src.analysis.los_breakdown_diagnostics import (
    LOS_PRED_BIN_COL,
    LOS_TRUE_BIN_COL,
    binary_metrics_from_arrays,
    build_los_confusion_tables,
    build_middle_to_long_flow_summary,
    build_population_contamination,
    compute_drift_decomposition,
    compute_outcome_metrics_by_los_bin,
    map_los_to_breakdown9,
    map_los_to_coarse6,
    merge_outcome_with_joint_predictions,
)


def _joint_df(pred_los: list[int], pred_sub1: list[int]) -> pd.DataFrame:
    true_los = [8, 9, 16, 24, 30, 36]
    return pd.DataFrame(
        {
            "row_idx": list(range(len(true_los))),
            "true_los": true_los,
            "pred_los": pred_los,
            "true_SUB1_D": [0, 0, 1, 1, 2, 2],
            "pred_SUB1_D": pred_sub1,
            "true_SERVICES_D": [1, 1, 1, 1, 1, 1],
            "pred_SERVICES_D": [1, 1, 1, 1, 1, 1],
        }
    )


def _outcome_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_idx": list(range(6)),
            "y_true": [0, 1, 0, 1, 1, 0],
            "y_pred": [0, 1, 0, 1, 0, 0],
            "y_score": [0.1, 0.8, 0.2, 0.7, 0.4, 0.3],
        }
    )


def test_los_mapping_supports_raw_representative_and_encoded_values() -> None:
    assert map_los_to_coarse6(1) == 0
    assert map_los_to_coarse6(4) == 1
    assert map_los_to_coarse6(33) == 5
    assert map_los_to_coarse6(5) == 1
    assert map_los_to_breakdown9(36) == 8
    assert map_los_to_breakdown9(31) == 5
    assert map_los_to_breakdown9(8) == 2
    assert map_los_to_breakdown9(7) == 1


def test_confusion_counts_and_row_percentages_are_correct() -> None:
    joint = _joint_df(pred_los=[2, 5, 6, 4, 5, 8], pred_sub1=[0, 0, 1, 1, 2, 2])
    merged, _ = merge_outcome_with_joint_predictions(_outcome_df(), joint, target_scheme="breakdown9", context="test")
    counts_df, row_pct_df = build_los_confusion_tables(merged, run_name="r", scheme="breakdown9")
    count_8_14_to_29_31 = counts_df[(counts_df["true_bin"] == 2) & (counts_df["pred_bin"] == 5)]["count"].iloc[0]
    assert count_8_14_to_29_31 == 1
    row_pct = row_pct_df[(row_pct_df["true_bin"] == 2) & (row_pct_df["pred_bin"] == 5)]["row_pct"].iloc[0]
    assert row_pct == 0.5


def test_population_contamination_is_column_normalized() -> None:
    joint = _joint_df(pred_los=[5, 5, 6, 4, 5, 8], pred_sub1=[0, 0, 1, 1, 2, 2])
    merged, _ = merge_outcome_with_joint_predictions(_outcome_df(), joint, target_scheme="breakdown9", context="test")
    counts_df, _ = build_los_confusion_tables(merged, run_name="r", scheme="breakdown9")
    contamination = build_population_contamination(counts_df, run_name="r", scheme="breakdown9")
    pred_29_31 = contamination[contamination["pred_bin"] == 5]
    assert np.isclose(pred_29_31["share_within_pred_bin"].sum(), 1.0)


def test_middle_to_long_flow_summary_tracks_middle_bins() -> None:
    joint = _joint_df(pred_los=[5, 5, 6, 4, 5, 8], pred_sub1=[0, 0, 1, 1, 2, 2])
    merged, _ = merge_outcome_with_joint_predictions(_outcome_df(), joint, target_scheme="breakdown9", context="test")
    counts_df, _ = build_los_confusion_tables(merged, run_name="r", scheme="breakdown9")
    flow = build_middle_to_long_flow_summary(counts_df, run_name="r", scheme="breakdown9")
    row = flow[flow["true_bin"] == 2].iloc[0]
    assert row["support"] == 2
    assert row["total_to_long_count"] == 2
    assert row["total_to_long_pct"] == 1.0


def test_binary_metrics_by_los_bin_sets_auc_nan_for_single_class_bins() -> None:
    joint = _joint_df(pred_los=[2, 2, 3, 4, 5, 8], pred_sub1=[0, 0, 1, 1, 2, 2])
    outcome = pd.DataFrame(
        {
            "row_idx": list(range(6)),
            "y_true": [1, 1, 1, 1, 0, 0],
            "y_pred": [1, 1, 1, 0, 0, 0],
            "y_score": [0.8, 0.9, 0.7, 0.4, 0.2, 0.3],
        }
    )
    merged, _ = merge_outcome_with_joint_predictions(outcome, joint, target_scheme="breakdown9", context="test")
    metrics_df = compute_outcome_metrics_by_los_bin(merged, run_name="r", split="test", scheme="breakdown9")
    row = metrics_df[(metrics_df["los_basis"] == "true_los_bin") & (metrics_df["los_bin"] == 2)].iloc[0]
    assert np.isnan(row["auc"])


def test_drift_decomposition_separates_los_only_and_structured_d_error() -> None:
    los_only = _joint_df(pred_los=[5, 5, 6, 4, 5, 8], pred_sub1=[0, 0, 1, 1, 2, 2])
    los_only_drift = compute_drift_decomposition(los_only, run_name="a", run_type="target", native_scheme="breakdown9")
    los_only_row = los_only_drift[(los_only_drift["head"] == "SUB1_D") & (los_only_drift["los_scheme_eval"] == "breakdown9")].iloc[0]
    assert abs(los_only_row["dV_D"]) < 1e-9
    assert abs(los_only_row["dV_LOS"]) > 0

    structured = _joint_df(pred_los=[5, 5, 6, 4, 5, 8], pred_sub1=[2, 2, 2, 1, 2, 2])
    structured_drift = compute_drift_decomposition(structured, run_name="b", run_type="target", native_scheme="breakdown9")
    structured_row = structured_drift[(structured_drift["head"] == "SUB1_D") & (structured_drift["los_scheme_eval"] == "breakdown9")].iloc[0]
    assert structured_row["dV_D"] > los_only_row["dV_D"]


def test_binary_metrics_from_arrays_empty_case() -> None:
    metrics = binary_metrics_from_arrays(np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=float))
    assert metrics["support"] == 0
    assert np.isnan(metrics["auc"])
