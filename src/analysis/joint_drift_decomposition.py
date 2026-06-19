from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


LOS_TRUE_BIN_COL = "true_los_bin"
LOS_PRED_BIN_COL = "pred_los_bin"


def js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1.0e-12) -> float:
    """Return Jensen-Shannon divergence using base-2 logarithms.

    Empty-vs-empty inputs return 0.0. Empty-vs-nonempty inputs are treated as a
    zero-mass vector versus the normalized nonempty vector, which yields 0.5 in
    base-2 units for identical support.
    """
    lhs = np.asarray(p, dtype=np.float64)
    rhs = np.asarray(q, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"JS inputs must have the same shape, got {lhs.shape} and {rhs.shape}.")
    if np.any(lhs < 0) or np.any(rhs < 0):
        raise ValueError("JS inputs must be non-negative.")

    lhs_sum = float(lhs.sum())
    rhs_sum = float(rhs.sum())
    if lhs_sum <= 0.0 and rhs_sum <= 0.0:
        return 0.0
    lhs_prob = lhs / lhs_sum if lhs_sum > 0.0 else np.zeros_like(lhs, dtype=np.float64)
    rhs_prob = rhs / rhs_sum if rhs_sum > 0.0 else np.zeros_like(rhs, dtype=np.float64)
    mid = 0.5 * (lhs_prob + rhs_prob)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0.0
        if not np.any(mask):
            return 0.0
        return float(np.sum(a[mask] * (np.log2(np.clip(a[mask], eps, None)) - np.log2(np.clip(b[mask], eps, None)))))

    return 0.5 * (_kl(lhs_prob, mid) + _kl(rhs_prob, mid))


def cramers_v(table: np.ndarray) -> float:
    """Compute uncorrected Cramer's V for a contingency table.

    All-zero rows and columns are removed before computing the statistic. Tables
    with no observations, one populated row, or one populated column return 0.0.
    """
    counts = np.asarray(table, dtype=np.float64)
    if counts.ndim != 2:
        raise ValueError(f"Cramer's V requires a 2D table, got shape {counts.shape}.")
    if np.any(counts < 0):
        raise ValueError("Cramer's V table must be non-negative.")

    row_mask = counts.sum(axis=1) > 0.0
    col_mask = counts.sum(axis=0) > 0.0
    counts = counts[np.ix_(row_mask, col_mask)]
    total = float(counts.sum())
    if total <= 0.0 or counts.shape[0] < 2 or counts.shape[1] < 2:
        return 0.0

    row_sum = counts.sum(axis=1, keepdims=True)
    col_sum = counts.sum(axis=0, keepdims=True)
    expected = row_sum @ col_sum / total
    valid = expected > 0.0
    chi2 = float(np.sum(((counts - expected) ** 2)[valid] / expected[valid]))
    denom = float(min(counts.shape[0] - 1, counts.shape[1] - 1))
    if denom <= 0.0:
        return 0.0
    return float(np.sqrt(max(chi2 / total / denom, 0.0)))


def _resolve_classes(d_classes: Sequence[Any] | int) -> list[Any]:
    if isinstance(d_classes, int):
        if d_classes < 0:
            raise ValueError(f"Number of D classes must be non-negative, got {d_classes}.")
        return list(range(d_classes))
    classes = list(d_classes)
    if len(classes) != len(set(classes)):
        raise ValueError("D classes must be unique.")
    return classes


def contingency_table(
    d_values: Sequence[Any] | np.ndarray | pd.Series,
    los_values: Sequence[Any] | np.ndarray | pd.Series,
    d_classes: Sequence[Any] | int,
    n_los_bins: int,
) -> np.ndarray:
    """Return a [num_d_classes, n_los_bins] count table."""
    classes = _resolve_classes(d_classes)
    if n_los_bins <= 0:
        raise ValueError(f"n_los_bins must be positive, got {n_los_bins}.")
    class_to_idx = {value: idx for idx, value in enumerate(classes)}
    table = np.zeros((len(classes), int(n_los_bins)), dtype=np.float64)
    d_arr = np.asarray(d_values)
    los_arr = np.asarray(los_values)
    if d_arr.shape[0] != los_arr.shape[0]:
        raise ValueError(f"D and LOS arrays must have equal length, got {d_arr.shape[0]} and {los_arr.shape[0]}.")

    for d_value, los_value in zip(d_arr.tolist(), los_arr.tolist()):
        if d_value not in class_to_idx:
            continue
        los_idx = int(los_value)
        if 0 <= los_idx < n_los_bins:
            table[class_to_idx[d_value], los_idx] += 1.0
    return table


def weighted_column_js(A: np.ndarray, B: np.ndarray, weight_table: np.ndarray | None = None) -> float:
    """Return support-weighted average JS over LOS columns.

    Each column is normalized independently inside ``js_divergence``. Empty
    columns according to ``weight_table`` are skipped. When all columns are empty,
    the function returns 0.0.
    """
    lhs = np.asarray(A, dtype=np.float64)
    rhs = np.asarray(B, dtype=np.float64)
    if lhs.shape != rhs.shape:
        raise ValueError(f"Column-JS tables must have the same shape, got {lhs.shape} and {rhs.shape}.")
    weights_source = lhs if weight_table is None else np.asarray(weight_table, dtype=np.float64)
    if weights_source.shape != lhs.shape:
        raise ValueError(
            f"weight_table shape must match input tables, got {weights_source.shape} and {lhs.shape}."
        )

    weighted_total = 0.0
    support_total = 0.0
    for col_idx in range(lhs.shape[1]):
        support = float(weights_source[:, col_idx].sum())
        if support <= 0.0:
            continue
        weighted_total += support * js_divergence(lhs[:, col_idx], rhs[:, col_idx])
        support_total += support
    if support_total <= 0.0:
        return 0.0
    return float(weighted_total / support_total)


def _sorted_unique(values: np.ndarray) -> list[Any]:
    uniques = pd.Series(values).dropna().unique().tolist()
    try:
        return sorted(uniques)
    except TypeError:
        return sorted(uniques, key=lambda x: str(x))


def _as_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def decompose_head_drift(df: pd.DataFrame, head: str, n_los_bins: int) -> dict[str, Any]:
    """Decompose one discharge head into LOS-attributable and D-attributable drift.

    The input dataframe must contain ``true_<head>``, ``pred_<head>``, and either
    normalized ``true_los_bin``/``pred_los_bin`` columns or raw ``true_los``/
    ``pred_los`` columns that are already in the same 0-based bin space.
    """
    true_col = f"true_{head}"
    pred_col = f"pred_{head}"
    missing = [col for col in (true_col, pred_col) if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required D columns for head {head!r}: {missing}")

    true_los_col = LOS_TRUE_BIN_COL if LOS_TRUE_BIN_COL in df.columns else "true_los"
    pred_los_col = LOS_PRED_BIN_COL if LOS_PRED_BIN_COL in df.columns else "pred_los"
    missing_los = [col for col in (true_los_col, pred_los_col) if col not in df.columns]
    if missing_los:
        raise ValueError(f"Missing required LOS columns for head {head!r}: {missing_los}")

    work = pd.DataFrame(
        {
            "td": _as_numeric_series(df, true_col),
            "pd": _as_numeric_series(df, pred_col),
            "tl": _as_numeric_series(df, true_los_col),
            "pl": _as_numeric_series(df, pred_los_col),
        }
    )
    original_rows = int(len(work))
    valid = work.notna().all(axis=1)
    valid &= work["tl"].between(0, n_los_bins - 1)
    valid &= work["pl"].between(0, n_los_bins - 1)
    work = work.loc[valid].copy()
    if work.empty:
        return {
            "head": head,
            "n_rows": 0,
            "num_classes": 0,
            "unique_true_classes": 0,
            "unique_pred_classes": 0,
            "coverage_ratio": 0.0 if original_rows else 1.0,
            "acc": np.nan,
            "V_oracle": np.nan,
            "V_mid": np.nan,
            "V_full": np.nan,
            "dV_LOS": np.nan,
            "dV_D": np.nan,
            "abs_dV_D": np.nan,
            "dV_total": np.nan,
            "js_LOS": np.nan,
            "js_D": np.nan,
            "js_total": np.nan,
            "support_min": 0.0,
            "support_max": 0.0,
        }

    for col in ("td", "pd", "tl", "pl"):
        work[col] = work[col].astype(int)

    d_classes = _sorted_unique(np.concatenate([work["td"].to_numpy(), work["pd"].to_numpy()]))
    td = work["td"].to_numpy()
    pd_values = work["pd"].to_numpy()
    tl = work["tl"].to_numpy()
    pl = work["pl"].to_numpy()

    T_oracle = contingency_table(td, tl, d_classes, n_los_bins)
    T_mid = contingency_table(td, pl, d_classes, n_los_bins)
    T_full = contingency_table(pd_values, pl, d_classes, n_los_bins)
    V_oracle = cramers_v(T_oracle)
    V_mid = cramers_v(T_mid)
    V_full = cramers_v(T_full)
    dV_LOS = V_mid - V_oracle
    dV_D = V_full - V_mid
    support = T_mid.sum(axis=0)

    return {
        "head": head,
        "n_rows": int(len(work)),
        "num_classes": int(len(d_classes)),
        "unique_true_classes": int(pd.Series(td).nunique()),
        "unique_pred_classes": int(pd.Series(pd_values).nunique()),
        "coverage_ratio": float(len(work) / original_rows) if original_rows else 1.0,
        "acc": float(np.mean(td == pd_values)),
        "V_oracle": float(V_oracle),
        "V_mid": float(V_mid),
        "V_full": float(V_full),
        "dV_LOS": float(dV_LOS),
        "dV_D": float(dV_D),
        "abs_dV_D": float(abs(dV_D)),
        "dV_total": float(V_full - V_oracle),
        "js_LOS": weighted_column_js(T_oracle, T_mid, T_oracle),
        "js_D": weighted_column_js(T_mid, T_full, T_mid),
        "js_total": weighted_column_js(T_oracle, T_full, T_oracle),
        "support_min": float(np.min(support)) if support.size else 0.0,
        "support_max": float(np.max(support)) if support.size else 0.0,
    }
