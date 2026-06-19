"""
los_group_detection.py

Detect LOS breakpoints from GatedFusion weight patterns using change-point detection.

Input:
    src/analysis/gated_fusion_w_los_seed_1.csv
    Multi-level columns: (w_ad, mean), (w_dis, mean), (w_merged, mean) for LOS 1-37

Algorithm:
    Primary:   ruptures.Pelt(model="rbf").predict(pen=beta) on 37x3 signal
               Selects the first penalty in pen_grid that yields 2-6 groups.
    Fallback:  sklearn KMeans(k=3) -> assign each LOS to nearest cluster,
               then convert consecutive cluster labels to LOS ranges.

Output:
    src/analysis/los_groups.json   — {"group_0": [1], "group_1": [2,...], ...}
    src/analysis/resources/los_group_detection.png  — visualization

Usage:
    python src/analysis/los_group_detection.py
    python src/analysis/los_group_detection.py --k_kmeans 4 --pen_grid "1,3,5,10"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
# Default: k-fold based CSV (preferred). Falls back to single-split CSV.
CSV_PATH = _THIS_DIR / "gated_fusion_w_los_kfold.csv"
CSV_PATH_FALLBACK = _THIS_DIR / "gated_fusion_w_los_seed_1.csv"
OUTPUT_JSON = _THIS_DIR / "los_groups.json"
OUTPUT_PLOT = _THIS_DIR / "resources" / "los_group_detection.png"


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def _load_flat_csv(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse flat CSV from extract_gated_fusion_kfold.py.
    Expected columns: LOS, w_ad_mean, w_dis_mean, w_merged_mean
    """
    df = df.sort_values("LOS")
    los_values = df["LOS"].astype(int).values
    signal = np.stack(
        [
            df["w_ad_mean"].values.astype(np.float64),
            df["w_dis_mean"].values.astype(np.float64),
            df["w_merged_mean"].values.astype(np.float64),
        ],
        axis=1,
    )
    return los_values, signal


def _load_multilevel_csv(df_raw: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse legacy multi-level CSV (gated_fusion_w_los_seed_1.csv).
    Header: header=[0,1], index_col=0 → MultiIndex columns (w_ad, mean) etc.
    """
    df_raw.index = df_raw.index.astype(float)
    df_raw = df_raw.sort_index()
    los_values = df_raw.index.astype(int).values
    signal = np.stack(
        [
            df_raw[("w_ad", "mean")].values.astype(np.float64),
            df_raw[("w_dis", "mean")].values.astype(np.float64),
            df_raw[("w_merged", "mean")].values.astype(np.float64),
        ],
        axis=1,
    )
    return los_values, signal


def load_signal(csv_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load GatedFusion weight summary CSV and extract 3D signal per LOS.

    Supports two formats:
      - Flat CSV (from extract_gated_fusion_kfold.py):
            columns: LOS, w_ad_mean, w_dis_mean, w_merged_mean, ...
      - Multi-level CSV (legacy gated_fusion_w_los_seed_1.csv):
            header=[0,1], index=LOS, columns: (w_ad,mean), (w_dis,mean), ...

    Returns:
        los_values: int array [T]
        signal:     float64 array [T, 3] — w_ad_mean, w_dis_mean, w_merged_mean
    """
    # Try flat format first (has explicit "LOS" and "w_ad_mean" columns)
    try:
        df_flat = pd.read_csv(csv_path)
        if "LOS" in df_flat.columns and "w_ad_mean" in df_flat.columns:
            print(f"  Detected flat CSV format (k-fold based)")
            los_values, signal = _load_flat_csv(df_flat)
        else:
            # Multi-level format
            print(f"  Detected multi-level CSV format (single-split based)")
            df_ml = pd.read_csv(csv_path, header=[0, 1], index_col=0)
            los_values, signal = _load_multilevel_csv(df_ml)
    except Exception as e:
        raise ValueError(f"Could not parse {csv_path}: {e}")

    # Drop rows with NaN
    valid = ~np.isnan(signal).any(axis=1)
    if not valid.all():
        print(f"  Warning: dropping {(~valid).sum()} LOS rows with NaN")
        los_values = los_values[valid]
        signal = signal[valid]

    return los_values, signal


# -----------------------------------------------------------------------
# Change-point detection
# -----------------------------------------------------------------------

def detect_breakpoints_ruptures(
    signal: np.ndarray,
    pen_grid: List[float],
) -> Tuple[List[int], float]:
    """
    Apply ruptures.Pelt to find change-points in the 3D weight signal.

    Args:
        signal:   [T, 3] float array
        pen_grid: list of penalty values to try

    Returns:
        breakpoints: list of 0-indexed positions (split BEFORE this position)
        pen_used:    the penalty value selected
    """
    import ruptures as rpt

    algo = rpt.Pelt(model="rbf").fit(signal)

    selected_bkps = None
    selected_pen = pen_grid[-1]

    for pen in pen_grid:
        bkps = algo.predict(pen=pen)  # last element is T (end marker)
        n_groups = len(bkps)          # includes end marker, so n_groups = actual groups
        if 2 <= n_groups <= 6:
            selected_bkps = bkps[:-1]  # remove end marker
            selected_pen = pen
            break

    if selected_bkps is None:
        # Fallback: use middle penalty
        mid_pen = pen_grid[len(pen_grid) // 2]
        bkps = algo.predict(pen=mid_pen)
        selected_bkps = bkps[:-1]
        selected_pen = mid_pen
        print(f"  No penalty gave 2-6 groups; using pen={mid_pen}, groups={len(bkps)}")

    return selected_bkps, selected_pen


def detect_breakpoints_kmeans(
    signal: np.ndarray,
    k: int,
) -> List[int]:
    """
    Fallback: KMeans clustering on the signal rows, then find where cluster labels change.

    Returns:
        breakpoints: list of 0-indexed positions where a new cluster begins
    """
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    labels = km.fit_predict(signal)  # [T]

    breakpoints = []
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            breakpoints.append(i)

    return breakpoints


# -----------------------------------------------------------------------
# Group construction
# -----------------------------------------------------------------------

def breakpoints_to_los_groups(
    breakpoints: List[int],
    los_values: np.ndarray,
) -> Dict[str, List[int]]:
    """
    Convert 0-indexed breakpoint positions into LOS groups.

    Example:
        los_values = [1, 2, ..., 37]
        breakpoints = [1, 7]
        -> group_0: [1]
        -> group_1: [2, 3, 4, 5, 6, 7]
        -> group_2: [8, ..., 37]
    """
    T = len(los_values)
    starts = [0] + sorted(breakpoints)
    ends = sorted(breakpoints) + [T]

    groups: Dict[str, List[int]] = {}
    for i, (s, e) in enumerate(zip(starts, ends)):
        groups[f"group_{i}"] = los_values[s:e].tolist()

    return groups


# -----------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------

def plot_groups(
    signal: np.ndarray,
    los_values: np.ndarray,
    groups: Dict[str, List[int]],
    output_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    weight_labels = ["w_ad (mean)", "w_dis (mean)", "w_merged (mean)"]
    colors = cm.tab10.colors

    for ax, sig_col, label in zip(axes, signal.T, weight_labels):
        ax.plot(los_values, sig_col, "k-", lw=1.5)
        ax.set_ylabel(label, fontsize=10)

        for i, (gname, glos) in enumerate(groups.items()):
            lo = min(glos) - 0.5
            hi = max(glos) + 0.5
            ax.axvspan(lo, hi, alpha=0.15, color=colors[i % 10],
                       label=gname if ax is axes[0] else None)

    # Vertical lines at group boundaries
    for ax in axes:
        for gname, glos in list(groups.items())[1:]:
            ax.axvline(min(glos) - 0.5, color="red", linestyle="--", lw=1)

    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("LOS (days)", fontsize=10)
    fig.suptitle("GatedFusion Weight Patterns and LOS Groups\n(Change-point Detection)", fontsize=12)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# -----------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------

def main():
    # Prefer k-fold based CSV; fall back to single-split if not yet generated
    default_csv = str(CSV_PATH) if CSV_PATH.exists() else str(CSV_PATH_FALLBACK)

    p = argparse.ArgumentParser(description="Detect LOS groups from GatedFusion weight patterns")
    p.add_argument("--csv", type=str, default=default_csv,
                   help="Path to GatedFusion LOS summary CSV (flat or multi-level format)")
    p.add_argument("--output", type=str, default=str(OUTPUT_JSON),
                   help="Output JSON path for LOS groups")
    p.add_argument("--plot", type=str, default=str(OUTPUT_PLOT),
                   help="Output path for visualization")
    p.add_argument("--pen_grid", type=str, default="1,2,3,5,8,10,15,20",
                   help="Comma-separated penalty values for ruptures.Pelt")
    p.add_argument("--k_kmeans", type=int, default=3,
                   help="Number of clusters for KMeans fallback")
    args = p.parse_args()

    print(f"Loading signal from: {args.csv}")
    los_values, signal = load_signal(Path(args.csv))
    print(f"  LOS range: {los_values[0]}..{los_values[-1]}, T={len(los_values)}")
    print(f"  Signal shape: {signal.shape}")

    pen_grid = [float(x.strip()) for x in args.pen_grid.split(",") if x.strip()]

    try:
        import ruptures  # noqa: F401
        breakpoints, pen_used = detect_breakpoints_ruptures(signal, pen_grid)
        print(f"ruptures.Pelt: pen_used={pen_used}, breakpoints at positions={breakpoints}")
        method = "ruptures"
    except ImportError:
        print("ruptures not available — falling back to KMeans")
        breakpoints = detect_breakpoints_kmeans(signal, k=args.k_kmeans)
        print(f"KMeans(k={args.k_kmeans}): breakpoints at positions={breakpoints}")
        method = f"kmeans(k={args.k_kmeans})"

    groups = breakpoints_to_los_groups(breakpoints, los_values)

    print(f"\nDetected {len(groups)} LOS groups (method={method}):")
    for gname, glos in groups.items():
        print(f"  {gname}: LOS {glos[0]}..{glos[-1]}  (n_los={len(glos)})")

    # Verify coverage
    all_covered = sorted([v for glos in groups.values() for v in glos])
    expected = sorted(los_values.tolist())
    if all_covered != expected:
        print(f"Warning: group coverage mismatch! covered={all_covered}, expected={expected}")

    # Save JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(groups, f, indent=2)
    print(f"\nSaved LOS groups: {output_path}")

    # Save plot
    try:
        plot_groups(signal, los_values, groups, Path(args.plot))
    except Exception as e:
        print(f"Warning: plot failed: {e}")


if __name__ == "__main__":
    main()
