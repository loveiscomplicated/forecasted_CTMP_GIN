import os
import sys
from pathlib import Path
import pandas as pd
import torch

# Ensure the project root is in sys.path to allow importing src modules
# This script is assumed to be in src/explainers/
# So, project_root should be two levels up from this script's directory.
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root))

# Import stability report functions
try:
    from src.explainers.stablity_report import (
        stability_report,
        print_stability_report,
        unstable_variables_report,
        print_unstable_report_with_names,
        importance_mean_std_table,
    )
except ImportError:
    # Fallback if the path assumption or import structure is different
    # Assumes stablity_report.py is in the same directory as this script
    print('Could not import from src.explainers, trying local stablity_report.py')
    from stablity_report import (
        stability_report,
        print_stability_report,
        unstable_variables_report,
        print_unstable_report_with_names,
        importance_mean_std_table,
    )


def df_to_importance_vector(df, V: int) -> torch.Tensor:
    """
    df: output of compute_permutation_importance
    returns vector of length (V + 1): [var0..var(V-1), LOS]
    """
    vec = torch.full((V + 1,), float("nan"), dtype=torch.float32)

    for _, row in df.iterrows():
        kind = row.get("kind", None)
        if kind == "node":
            j = int(row["index"])
            vec[j] = float(row["importance_mean"])
        elif kind == "los":
            vec[V] = float(row["importance_mean"])

    return vec


def permutation_stability_analysis():
    # --- Configuration ---
    # The path to the results directory, relative to this script's location.
    save_path = script_dir / 'results'
    csv_files = [
        save_path / 'permutation_seed0_ratio0.1.csv',
        save_path / 'permutation_seed1_ratio0.1.csv',
        save_path / 'permutation_seed2_ratio0.1.csv',
    ]
    sample_ratio = 0.1
    # This prefix is used for saving the new mean_std table.
    save_prefix = 'permutation_re-analysis'

    # --- Load data and reconstruct variables ---
    dfs = [pd.read_csv(f) for f in csv_files]

    # Reconstruct names_with_los and V from the first dataframe
    df_sample = dfs[0]
    # Filter for node features and sort by index to ensure correct order
    node_df = df_sample[df_sample['kind'] == 'node'].sort_values('index')
    col_names = node_df['feature'].tolist()
    V = len(col_names)
    names_with_los = col_names + ["LOS"]

    # Reconstruct the 'outs' tensor list from the dataframes
    outs = [df_to_importance_vector(df, V) for df in dfs]

    print(f"Loaded {len(outs)} importance vectors from {len(csv_files)} files.")
    print(f"Number of variables (V) inferred: {V}")

    # --- Generate and Print Stability Reports ---

    print("\n=== Building mean±std table across seeds ===")
    df_ms = importance_mean_std_table(outs, names_with_los)
    out_ms_csv = save_path / f"{save_prefix}_mean_std_ratio{sample_ratio}.csv"
    df_ms.to_csv(out_ms_csv, index=False)
    print(f"Saved mean±std table to: {out_ms_csv}")

    print("\n=== Top 30 Features (mean±std) ===")
    print(df_ms.head(30).to_string(index=False))

    print("\n=== Stability Report (Jaccard Index) ===")
    report = stability_report(outs, ks=[10, 20, 30])
    print_stability_report(report, ks=[10, 20, 30])

    print("\n=== Unstable Variables Report (k=20) ===")
    rep20 = unstable_variables_report(outs, k=20)
    print_unstable_report_with_names(rep20, names_with_los)

    print("\n=== Unstable Variables Report (k=30) ===")
    rep30 = unstable_variables_report(outs, k=30)
    print_unstable_report_with_names(rep30, names_with_los)


def gbig_stability_analysis():
    # --- Configuration ---
    # The path to the results directory, relative to this script's location.
    save_path = script_dir / 'results'
    csv_files = [
        save_path / 'permutation_seed0_ratio0.1.csv',
        save_path / 'permutation_seed1_ratio0.1.csv',
        save_path / 'permutation_seed2_ratio0.1.csv',
    ]
    sample_ratio = 0.1
    # This prefix is used for saving the new mean_std table.
    save_prefix = 'permutation_re-analysis'

    # --- Load data and reconstruct variables ---
    dfs = [pd.read_csv(f) for f in csv_files]

    # Reconstruct names_with_los and V from the first dataframe
    df_sample = dfs[0]
    # Filter for node features and sort by index to ensure correct order
    node_df = df_sample[df_sample['kind'] == 'node'].sort_values('index')
    col_names = node_df['feature'].tolist()
    V = len(col_names)
    names_with_los = col_names + ["LOS"]

    # Reconstruct the 'outs' tensor list from the dataframes
    outs = [df_to_importance_vector(df, V) for df in dfs]

    print(f"Loaded {len(outs)} importance vectors from {len(csv_files)} files.")
    print(f"Number of variables (V) inferred: {V}")

    # --- Generate and Print Stability Reports ---

    print("\n=== Building mean±std table across seeds ===")
    df_ms = importance_mean_std_table(outs, names_with_los)
    out_ms_csv = save_path / f"{save_prefix}_mean_std_ratio{sample_ratio}.csv"
    df_ms.to_csv(out_ms_csv, index=False)
    print(f"Saved mean±std table to: {out_ms_csv}")

    print("\n=== Top 30 Features (mean±std) ===")
    print(df_ms.head(30).to_string(index=False))

    print("\n=== Stability Report (Jaccard Index) ===")
    report = stability_report(outs, ks=[10, 20, 30])
    print_stability_report(report, ks=[10, 20, 30])

    print("\n=== Unstable Variables Report (k=20) ===")
    rep20 = unstable_variables_report(outs, k=20)
    print_unstable_report_with_names(rep20, names_with_los)

    print("\n=== Unstable Variables Report (k=30) ===")
    rep30 = unstable_variables_report(outs, k=30)
    print_unstable_report_with_names(rep30, names_with_los)

if __name__ == "__main__":
    permutation_stability_analysis()
