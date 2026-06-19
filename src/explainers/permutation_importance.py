"""
permutation_importance.py

Permutation-based global importance for CTMP-GIN style models where:

Model forward signature:
    model(x, los, edge_index, device) -> logits (or (logits, ...))

Dataloader yields batches unpackable as:
    x, y, los

Edge index is fixed for the whole dataset and provided once to the module.

Key design: shuffle is performed GLOBALLY across the entire dataset (not within
each batch), which avoids underestimation of importance that occurs with
batch-local permutation.

Outputs:
    pandas.DataFrame sorted by importance (descending), including:
      - each node(variable) importance via permuting x[:, j] across all patients
      - LOS importance via permuting los across all patients

Metric:
    Binary ROC-AUC (default).

Progress:
    Uses a single tqdm bar over the total number of permutations:
        total = V * R + R(LOS) = (V + 1) * R
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Tuple, Union, Dict
import sys
import numpy as np
import pandas as pd
import torch

from tqdm.auto import tqdm
from sklearn.metrics import roc_auc_score

Tensor = torch.Tensor


# -----------------------------
# Config
# -----------------------------

@dataclass
class PermutationImportanceConfig:
    num_repeats: int = 10
    seed: int = 42
    variable_names: Optional[List[str]] = None  # length V if provided


# -----------------------------
# Helpers
# -----------------------------

def _to_device(t: Tensor, device: Union[str, torch.device]) -> Tensor:
    return t.to(device)


def _predict_proba(
    model: torch.nn.Module,
    x: Tensor,
    los: Tensor,
    edge_index: Tensor,
    device: Union[str, torch.device],
) -> Tensor:
    """
    Calls model(x, los, edge_index, device) and returns prob for class 1.

    Supported logits shapes:
      - [B] or [B, 1] : sigmoid
      - [B, 2]        : softmax -> [:, 1]
    """
    out = model(x, los, edge_index, device=device)
    if isinstance(out, (tuple, list)):
        out = out[0]

    logits = out
    if logits.dim() == 2 and logits.size(-1) == 2:
        return torch.softmax(logits, dim=-1)[:, 1]
    if logits.dim() == 2 and logits.size(-1) == 1:
        logits = logits.squeeze(-1)
    if logits.dim() == 1:
        return torch.sigmoid(logits)

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def _to_1d_binary_labels(y: Tensor) -> Tensor:
    """
    Ensure y is 1D binary labels [N] with values in {0,1}.
    Accepts:
      - [N]
      - [N,2] one-hot or logits/probs
      - [2,N] transposed one-hot
    """
    if y.dim() == 1:
        return y.long().view(-1)

    if y.dim() == 2:
        if y.size(1) == 2:
            return torch.argmax(y, dim=1).long().view(-1)
        if y.size(0) == 2:
            return torch.argmax(y, dim=0).long().view(-1)

    raise ValueError(f"Unsupported y shape for binary labels: {tuple(y.shape)}")


def _to_1d_pos_scores(p: Tensor) -> Tensor:
    """
    Ensure predicted scores are 1D positive-class scores [N].
    Accepts:
      - [N] (already positive-class prob/logit)
      - [N,2] (class probs/logits)
      - [2,N] (transposed)
    """
    if p.dim() == 1:
        return p.view(-1)

    if p.dim() == 2:
        if p.size(1) == 2:
            return p[:, 1].contiguous().view(-1)
        if p.size(0) == 2:
            return p[1, :].contiguous().view(-1)

    raise ValueError(f"Unsupported prediction shape for binary AUC: {tuple(p.shape)}")


def _collect_tensors(
    dataloader: Iterable,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Iterate the dataloader once and collect all (x, y, los) into CPU tensors.

    Returns:
        all_x   [N, V]  CPU tensor
        all_y   [N]     CPU tensor
        all_los [N]     CPU tensor
    """
    xs, ys, lss = [], [], []
    for batch in dataloader:
        x, y, los = batch
        if not isinstance(x, Tensor) or x.dim() != 2:
            raise ValueError(
                f"Expected x to have shape [B, V], got {type(x)} with shape "
                f"{getattr(x, 'shape', None)}"
            )
        xs.append(x.cpu())
        ys.append(y.cpu())
        lss.append(los.cpu())

    all_x = torch.cat(xs, dim=0)    # [N, V]
    all_y = torch.cat(ys, dim=0)    # [N]
    all_los = torch.cat(lss, dim=0) # [N]
    return all_x, all_y, all_los


def _run_auc_on_tensors(
    model: torch.nn.Module,
    all_x: Tensor,
    all_y: Tensor,
    all_los: Tensor,
    edge_index: Tensor,
    device: Union[str, torch.device],
    batch_size: int,
    y_cat: Optional[np.ndarray] = None,
) -> float:
    """
    Split [N] CPU tensors into batches, run model forward, compute ROC-AUC.
    edge_index must already be on device.
    """
    model.eval()
    N = all_x.size(0)
    if N == 0:
        return float("nan")

    p_all: List[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            x_b = all_x[start:end].to(device, non_blocking=True)
            los_b = all_los[start:end].to(device, non_blocking=True)

            probs = _predict_proba(model, x_b, los_b, edge_index, device=device)
            p_1d = _to_1d_pos_scores(probs)

            p_all.append(p_1d.detach().cpu().numpy())

    if y_cat is None:
        y_cat = _to_1d_binary_labels(all_y).detach().cpu().numpy().reshape(-1)
    p_cat = np.concatenate(p_all, axis=0).reshape(-1)

    if np.unique(y_cat).size < 2:
        return float("nan")
    return float(roc_auc_score(y_cat, p_cat))


def _cache_tensors_on_device(
    all_x: Tensor,
    all_y: Tensor,
    all_los: Tensor,
    device: Union[str, torch.device],
) -> Tuple[Tensor, Tensor, Tensor]:
    device_obj = torch.device(device)
    if device_obj.type == "cpu":
        return all_x, all_y, all_los

    try:
        return (
            all_x.to(device_obj, non_blocking=True),
            all_y.to(device_obj, non_blocking=True),
            all_los.to(device_obj, non_blocking=True),
        )
    except RuntimeError as exc:
        print(f"  [Warning] Could not cache tensors on {device_obj}: {exc}")
        print("  [Warning] Falling back to per-batch device transfers.")
        return all_x, all_y, all_los


# -----------------------------
# Public API
# -----------------------------

def compute_permutation_importance(
    model: torch.nn.Module,
    dataloader: Iterable,
    edge_index: Tensor,
    device: Union[str, torch.device] = "cuda",
    config: Optional[PermutationImportanceConfig] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Compute global permutation importance with GLOBAL shuffle (not batch-local).

    For each feature j:
      1. Permute x[:, j] across ALL N samples (global randperm)
      2. Run forward pass in batches
      3. importance = baseline_auc - permuted_auc

    This avoids the underestimation that occurs when shuffling within small batches.
    """
    if config is None:
        config = PermutationImportanceConfig()

    # Collect entire dataset into memory (CPU)
    print("Collecting dataset into memory...")
    all_x, all_y, all_los = _collect_tensors(dataloader)
    N, V = all_x.shape
    print(f"  N={N}, V={V}")
    y_cat = _to_1d_binary_labels(all_y).detach().cpu().numpy().reshape(-1)
    all_x, all_y, all_los = _cache_tensors_on_device(all_x, all_y, all_los, device)

    # Batch size for forward passes
    batch_size: int = getattr(dataloader, "batch_size", None) or 256

    # Variable names
    if config.variable_names is not None:
        if len(config.variable_names) != V:
            raise ValueError(
                f"variable_names length ({len(config.variable_names)}) must match V ({V})."
            )
        var_names = config.variable_names
    else:
        var_names = [f"var_{j}" for j in range(V)]

    # Progress bar: 1 baseline + (V+1) features × num_repeats
    total_steps = 1 + (V + 1) * config.num_repeats
    pbar = tqdm(
        total=total_steps,
        desc="Permutation importance",
        disable=not show_progress,
        file=sys.stderr,
        dynamic_ncols=True,
        mininterval=0.1,
    )

    edge_index = edge_index.to(device)

    try:
        # Baseline AUC
        print("Computing baseline AUC...")
        baseline_auc = _run_auc_on_tensors(
            model, all_x, all_y, all_los, edge_index, device, batch_size, y_cat=y_cat
        )
        pbar.update(1)
        pbar.set_postfix_str(f"baseline_auc={baseline_auc:.4f}")

        rows: List[Dict[str, Any]] = []

        # Node (variable) importances — global shuffle
        for j in range(V):
            perm_aucs: List[float] = []
            saved_col = all_x[:, j].clone()  # save original column once

            for r in range(config.num_repeats):
                rng = torch.Generator()
                rng.manual_seed(config.seed + 1000 * j + r)

                # Global permutation of variable j — in-place to avoid full clone
                perm = torch.randperm(N, generator=rng).to(all_x.device)
                all_x[:, j] = saved_col[perm]

                auc_p = _run_auc_on_tensors(
                    model, all_x, all_y, all_los, edge_index, device, batch_size, y_cat=y_cat
                )
                perm_aucs.append(auc_p)
                pbar.update(1)

            all_x[:, j] = saved_col  # restore column

            imp_vals = [
                baseline_auc - a
                for a in perm_aucs
                if not (np.isnan(a) or np.isnan(baseline_auc))
            ]
            rows.append(
                dict(
                    feature=var_names[j],
                    kind="node",
                    index=j,
                    baseline_auc=baseline_auc,
                    perm_auc_mean=float(np.nanmean(perm_aucs)),
                    importance_mean=float(np.nanmean(imp_vals)) if imp_vals else float("nan"),
                    importance_std=float(np.nanstd(imp_vals)) if imp_vals else float("nan"),
                    num_repeats=config.num_repeats,
                )
            )

        # LOS importance — global shuffle
        perm_aucs_los: List[float] = []
        for r in range(config.num_repeats):
            rng = torch.Generator()
            rng.manual_seed(config.seed + 999999 + r)

            perm = torch.randperm(N, generator=rng).to(all_los.device)
            los_perm = all_los[perm]

            auc_p = _run_auc_on_tensors(
                model, all_x, all_y, los_perm, edge_index, device, batch_size, y_cat=y_cat
            )
            perm_aucs_los.append(auc_p)
            pbar.update(1)

        imp_vals_los = [
            baseline_auc - a
            for a in perm_aucs_los
            if not (np.isnan(a) or np.isnan(baseline_auc))
        ]
        rows.append(
            dict(
                feature="LOS",
                kind="los",
                index=None,
                baseline_auc=baseline_auc,
                perm_auc_mean=float(np.nanmean(perm_aucs_los)),
                importance_mean=float(np.nanmean(imp_vals_los)) if imp_vals_los else float("nan"),
                importance_std=float(np.nanstd(imp_vals_los)) if imp_vals_los else float("nan"),
                num_repeats=config.num_repeats,
            )
        )

    finally:
        pbar.close()

    df = pd.DataFrame(rows)
    df = df.sort_values("importance_mean", ascending=False, na_position="last").reset_index(drop=True)
    return df
