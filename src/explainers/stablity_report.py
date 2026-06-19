# stablity_report.py
from typing import List, Dict, Tuple
import torch
import pandas as pd
import numpy as np
import os

def idx_to_name(idx: int, col_names: list[str]) -> str:
    if idx < 0 or idx >= len(col_names):
        return f"<idx:{idx}>"
    return col_names[idx]


def topk_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return top-k indices by descending score."""
    return torch.topk(x, k=k, largest=True).indices


def jaccard(a: torch.Tensor, b: torch.Tensor) -> float:
    """Jaccard similarity between two 1D index tensors."""
    sa = set(a.tolist())
    sb = set(b.tolist())
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def overlap_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    """|A ∩ B| / k for top-k overlap."""
    sa = set(a.tolist())
    sb = set(b.tolist())
    inter = len(sa & sb)
    k = max(len(sa), 1)
    return inter / k


def spearman_corr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Spearman rank correlation (no scipy).
    Ties are handled approximately via argsort ranks (OK for most importance vectors).
    """
    x = x.flatten().float()
    y = y.flatten().float()
    n = x.numel()
    if n <= 1:
        return 0.0

    # ranks: 0..n-1 (ascending). For Spearman we just need monotonic ranks.
    rx = torch.argsort(torch.argsort(x))
    ry = torch.argsort(torch.argsort(y))

    rx = rx.float()
    ry = ry.float()

    rx = (rx - rx.mean()) / (rx.std(unbiased=False) + 1e-12)
    ry = (ry - ry.mean()) / (ry.std(unbiased=False) + 1e-12)
    return float((rx * ry).mean().item())


def stability_report(
    outs: List[torch.Tensor],
    ks: List[int] = [10, 20, 30],
) -> Dict[str, object]:
    """
    outs: list of importance vectors [N] (CPU or GPU ok)
    NOTE: This function is for VECTORS only. (LOS scalar should be handled separately.)
    """
    outs = [o.detach().cpu().view(-1) for o in outs]  # ✅ 항상 1D로

    # --- guard: scalar 들어오면 바로 잡아내기 ---
    bad = [(i, tuple(o.shape), int(o.numel())) for i, o in enumerate(outs) if o.numel() <= 1]
    if bad:
        raise ValueError(
            f"stability_report expects vectors with numel>1, but got scalar/too-small entries: {bad}. "
            "Move LOS scalars to stability_report_scalars()."
        )

    m = len(outs)

    # --- safety: no pairwise comparison possible ---
    if m < 2:
        return {
            "pair_spearman": [],
            "spearman_avg": 0.0,
            "pair_topk": {k: [] for k in ks},
            "topk_avg": {k: {"overlap_avg": 0.0, "jaccard_avg": 0.0} for k in ks},
        }

    pair_spearman: List[Tuple[Tuple[int, int], float]] = []
    pair_topk: Dict[int, List[Tuple[Tuple[int, int], float, float]]] = {k: [] for k in ks}

    for i in range(m):
        for j in range(i + 1, m):
            sp = spearman_corr_torch(outs[i], outs[j])
            pair_spearman.append(((i, j), sp))

            # ✅ 길이 안전: len() 금지(0-d 터짐) -> numel()
            N = min(int(outs[i].numel()), int(outs[j].numel()))

            for k in ks:
                effective_k = min(int(k), N)

                if effective_k <= 0:
                    pair_topk[k].append(((i, j), 0.0, 0.0))
                    continue

                ti = topk_indices(outs[i], effective_k)
                tj = topk_indices(outs[j], effective_k)

                ov = overlap_ratio(ti, tj)
                jac = jaccard(ti, tj)

                pair_topk[k].append(((i, j), ov, jac))

    spearman_avg = sum(v for _, v in pair_spearman) / max(len(pair_spearman), 1)

    topk_avg = {}
    for k, lst in pair_topk.items():
        if not lst:
            topk_avg[k] = {"overlap_avg": 0.0, "jaccard_avg": 0.0}
        else:
            overlap_avg = sum(v[1] for v in lst) / len(lst)
            jaccard_avg = sum(v[2] for v in lst) / len(lst)
            topk_avg[k] = {"overlap_avg": overlap_avg, "jaccard_avg": jaccard_avg}

    return {
        "pair_spearman": pair_spearman,
        "spearman_avg": spearman_avg,
        "pair_topk": pair_topk,
        "topk_avg": topk_avg,
    }



def print_stability_report(report: Dict[str, object], ks: List[int] = [10, 20, 30]) -> None:
    print("\n=== Stability Report ===")
    print(f"Spearman avg: {report['spearman_avg']:.4f}")
    print("Spearman pairs:", report["pair_spearman"])

    print("\nTop-k overlap/Jaccard (pairwise):")
    pair_topk = report["pair_topk"]
    for k in ks:
        print(f"  k={k}")
        for (i, j), ov, jac in pair_topk[k]:
            print(f"    seeds({i},{j}) overlap={ov:.3f}  jaccard={jac:.3f}")

    print("\nTop-k averages:")
    for k in ks:
        d = report["topk_avg"][k]
        print(f"  k={k} overlap_avg={d['overlap_avg']:.3f}  jaccard_avg={d['jaccard_avg']:.3f}")

def ranks_ascending(x: torch.Tensor) -> torch.Tensor:
    """
    Return rank positions (0..N-1) where 0 = smallest.
    If you want 0 = largest, use ranks_descending().
    """
    x = x.flatten()
    return torch.argsort(torch.argsort(x))

def ranks_descending(x: torch.Tensor) -> torch.Tensor:
    """
    Return rank positions (0..N-1) where 0 = largest importance.
    """
    x = x.flatten()
    return torch.argsort(torch.argsort(-x))

def topk_set(x: torch.Tensor, k: int) -> set:
    return set(torch.topk(x, k=k, largest=True).indices.detach().cpu().tolist())

def unstable_variables_report(
    outs: List[torch.Tensor],
    k: int = 20,
) -> Dict[str, object]:
    """
    Identify stable/unstable variables across multiple seeds.
    VECTORS ONLY. Scalars (e.g. LOS) are not allowed.
    """

    outs = [o.detach().cpu().flatten() for o in outs]

    if len(outs) == 0:
        raise ValueError("outs is empty.")
    
    # --- guard: scalar protection ---
    bad = [(i, tuple(o.shape), int(o.numel()))
           for i, o in enumerate(outs)
           if o.numel() <= 1]

    if bad:
        raise ValueError(
            f"unstable_variables_report expects vectors with numel>1, "
            f"but got scalar/too-small entries: {bad}. "
            "Use stability_report_scalars for LOS."
        )

    m = len(outs)
    N = outs[0].numel()

    if any(o.numel() != N for o in outs):
        raise ValueError("All outs must have the same vector length.")

    # k safety
    k = min(k, N)

    if k <= 0:
        # Top-k 자체가 의미 없으니 빈 결과 반환 or 에러(취향)
        raise ValueError(f"k must be >=1 after clipping. Got k={k}, N={N}.")
    
    # 1) Top-K membership frequency
    topk_sets = [topk_set(o, k) for o in outs]
    freq = torch.zeros(N, dtype=torch.long)
    for s in topk_sets:
        for idx in s:
            freq[idx] += 1

    stable = (freq == m).nonzero(as_tuple=True)[0].tolist()
    unstable = ((freq > 0) & (freq < m)).nonzero(as_tuple=True)[0].tolist()
    outside = (freq == 0).nonzero(as_tuple=True)[0].tolist()

    # 2) Rank variability (descending rank: 0 is best)
    rank_mat = torch.stack([ranks_descending(o) for o in outs], dim=0)  # [m, N]
    rank_min = rank_mat.min(dim=0).values
    rank_max = rank_mat.max(dim=0).values
    rank_range = (rank_max - rank_min)  # [N]

    # Only care about variables that appear at least once in Top-K
    candidates = (freq > 0).nonzero(as_tuple=True)[0]
    cand_ranges = rank_range[candidates]
    # sort candidates by rank_range desc
    order = torch.argsort(cand_ranges, descending=True)
    most_unstable_by_rank = candidates[order].tolist()

    # For convenience: per-variable summary for unstable ones
    unstable_summaries: List[Tuple[int, int, int]] = []
    # (var_idx, freq, rank_range)
    for idx in unstable:
        unstable_summaries.append((idx, int(freq[idx].item()), int(rank_range[idx].item())))
    unstable_summaries.sort(key=lambda t: (t[1], -t[2]))  # low freq first, then big rank swings

    return {
        "k": k,
        "num_vars": N,
        "num_seeds": m,
        "topk_sets": topk_sets,
        "freq": freq,  # tensor [N]
        "stable_in_topk": stable,
        "unstable_in_topk": unstable,
        "outside_topk": outside,
        "rank_range": rank_range,  # tensor [N]
        "most_unstable_by_rank": most_unstable_by_rank,  # candidates sorted by rank swing
        "unstable_summaries": unstable_summaries,  # list[(idx, freq, rank_range)]
    }


def print_unstable_report_with_names(
    rep: dict,
    col_names: list[str],
    top_n_rank_swing: int = 20,
) -> None:
    k = rep["k"]
    m = rep["num_seeds"]

    print(f"\n=== Unstable Variables Report (Top-{k}, seeds={m}) ===")

    stable = rep["stable_in_topk"]
    unstable = rep["unstable_in_topk"]

    print(f"Stable in Top-{k} (appear in all seeds): {len(stable)}")
    print(f"Unstable in Top-{k} (appear in some seeds): {len(unstable)}")

    print("\n[Stable-in-TopK]")
    for idx in stable:
        name = idx_to_name(idx, col_names)
        print(f"  {idx:>3d} | {name}")

    print("\n[Unstable-in-TopK] (idx | name | freq | rank_range)")
    for idx, freq, rr in rep["unstable_summaries"][:50]:
        name = idx_to_name(idx, col_names)
        print(f"  {idx:>3d} | {name:<30} | {freq}/{m} | {rr}")

    print(f"\n[Most rank-unstable among vars appearing in Top-{k}] (top {top_n_rank_swing})")
    freq_t = rep["freq"]
    rank_range = rep["rank_range"]

    for idx in rep["most_unstable_by_rank"][:top_n_rank_swing]:
        name = idx_to_name(idx, col_names)
        print(
            f"  {idx:>3d} | {name:<30} | "
            f"freq={int(freq_t[idx])}/{m} | rank_range={int(rank_range[idx])}"
        )

def full_importance_table(mean_importance: torch.Tensor, col_names: list[str]) -> pd.DataFrame:
    """
    Create a full variable-importance table for all variables.

    Args:
        mean_importance: Tensor [N], global importance per variable (already aggregated).
        col_names: list of variable names (length N).

    Returns:
        DataFrame with columns: idx, name, importance, rank
        (rank: 1 is most important)
    """
    imp = mean_importance.detach().cpu().flatten()
    N = imp.numel()
    assert len(col_names) == N, f"col_names length ({len(col_names)}) != N ({N})"

    order = torch.argsort(imp, descending=True)
    ranks = torch.empty_like(order)
    ranks[order] = torch.arange(1, N + 1)  # rank 1 is best

    df = pd.DataFrame({
        "idx": torch.arange(N).tolist(),
        "name": col_names,
        "importance": imp.tolist(),
        "rank": ranks.tolist(),
    })
    df = df.sort_values("rank").reset_index(drop=True)
    return df


def importance_mean_std_table(
    outs: list[torch.Tensor],
    col_names: list[str],
    save_path, 
    filename,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """
    Build a full variable-importance table with mean±std across seeds.

    Args:
        outs: list of tensors, each [N] global importance vector from a different seed.
        col_names: list of variable names (length N).
        eps: small constant for numerical stability in CV computation.

    Returns:
        DataFrame sorted by mean importance (descending) with:
          idx, name, mean, std, mean_minus_std, mean_plus_std, cv, rank_mean
    """
    # TODO: ValueError: col_names length (72) != N (60)
    # scores_b = res.gbig_var  # [B, N] <--error

    outs_cpu = [o.detach().cpu().flatten() for o in outs]
    m = len(outs_cpu)
    if m == 0:
        raise ValueError("outs is empty.")
    N = outs_cpu[0].numel()
    if any(o.numel() != N for o in outs_cpu):
        raise ValueError("All outs must have the same length N.")
    if len(col_names) != N:
        raise ValueError(f"col_names length ({len(col_names)}) != N ({N})")
    

    mat = torch.stack(outs_cpu, dim=0).to(torch.float64)  # [m, N]
    mean = mat.mean(dim=0)                                # [N]
    std = mat.std(dim=0, unbiased=False)                  # [N]

    # CV = std / mean (avoid division blow-up)
    mean_abs = mean.abs()
    cv = torch.where(mean_abs > eps, std / mean_abs, torch.full_like(std, float("nan")))

    # Rank by mean (1 = best)
    order = torch.argsort(mean, descending=True)
    rank = torch.empty_like(order)
    rank[order] = torch.arange(1, N + 1, dtype=rank.dtype)

    df = pd.DataFrame({
        "idx": np.arange(N),
        "name": col_names,
        "mean": mean.numpy(),
        "std": std.numpy(),
        "mean_minus_std": (mean - std).numpy(),
        "mean_plus_std": (mean + std).numpy(),
        "cv": cv.numpy(),
        "rank_mean": rank.numpy(),
    })

    df = df.sort_values("rank_mean").reset_index(drop=True)

    # CSV 저장
    out_csv = os.path.join(save_path, filename)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    
    return df

def stability_report_scalars(
    vals: List[torch.Tensor],
) -> Dict[str, object]:
    """
    vals: list of scalar tensors (e.g., LOS attribution per seed).
    Returns mean/std/cv stability summary.
    """
    xs: List[float] = []
    for i, v in enumerate(vals):
        t = v.detach().cpu().view(-1)
        if int(t.numel()) != 1:
            raise ValueError(
                f"stability_report_scalars expects scalars (numel==1). "
                f"Got vals[{i}] shape={tuple(v.shape)}, numel={int(t.numel())}."
            )
        xs.append(float(t.item()))

    if len(xs) == 0:
        return {"values": [], "mean": 0.0, "std": 0.0, "cv": 0.0}

    x = torch.tensor(xs, dtype=torch.float32)
    mean = float(x.mean().item())
    std = float(x.std(unbiased=False).item()) if x.numel() > 1 else 0.0
    cv = float(std / (abs(mean) + 1e-12))
    return {"values": xs, "mean": mean, "std": std, "cv": cv}


def print_scalar_stability_report(rep: Dict[str, object], name: str = "LOS") -> None:
    print(f"\n=== Scalar Stability Report: {name} ===")
    print(f"values: {rep['values']}")
    print(f"mean: {rep['mean']:.6f}  std: {rep['std']:.6f}  cv: {rep['cv']:.6f}")


def report(df_ms, outs, col_names, scalar=False):
    # 보기 편하게 top/bottom
    print("\n=== Top 20 (mean ± std) ===")
    print(df_ms.head(20).to_string(index=False))

    print("\n=== Bottom 20 (mean ± std) ===")
    print(df_ms.tail(20).to_string(index=False))



    # --- scalar reports (LOS) ---
    if scalar:
        rep_los = stability_report_scalars(outs)
        print_scalar_stability_report(rep_los, name=col_names[0])
        return
    
    # --- vector reports (variables) ---
    rep_vec = stability_report(outs, ks=[10, 20, 30])
    print_stability_report(rep_vec, ks=[10, 20, 30])

    rep20 = unstable_variables_report(outs, k=20)
    print_unstable_report_with_names(rep20, col_names)

    rep30 = unstable_variables_report(outs, k=30)
    print_unstable_report_with_names(rep30, col_names)


