import torch
from typing import Dict, Optional

@torch.no_grad()
def _make_baseline_x_like(x_idx: torch.Tensor) -> torch.Tensor:
    # baseline category = 0
    return torch.zeros_like(x_idx, dtype=torch.long)

@torch.no_grad()
def _make_baseline_los_like(los_idx: torch.Tensor) -> torch.Tensor:
    # baseline LOS token = 0 (NONE)
    return torch.zeros_like(los_idx, dtype=torch.long)


def manual_ig_embeddings(
    model,
    x_idx,
    los_idx,
    edge_index,
    target="logit",
    n_steps=50,
    return_full=False,
):

    assert target == "logit"
    eps = 1e-12  # safeguard for rare zero-total cases

    model.eval()
    device = next(model.parameters()).device

    x_idx = x_idx.to(device).long()
    los_idx = los_idx.to(device).long()
    edge_index = edge_index.to(device)

    B = x_idx.size(0)
    num_nodes = len(model.ad_col_index)

    # ---------- baseline indices ----------
    base_x_idx = torch.zeros_like(x_idx)
    base_los_idx = torch.zeros_like(los_idx)

    # ---------- embeddings ----------
    x_emb = model.entity_embedding_layer(x_idx)
    base_x_emb = model.entity_embedding_layer(base_x_idx)

    dx = x_emb - base_x_emb

    edge_index_2 = model.get_edge_index_2(
        edge_index=edge_index,
        num_nodes=num_nodes,
        batch_size=B,
    ).to(device)

    edge_attr_1 = model.get_edge_attr(
        los=los_idx,
        edge_index=edge_index,
        batch_size=B,
        num_nodes=num_nodes,
    )

    edge_attr_0 = model.get_edge_attr(
        los=base_los_idx,
        edge_index=edge_index,
        batch_size=B,
        num_nodes=num_nodes,
    )

    d_edge = edge_attr_1 - edge_attr_0

    # ---------- accumulators ----------
    total_grad_x = torch.zeros_like(x_emb)
    total_grad_e = torch.zeros_like(edge_attr_1)

    # ==================================================
    # JOINT IG LOOP
    # ==================================================
    for s in range(1, n_steps + 1):

        alpha = s / n_steps

        x_s = (base_x_emb + alpha * dx).detach().requires_grad_(True)
        e_s = (edge_attr_0 + alpha * d_edge).detach().requires_grad_(True)

        out = model.forward_from_x_emb_with_edge_attr(
            x_embedded=x_s,
            edge_index=edge_index,
            edge_index_2=edge_index_2,
            edge_attr=e_s,
        ).squeeze(-1)

        grad_x, grad_e = torch.autograd.grad(
            out.sum(),
            [x_s, e_s],
            retain_graph=False,
            create_graph=False,
        )

        total_grad_x += grad_x
        total_grad_e += grad_e

    # ---------- IG ----------
    avg_grad_x = total_grad_x / n_steps
    avg_grad_e = total_grad_e / n_steps

    ig_x = dx * avg_grad_x
    ig_edge = d_edge * avg_grad_e

    # LOS aggregation
    E_cross = B * num_nodes
    cross = ig_edge[-E_cross:]
    ig_los = cross.reshape(B, num_nodes, -1).sum(dim=1)
    ig_abs_los = torch.abs(cross.reshape(B, num_nodes, -1)).sum(dim=1)

    # ---------- reductions ----------
    imp_abs_x = ig_x.abs().sum(dim=-1)
    imp_signed_x = ig_x.sum(dim=-1)

    imp_abs_los = ig_abs_los.sum(dim=-1)
    imp_signed_los = ig_los.sum(dim=-1)

    # ---------- delta ----------
    with torch.no_grad():

        fx = model.forward_from_x_emb_with_edge_attr(
            x_embedded=x_emb,
            edge_index=edge_index,
            edge_index_2=edge_index_2,
            edge_attr=edge_attr_1,
        ).squeeze(-1)

        f0 = model.forward_from_x_emb_with_edge_attr(
            x_embedded=base_x_emb,
            edge_index=edge_index,
            edge_index_2=edge_index_2,
            edge_attr=edge_attr_0,
        ).squeeze(-1)

        ig_scalar = imp_signed_x.sum(dim=1) + imp_signed_los
        delta = (fx - f0) - ig_scalar
        delta_ratio = torch.abs(delta) / (torch.abs(fx - f0) + eps)

    out = {
        "imp_abs_x": imp_abs_x,
        "imp_signed_x": imp_signed_x,
        "imp_abs_los": imp_abs_los,
        "imp_signed_los": imp_signed_los,
        "delta": delta,
        "delta_ratio": delta_ratio,
    }

    if return_full:
        out["ig_x"] = ig_x
        out["ig_los"] = ig_los
        out["ig_edge"] = ig_edge

    return out



from tqdm import tqdm
from src.explainers.utils import _iter_selected_batches
from typing import Optional, List, Literal


"""
      ig_x_emb:  [B, 72, emb_dim]  (variable attribution in embedding space)
      ig_los_emb:[B, los_emb_dim]  (LOS attribution in embedding space)
      delta:     [B] (approx convergence check: f(x)-f(baseline) - sum(IG))
"""


def explain_ig_for_dataset(
    dataloader,
    model,
    edge_index: torch.Tensor,     # [2, E]
    target: str = "logit",        # "logit" only for now
    n_steps: int = 50,
    reduce: Literal["mean", "median"] = "mean",
    keep_all: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    sample_ratio: float = 0.1,
    seed: int = 0,
    ):

    model.eval()

    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    # --- storage ---
    scores_abs_x: List[torch.Tensor] = []
    scores_abs_los: List[torch.Tensor] = []
    scores_signed_x: List[torch.Tensor] = []
    scores_signed_los: List[torch.Tensor] = []
    scores_delta: List[torch.Tensor] = []
    scores_delta_ratio: List[torch.Tensor] = []

    running_sum_abs_x: Optional[torch.Tensor] = None
    running_sum_abs_los: Optional[torch.Tensor] = None
    running_sum_signed_x: Optional[torch.Tensor] = None
    running_sum_signed_los: Optional[torch.Tensor] = None
    running_sum_delta: Optional[torch.Tensor] = None
    running_sum_delta_ratio: Optional[torch.Tensor] = None

    n_seen = 0

    if not hasattr(dataloader, "__len__"):
        raise ValueError("Sampling requires dataloader with __len__().")

    total_batches = len(dataloader)
    effective_batches = min(total_batches, max_batches) if max_batches is not None else total_batches

    if sample_ratio < 1.0:
        g = torch.Generator().manual_seed(seed)
        n_pick = max(1, int(round(effective_batches * sample_ratio)))
        perm = torch.randperm(effective_batches, generator=g).tolist()
        selected = sorted(perm[:n_pick])
        if verbose:
            print(f"[Integrated Gradients] Sampling batches: {n_pick}/{effective_batches} (ratio={sample_ratio}, seed={seed})")
    else:
        selected = list(range(effective_batches))

    iterator = _iter_selected_batches(dataloader, selected)
    pbar = tqdm(iterator, total=len(selected), desc="IG (dataset)")

    eps = 1e-12  # safeguard for rare zero-total cases

    for b_idx, batch in pbar:
        x_idx, y_idx, los_idx = batch

        res = manual_ig_embeddings(
            model=model,
            x_idx=x_idx,
            los_idx=los_idx,
            edge_index=edge_index,
            target=target,
            n_steps=n_steps,
        )

        imp_abs_x = res["imp_abs_x"]           # [B,72]
        imp_abs_los = res["imp_abs_los"]       # [B]
        imp_signed_x = res["imp_signed_x"]     # [B,72] (leave as-is)
        imp_signed_los = res["imp_signed_los"] # [B]    (leave as-is)
        delta = res["delta"]                   # [B]    (leave as-is)
        delta_ratio = res["delta_ratio"]

        # -----------------------------
        # NEW: per-sample share (ABS only)
        # -----------------------------
        # total_abs per sample: [B]
        total_abs = imp_abs_x.sum(dim=1) + imp_abs_los
        total_abs = total_abs + eps  # avoid divide-by-zero

        share_x = imp_abs_x / total_abs.unsqueeze(1)  # [B,72]
        share_los = imp_abs_los / total_abs           # [B]

        B = share_x.size(0)
        n_seen += B

        if reduce == "mean" and not keep_all:
            # NOTE: we now accumulate shares instead of raw abs importances
            if running_sum_abs_x is None:
                running_sum_abs_x = share_x.detach().sum(dim=0)          # [72]
                running_sum_abs_los = share_los.detach().sum(dim=0)      # scalar tensor
                running_sum_signed_x = imp_signed_x.detach().sum(dim=0)  # [72]
                running_sum_signed_los = imp_signed_los.detach().sum(dim=0)  # scalar tensor
                running_sum_delta = delta.detach().sum(dim=0)            # scalar tensor
                running_sum_delta_ratio = delta_ratio.detach().sum(dim=0) # scalar tensor
            else:
                running_sum_abs_x += share_x.detach().sum(dim=0)
                running_sum_abs_los += share_los.detach().sum(dim=0)
                running_sum_signed_x += imp_signed_x.detach().sum(dim=0)
                running_sum_signed_los += imp_signed_los.detach().sum(dim=0)
                running_sum_delta += delta.detach().sum(dim=0)
                running_sum_delta_ratio += delta_ratio.detach().sum(dim=0)
        else:
            # keep_all path stores per-sample shares for ABS, raw for signed/delta
            scores_abs_x.append(share_x.detach().cpu())
            scores_abs_los.append(share_los.detach().cpu())
            scores_signed_x.append(imp_signed_x.detach().cpu())
            scores_signed_los.append(imp_signed_los.detach().cpu())
            scores_delta.append(delta.detach().cpu())
            scores_delta_ratio.append(delta_ratio.detach().cpu())


    # ---- aggregate ----
    if n_seen == 0:
        raise RuntimeError("No samples processed.")

    # fast path: mean without keep_all
    if reduce == "mean" and not keep_all:
        if running_sum_abs_x is None:
            raise RuntimeError("No samples processed (running_sum_abs is None).")
        if running_sum_signed_x is None:
            raise RuntimeError("No samples processed (running_sum_signed is None).")

        # ABS outputs are now mean shares
        global_abs_x = (running_sum_abs_x / float(n_seen)).cpu()          # [72], mean share
        global_abs_los = (running_sum_abs_los / float(n_seen)).cpu()      # scalar, mean share

        # signed + delta unchanged
        global_signed_x = (running_sum_signed_x / float(n_seen)).cpu()
        global_signed_los = (running_sum_signed_los / float(n_seen)).cpu()
        global_delta = (running_sum_delta / float(n_seen)).cpu()
        global_delta_ratio = (running_sum_delta_ratio / float(n_seen)).cpu()

        res = {
            "global_abs_x": global_abs_x,
            "global_abs_los": global_abs_los,
            "global_signed_x": global_signed_x,
            "global_signed_los": global_signed_los,
            "global_delta": global_delta,
            "global_delta_ratio": global_delta_ratio,
        }

        return res

    # otherwise, concatenate and reduce
    if len(scores_abs_x) == 0:
        raise RuntimeError("No scores collected.")

    all_abs_x = torch.cat(scores_abs_x, dim=0)        # [n_samples, 72] (shares)
    all_abs_los = torch.cat(scores_abs_los, dim=0)    # [n_samples]      (shares)
    all_signed_x = torch.cat(scores_signed_x, dim=0)  # [n_samples, 72]
    all_signed_los = torch.cat(scores_signed_los, dim=0)  # [n_samples]
    all_delta = torch.cat(scores_delta, dim=0)        # [n_samples]
    all_delta_ratio = torch.cat(scores_delta_ratio, dim=0) # [n_samples]

    if reduce == "mean":
        global_abs_x = all_abs_x.mean(dim=0)          # mean share per feature
        global_abs_los = all_abs_los.mean(dim=0)      # mean share for LOS
        global_signed_x = all_signed_x.mean(dim=0)
        global_signed_los = all_signed_los.mean(dim=0)
        global_delta = all_delta.mean(dim=0)
        global_delta_ratio = all_delta_ratio.mean(dim=0)
    elif reduce == "median":
        global_abs_x = all_abs_x.median(dim=0).values
        global_abs_los = all_abs_los.median(dim=0).values
        global_signed_x = all_signed_x.median(dim=0).values
        global_signed_los = all_signed_los.median(dim=0).values
        global_delta = all_delta.median(dim=0).values
        global_delta_ratio = all_delta_ratio.median(dim=0).values
    else:
        raise ValueError(reduce)
    
    

    def _make_stat_dict(_all_scores):
        stats = {
            "mean": _all_scores.mean().item(),
            "median": _all_scores.median().item(),
            "p90": torch.quantile(_all_scores, 0.90).item(),
            "p95": torch.quantile(_all_scores, 0.95).item(),
            "p99": torch.quantile(_all_scores, 0.99).item(),
            "max": _all_scores.max().item(),
        }
        return stats
    
    los_share = all_abs_los.detach().cpu().float()
    delta_share = all_delta.detach().cpu().float()
    delta_ratio_share = all_delta_ratio.detach().cpu().float()

    los_stats = _make_stat_dict(los_share)
    delta_stats = _make_stat_dict(delta_share)
    delta_ratio_stats = _make_stat_dict(delta_ratio_share)

    res = {
        "global_abs_x": global_abs_x,
        "global_abs_los": global_abs_los,
        "global_signed_x": global_signed_x,
        "global_signed_los": global_signed_los,
        "global_delta": global_delta,
        "global_delta_ratio": global_delta_ratio,
        "los_abs_stats": los_stats,
        "delta_stats": delta_stats,
        "delta_ratio_stats": delta_ratio_stats,

    }

    return res # abs: share (ratio) 

def manual_ig_embeddings_gin(
    model,
    x_idx,
    los_idx,
    edge_index,
    target="logit",
    n_steps=50,
    return_full=False,
):
    """
    GIN용 Integrated Gradients.
    GIN은 x와 los를 concat해서 단일 entity_embedding_layer로 임베딩한다.
    따라서 x_idx: [B, num_x_vars], los_idx: [B] → concat → [B, num_x_vars+1]
    로 처리하고 전체 임베딩에 대해 IG를 계산한다.
    마지막 컬럼(num_x_vars 인덱스)이 LOS에 해당한다.
    """
    assert target == "logit"
    eps = 1e-12

    model.eval()
    device = next(model.parameters()).device

    x_idx = x_idx.to(device).long()
    los_idx = los_idx.to(device).long()
    edge_index = edge_index.to(device)

    B = x_idx.size(0)

    # GIN forward에서 los를 unsqueeze(1)하고 cat하는 방식 그대로 재현
    los_idx_2d = los_idx.unsqueeze(dim=1)                     # [B, 1]
    full_idx = torch.cat((x_idx, los_idx_2d), dim=1)          # [B, num_vars+1]

    base_full_idx = torch.zeros_like(full_idx)                 # baseline = 0

    # embeddings
    x_emb = model.entity_embedding_layer(full_idx)            # [B, num_vars+1, emb_dim]
    base_x_emb = model.entity_embedding_layer(base_full_idx)  # [B, num_vars+1, emb_dim]
    dx = x_emb - base_x_emb                                   # [B, num_vars+1, emb_dim]

    # forward_from_full_emb: GIN 고유 forward (embedding 이후)
    def _forward_from_emb(x_embedded):
        # x_embedded: [B, num_vars+1, emb_dim]
        num_nodes = x_embedded.size(1)
        batch_size = x_embedded.size(0)
        node_emb = x_embedded.reshape(batch_size * num_nodes, -1)
        sum_pooled = []
        for layer in model.gin_layers:
            node_emb = layer(node_emb, edge_index)
            x_temp = node_emb.reshape(batch_size, num_nodes, -1)
            x_sum = torch.sum(x_temp, dim=1)
            sum_pooled.append(x_sum)
        graph_emb = torch.cat(sum_pooled, dim=1)
        return model.classifier(graph_emb).squeeze(-1)

    # accumulate gradients
    total_grad = torch.zeros_like(x_emb)  # [B, num_vars+1, emb_dim]

    for s in range(1, n_steps + 1):
        alpha = s / n_steps
        x_s = (base_x_emb + alpha * dx).detach().requires_grad_(True)

        out = _forward_from_emb(x_s)

        grad, = torch.autograd.grad(
            out.sum(),
            x_s,
            retain_graph=False,
            create_graph=False,
        )
        total_grad += grad

    avg_grad = total_grad / n_steps                   # [B, num_vars+1, emb_dim]
    ig = dx * avg_grad                                # [B, num_vars+1, emb_dim]

    # reduce: abs & signed per node
    imp_abs = ig.abs().sum(dim=-1)                    # [B, num_vars+1]
    imp_signed = ig.sum(dim=-1)                       # [B, num_vars+1]

    # split: last column is LOS
    imp_abs_x = imp_abs[:, :-1]                       # [B, num_x_vars]
    imp_abs_los = imp_abs[:, -1]                      # [B]
    imp_signed_x = imp_signed[:, :-1]                 # [B, num_x_vars]
    imp_signed_los = imp_signed[:, -1]                # [B]

    # delta (completeness check)
    with torch.no_grad():
        fx = _forward_from_emb(x_emb)
        f0 = _forward_from_emb(base_x_emb)
        ig_scalar = imp_signed_x.sum(dim=1) + imp_signed_los
        delta = (fx - f0) - ig_scalar
        delta_ratio = torch.abs(delta) / (torch.abs(fx - f0) + eps)

    out_dict = {
        "imp_abs_x": imp_abs_x,
        "imp_signed_x": imp_signed_x,
        "imp_abs_los": imp_abs_los,
        "imp_signed_los": imp_signed_los,
        "delta": delta,
        "delta_ratio": delta_ratio,
    }

    if return_full:
        out_dict["ig"] = ig

    return out_dict


def explain_ig_for_dataset_gin(
    dataloader,
    model,
    edge_index: torch.Tensor,
    target: str = "logit",
    n_steps: int = 50,
    reduce="mean",
    keep_all: bool = False,
    max_batches=None,
    verbose: bool = True,
    sample_ratio: float = 1.0,
    seed: int = 0,
):
    """GIN 전용 explain_ig_for_dataset."""
    from tqdm import tqdm
    from src.explainers.utils import _iter_selected_batches

    model.eval()

    if not (0.0 < sample_ratio <= 1.0):
        raise ValueError("sample_ratio must be in (0, 1].")

    scores_abs_x = []
    scores_abs_los = []
    scores_signed_x = []
    scores_signed_los = []
    scores_delta = []
    scores_delta_ratio = []

    running_sum_abs_x = None
    running_sum_abs_los = None
    running_sum_signed_x = None
    running_sum_signed_los = None
    running_sum_delta = None
    running_sum_delta_ratio = None

    n_seen = 0

    total_batches = len(dataloader)
    effective_batches = min(total_batches, max_batches) if max_batches is not None else total_batches

    if sample_ratio < 1.0:
        g = torch.Generator().manual_seed(seed)
        n_pick = max(1, int(round(effective_batches * sample_ratio)))
        perm = torch.randperm(effective_batches, generator=g).tolist()
        selected = sorted(perm[:n_pick])
        if verbose:
            print(f"[GIN-IG] Sampling {n_pick}/{effective_batches} batches (ratio={sample_ratio}, seed={seed})")
    else:
        selected = list(range(effective_batches))

    iterator = _iter_selected_batches(dataloader, selected)
    pbar = tqdm(iterator, total=len(selected), desc="GIN-IG (dataset)")

    eps = 1e-12

    for b_idx, batch in pbar:
        x_idx, y_idx, los_idx = batch

        res = manual_ig_embeddings_gin(
            model=model,
            x_idx=x_idx,
            los_idx=los_idx,
            edge_index=edge_index,
            target=target,
            n_steps=n_steps,
        )

        imp_abs_x = res["imp_abs_x"]
        imp_abs_los = res["imp_abs_los"]
        imp_signed_x = res["imp_signed_x"]
        imp_signed_los = res["imp_signed_los"]
        delta = res["delta"]
        delta_ratio = res["delta_ratio"]

        total_abs = imp_abs_x.sum(dim=1) + imp_abs_los + eps
        share_x = imp_abs_x / total_abs.unsqueeze(1)
        share_los = imp_abs_los / total_abs

        B = share_x.size(0)
        n_seen += B

        if reduce == "mean" and not keep_all:
            if running_sum_abs_x is None:
                running_sum_abs_x = share_x.detach().sum(dim=0)
                running_sum_abs_los = share_los.detach().sum(dim=0)
                running_sum_signed_x = imp_signed_x.detach().sum(dim=0)
                running_sum_signed_los = imp_signed_los.detach().sum(dim=0)
                running_sum_delta = delta.detach().sum(dim=0)
                running_sum_delta_ratio = delta_ratio.detach().sum(dim=0)
            else:
                running_sum_abs_x += share_x.detach().sum(dim=0)
                running_sum_abs_los += share_los.detach().sum(dim=0)
                running_sum_signed_x += imp_signed_x.detach().sum(dim=0)
                running_sum_signed_los += imp_signed_los.detach().sum(dim=0)
                running_sum_delta += delta.detach().sum(dim=0)
                running_sum_delta_ratio += delta_ratio.detach().sum(dim=0)
        else:
            scores_abs_x.append(share_x.detach().cpu())
            scores_abs_los.append(share_los.detach().cpu())
            scores_signed_x.append(imp_signed_x.detach().cpu())
            scores_signed_los.append(imp_signed_los.detach().cpu())
            scores_delta.append(delta.detach().cpu())
            scores_delta_ratio.append(delta_ratio.detach().cpu())

    if n_seen == 0:
        raise RuntimeError("No samples processed.")

    if reduce == "mean" and not keep_all:
        if running_sum_abs_x is None:
            raise RuntimeError("No samples processed.")
        return {
            "global_abs_x": (running_sum_abs_x / float(n_seen)).cpu(),
            "global_abs_los": (running_sum_abs_los / float(n_seen)).cpu(),
            "global_signed_x": (running_sum_signed_x / float(n_seen)).cpu(),
            "global_signed_los": (running_sum_signed_los / float(n_seen)).cpu(),
            "global_delta": (running_sum_delta / float(n_seen)).cpu(),
            "global_delta_ratio": (running_sum_delta_ratio / float(n_seen)).cpu(),
        }

    if len(scores_abs_x) == 0:
        raise RuntimeError("No scores collected.")

    all_abs_x = torch.cat(scores_abs_x, dim=0)
    all_abs_los = torch.cat(scores_abs_los, dim=0)
    all_signed_x = torch.cat(scores_signed_x, dim=0)
    all_signed_los = torch.cat(scores_signed_los, dim=0)
    all_delta = torch.cat(scores_delta, dim=0)
    all_delta_ratio = torch.cat(scores_delta_ratio, dim=0)

    def _reduce(t):
        if reduce == "mean":
            return t.mean(dim=0)
        elif reduce == "median":
            return t.median(dim=0).values
        raise ValueError(reduce)

    def _make_stat_dict(_all_scores):
        return {
            "mean": _all_scores.mean().item(),
            "median": _all_scores.median().item(),
            "p90": torch.quantile(_all_scores, 0.90).item(),
            "p95": torch.quantile(_all_scores, 0.95).item(),
            "p99": torch.quantile(_all_scores, 0.99).item(),
            "max": _all_scores.max().item(),
        }

    return {
        "global_abs_x": _reduce(all_abs_x),
        "global_abs_los": _reduce(all_abs_los),
        "global_signed_x": _reduce(all_signed_x),
        "global_signed_los": _reduce(all_signed_los),
        "global_delta": _reduce(all_delta),
        "global_delta_ratio": _reduce(all_delta_ratio),
        "los_abs_stats": _make_stat_dict(all_abs_los.float()),
        "delta_stats": _make_stat_dict(all_delta.float()),
        "delta_ratio_stats": _make_stat_dict(all_delta_ratio.float()),
    }


def gin_ig_main(
    dataset,
    dataloader,
    model,
    save_path: str,
    edge_index: torch.Tensor,
    target: str = "logit",
    n_steps: int = 400,
    reduce="mean",
    keep_all: bool = False,
    max_batches=None,
    verbose: bool = True,
    sample_ratio: float = 1.0,
):
    """GIN 전용 ig_main. explainer_main.py에서 호출."""
    from src.utils.seed_set import set_seed
    from src.explainers.stablity_report import importance_mean_std_table, report

    outs_abs_x = []
    outs_abs_los = []
    outs_signed_x = []
    outs_signed_los = []
    outs_delta = []
    outs_delta_ratio = []
    los_abs_stats_list = []
    delta_stats_list = []
    delta_ratio_stats_list = []

    for s in [0, 1, 2]:
        set_seed(s)
        res = explain_ig_for_dataset_gin(
            dataloader=dataloader,
            model=model,
            edge_index=edge_index,
            target=target,
            n_steps=n_steps,
            reduce=reduce,
            keep_all=keep_all,
            max_batches=max_batches,
            verbose=verbose,
            sample_ratio=sample_ratio,
            seed=s,
        )

        outs_abs_x.append(res["global_abs_x"].cpu().float())
        outs_abs_los.append(res["global_abs_los"].cpu().float())
        outs_signed_x.append(res["global_signed_x"].cpu().float())
        outs_signed_los.append(res["global_signed_los"].cpu().float())
        outs_delta.append(res["global_delta"].cpu().float())
        outs_delta_ratio.append(res["global_delta_ratio"].cpu().float())
        los_abs_stats_list.append(res.get("los_abs_stats", None))
        delta_stats_list.append(res.get("delta_stats", None))
        delta_ratio_stats_list.append(res.get("delta_ratio_stats", None))

        if sample_ratio == 1:
            break

    col_names = dataset.col_info[0]  # col_info: (col_list, col_dims, ad_col_index, dis_col_index)

    df_abs_x = importance_mean_std_table(outs_abs_x, col_names, save_path, f"GIN_IG_abs_x_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_abs_los = importance_mean_std_table(outs_abs_los, ["LOS"], save_path, f"GIN_IG_abs_los_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_signed_x = importance_mean_std_table(outs_signed_x, col_names, save_path, f"GIN_IG_signed_x_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_signed_los = importance_mean_std_table(outs_signed_los, ["LOS"], save_path, f"GIN_IG_signed_los_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_delta = importance_mean_std_table(outs_delta, ["delta"], save_path, f"GIN_IG_delta_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_delta_ratio = importance_mean_std_table(outs_delta_ratio, ["delta_ratio"], save_path, f"GIN_IG_delta_ratio_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")

    report(df_abs_x, outs_abs_x, col_names)
    report(df_abs_los, outs_abs_los, ["LOS"], scalar=True)
    report(df_signed_x, outs_signed_x, col_names)
    report(df_signed_los, outs_signed_los, ["LOS"], scalar=True)
    report(df_delta, outs_delta, ["delta"], scalar=True)
    report(df_delta_ratio, outs_delta_ratio, ["delta_ratio"], scalar=True)

    if keep_all:
        def _print_stats(name, stats_list):
            if stats_list[0] is None:
                return
            for seed, stats in enumerate(stats_list):
                print(f"\n[{name} distribution](seed: {seed})")
                for k, v in stats.items():
                    print(f"  {k:>6}: {v:.6f}")

        _print_stats("LOS_abs", los_abs_stats_list)
        _print_stats("delta", delta_stats_list)
        _print_stats("delta_ratio", delta_ratio_stats_list)


from src.utils.seed_set import set_seed
from src.explainers.stablity_report import (
    importance_mean_std_table,
    report
)

def ig_main(
    dataset,
    dataloader,
    model,
    save_path: str,
    edge_index: torch.Tensor,     # [2, E]
    target: str = "logit",        # "logit" only for now
    n_steps: int = 50,
    reduce: Literal["mean", "median"] = "mean",
    keep_all: bool = False,
    max_batches: Optional[int] = None,
    verbose: bool = True,
    sample_ratio: float = 0.1,
    ):
    
    outs_abs_x = []
    outs_abs_los = []
    outs_signed_x = []
    outs_signed_los = []
    outs_delta = []
    outs_delta_ratio = []
    los_abs_stats_list = []
    delta_stats_list = []
    delta_ratio_stats_list = []
    
    for s in [0, 1, 2]:
        set_seed(s)
        res = explain_ig_for_dataset(
            dataloader=dataloader,
            model=model,
            edge_index=edge_index,
            target=target,
            n_steps=n_steps,
            reduce=reduce,
            keep_all=keep_all,
            max_batches=max_batches,
            verbose=verbose,
            sample_ratio=sample_ratio,
            seed=s
        )
        
        outs_abs_x.append(res["global_abs_x"].cpu().float())
        outs_abs_los.append(res["global_abs_los"].cpu().float())
        outs_signed_x.append(res["global_signed_x"].cpu().float())
        outs_signed_los.append(res["global_signed_los"].cpu().float())
        outs_delta.append(res["global_delta"].cpu().float())
        outs_delta_ratio.append(res["global_delta_ratio"].cpu().float())
        los_abs_stats_list.append(res.get("los_abs_stats", None))
        delta_stats_list.append(res.get("delta_stats", None))
        delta_ratio_stats_list.append(res.get("delta_ratio_stats", None))
        
        if sample_ratio == 1:
            break

    col_names, col_dims, ad_col_index, dis_col_index = dataset.col_info

    df_abs_x = importance_mean_std_table(outs_abs_x, col_names, save_path, f"IG_abs_x_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_abs_los = importance_mean_std_table(outs_abs_los, ["LOS"], save_path, f"IG_abs_los_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv",)
    df_signed_x = importance_mean_std_table(outs_signed_x, col_names, save_path, f"IG_signed_x_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_signed_los = importance_mean_std_table(outs_signed_los, ["LOS"], save_path, f"IG_signed_los_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_delta = importance_mean_std_table(outs_delta, ["delta"], save_path, f"IG_delta_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")
    df_delta_ratio = importance_mean_std_table(outs_delta_ratio, ["delta_ratio"], save_path, f"IG_delta_ratio_{reduce}_step{n_steps}_{sample_ratio}_global_importance.csv")



    report(df_abs_x, outs_abs_x, col_names)
    report(df_abs_los, outs_abs_los, ["LOS"], scalar=True)
    report(df_signed_x, outs_signed_x, col_names)
    report(df_signed_los, outs_signed_los, ["LOS"], scalar=True)
    report(df_delta, outs_delta, ["delta"], scalar=True)
    report(df_delta_ratio, outs_delta_ratio, ["delta_ratio"], scalar=True)

    if keep_all:
        def _print_stats(name, stats_list):
            if stats_list[0] is None: 
                return
            
            for seed, stats in enumerate(stats_list):
                print(f"\n[{name} distribution](seed: {seed})")
                for k, v in stats.items():
                    print(f"  {k:>6}: {v:.6f}")

        _print_stats("LOS_abs", los_abs_stats_list)
        _print_stats("delta", delta_stats_list)
        _print_stats("delta_ratio", delta_ratio_stats_list)