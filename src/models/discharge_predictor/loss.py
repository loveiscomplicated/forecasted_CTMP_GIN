from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskCategoricalLoss(nn.Module):
    """Weighted sum of per-target cross-entropy losses."""

    def __init__(
        self,
        task_weights: Optional[Dict[str, float]] = None,
        ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.task_weights = task_weights or {}
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = next(iter(logits.values())).device
        total_loss = torch.zeros(1, device=device).squeeze()
        loss_dict: Dict[str, float] = {}

        for name, pred in logits.items():
            target = targets[name].long()
            weight = self.task_weights.get(name, 1.0)
            loss = self.ce(pred, target)
            total_loss = total_loss + weight * loss
            loss_dict[f"loss_{name}"] = float(loss.detach().cpu())

        loss_dict["loss_total"] = float(total_loss.detach().cpu())
        return total_loss, loss_dict


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    gamma: float = 2.0,
    alpha: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Multiclass focal loss computed from cross-entropy terms."""
    if reduction not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduction}")

    ce = F.cross_entropy(
        logits,
        targets,
        weight=None,
        ignore_index=ignore_index,
        reduction="none",
        label_smoothing=label_smoothing,
    )

    valid_mask = targets != ignore_index
    if not torch.any(valid_mask):
        return logits.new_zeros(())

    ce_valid = ce[valid_mask]
    target_valid = targets[valid_mask]
    pt = torch.exp(-ce_valid)
    focal_factor = torch.pow((1.0 - pt).clamp_min(0.0), gamma)
    loss = focal_factor * ce_valid

    if alpha is not None:
        alpha_t = alpha.to(device=logits.device, dtype=logits.dtype)[target_valid]
        loss = loss * alpha_t

    if reduction == "none":
        out = torch.zeros_like(ce)
        out[valid_mask] = loss
        return out
    if reduction == "sum":
        return loss.sum()
    return loss.mean()
