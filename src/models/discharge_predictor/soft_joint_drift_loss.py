from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discharge_predictor.risk_heads import resolve_risk_head_selection


def _js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float) -> torch.Tensor:
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p.clamp_min(eps)) - torch.log(m.clamp_min(eps))))
    kl_qm = torch.sum(q * (torch.log(q.clamp_min(eps)) - torch.log(m.clamp_min(eps))))
    return 0.5 * (kl_pm + kl_qm)


class SoftJointDriftLoss(nn.Module):
    def __init__(
        self,
        *,
        risk_head_set: str,
        available_heads: Sequence[str],
        stopgrad_los: bool = True,
        min_los_support: float = 1.0e-6,
        eps: float = 1.0e-8,
        weight_by_los_support: bool = True,
        use_ema: bool = False,
        ema_momentum: float = 0.95,
    ) -> None:
        super().__init__()
        if use_ema:
            raise ValueError(
                "joint_struct_loss.use_ema=true is reserved for a future version. "
                "v1 supports only batch-local structured loss."
            )
        self.risk_head_set = str(risk_head_set)
        self.resolved_risk_heads = resolve_risk_head_selection(
            self.risk_head_set,
            available_heads=available_heads,
            mode="strict_named_set",
            field_name="joint_struct_loss.risk_head_set",
        )
        if not self.resolved_risk_heads:
            raise ValueError("joint_struct_loss resolved zero risk heads.")
        self.stopgrad_los = bool(stopgrad_los)
        self.min_los_support = float(min_los_support)
        self.eps = float(eps)
        self.weight_by_los_support = bool(weight_by_los_support)
        self.use_ema = bool(use_ema)
        self.ema_momentum = float(ema_momentum)

    def forward(
        self,
        *,
        los_probs: torch.Tensor,
        d_logits: dict[str, torch.Tensor],
        d_targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if los_probs.ndim != 2:
            raise ValueError(f"los_probs must have shape [B, L], got {tuple(los_probs.shape)}")
        los_w = los_probs.detach() if self.stopgrad_los else los_probs

        per_head_losses: dict[str, torch.Tensor] = {}
        for head_name in self.resolved_risk_heads:
            if head_name not in d_logits or head_name not in d_targets:
                raise ValueError(f"Missing logits or targets for structured-loss head {head_name}")
            logits_h = d_logits[head_name]
            target_h = d_targets[head_name].long()
            if logits_h.ndim != 2:
                raise ValueError(f"{head_name}: logits must have shape [B, C]")
            if target_h.ndim != 1:
                raise ValueError(f"{head_name}: targets must have shape [B]")
            if logits_h.shape[0] != los_w.shape[0] or target_h.shape[0] != los_w.shape[0]:
                raise ValueError(f"{head_name}: batch dimension mismatch across logits/targets/los_probs")

            p_d_h = F.softmax(logits_h, dim=1)
            y_onehot_h = F.one_hot(target_h, num_classes=logits_h.shape[1]).to(dtype=p_d_h.dtype)
            t_mid = y_onehot_h.transpose(0, 1) @ los_w
            t_full = p_d_h.transpose(0, 1) @ los_w
            support = los_w.sum(dim=0)

            per_bin_terms: list[torch.Tensor] = []
            per_bin_weights: list[torch.Tensor] = []
            for los_idx in range(los_w.shape[1]):
                bin_support = support[los_idx]
                if float(bin_support.detach().cpu()) < self.min_los_support:
                    continue
                js = _js_divergence(t_mid[:, los_idx], t_full[:, los_idx], self.eps)
                per_bin_terms.append(js)
                per_bin_weights.append(bin_support)

            if not per_bin_terms:
                per_head_losses[head_name] = logits_h.new_zeros(())
                continue

            stacked_terms = torch.stack(per_bin_terms)
            if self.weight_by_los_support:
                weights = torch.stack(per_bin_weights).to(dtype=stacked_terms.dtype)
                per_head_losses[head_name] = (stacked_terms * weights).sum() / weights.sum().clamp_min(self.eps)
            else:
                per_head_losses[head_name] = stacked_terms.mean()

        total = torch.stack([per_head_losses[name] for name in self.resolved_risk_heads]).mean()
        metrics = {"loss_struct": float(total.detach().cpu())}
        for head_name in self.resolved_risk_heads:
            metrics[f"loss_struct_{head_name}"] = float(per_head_losses[head_name].detach().cpu())
        return total, metrics

    def settings_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "lambda_struct": None,
            "loss_type": "soft_js_d",
            "risk_head_set": self.risk_head_set,
            "resolved_risk_heads": list(self.resolved_risk_heads),
            "stopgrad_los": self.stopgrad_los,
            "min_los_support": self.min_los_support,
            "eps": self.eps,
            "weight_by_los_support": self.weight_by_los_support,
            "use_ema": self.use_ema,
            "ema_momentum": self.ema_momentum,
        }
