from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from src.models.discharge_predictor.joint_consistent_predictor import JointPredictorOutput
from src.models.discharge_predictor.soft_joint_drift_loss import SoftJointDriftLoss


def _safe_js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    eps = 1.0e-12
    p = p / p.sum().clamp_min(eps)
    q = q / q.sum().clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum()
    kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum()
    return 0.5 * (kl_pm + kl_qm)


def _conditional_distribution_from_targets(
    targets: torch.Tensor,
    conditions: torch.Tensor,
    *,
    num_target_classes: int,
    num_condition_classes: int,
    output_device: torch.device | None = None,
) -> torch.Tensor:
    result_device = output_device or targets.device
    if targets.device.type == "cpu":
        targets_cpu = targets.detach().long().contiguous().view(-1)
    else:
        targets_cpu = targets.detach().long().contiguous().cpu().view(-1)
    if conditions.device.type == "cpu":
        conditions_cpu = conditions.detach().long().contiguous().view(-1)
    else:
        conditions_cpu = conditions.detach().long().contiguous().cpu().view(-1)
    if targets_cpu.numel() != conditions_cpu.numel():
        raise ValueError(
            "targets and conditions must contain the same number of elements: "
            f"got {targets_cpu.numel()} and {conditions_cpu.numel()}"
        )
    if targets_cpu.numel() > 0:
        target_min = int(targets_cpu.min().item())
        target_max = int(targets_cpu.max().item())
        condition_min = int(conditions_cpu.min().item())
        condition_max = int(conditions_cpu.max().item())
        if target_min < 0 or target_max >= num_target_classes:
            raise ValueError(
                "target labels are outside the expected class range "
                f"[0, {num_target_classes - 1}]: min={target_min}, max={target_max}"
            )
        if condition_min < 0 or condition_max >= num_condition_classes:
            raise ValueError(
                "condition labels are outside the expected class range "
                f"[0, {num_condition_classes - 1}]: "
                f"min={condition_min}, max={condition_max}"
            )
    table = torch.zeros(
        (num_condition_classes, num_target_classes),
        device="cpu",
        dtype=torch.float32,
    )
    for cond_idx in range(num_condition_classes):
        mask = conditions_cpu == cond_idx
        if not torch.any(mask):
            continue
        counts = torch.bincount(
            targets_cpu[mask],
            minlength=num_target_classes,
        ).to(dtype=torch.float32)
        table[cond_idx] = counts / counts.sum().clamp_min(1.0)
    return table.to(device=result_device)


def _conditional_distribution_from_probs(
    target_probs: torch.Tensor,
    condition_probs: torch.Tensor,
) -> torch.Tensor:
    weight = condition_probs.sum(dim=0, keepdim=True).transpose(0, 1).clamp_min(1.0e-12)
    return (condition_probs.transpose(0, 1) @ target_probs) / weight


def compute_joint_consistency_penalty(
    output: JointPredictorOutput,
    d_targets: Dict[str, torch.Tensor],
    los_targets: torch.Tensor,
    *,
    joint_head_names: Sequence[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    loss = output.final_los_probs.new_zeros(())
    diagnostics: dict[str, float] = {}
    num_terms = 0

    los_pred_hard = torch.argmax(output.final_los_probs, dim=1)
    for head_name in joint_head_names:
        d_pred_probs = output.final_d_probs[head_name]
        d_target = d_targets[head_name].long()
        d_cond_pred = _conditional_distribution_from_probs(d_pred_probs, output.final_los_probs)
        d_cond_true = _conditional_distribution_from_targets(
            d_target,
            los_targets.long(),
            num_target_classes=d_pred_probs.shape[1],
            num_condition_classes=output.final_los_probs.shape[1],
            output_device=d_pred_probs.device,
        )
        term = torch.stack(
            [
                _safe_js_divergence(d_cond_pred[idx], d_cond_true[idx])
                for idx in range(d_cond_pred.shape[0])
            ]
        ).mean()
        diagnostics[f"joint_js_d_given_los_{head_name}"] = float(term.detach().cpu())
        loss = loss + term
        num_terms += 1

        los_cond_pred = _conditional_distribution_from_probs(output.final_los_probs, d_pred_probs)
        los_cond_true = _conditional_distribution_from_targets(
            los_targets.long(),
            d_target,
            num_target_classes=output.final_los_probs.shape[1],
            num_condition_classes=d_pred_probs.shape[1],
            output_device=output.final_los_probs.device,
        )
        term = torch.stack(
            [
                _safe_js_divergence(los_cond_pred[idx], los_cond_true[idx])
                for idx in range(los_cond_pred.shape[0])
            ]
        ).mean()
        diagnostics[f"joint_js_los_given_d_{head_name}"] = float(term.detach().cpu())
        loss = loss + term
        num_terms += 1

    diagnostics["joint_consistency_terms"] = float(num_terms)
    if num_terms == 0:
        return loss, diagnostics
    return loss / float(num_terms), diagnostics


class JointConsistencyLoss(nn.Module):
    def __init__(
        self,
        *,
        lambda_los: float = 1.0,
        lambda_aux: float = 0.3,
        lambda_joint: float = 0.0,
        joint_head_names: Sequence[str],
        lambda_struct: float = 0.0,
        struct_loss_module: SoftJointDriftLoss | None = None,
    ) -> None:
        super().__init__()
        self.lambda_los = float(lambda_los)
        self.lambda_aux = float(lambda_aux)
        self.lambda_joint = float(lambda_joint)
        self.joint_head_names = list(joint_head_names)
        self.lambda_struct = float(lambda_struct)
        self.struct_loss_module = struct_loss_module
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        output: JointPredictorOutput,
        *,
        d_targets: Dict[str, torch.Tensor],
        los_targets: torch.Tensor,
        d_targets_for_joint: Dict[str, torch.Tensor] | None = None,
        los_targets_for_joint: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        final_d_loss = output.final_los_logits.new_zeros(())
        base_d_loss = output.final_los_logits.new_zeros(())
        for head_name, target in d_targets.items():
            final_d_loss = final_d_loss + self.ce(output.final_d_logits[head_name], target.long())
            base_d_loss = base_d_loss + self.ce(output.base_d_logits[head_name], target.long())

        final_los_loss = self.ce(output.final_los_logits, los_targets.long())
        base_los_loss = self.ce(output.base_los_logits, los_targets.long())
        aux_loss = base_d_loss + self.lambda_los * base_los_loss

        joint_penalty = output.final_los_logits.new_zeros(())
        joint_metrics: dict[str, float] = {}
        if self.lambda_joint > 0.0:
            joint_penalty, joint_metrics = compute_joint_consistency_penalty(
                output,
                d_targets_for_joint or d_targets,
                los_targets if los_targets_for_joint is None else los_targets_for_joint,
                joint_head_names=self.joint_head_names,
            )

        struct_penalty = output.final_los_logits.new_zeros(())
        struct_metrics: dict[str, float] = {}
        if self.lambda_struct > 0.0:
            if self.struct_loss_module is None:
                raise ValueError("lambda_struct > 0 requires struct_loss_module")
            struct_penalty, struct_metrics = self.struct_loss_module(
                los_probs=output.final_los_probs,
                d_logits=output.final_d_logits,
                d_targets=d_targets,
            )

        total = (
            final_d_loss
            + self.lambda_los * final_los_loss
            + self.lambda_aux * aux_loss
            + self.lambda_joint * joint_penalty
            + self.lambda_struct * struct_penalty
        )
        metrics = {
            "loss_total": float(total.detach().cpu()),
            "loss_final_d": float(final_d_loss.detach().cpu()),
            "loss_final_los": float(final_los_loss.detach().cpu()),
            "loss_base_d": float(base_d_loss.detach().cpu()),
            "loss_base_los": float(base_los_loss.detach().cpu()),
            "loss_aux": float(aux_loss.detach().cpu()),
            "loss_joint": float(joint_penalty.detach().cpu()),
            **joint_metrics,
        }
        if self.lambda_struct > 0.0:
            metrics["loss_struct"] = float(struct_penalty.detach().cpu())
            metrics.update(struct_metrics)
        return total, metrics
