from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error


def make_ordinal_targets(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class labels to cumulative binary targets for ordinal regression.

    y = k  →  first k thresholds are 1, rest are 0
    Output shape: [B, num_classes - 1]
    """
    thresholds = torch.arange(num_classes - 1, device=y.device)
    return (y.unsqueeze(1) > thresholds.unsqueeze(0)).float()


def ordinal_logits_to_class(
    logits: torch.Tensor, threshold: Union[float, torch.Tensor] = 0.5
) -> torch.Tensor:
    """Decode ordinal logits to predicted class index. Output shape: [B]."""
    probs = torch.sigmoid(logits)
    if not torch.is_tensor(threshold):
        threshold = torch.full(
            (probs.shape[1],),
            float(threshold),
            device=probs.device,
            dtype=probs.dtype,
        )
    else:
        threshold = threshold.to(device=probs.device, dtype=probs.dtype)
        if threshold.ndim == 0:
            threshold = threshold.repeat(probs.shape[1])
    if threshold.numel() != probs.shape[1]:
        raise ValueError(
            f"threshold must have {probs.shape[1]} elements, got {threshold.numel()}"
        )
    return (probs > threshold).sum(dim=1)


def score_ordinal_objective(
    y_true: np.ndarray, y_pred: np.ndarray, objective: str
) -> float:
    """Score ordinal predictions for calibration search."""
    objective = objective.lower()
    if objective == "qwk":
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    if objective == "mae":
        return -float(mean_absolute_error(y_true, y_pred))
    if objective == "within_1_acc":
        return float(np.mean(np.abs(y_true - y_pred) <= 1))
    if objective == "acc":
        return float(accuracy_score(y_true, y_pred))
    if objective == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    raise ValueError(f"Unsupported calibration objective: {objective}")


def compute_ordinal_constraint_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within_1_acc": float(np.mean(abs_err <= 1)),
        "within_2_acc": float(np.mean(abs_err <= 2)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }


def _ordinal_probs_to_class(
    probs_np: np.ndarray, thresholds_np: np.ndarray
) -> np.ndarray:
    if probs_np.ndim != 2:
        raise ValueError(f"probs_np must be 2D, got shape {probs_np.shape}")
    if thresholds_np.shape != (probs_np.shape[1],):
        raise ValueError(
            f"thresholds shape must be {(probs_np.shape[1],)}, got {thresholds_np.shape}"
        )
    return (probs_np > thresholds_np.reshape(1, -1)).sum(axis=1).astype(int)


def fit_ordinal_thresholds(
    logits_np: np.ndarray,
    targets_np: np.ndarray,
    objective: str = "qwk",
    min_metrics: Optional[Dict[str, float]] = None,
) -> Dict[str, np.ndarray | float | Dict[str, float]]:
    """Fit per-threshold cutoffs on validation logits via coordinate search."""
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    thresholds = np.full(probs_np.shape[1], 0.5, dtype=np.float32)
    preds = _ordinal_probs_to_class(probs_np, thresholds)
    best_metrics = compute_ordinal_constraint_metrics(targets_np, preds)
    best_score = score_ordinal_objective(targets_np, preds, objective)

    def satisfies_constraints(metrics: Dict[str, float]) -> bool:
        if not min_metrics:
            return True
        for metric_name, minimum in min_metrics.items():
            if metrics.get(metric_name, -float("inf")) < minimum:
                return False
        return True

    coarse_grid = np.arange(0.05, 0.951, 0.05, dtype=np.float32)
    for idx in range(thresholds.shape[0]):
        local_best_threshold = float(thresholds[idx])
        local_best_score = float(best_score)
        for candidate in coarse_grid:
            candidate_thresholds = thresholds.copy()
            candidate_thresholds[idx] = float(candidate)
            candidate_preds = _ordinal_probs_to_class(probs_np, candidate_thresholds)
            candidate_metrics = compute_ordinal_constraint_metrics(targets_np, candidate_preds)
            if not satisfies_constraints(candidate_metrics):
                continue
            candidate_score = score_ordinal_objective(targets_np, candidate_preds, objective)
            if candidate_score > local_best_score:
                local_best_score = float(candidate_score)
                local_best_threshold = float(candidate)
                best_metrics = candidate_metrics
        thresholds[idx] = local_best_threshold
        best_score = local_best_score

        refine_start = max(0.01, local_best_threshold - 0.05)
        refine_end = min(0.99, local_best_threshold + 0.05)
        refine_grid = np.arange(refine_start, refine_end + 0.001, 0.01, dtype=np.float32)
        for candidate in refine_grid:
            candidate_thresholds = thresholds.copy()
            candidate_thresholds[idx] = float(candidate)
            candidate_preds = _ordinal_probs_to_class(probs_np, candidate_thresholds)
            candidate_metrics = compute_ordinal_constraint_metrics(targets_np, candidate_preds)
            if not satisfies_constraints(candidate_metrics):
                continue
            candidate_score = score_ordinal_objective(targets_np, candidate_preds, objective)
            if candidate_score > best_score:
                best_score = float(candidate_score)
                thresholds[idx] = float(candidate)
                best_metrics = candidate_metrics

    calibrated_preds = _ordinal_probs_to_class(probs_np, thresholds)
    calibrated_metrics = compute_ordinal_constraint_metrics(targets_np, calibrated_preds)
    return {
        "thresholds": thresholds.astype(np.float32),
        "best_score": float(best_score),
        "preds": calibrated_preds,
        "metrics": calibrated_metrics,
    }


def compute_ordinal_pos_weight(
    y: torch.Tensor, num_classes: int, max_weight: float = 10.0
) -> torch.Tensor:
    """Per-threshold positive weights for BCEWithLogitsLoss. Output shape: [num_classes - 1]."""
    targets = make_ordinal_targets(y, num_classes)
    pos = targets.sum(dim=0)
    neg = targets.shape[0] - pos
    pos_weight = neg / pos.clamp_min(1.0)
    return pos_weight.clamp(max=max_weight)


def compute_ce_class_weight(
    y: torch.Tensor,
    num_classes: int,
    mode: str = "inverse",
    beta: float = 0.999,
    max_weight: Optional[float] = None,
) -> torch.Tensor:
    """Compute class weights for CrossEntropyLoss from training labels only.

    Args:
        y: Integer labels [N].
        num_classes: Total number of LOS classes.
        mode:
            - ``inverse``: 1 / count
            - ``inverse_sqrt``: 1 / sqrt(count)
            - ``effective_num``: Class-Balanced Loss style weight
        beta: Effective-number beta when ``mode == "effective_num"``.
        max_weight: Optional upper clip after mean-normalization.
    """
    counts = torch.bincount(y.long(), minlength=num_classes).float()
    safe_counts = counts.clamp_min(1.0)

    mode = mode.lower()
    if mode == "inverse":
        weights = 1.0 / safe_counts
    elif mode == "inverse_sqrt":
        weights = 1.0 / torch.sqrt(safe_counts)
    elif mode == "effective_num":
        beta = float(beta)
        effective_num = 1.0 - torch.pow(torch.full_like(safe_counts, beta), safe_counts)
        weights = (1.0 - beta) / effective_num.clamp_min(1.0e-12)
    else:
        raise ValueError(f"Unsupported CE class-weight mode: {mode}")

    weights = weights / weights.mean().clamp_min(1.0e-12)
    if max_weight is not None:
        weights = weights.clamp(max=float(max_weight))
        weights = weights / weights.mean().clamp_min(1.0e-12)
    return weights


class OrdinalBCELoss(nn.Module):
    """BCEWithLogitsLoss over cumulative ordinal targets.

    Args:
        num_classes: Total number of LOS classes K. The model outputs K-1 logits.
        pos_weight: Optional [K-1] tensor pre-computed on the training set.
    """

    def __init__(self, num_classes: int, pos_weight: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, K-1] raw ordinal logits
            y:      [B]      integer class labels in [0, K-1]
        Returns:
            Scalar loss.
        """
        targets = make_ordinal_targets(y, self.num_classes)
        pw = self.pos_weight.to(logits.device) if self.pos_weight is not None else None
        return nn.functional.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw)
