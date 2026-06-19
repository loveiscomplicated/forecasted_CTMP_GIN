from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, mean_absolute_error


def compute_discharge_metrics(
    all_logits: Dict[str, np.ndarray],
    all_targets: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute per-target accuracy and macro-F1, plus dataset-level means.

    Args:
        all_logits: name → [N, K] softmax-logit array
        all_targets: name → [N,] integer label array

    Returns:
        Flat dict with acc_{name}, f1_{name}, mean_accuracy, mean_macro_f1.
    """
    result: Dict[str, float] = {}
    accuracies = []
    f1s = []

    for name in all_logits:
        preds = np.argmax(all_logits[name], axis=1)
        targets = all_targets[name]
        acc = float(accuracy_score(targets, preds))
        f1 = float(f1_score(targets, preds, average="macro", zero_division=0))
        result[f"acc_{name}"] = acc
        result[f"f1_{name}"] = f1
        accuracies.append(acc)
        f1s.append(f1)

    result["mean_accuracy"] = float(np.mean(accuracies)) if accuracies else 0.0
    result["mean_macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return result


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Ordinal-aware metrics for LOS prediction."""
    abs_err = np.abs(y_true - y_pred)
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "within_1_acc": float(np.mean(abs_err <= 1)),
        "within_2_acc": float(np.mean(abs_err <= 2)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
