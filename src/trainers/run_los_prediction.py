from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data_processing.los_prediction_dataset import LOSPredictionDataset, split_los_dataset
from src.models.discharge_predictor import (
    LOSCoarsePredictor,
    LOSOrdinalPredictor,
    OrdinalBCELoss,
    compute_ordinal_metrics,
    get_los_coarse_class_labels,
    get_los_coarse_num_classes,
    infer_los_coarse_breakdown_from_cfg,
    infer_los_target_from_cfg,
    los_binning_metadata_dict,
    map_los_array_to_coarse_bins,
    multiclass_focal_loss,
)
from src.models.discharge_predictor.ordinal_loss import (
    compute_ce_class_weight,
    compute_ordinal_pos_weight,
    fit_ordinal_thresholds,
    ordinal_logits_to_class,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.device_set import device_set
from src.utils.experiment import ExperimentLogger, ensure_run_dir, make_run_id
from src.utils.seed_set import set_seed


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for LOS prediction training."""
    parser = argparse.ArgumentParser(description="Train LOS predictor")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def load_yaml(path: str) -> dict:
    """Load a YAML config from disk."""
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_cfg(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply CLI overrides to a config dict."""
    if args.device is not None:
        cfg["device"] = args.device
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    return cfg


def _coarse_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[str, float]:
    abs_err = np.abs(y_true - y_pred)
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        output_dict=True,
        zero_division=0,
    )
    metrics: Dict[str, float] = {
        "acc": float((y_true == y_pred).mean()),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mae": float(abs_err.mean()),
        "within_1_acc": float((abs_err <= 1).mean()),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
    }
    for cls in range(num_classes):
        cls_key = str(cls)
        cls_report = report.get(cls_key, {})
        metrics[f"precision_{cls}"] = float(cls_report.get("precision", 0.0))
        metrics[f"recall_{cls}"] = float(cls_report.get("recall", 0.0))
        metrics[f"f1_{cls}"] = float(cls_report.get("f1-score", 0.0))
    return metrics


def _is_ce_like_loss(loss_type: str) -> bool:
    return loss_type in {"ce", "focal", "focal_alpha", "cb_focal"}


def _softmax_np(logits_np: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    probs_t = F.softmax(logits_t / float(temperature), dim=1)
    return probs_t.cpu().numpy()


def _compute_multiclass_nll(logits_np: np.ndarray, targets_np: np.ndarray, temperature: float = 1.0) -> float:
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    targets_t = torch.tensor(targets_np, dtype=torch.long)
    loss = F.cross_entropy(logits_t / float(temperature), targets_t, reduction="mean")
    return float(loss.detach().cpu())


def _compute_top_label_ece(probs_np: np.ndarray, targets_np: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs_np.max(axis=1)
    predictions = probs_np.argmax(axis=1)
    correctness = (predictions == targets_np).astype(np.float64)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for idx in range(n_bins):
        lo = bin_edges[idx]
        hi = bin_edges[idx + 1]
        if idx == n_bins - 1:
            in_bin = (confidences >= lo) & (confidences <= hi)
        else:
            in_bin = (confidences >= lo) & (confidences < hi)
        if not np.any(in_bin):
            continue
        bin_acc = correctness[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += (in_bin.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def _fit_temperature_scaling(logits_np: np.ndarray, targets_np: np.ndarray) -> float:
    logits_t = torch.tensor(logits_np, dtype=torch.float32)
    targets_t = torch.tensor(targets_np, dtype=torch.long)
    log_t = torch.zeros((), dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=50, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = log_t.exp().clamp_min(1.0e-6)
        loss = F.cross_entropy(logits_t / temperature, targets_t, reduction="mean")
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().cpu())


def _build_coarse_metrics_from_logits(
    logits_np: np.ndarray,
    targets_np: np.ndarray,
    *,
    num_classes: int,
    temperature: float = 1.0,
    ece_bins: int = 15,
) -> tuple[Dict[str, float], np.ndarray]:
    probs_np = _softmax_np(logits_np, temperature=temperature)
    preds_np = probs_np.argmax(axis=1)
    metrics = _coarse_metrics(targets_np, preds_np, num_classes)
    metrics["nll"] = _compute_multiclass_nll(logits_np, targets_np, temperature=temperature)
    metrics["ece"] = _compute_top_label_ece(probs_np, targets_np, n_bins=ece_bins)
    return metrics, probs_np


def _build_coarse_targets(raw_los: torch.Tensor, *, breakdown: bool = False) -> torch.Tensor:
    return map_los_array_to_coarse_bins(raw_los, breakdown=breakdown).long()


def _select_logits(outputs, loss_type: str):
    if _is_ce_like_loss(loss_type):
        return outputs
    if loss_type == "hybrid_ce_ordinal":
        return outputs["ordinal"]
    return outputs


def _num_prediction_classes(logits_np: np.ndarray, loss_type: str) -> int:
    return int(logits_np.shape[1]) if _is_ce_like_loss(loss_type) else int(logits_np.shape[1] + 1)


COARSE_CLASS_LABELS = get_los_coarse_class_labels()


def _build_coarse_label_counts(y_true: np.ndarray, num_classes: int) -> Dict[str, int]:
    counts = np.bincount(y_true.astype(int), minlength=num_classes)
    return {f"class_{cls}": int(count) for cls, count in enumerate(counts)}


def _build_class_count_list(y_true: np.ndarray, num_classes: int) -> list[int]:
    counts = np.bincount(y_true.astype(int), minlength=num_classes)
    return [int(count) for count in counts]


def _majority_baseline_predictions(train_y: np.ndarray, size: int) -> np.ndarray:
    counts = np.bincount(train_y.astype(int))
    majority_class = int(np.argmax(counts))
    return np.full(size, majority_class, dtype=np.int64)


def _stratified_baseline_predictions(
    train_y: np.ndarray,
    size: int,
    *,
    seed: int,
    num_classes: int,
) -> np.ndarray:
    counts = np.bincount(train_y.astype(int), minlength=num_classes).astype(np.float64)
    probs = counts / counts.sum()
    rng = np.random.default_rng(seed)
    return rng.choice(np.arange(num_classes), size=size, p=probs).astype(np.int64)


def _print_coarse_per_class_metrics(
    label: str,
    metrics: Dict[str, float],
    class_labels: tuple[str, ...] = COARSE_CLASS_LABELS,
) -> None:
    print(label)
    for cls, class_label in enumerate(class_labels):
        print(
            f"  class_{cls} [{class_label}] "
            f"precision={metrics[f'precision_{cls}']:.4f} | "
            f"recall={metrics[f'recall_{cls}']:.4f} | "
            f"f1={metrics[f'f1_{cls}']:.4f}"
        )
    if len(class_labels) == len(COARSE_CLASS_LABELS):
        print(
            "  long_stay [29-37] "
            f"precision={metrics['precision_5']:.4f} | "
            f"recall={metrics['recall_5']:.4f} | "
            f"f1={metrics['f1_5']:.4f}"
        )


def _print_coarse_test_summary(label: str, metrics: Dict[str, float]) -> None:
    summary = (
        f"{label} loss={metrics['loss']:.4f} | "
        f"acc={metrics['acc']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"weighted_f1={metrics['weighted_f1']:.4f} | "
        f"mae={metrics['mae']:.3f} | "
        f"within_1={metrics['within_1_acc']:.4f} | "
        f"qwk={metrics['qwk']:.4f}"
    )
    if "nll" in metrics:
        summary += f" | nll={metrics['nll']:.4f}"
    if "ece" in metrics:
        summary += f" | ece={metrics['ece']:.4f}"
    print(summary)


def _save_coarse_baselines_artifact(
    run_dir: str,
    *,
    model_metrics: Dict[str, float],
    majority_metrics: Dict[str, float],
    stratified_metrics: Dict[str, float],
    train_counts: Dict[str, int],
    test_counts: Dict[str, int],
    class_labels: tuple[str, ...] = COARSE_CLASS_LABELS,
) -> None:
    payload = {
        "model_metrics": {k: float(v) for k, v in model_metrics.items()},
        "majority_metrics": {k: float(v) for k, v in majority_metrics.items()},
        "stratified_metrics": {k: float(v) for k, v in stratified_metrics.items()},
        "train_class_counts": train_counts,
        "test_class_counts": test_counts,
        "class_labels": {str(idx): label for idx, label in enumerate(class_labels)},
    }
    out_path = os.path.join(run_dir, "test_baselines_coarse.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Baseline metrics saved: {out_path}")


def _train_one_epoch(
    model,
    loader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_type: str,
    ce_weight: float,
    ce_class_weight: torch.Tensor | None,
    target_mode: str,
    coarse_breakdown: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y, raw_y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        raw_y = raw_y.to(device, non_blocking=True)
        if target_mode == "coarse":
            y = _build_coarse_targets(raw_y, breakdown=coarse_breakdown)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(x)
        loss = _compute_los_loss(outputs, y, criterion, loss_type, ce_weight, ce_class_weight)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _evaluate(
    model,
    loader,
    criterion,
    device: torch.device,
    loss_type: str,
    ce_weight: float,
    ce_class_weight: torch.Tensor | None,
    target_mode: str,
    coarse_breakdown: bool = False,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_targets_raw: List[np.ndarray] = []
    all_targets: List[np.ndarray] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y, raw_y in tqdm(loader, desc="eval", leave=False):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            raw_y = raw_y.to(device, non_blocking=True)
            targets = (
                _build_coarse_targets(raw_y, breakdown=coarse_breakdown)
                if target_mode == "coarse"
                else y
            )
            outputs = model(x)
            loss = _compute_los_loss(outputs, targets, criterion, loss_type, ce_weight, ce_class_weight)
            logits = _select_logits(outputs, loss_type)
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            all_logits.append(logits.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_targets_raw.append(raw_y.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)
    raw_targets_np = np.concatenate(all_targets_raw, axis=0)

    if _is_ce_like_loss(loss_type):
        preds_np = np.argmax(logits_np, axis=1)
    else:
        preds_np = ordinal_logits_to_class(torch.tensor(logits_np), threshold=0.5).numpy()

    if target_mode == "coarse":
        metrics = _coarse_metrics(targets_np, preds_np, _num_prediction_classes(logits_np, loss_type))
    else:
        metrics = compute_ordinal_metrics(targets_np, preds_np)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics, logits_np, raw_targets_np, preds_np


def _compute_los_loss(outputs, y: torch.Tensor, criterion, loss_type: str, ce_weight: float, ce_class_weight: torch.Tensor | None) -> torch.Tensor:
    ce_weight_tensor = None
    if ce_class_weight is not None:
        ce_weight_tensor = ce_class_weight.to(device=y.device, dtype=torch.float32)
    if loss_type == "ce":
        return F.cross_entropy(outputs, y, weight=ce_weight_tensor)
    if loss_type in {"focal", "focal_alpha", "cb_focal"}:
        gamma = float(criterion["gamma"])
        alpha = criterion.get("alpha")
        label_smoothing = float(criterion.get("label_smoothing", 0.0))
        return multiclass_focal_loss(
            outputs,
            y,
            gamma=gamma,
            alpha=alpha,
            label_smoothing=label_smoothing,
            reduction="mean",
        )
    if loss_type == "hybrid_ce_ordinal":
        ordinal_loss = criterion(outputs["ordinal"], y)
        ce_loss = F.cross_entropy(outputs["ce"], y, weight=ce_weight_tensor)
        return ordinal_loss + ce_weight * ce_loss
    if loss_type == "ordinal_bce":
        return criterion(outputs, y)
    raise ValueError(f"Unsupported LOS loss type: {loss_type}")


def _decode_logits_np(
    logits_np: np.ndarray,
    thresholds: float | np.ndarray = 0.5,
) -> np.ndarray:
    threshold_tensor = torch.tensor(thresholds, dtype=torch.float32)
    return ordinal_logits_to_class(
        torch.tensor(logits_np, dtype=torch.float32), threshold=threshold_tensor
    ).numpy()


def _save_predictions_csv(
    run_dir: str,
    split_name: str,
    logits_np: np.ndarray,
    raw_targets_np: np.ndarray,
    predictions_np: np.ndarray | Dict[str, np.ndarray],
    target_mode: str,
    ordinal_logits: bool | None = None,
    coarse_probability_payload: Dict[str, np.ndarray] | None = None,
    coarse_breakdown: bool = False,
) -> None:
    rows: Dict[str, np.ndarray] = {}
    if target_mode == "coarse":
        assert not isinstance(predictions_np, dict)
        rows["true_los_raw"] = raw_targets_np.astype(int)
        rows["true_los_coarse"] = map_los_array_to_coarse_bins(
            raw_targets_np,
            breakdown=coarse_breakdown,
        ).astype(int)
        if coarse_probability_payload is not None:
            rows["pred_los_coarse_raw"] = coarse_probability_payload["pred_raw"].astype(int)
            rows["pred_los_coarse_calibrated"] = coarse_probability_payload["pred_calibrated"].astype(int)
            for k in range(logits_np.shape[1]):
                rows[f"logit_{k}"] = logits_np[:, k]
            for k in range(coarse_probability_payload["prob_raw"].shape[1]):
                rows[f"prob_raw_{k}"] = coarse_probability_payload["prob_raw"][:, k]
                rows[f"prob_cal_{k}"] = coarse_probability_payload["prob_calibrated"][:, k]
        else:
            rows["pred_los_coarse_raw"] = predictions_np.astype(int)
            rows["pred_los_coarse_calibrated"] = predictions_np.astype(int)
        if ordinal_logits:
            probs = 1 / (1 + np.exp(-logits_np))
            for k in range(probs.shape[1]):
                rows[f"prob_gt_{k}"] = probs[:, k]
        elif coarse_probability_payload is None:
            probs = F.softmax(torch.tensor(logits_np), dim=1).cpu().numpy()
            for k in range(probs.shape[1]):
                rows[f"prob_class_{k}"] = probs[:, k]
        file_name = "predictions_los_coarse.csv" if split_name == "test" else f"{split_name}_predictions_los_coarse.csv"
    else:
        rows["true_LOS"] = raw_targets_np.astype(int)
        if isinstance(predictions_np, dict):
            for mode_name, preds_np in predictions_np.items():
                rows[f"pred_LOS_{mode_name}"] = preds_np.astype(int)
                rows[f"abs_error_{mode_name}"] = np.abs(raw_targets_np - preds_np).astype(int)
        else:
            rows["pred_LOS"] = predictions_np.astype(int)
        if ordinal_logits is None:
            ordinal_logits = True
        if ordinal_logits:
            probs = 1 / (1 + np.exp(-logits_np))
            for k in range(probs.shape[1]):
                rows[f"prob_gt_{k}"] = probs[:, k]
        else:
            probs = F.softmax(torch.tensor(logits_np), dim=1).cpu().numpy()
            for k in range(probs.shape[1]):
                rows[f"prob_class_{k}"] = probs[:, k]
        file_name = "predictions.csv" if split_name == "test" else f"{split_name}_predictions.csv"

    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, file_name)
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved: {csv_path}  ({len(df):,} rows)")


def _save_calibration_json(run_dir: str, payload: Dict[str, object]) -> None:
    json_path = os.path.join(run_dir, "calibration.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Calibration saved: {json_path}")


def _save_distribution_csv(
    run_dir: str,
    split_name: str,
    y_true: np.ndarray,
    predictions_by_mode: Dict[str, np.ndarray],
    num_classes: int,
) -> None:
    true_counts = np.bincount(y_true.astype(int), minlength=num_classes)
    rows: List[Dict[str, int]] = []
    for class_idx in range(num_classes):
        row: Dict[str, int] = {
            "class_idx": int(class_idx),
            "true_count": int(true_counts[class_idx]),
        }
        for mode_name, preds_np in predictions_by_mode.items():
            pred_counts = np.bincount(preds_np.astype(int), minlength=num_classes)
            row[f"pred_count_{mode_name}"] = int(pred_counts[class_idx])
            row[f"pred_minus_true_{mode_name}"] = int(
                pred_counts[class_idx] - true_counts[class_idx]
            )
        rows.append(row)
    csv_path = os.path.join(run_dir, f"{split_name}_prediction_distribution.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"Distribution saved: {csv_path}")


def _save_confusion_matrix(run_dir: str, split_name: str, y_true: np.ndarray, y_pred: np.ndarray, num_classes: int, suffix: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    csv_path = os.path.join(run_dir, f"{split_name}_confusion_matrix_{suffix}.csv")
    pd.DataFrame(cm, index=range(num_classes), columns=range(num_classes)).to_csv(csv_path, index_label="true_class")
    print(f"Confusion matrix saved: {csv_path}")


def _save_confusion_matrices(
    run_dir: str,
    split_name: str,
    y_true: np.ndarray,
    predictions_by_mode: Dict[str, np.ndarray],
    num_classes: int,
) -> None:
    for mode_name, preds_np in predictions_by_mode.items():
        _save_confusion_matrix(run_dir, split_name, y_true, preds_np, num_classes, mode_name)


def _print_metrics(label: str, metrics: Dict[str, float]) -> None:
    print(
        f"{label} acc={metrics['acc']:.4f} | "
        f"macro_f1={metrics['macro_f1']:.4f} | "
        f"mae={metrics['mae']:.3f} | "
        f"within_1={metrics['within_1_acc']:.4f} | "
        f"within_2={metrics['within_2_acc']:.4f} | "
        f"qwk={metrics['qwk']:.4f}"
    )


def _print_epoch_metrics(epoch: int, epochs: int, train_loss: float, val_metrics: Dict[str, float], target_mode: str) -> None:
    """Print a compact per-epoch metrics summary."""
    if target_mode == "coarse":
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['acc']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"weighted_f1={val_metrics['weighted_f1']:.4f} | "
            f"mae={val_metrics['mae']:.3f} | "
            f"within_1={val_metrics['within_1_acc']:.4f} | "
            f"qwk={val_metrics['qwk']:.4f}"
        )
    else:
        print(
            f"[Epoch {epoch}/{epochs}] "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['acc']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"mae={val_metrics['mae']:.3f} | "
            f"within_1={val_metrics['within_1_acc']:.4f} | "
            f"qwk={val_metrics['qwk']:.4f}"
        )


def run_los_prediction(cfg: dict, root: str) -> dict:
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)
    device = device_set(cfg.get("device"))
    target_mode = infer_los_target_from_cfg(cfg)
    coarse_breakdown = (
        infer_los_coarse_breakdown_from_cfg(cfg) if target_mode == "coarse" else False
    )
    coarse_num_classes = get_los_coarse_num_classes(breakdown=coarse_breakdown)
    coarse_class_labels = get_los_coarse_class_labels(breakdown=coarse_breakdown)
    if target_mode == "coarse":
        coarse_metadata = los_binning_metadata_dict(breakdown=coarse_breakdown)
        cfg["los_coarse_breakdown"] = coarse_breakdown
        cfg["num_classes"] = coarse_num_classes
        cfg["los_bins"] = coarse_metadata["los_bins"]
    run_id = make_run_id(cfg)
    run_dir = ensure_run_dir("runs", run_id)
    logger = ExperimentLogger(cfg, run_dir)
    calibration_cfg = cfg.get("calibration", {})
    calibration_enabled = bool(calibration_cfg.get("enabled", True))
    export_val_predictions = bool(calibration_cfg.get("export_val_predictions", False))
    export_test_predictions = bool(calibration_cfg.get("export_test_predictions", True))
    ece_bins = int(calibration_cfg.get("ece_bins", 15))

    loss_type = str(cfg.get("loss", {}).get("type", "ordinal_bce"))
    ce_weight = float(cfg.get("loss", {}).get("ce_weight", 1.0))
    output_mode = {
        "ordinal_bce": "ordinal",
        "ce": "ce",
        "focal": "ce",
        "focal_alpha": "ce",
        "cb_focal": "ce",
        "hybrid_ce_ordinal": "hybrid",
    }.get(loss_type)
    if output_mode is None:
        raise ValueError(f"Unsupported LOS loss type: {loss_type}")

    dataset = LOSPredictionDataset(root=root, do_preprocess=cfg["train"].get("do_preprocess", False))
    schema_metadata = dict(dataset.schema_metadata)
    schema_metadata["los_num_classes"] = int(dataset.los_num_classes)
    schema_metadata["los_target_mode"] = target_mode
    if target_mode == "coarse":
        schema_metadata.update(los_binning_metadata_dict(breakdown=coarse_breakdown))

    print(f"Admission variables : {len(dataset.ad_col_names)}")
    print(f"LOS num classes     : {dataset.los_num_classes}")
    print(f"Target mode         : {target_mode}")
    if target_mode == "coarse":
        print(f"Coarse breakdown    : {coarse_breakdown}")
        print(f"Coarse num classes  : {coarse_num_classes}")
    print(f"Dataset size        : {len(dataset):,}")

    split_ratio = [cfg["train"]["train_ratio"], cfg["train"]["val_ratio"], cfg["train"]["test_ratio"]]
    train_loader, val_loader, test_loader, (train_idx, _, _) = split_los_dataset(
        dataset=dataset,
        batch_size=cfg["train"]["batch_size"],
        ratio=split_ratio,
        seed=seed,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    train_y_raw = dataset.los_raw[train_idx]
    train_y = (
        _build_coarse_targets(train_y_raw, breakdown=coarse_breakdown)
        if target_mode == "coarse"
        else dataset.y[train_idx]
    )

    use_pos_weight = cfg.get("loss", {}).get("use_pos_weight", True)
    pos_weight = None
    ce_class_weight = None
    if use_pos_weight and not _is_ce_like_loss(loss_type):
        pw_clip = float(cfg.get("loss", {}).get("pos_weight_clip", 10.0))
        pos_weight = compute_ordinal_pos_weight(
            train_y,
            coarse_num_classes if target_mode == "coarse" else dataset.los_num_classes,
            max_weight=pw_clip,
        )
        print(f"pos_weight (first 5): {pos_weight[:5].tolist()}")

    use_ce_class_weight = bool(cfg.get("loss", {}).get("use_ce_class_weight", False))
    alpha_mode = str(cfg.get("loss", {}).get("alpha_mode", "none")).lower()
    alpha_clip = cfg.get("loss", {}).get("alpha_clip")
    label_smoothing = float(cfg.get("loss", {}).get("label_smoothing", 0.0))
    focal_gamma = float(cfg.get("loss", {}).get("gamma", 2.0))
    alpha_tensor = None

    if use_ce_class_weight and (loss_type == "ce" or loss_type == "hybrid_ce_ordinal"):
        ce_class_weight = compute_ce_class_weight(
            train_y,
            coarse_num_classes if target_mode == "coarse" else dataset.los_num_classes,
            mode=str(cfg.get("loss", {}).get("ce_class_weight_mode", "inverse")),
            beta=float(cfg.get("loss", {}).get("ce_class_weight_beta", 0.999)),
            max_weight=cfg.get("loss", {}).get("ce_class_weight_clip"),
        )
        print(f"ce_class_weight: {ce_class_weight.tolist()}")

    if loss_type in {"focal_alpha", "cb_focal"}:
        resolved_alpha_mode = alpha_mode
        if loss_type == "focal_alpha" and resolved_alpha_mode == "none":
            resolved_alpha_mode = "inverse_sqrt"
        if loss_type == "cb_focal":
            resolved_alpha_mode = "effective_num"
        alpha_tensor = compute_ce_class_weight(
            train_y,
            coarse_num_classes if target_mode == "coarse" else dataset.los_num_classes,
            mode=resolved_alpha_mode,
            beta=float(cfg.get("loss", {}).get("beta", cfg.get("loss", {}).get("ce_class_weight_beta", 0.999))),
            max_weight=alpha_clip,
        )
        print(f"focal_alpha: {alpha_tensor.tolist()}")

    if target_mode == "coarse":
        if _is_ce_like_loss(loss_type):
            model = LOSCoarsePredictor(
                ad_col_dims=dataset.ad_col_dims,
                num_classes=coarse_num_classes,
                **cfg["model"].get("params", {}),
            ).to(device)
            if loss_type == "ce":
                criterion = torch.nn.CrossEntropyLoss(
                    weight=ce_class_weight.to(device) if ce_class_weight is not None else None,
                    label_smoothing=label_smoothing,
                )
            else:
                criterion = {
                    "gamma": focal_gamma,
                    "alpha": alpha_tensor,
                    "label_smoothing": label_smoothing,
                }
        else:
            model = LOSOrdinalPredictor(
                ad_col_dims=dataset.ad_col_dims,
                los_num_classes=coarse_num_classes,
                output_mode=output_mode,
                **cfg["model"].get("params", {}),
            ).to(device)
            criterion = OrdinalBCELoss(
                num_classes=coarse_num_classes,
                pos_weight=pos_weight,
            ).to(device)
        monitor_metric = "valid_macro_f1"
        monitor_mode = "max"
    else:
        model = LOSOrdinalPredictor(
            ad_col_dims=dataset.ad_col_dims,
            los_num_classes=dataset.los_num_classes,
            output_mode=output_mode,
            **cfg["model"].get("params", {}),
        ).to(device)
        criterion = OrdinalBCELoss(num_classes=dataset.los_num_classes, pos_weight=pos_weight).to(device)
        monitor_metric = cfg["train"].get("monitor_metric", "valid_qwk")
        monitor_mode = str(cfg["train"].get("monitor_mode", "max")).lower()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {total_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    ) if cfg["train"].get("optimizer", "adamw") == "adamw" else torch.optim.Adam(
        model.parameters(),
        lr=cfg["train"]["learning_rate"],
        weight_decay=cfg["train"].get("weight_decay", 0.0),
    )

    scheduler = ReduceLROnPlateau(optimizer, monitor_mode, patience=cfg["train"]["lr_scheduler_patience"])
    early_stopper = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])
    epochs = cfg["train"]["epochs"]
    best_val = -float("inf")

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = _train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            loss_type,
            ce_weight,
            ce_class_weight,
            target_mode,
            coarse_breakdown,
        )
        val_metrics, _, _, _ = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            loss_type,
            ce_weight,
            ce_class_weight,
            target_mode,
            coarse_breakdown,
        )
        val_loss = val_metrics["loss"]
        monitored_value = float(val_metrics.get("macro_f1" if target_mode == "coarse" else "qwk", val_metrics["loss"]))
        scheduler.step(monitored_value if monitor_mode == "max" else val_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        log_metrics = {"train_loss": float(train_loss), "valid_loss": float(val_loss), "lr": float(current_lr), **{f"valid_{k}": float(v) for k, v in val_metrics.items()}}
        logger.log_metrics(epoch, log_metrics)
        logger.maybe_save_checkpoint(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, metrics=log_metrics, extra={"schema": schema_metadata})
        _print_epoch_metrics(epoch, epochs, train_loss, val_metrics, target_mode)
        cur_obj = monitored_value if monitor_mode == "max" else -val_loss
        if cur_obj > best_val:
            best_val = cur_obj
        if early_stopper(-cur_obj):
            print("--- Early Stopping activated ---")
            break

    best_ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    val_metrics, val_logits_np, val_raw_np, val_preds_np = _evaluate(
        model,
        val_loader,
        criterion,
        device,
        loss_type,
        ce_weight,
        ce_class_weight,
        target_mode,
        coarse_breakdown,
    )
    test_metrics, test_logits_np, test_raw_np, test_preds_np = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        loss_type,
        ce_weight,
        ce_class_weight,
        target_mode,
        coarse_breakdown,
    )

    if target_mode == "coarse":
        num_classes = coarse_num_classes
        train_targets_np = map_los_array_to_coarse_bins(
            train_y_raw.cpu().numpy(),
            breakdown=coarse_breakdown,
        ).astype(np.int64)
        val_targets_np = map_los_array_to_coarse_bins(
            val_raw_np,
            breakdown=coarse_breakdown,
        ).astype(np.int64)
        test_targets_np = map_los_array_to_coarse_bins(
            test_raw_np,
            breakdown=coarse_breakdown,
        ).astype(np.int64)
        majority_preds_np = _majority_baseline_predictions(train_targets_np, size=len(test_targets_np))
        stratified_preds_np = _stratified_baseline_predictions(
            train_targets_np,
            size=len(test_targets_np),
            seed=seed,
            num_classes=num_classes,
        )
        majority_metrics = _coarse_metrics(test_targets_np, majority_preds_np, num_classes)
        stratified_metrics = _coarse_metrics(test_targets_np, stratified_preds_np, num_classes)
        majority_metrics["loss"] = float("nan")
        stratified_metrics["loss"] = float("nan")
        train_count_metrics = _build_coarse_label_counts(train_targets_np, num_classes)
        val_count_metrics = _build_coarse_label_counts(val_targets_np, num_classes)
        test_count_metrics = _build_coarse_label_counts(test_targets_np, num_classes)

        if _is_ce_like_loss(loss_type):
            raw_val_metrics, val_prob_raw_np = _build_coarse_metrics_from_logits(
                val_logits_np,
                val_targets_np,
                num_classes=num_classes,
                temperature=1.0,
                ece_bins=ece_bins,
            )
            raw_val_metrics["loss"] = float(val_metrics["loss"])
            raw_test_metrics, test_prob_raw_np = _build_coarse_metrics_from_logits(
                test_logits_np,
                test_targets_np,
                num_classes=num_classes,
                temperature=1.0,
                ece_bins=ece_bins,
            )
            raw_test_metrics["loss"] = float(test_metrics["loss"])

            fitted_temperature = 1.0
            if calibration_enabled:
                fitted_temperature = _fit_temperature_scaling(val_logits_np, val_targets_np)

            calibrated_val_metrics, val_prob_cal_np = _build_coarse_metrics_from_logits(
                val_logits_np,
                val_targets_np,
                num_classes=num_classes,
                temperature=fitted_temperature,
                ece_bins=ece_bins,
            )
            calibrated_val_metrics["loss"] = float(val_metrics["loss"])
            calibrated_test_metrics, test_prob_cal_np = _build_coarse_metrics_from_logits(
                test_logits_np,
                test_targets_np,
                num_classes=num_classes,
                temperature=fitted_temperature,
                ece_bins=ece_bins,
            )
            calibrated_test_metrics["loss"] = float(test_metrics["loss"])

            print("")
            _print_coarse_test_summary("[Test coarse raw]", raw_test_metrics)
            _print_coarse_per_class_metrics(
                "[Test coarse raw per-class]",
                raw_test_metrics,
                coarse_class_labels,
            )
            print("")
            _print_coarse_test_summary("[Test coarse calibrated]", calibrated_test_metrics)
            _print_coarse_per_class_metrics(
                "[Test coarse calibrated per-class]",
                calibrated_test_metrics,
                coarse_class_labels,
            )
        else:
            raw_val_metrics = dict(val_metrics)
            raw_test_metrics = dict(test_metrics)
            calibrated_val_metrics = dict(val_metrics)
            calibrated_test_metrics = dict(test_metrics)
            fitted_temperature = 1.0
            val_prob_raw_np = np.empty((len(val_targets_np), 0), dtype=np.float32)
            test_prob_raw_np = np.empty((len(test_targets_np), 0), dtype=np.float32)
            val_prob_cal_np = val_prob_raw_np
            test_prob_cal_np = test_prob_raw_np

            print("")
            _print_coarse_test_summary("[Test coarse model]", test_metrics)
            _print_coarse_per_class_metrics(
                "[Test coarse model per-class]",
                test_metrics,
                coarse_class_labels,
            )

        print("")
        _print_coarse_test_summary("[Test coarse majority baseline]", majority_metrics)
        _print_coarse_per_class_metrics(
            "[Test coarse majority baseline per-class]",
            majority_metrics,
            coarse_class_labels,
        )
        print("")
        _print_coarse_test_summary("[Test coarse stratified baseline]", stratified_metrics)
        _print_coarse_per_class_metrics(
            "[Test coarse stratified baseline per-class]",
            stratified_metrics,
            coarse_class_labels,
        )

        test_log_metrics = {
            "split": "test",
            **{f"test_raw_{k}": float(v) for k, v in raw_test_metrics.items()},
            **{f"test_calibrated_{k}": float(v) for k, v in calibrated_test_metrics.items()},
            **{f"valid_raw_{k}": float(v) for k, v in raw_val_metrics.items() if k != "loss"},
            **{f"valid_calibrated_{k}": float(v) for k, v in calibrated_val_metrics.items() if k != "loss"},
            "calibrated_temperature": float(fitted_temperature),
        }
        test_log_metrics.update({f"test_majority_{k}": float(v) for k, v in majority_metrics.items() if k != "loss"})
        test_log_metrics.update({f"test_stratified_{k}": float(v) for k, v in stratified_metrics.items() if k != "loss"})
        test_log_metrics.update({f"train_count_{k}": float(v) for k, v in train_count_metrics.items()})
        test_log_metrics.update({f"valid_count_{k}": float(v) for k, v in val_count_metrics.items()})
        test_log_metrics.update({f"test_count_{k}": float(v) for k, v in test_count_metrics.items()})
        logger.log_metrics(epochs, test_log_metrics)

        if _is_ce_like_loss(loss_type):
            _save_confusion_matrix(run_dir, "val", val_targets_np, val_prob_raw_np.argmax(axis=1), num_classes, "coarse_raw")
            _save_confusion_matrix(run_dir, "test", test_targets_np, test_prob_raw_np.argmax(axis=1), num_classes, "coarse_raw")
            _save_confusion_matrix(
                run_dir,
                "val",
                val_targets_np,
                val_prob_cal_np.argmax(axis=1),
                num_classes,
                "coarse_calibrated",
            )
            _save_confusion_matrix(
                run_dir,
                "test",
                test_targets_np,
                test_prob_cal_np.argmax(axis=1),
                num_classes,
                "coarse_calibrated",
            )
            calibration_payload: Dict[str, object] = {
                "target_mode": "coarse",
                "num_classes": num_classes,
                "breakdown": bool(coarse_breakdown),
                "los_coarse_breakdown": bool(coarse_breakdown),
                "los_bins": {
                    str(idx): label for idx, label in enumerate(coarse_class_labels)
                },
                "loss": {
                    "type": loss_type,
                    "gamma": float(focal_gamma),
                    "alpha_mode": alpha_mode,
                    "alpha_clip": None if alpha_clip is None else float(alpha_clip),
                    "label_smoothing": float(label_smoothing),
                },
                "temperature": {
                    "fitted": float(fitted_temperature),
                    "source": "validation_nll",
                },
                "raw": {
                    "valid": {k: float(v) for k, v in raw_val_metrics.items()},
                    "test": {k: float(v) for k, v in raw_test_metrics.items()},
                },
                "calibrated": {
                    "valid": {k: float(v) for k, v in calibrated_val_metrics.items()},
                    "test": {k: float(v) for k, v in calibrated_test_metrics.items()},
                },
                "class_counts": {
                    "train": _build_class_count_list(train_targets_np, num_classes),
                    "valid": _build_class_count_list(val_targets_np, num_classes),
                    "test": _build_class_count_list(test_targets_np, num_classes),
                },
            }
            _save_calibration_json(run_dir, calibration_payload)
            if export_val_predictions:
                _save_predictions_csv(
                    run_dir,
                    "val",
                    val_logits_np,
                    val_raw_np,
                    val_prob_raw_np.argmax(axis=1),
                    target_mode,
                    ordinal_logits=False,
                    coarse_probability_payload={
                        "pred_raw": val_prob_raw_np.argmax(axis=1),
                        "pred_calibrated": val_prob_cal_np.argmax(axis=1),
                        "prob_raw": val_prob_raw_np,
                        "prob_calibrated": val_prob_cal_np,
                    },
                    coarse_breakdown=coarse_breakdown,
                )
            if export_test_predictions:
                _save_predictions_csv(
                    run_dir,
                    "test",
                    test_logits_np,
                    test_raw_np,
                    test_prob_raw_np.argmax(axis=1),
                    target_mode,
                    ordinal_logits=False,
                    coarse_probability_payload={
                        "pred_raw": test_prob_raw_np.argmax(axis=1),
                        "pred_calibrated": test_prob_cal_np.argmax(axis=1),
                        "prob_raw": test_prob_raw_np,
                        "prob_calibrated": test_prob_cal_np,
                    },
                    coarse_breakdown=coarse_breakdown,
                )
        else:
            _save_confusion_matrix(run_dir, "val", val_targets_np, val_preds_np, num_classes, "coarse")
            _save_confusion_matrix(run_dir, "test", test_targets_np, test_preds_np, num_classes, "coarse")
            if export_test_predictions:
                _save_predictions_csv(
                    run_dir,
                    "test",
                    test_logits_np,
                    test_raw_np,
                    test_preds_np,
                    target_mode,
                    ordinal_logits=not _is_ce_like_loss(loss_type),
                    coarse_breakdown=coarse_breakdown,
                )
        _save_coarse_baselines_artifact(
            run_dir,
            model_metrics=calibrated_test_metrics if _is_ce_like_loss(loss_type) else test_metrics,
            majority_metrics=majority_metrics,
            stratified_metrics=stratified_metrics,
            train_counts=train_count_metrics,
            test_counts=test_count_metrics,
            class_labels=coarse_class_labels,
        )
        with open(os.path.join(run_dir, "los_binning.json"), "w", encoding="utf-8") as f:
            json.dump(los_binning_metadata_dict(breakdown=coarse_breakdown), f, indent=2)
        return {
            "best_valid_metric": float(best_val),
            "test_acc": float(raw_test_metrics["acc"]),
            "test_macro_f1": float(raw_test_metrics["macro_f1"]),
            "test_calibrated_nll": float(calibrated_test_metrics.get("nll", float("nan"))),
            "test_calibrated_ece": float(calibrated_test_metrics.get("ece", float("nan"))),
            "test_majority_acc": float(majority_metrics["acc"]),
            "test_majority_macro_f1": float(majority_metrics["macro_f1"]),
            "test_stratified_acc": float(stratified_metrics["acc"]),
            "test_stratified_macro_f1": float(stratified_metrics["macro_f1"]),
            "run_dir": run_dir,
        }

    baseline_val_metrics = dict(val_metrics)
    baseline_test_metrics = dict(test_metrics)
    val_targets_np = val_raw_np.astype(int) - 1
    test_targets_np = test_raw_np.astype(int) - 1
    if loss_type == "ce":
        baseline_thresholds = np.array([], dtype=np.float32)
        num_classes = int(val_logits_np.shape[1])
    else:
        baseline_thresholds = np.full(val_logits_np.shape[1], 0.5, dtype=np.float32)
        num_classes = int(val_logits_np.shape[1] + 1)
    calibration_variants: Dict[str, Dict[str, object]] = {
        "baseline": {
            "thresholds": baseline_thresholds,
            "val_preds": val_preds_np,
            "test_preds": test_preds_np,
            "val_metrics": baseline_val_metrics,
            "test_metrics": baseline_test_metrics,
            "objective": "baseline",
            "constraints": {},
        }
    }

    if calibration_enabled and loss_type != "ce":
        variant_specs = [
            ("qwk_first", "qwk", None),
            (
                "qwk_constrained",
                "qwk",
                {
                    "within_1_acc": float(
                        baseline_val_metrics["within_1_acc"]
                        + calibration_cfg.get("constrained_qwk", {}).get("min_within_1_delta", 0.0)
                    ),
                    "within_2_acc": float(
                        baseline_val_metrics["within_2_acc"]
                        + calibration_cfg.get("constrained_qwk", {}).get("min_within_2_delta", 0.0)
                    ),
                },
            ),
            ("mae_first", "mae", None),
        ]

        for variant_name, objective_name, min_metrics in variant_specs:
            calibration_result = fit_ordinal_thresholds(
                logits_np=val_logits_np,
                targets_np=val_targets_np,
                objective=objective_name,
                min_metrics=min_metrics,
            )
            variant_val_preds = calibration_result["preds"]
            variant_test_preds = _decode_logits_np(
                test_logits_np, thresholds=calibration_result["thresholds"]
            )
            variant_val_metrics = compute_ordinal_metrics(val_targets_np, variant_val_preds)
            variant_val_metrics["loss"] = baseline_val_metrics["loss"]
            variant_test_metrics = compute_ordinal_metrics(test_targets_np, variant_test_preds)
            variant_test_metrics["loss"] = baseline_test_metrics["loss"]
            calibration_variants[variant_name] = {
                "thresholds": calibration_result["thresholds"],
                "val_preds": variant_val_preds,
                "test_preds": variant_test_preds,
                "val_metrics": variant_val_metrics,
                "test_metrics": variant_test_metrics,
                "objective": objective_name,
                "constraints": min_metrics or {},
                "best_score": float(calibration_result["best_score"]),
            }

    calibration_payload: Dict[str, object] = {
        "enabled": calibration_enabled,
        "variants": {
            variant_name: {
                "objective": variant_data["objective"],
                "constraints": variant_data["constraints"],
                "thresholds": np.asarray(variant_data["thresholds"]).tolist(),
                "validation_metrics": {
                    k: float(v) for k, v in dict(variant_data["val_metrics"]).items()
                },
                "test_metrics": {
                    k: float(v) for k, v in dict(variant_data["test_metrics"]).items()
                },
                **(
                    {"best_score": float(variant_data["best_score"])}
                    if "best_score" in variant_data
                    else {}
                ),
            }
            for variant_name, variant_data in calibration_variants.items()
        },
    }
    _save_calibration_json(run_dir, calibration_payload)

    print("")
    for variant_name, variant_data in calibration_variants.items():
        _print_metrics(f"[Valid {variant_name}]", dict(variant_data["val_metrics"]))
    print("")
    for variant_name, variant_data in calibration_variants.items():
        _print_metrics(f"[Test {variant_name}]", dict(variant_data["test_metrics"]))

    test_log_metrics = {"split": "test"}
    for variant_name, variant_data in calibration_variants.items():
        prefix = "test" if variant_name == "baseline" else f"test_{variant_name}"
        test_log_metrics.update(
            {f"{prefix}_{k}": float(v) for k, v in dict(variant_data["test_metrics"]).items()}
        )
        valid_prefix = "valid" if variant_name == "baseline" else f"valid_{variant_name}"
        test_log_metrics.update(
            {
                f"{valid_prefix}_{k}": float(v)
                for k, v in dict(variant_data["val_metrics"]).items()
                if k != "loss"
            }
        )
    logger.log_metrics(epochs, test_log_metrics)

    val_predictions_by_mode = {
        variant_name: np.asarray(variant_data["val_preds"]).astype(int)
        for variant_name, variant_data in calibration_variants.items()
    }
    test_predictions_by_mode = {
        variant_name: np.asarray(variant_data["test_preds"]).astype(int)
        for variant_name, variant_data in calibration_variants.items()
    }
    _save_distribution_csv(run_dir, "val", val_targets_np, val_predictions_by_mode, num_classes)
    _save_distribution_csv(run_dir, "test", test_targets_np, test_predictions_by_mode, num_classes)
    _save_confusion_matrices(run_dir, "val", val_targets_np, val_predictions_by_mode, num_classes)
    _save_confusion_matrices(run_dir, "test", test_targets_np, test_predictions_by_mode, num_classes)

    if export_val_predictions:
        _save_predictions_csv(
            run_dir,
            "val",
            val_logits_np,
            val_targets_np,
            val_predictions_by_mode,
            target_mode,
            ordinal_logits=loss_type != "ce",
        )
    if export_test_predictions:
        _save_predictions_csv(
            run_dir,
            "test",
            test_logits_np,
            test_targets_np,
            test_predictions_by_mode,
            target_mode,
            ordinal_logits=loss_type != "ce",
        )
    return {
        "best_valid_metric": float(best_val),
        "test_qwk": float(baseline_test_metrics["qwk"]),
        "test_mae": float(baseline_test_metrics["mae"]),
        "test_within_1_acc": float(baseline_test_metrics["within_1_acc"]),
        "test_qwk_first_qwk": float(
            dict(calibration_variants["qwk_first"]["test_metrics"])["qwk"]
        ) if "qwk_first" in calibration_variants else float(baseline_test_metrics["qwk"]),
        "test_qwk_constrained_qwk": float(
            dict(calibration_variants["qwk_constrained"]["test_metrics"])["qwk"]
        ) if "qwk_constrained" in calibration_variants else float(baseline_test_metrics["qwk"]),
        "test_mae_first_mae": float(
            dict(calibration_variants["mae_first"]["test_metrics"])["mae"]
        ) if "mae_first" in calibration_variants else float(baseline_test_metrics["mae"]),
        "run_dir": run_dir,
    }


def main() -> None:
    """CLI entrypoint for LOS prediction training."""
    args = parse_args()
    cfg = override_cfg(load_yaml(args.config), args)
    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, "..", "data")
    run_los_prediction(cfg, os.path.abspath(root))


if __name__ == "__main__":
    main()
