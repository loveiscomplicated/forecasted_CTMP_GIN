from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.data_processing.discharge_prediction_dataset import DischargePredictionDataset
from src.data_processing.los_prediction_dataset import LOSPredictionDataset
from src.models.discharge_predictor import (
    LEGACY_TOP3_HEADS,
    LOSCoarsePredictor,
    LOSOrdinalPredictor,
    MultiTaskCategoricalLoss,
    MultiTaskDischargePredictor,
    OrdinalBCELoss,
    compute_discharge_metrics,
    compute_ordinal_metrics,
    expand_coarse_distribution_to_raw_los,
    get_los_coarse_class_labels,
    get_los_coarse_num_classes,
    infer_los_coarse_breakdown_from_cfg,
    infer_los_target_from_cfg,
    los_binning_metadata_dict,
    map_coarse_array_to_raw_los,
    map_los_array_to_coarse_bins,
)
from src.models.discharge_predictor.ordinal_loss import (
    compute_ce_class_weight,
    compute_ordinal_pos_weight,
    fit_ordinal_thresholds,
    ordinal_logits_to_class,
)
from src.models.forecast_inputs import resolve_model_forecast_input_metadata
from src.models.forecasted_ctmp_gin import resolve_joint_forecast_contract
from src.trainers.forecasted_discharge import ForecastedDischargeProvider
from src.trainers.forecasted_discharge import normalize_forecasted_discharge_cfg
from src.trainers.forecasted_los import ForecastedLOSProvider, normalize_forecasted_los_cfg
from src.trainers.run_joint_consistent_predictor import run_joint_consistent_predictor
from src.trainers.run_los_prediction import (
    _coarse_metrics,
    _compute_los_loss,
    _fit_temperature_scaling,
    _is_ce_like_loss,
    _num_prediction_classes,
    _select_logits,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.experiment import ExperimentLogger, append_jsonl, save_yaml
from src.utils.seed_set import set_seed


@dataclass
class ForecastedFoldData:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    split_payload: dict[str, Any]


class ForecastCacheDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        x_cache: torch.Tensor,
        los_cache: torch.Tensor,
        soft_discharge_cache: dict[str, Any] | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.x_cache = x_cache
        self.los_cache = los_cache
        self.soft_discharge_cache = soft_discharge_cache
        self.processed_df = base_dataset.processed_df
        self.col_info = base_dataset.col_info
        self.num_classes = base_dataset.num_classes

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        _, y, _ = self.base_dataset[index]
        if self.soft_discharge_cache is None:
            return self.x_cache[index], y, self.los_cache[index]

        soft_discharge: dict[str, dict[str, torch.Tensor]] = {}
        soft_discharge_mask: dict[str, torch.Tensor] = {}
        for head_name, head_payload in self.soft_discharge_cache["heads"].items():
            soft_discharge[head_name] = {
                "probs": head_payload["probs"][index],
                "target_col_idx": head_payload["target_col_idx"],
                "class_to_embedding_idx": head_payload["class_to_embedding_idx"],
                "num_classes": head_payload["num_classes"],
            }
            soft_discharge_mask[head_name] = head_payload["mask"][index]

        forecast_meta = {
            "soft_discharge": soft_discharge,
            "soft_discharge_mask": soft_discharge_mask,
            "metadata": copy.deepcopy(self.soft_discharge_cache.get("metadata", {})),
        }
        return self.x_cache[index], y, self.los_cache[index], forecast_meta


def joint_forecast_pipeline_enabled(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("joint_forecast_pipeline", {}).get("enabled", False))


def _deep_merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge_dict(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _inherit_dataset_settings(parent_cfg: dict[str, Any], child_cfg: dict[str, Any]) -> None:
    parent_train = parent_cfg.get("train", {})
    child_train = child_cfg.setdefault("train", {})

    # Forecasted predictors must see the same canonical dataset variant as the
    # outer GNN dataset, otherwise categorical cardinalities can diverge.
    child_train["do_preprocess"] = bool(parent_train.get("do_preprocess", True))


def _dataset_do_preprocess(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("train", {}).get("do_preprocess", True))


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _jsonable_metrics(metrics: dict[str, Any]) -> dict[str, float | str | int | None]:
    out: dict[str, float | str | int | None] = {}
    for key, value in metrics.items():
        if isinstance(value, (str, int)) or value is None:
            out[key] = value
        elif isinstance(value, (float, np.floating, np.integer)):
            out[key] = float(value)
    return out


def _metric(metrics: dict[str, Any], key: str, default: float = float("nan")) -> float:
    try:
        return float(metrics[key])
    except Exception:
        return default


def _print_discharge_epoch(role: str, epoch: int, epochs: int, metrics: dict[str, Any], lr: float, is_best: bool) -> None:
    msg = (
        f"[{role}] epoch {epoch}/{epochs} "
        f"lr={lr:.6g} train_loss={_metric(metrics, 'train_loss'):.4f}"
    )
    if "valid_loss" in metrics:
        msg += (
            f" | valid_loss={_metric(metrics, 'valid_loss'):.4f}"
            f" acc={_metric(metrics, 'valid_mean_accuracy'):.4f}"
            f" macro_f1={_metric(metrics, 'valid_mean_macro_f1'):.4f}"
        )
    if is_best:
        msg += " | best"
    tqdm.write(msg)


def _print_los_epoch(
    role: str,
    epoch: int,
    epochs: int,
    metrics: dict[str, Any],
    lr: float,
    target_mode: str,
    is_best: bool,
) -> None:
    msg = (
        f"[{role}] epoch {epoch}/{epochs} "
        f"lr={lr:.6g} train_loss={_metric(metrics, 'train_loss'):.4f}"
    )
    if "valid_loss" in metrics:
        if target_mode == "coarse":
            msg += (
                f" | valid_loss={_metric(metrics, 'valid_loss'):.4f}"
                f" acc={_metric(metrics, 'valid_acc'):.4f}"
                f" macro_f1={_metric(metrics, 'valid_macro_f1'):.4f}"
                f" weighted_f1={_metric(metrics, 'valid_weighted_f1'):.4f}"
                f" mae={_metric(metrics, 'valid_mae'):.3f}"
                f" within_1={_metric(metrics, 'valid_within_1_acc'):.4f}"
                f" qwk={_metric(metrics, 'valid_qwk'):.4f}"
            )
        else:
            msg += (
                f" | valid_loss={_metric(metrics, 'valid_loss'):.4f}"
                f" acc={_metric(metrics, 'valid_acc'):.4f}"
                f" macro_f1={_metric(metrics, 'valid_macro_f1'):.4f}"
                f" mae={_metric(metrics, 'valid_mae'):.3f}"
                f" within_1={_metric(metrics, 'valid_within_1_acc'):.4f}"
                f" qwk={_metric(metrics, 'valid_qwk'):.4f}"
            )
    if is_best:
        msg += " | best"
    tqdm.write(msg)


def _print_final_metrics(role: str, split_name: str, metrics: dict[str, Any]) -> None:
    if "mean_macro_f1" in metrics:
        print(
            f"[{role}] {split_name} final: "
            f"loss={_metric(metrics, 'loss'):.4f} "
            f"mean_acc={_metric(metrics, 'mean_accuracy'):.4f} "
            f"mean_macro_f1={_metric(metrics, 'mean_macro_f1'):.4f}"
        )
        return
    if "qwk" in metrics:
        print(
            f"[{role}] {split_name} final: "
            f"loss={_metric(metrics, 'loss'):.4f} "
            f"acc={_metric(metrics, 'acc'):.4f} "
            f"macro_f1={_metric(metrics, 'macro_f1'):.4f} "
            f"mae={_metric(metrics, 'mae'):.3f} "
            f"within_1={_metric(metrics, 'within_1_acc'):.4f} "
            f"qwk={_metric(metrics, 'qwk'):.4f}"
        )
        return
    print(f"[{role}] {split_name} final: {_jsonable_metrics(metrics)}")


def _make_loader(dataset, indices, batch_size: int, num_workers: int, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        Subset(dataset, np.asarray(indices, dtype=np.int64).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def _labels_for_indices(dataset, indices: np.ndarray) -> np.ndarray:
    return np.asarray([int(dataset[int(i)][1]) for i in indices], dtype=np.int64)


def _coarse_los_for_indices(dataset, indices: np.ndarray) -> np.ndarray:
    raw = np.asarray([int(dataset[int(i)][2]) for i in indices], dtype=np.int64)
    return map_los_array_to_coarse_bins(raw).astype(np.int64)


def _composite_stratification_labels(dataset, indices: np.ndarray, split_cfg: dict[str, Any]) -> np.ndarray:
    labels = _labels_for_indices(dataset, indices).astype(str)
    aux = split_cfg.get("stratify_aux", []) or []
    if "LOS_coarse" in aux:
        los_labels = _coarse_los_for_indices(dataset, indices).astype(str)
        labels = np.asarray([f"{a}_{b}" for a, b in zip(labels, los_labels)])
        _, counts = np.unique(labels, return_counts=True)
        if counts.min(initial=0) < 2:
            labels = _labels_for_indices(dataset, indices).astype(str)
    return labels


def split_outer_train_for_forecasted_pipeline(
    dataset,
    outer_train_idx: np.ndarray,
    cfg: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    split_cfg = cfg.get("forecasted_pipeline", {}).get("split", {})
    train_core_ratio = float(split_cfg.get("train_core_ratio", 0.8))
    predictor_val_ratio = float(split_cfg.get("predictor_val_ratio", 0.1))
    gnn_val_ratio = float(split_cfg.get("gnn_val_ratio", 0.1))
    total = train_core_ratio + predictor_val_ratio + gnn_val_ratio
    if abs(total - 1.0) > 1.0e-6:
        raise ValueError(f"forecasted_pipeline split ratios must sum to 1.0, got {total}")

    outer_train_idx = np.asarray(outer_train_idx, dtype=np.int64)
    labels = _composite_stratification_labels(dataset, outer_train_idx, split_cfg)
    temp_ratio = predictor_val_ratio + gnn_val_ratio
    sss = StratifiedShuffleSplit(n_splits=1, test_size=temp_ratio, random_state=seed)
    train_pos, temp_pos = next(sss.split(np.zeros(len(outer_train_idx)), labels))
    train_core_idx = outer_train_idx[train_pos]
    temp_idx = outer_train_idx[temp_pos]

    temp_labels = _composite_stratification_labels(dataset, temp_idx, split_cfg)
    gnn_fraction_of_temp = gnn_val_ratio / temp_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=gnn_fraction_of_temp, random_state=seed + 1)
    predictor_pos, gnn_pos = next(sss2.split(np.zeros(len(temp_idx)), temp_labels))
    predictor_val_idx = temp_idx[predictor_pos]
    gnn_val_idx = temp_idx[gnn_pos]
    return (
        train_core_idx.astype(np.int64),
        predictor_val_idx.astype(np.int64),
        gnn_val_idx.astype(np.int64),
    )


def _assert_disjoint(split_map: dict[str, np.ndarray]) -> None:
    seen: dict[int, str] = {}
    for name, values in split_map.items():
        for idx in np.asarray(values, dtype=np.int64).tolist():
            if idx in seen:
                raise ValueError(f"Split leakage: index {idx} appears in {seen[idx]} and {name}")
            seen[idx] = name


def _train_discharge_predictor(
    cfg: dict[str, Any],
    root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    run_dir: str,
    device: torch.device,
    *,
    fixed_epochs: int | None = None,
    role: str = "discharge predictor",
    verbose: bool = True,
) -> dict[str, Any]:
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {})["monitor_metric"] = "valid_mean_macro_f1"
    cfg["train"]["monitor_mode"] = "max"
    save_yaml(os.path.join(run_dir, "config.final.yaml"), cfg)
    logger = ExperimentLogger(cfg, run_dir)
    dataset = DischargePredictionDataset(
        root=root,
        do_preprocess=_dataset_do_preprocess(cfg),
        include_los_in_targets=cfg.get("targets", {}).get("include_los", False),
    )
    schema_metadata = dict(dataset.schema_metadata)
    schema_metadata["target_col_names"] = list(dataset.target_col_names)
    schema_metadata["target_col_dims"] = [int(v) for v in dataset.target_col_dims]

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    train_loader = _make_loader(dataset, train_idx, batch_size, num_workers, shuffle=True)
    val_loader = (
        _make_loader(dataset, val_idx, batch_size, num_workers, shuffle=False)
        if val_idx is not None and len(val_idx) >= batch_size
        else None
    )

    model = MultiTaskDischargePredictor(
        ad_col_dims=dataset.ad_col_dims,
        target_col_names=dataset.target_col_names,
        target_col_dims=dataset.target_col_dims,
        **cfg["model"].get("params", {}),
    ).to(device)
    criterion = MultiTaskCategoricalLoss()
    if cfg["train"].get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=cfg["train"].get("lr_scheduler_patience", 5)
    )
    early_stopper = EarlyStopper(patience=cfg["train"].get("early_stopping_patience", 5))
    target_names = dataset.target_col_names
    epochs = int(fixed_epochs or cfg["train"]["epochs"])
    best_value = -float("inf")
    best_epoch = epochs

    if verbose:
        print(
            f"\n[{role}] start: train={len(train_idx)} "
            f"valid={0 if val_idx is None else len(val_idx)} epochs={epochs} run_dir={run_dir}"
        )

    for epoch in tqdm(
        range(1, epochs + 1),
        desc=role,
        leave=verbose,
        disable=not verbose,
    ):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            y_dict = {name: y[:, i] for i, name in enumerate(target_names)}
            loss, _ = criterion(logits, y_dict)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)

        metrics = {"train_loss": float(train_loss)}
        cur_obj = -train_loss
        if val_loader is not None:
            val_metrics, _, _ = _evaluate_discharge(model, val_loader, criterion, device, target_names)
            cur_obj = float(val_metrics["mean_macro_f1"])
            metrics.update(
                {
                    "valid_loss": float(val_metrics["loss"]),
                    "valid_mean_accuracy": float(val_metrics["mean_accuracy"]),
                    "valid_mean_macro_f1": float(val_metrics["mean_macro_f1"]),
                }
            )
            scheduler.step(cur_obj)
        logger.log_metrics(epoch, metrics)
        logger.maybe_save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            extra={"schema": schema_metadata},
        )
        is_best = cur_obj > best_value
        if is_best:
            best_value = cur_obj
            best_epoch = epoch
        if verbose:
            _print_discharge_epoch(
                role,
                epoch,
                epochs,
                metrics,
                float(optimizer.param_groups[0]["lr"]),
                is_best,
            )
        if val_loader is not None and early_stopper(-cur_obj):
            if verbose:
                print(f"[{role}] early stopping at epoch={epoch}")
            break

    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_path):
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {"train_loss": float(train_loss)},
                "cfg": cfg,
                "schema": schema_metadata,
            },
            best_path,
        )
    else:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    if val_loader is not None:
        final_metrics, _, _ = _evaluate_discharge(model, val_loader, criterion, device, target_names)
        final_metrics = {"split": "predictor_val_final", **_jsonable_metrics(final_metrics)}
        logger.log_metrics(best_epoch, final_metrics)
        _save_json(os.path.join(run_dir, "predictor_val_final_metrics.json"), final_metrics)
        if verbose:
            _print_final_metrics(role, "predictor_val", final_metrics)
    if verbose:
        print(f"[{role}] finished: best_epoch={best_epoch} best_metric={best_value:.4f}")
    return {"checkpoint_path": best_path, "best_epoch": int(best_epoch), "run_dir": run_dir}


def _evaluate_discharge(model, loader, criterion, device, target_col_names):
    model.eval()
    all_logits = {name: [] for name in target_col_names}
    all_targets = {name: [] for name in target_col_names}
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            y_dict = {name: y[:, i] for i, name in enumerate(target_col_names)}
            loss, _ = criterion(logits, y_dict)
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            for name in target_col_names:
                all_logits[name].append(logits[name].cpu().numpy())
                all_targets[name].append(y_dict[name].cpu().numpy())
    logits_np = {name: np.concatenate(v, axis=0) for name, v in all_logits.items()}
    targets_np = {name: np.concatenate(v, axis=0) for name, v in all_targets.items()}
    metrics = compute_discharge_metrics(logits_np, targets_np)
    metrics["loss"] = total_loss / max(n_batches, 1)
    return metrics, logits_np, targets_np


def _resolve_los_coarse_settings(
    cfg: dict[str, Any],
) -> tuple[str, bool, int, tuple[str, ...]]:
    target_mode = infer_los_target_from_cfg(cfg)
    coarse_breakdown = (
        infer_los_coarse_breakdown_from_cfg(cfg) if target_mode == "coarse" else False
    )
    coarse_num_classes = get_los_coarse_num_classes(breakdown=coarse_breakdown)
    coarse_class_labels = get_los_coarse_class_labels(breakdown=coarse_breakdown)
    if target_mode == "coarse":
        metadata = los_binning_metadata_dict(breakdown=coarse_breakdown)
        cfg["los_coarse_breakdown"] = coarse_breakdown
        cfg["num_classes"] = coarse_num_classes
        cfg["los_bins"] = metadata["los_bins"]
    return target_mode, coarse_breakdown, coarse_num_classes, coarse_class_labels


def _build_los_model_and_loss(cfg: dict[str, Any], dataset: LOSPredictionDataset, train_idx, device):
    target_mode, coarse_breakdown, coarse_num_classes, _ = _resolve_los_coarse_settings(cfg)
    loss_type = str(cfg.get("loss", {}).get("type", "ordinal_bce"))
    output_mode = {
        "ordinal_bce": "ordinal",
        "ce": "ce",
        "focal": "ce",
        "focal_alpha": "ce",
        "cb_focal": "ce",
        "hybrid_ce_ordinal": "hybrid",
    }[loss_type]
    train_y_raw = dataset.los_raw[train_idx]
    train_y = (
        _build_coarse_targets(train_y_raw, breakdown=coarse_breakdown)
        if target_mode == "coarse"
        else dataset.y[train_idx]
    )

    pos_weight = None
    ce_class_weight = None
    use_pos_weight = cfg.get("loss", {}).get("use_pos_weight", True)
    if use_pos_weight and not _is_ce_like_loss(loss_type):
        pos_weight = compute_ordinal_pos_weight(
            train_y,
            coarse_num_classes if target_mode == "coarse" else dataset.los_num_classes,
            max_weight=float(cfg.get("loss", {}).get("pos_weight_clip", 10.0)),
        )
    if bool(cfg.get("loss", {}).get("use_ce_class_weight", False)) and (
        loss_type == "ce" or loss_type == "hybrid_ce_ordinal"
    ):
        ce_class_weight = compute_ce_class_weight(
            train_y,
            coarse_num_classes if target_mode == "coarse" else dataset.los_num_classes,
            mode=str(cfg.get("loss", {}).get("ce_class_weight_mode", "inverse")),
            beta=float(cfg.get("loss", {}).get("ce_class_weight_beta", 0.999)),
            max_weight=cfg.get("loss", {}).get("ce_class_weight_clip"),
        )

    alpha_tensor = None
    if loss_type in {"focal_alpha", "cb_focal"}:
        alpha_mode = str(cfg.get("loss", {}).get("alpha_mode", "none")).lower()
        if loss_type == "focal_alpha" and alpha_mode == "none":
            alpha_mode = "inverse_sqrt"
        if loss_type == "cb_focal":
            alpha_mode = "effective_num"
        alpha_tensor = compute_ce_class_weight(
            train_y,
            coarse_num_classes if target_mode == "coarse" else dataset.los_num_classes,
            mode=alpha_mode,
            beta=float(cfg.get("loss", {}).get("beta", cfg.get("loss", {}).get("ce_class_weight_beta", 0.999))),
            max_weight=cfg.get("loss", {}).get("alpha_clip"),
        )

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
                    label_smoothing=float(cfg.get("loss", {}).get("label_smoothing", 0.0)),
                )
            else:
                criterion = {
                    "gamma": float(cfg.get("loss", {}).get("gamma", 2.0)),
                    "alpha": alpha_tensor,
                    "label_smoothing": float(cfg.get("loss", {}).get("label_smoothing", 0.0)),
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
    else:
        model = LOSOrdinalPredictor(
            ad_col_dims=dataset.ad_col_dims,
            los_num_classes=dataset.los_num_classes,
            output_mode=output_mode,
            **cfg["model"].get("params", {}),
        ).to(device)
        criterion = OrdinalBCELoss(num_classes=dataset.los_num_classes, pos_weight=pos_weight).to(device)

    return model, criterion, ce_class_weight, target_mode, loss_type, coarse_breakdown


def _build_coarse_targets(raw_los: torch.Tensor, *, breakdown: bool = False) -> torch.Tensor:
    return map_los_array_to_coarse_bins(raw_los, breakdown=breakdown).long()


def _train_los_predictor(
    cfg: dict[str, Any],
    root: str,
    train_idx: np.ndarray,
    val_idx: np.ndarray | None,
    run_dir: str,
    device: torch.device,
    *,
    fixed_epochs: int | None = None,
    role: str = "LOS predictor",
    verbose: bool = True,
) -> dict[str, Any]:
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    cfg = copy.deepcopy(cfg)
    target_mode, coarse_breakdown, _coarse_num_classes, _coarse_class_labels = (
        _resolve_los_coarse_settings(cfg)
    )
    save_yaml(os.path.join(run_dir, "config.final.yaml"), cfg)
    logger = ExperimentLogger(cfg, run_dir)
    dataset = LOSPredictionDataset(root=root, do_preprocess=_dataset_do_preprocess(cfg))
    model, criterion, ce_class_weight, target_mode, loss_type, coarse_breakdown = _build_los_model_and_loss(
        cfg, dataset, train_idx, device
    )
    schema_metadata = dict(dataset.schema_metadata)
    schema_metadata["los_num_classes"] = int(dataset.los_num_classes)
    schema_metadata["los_target_mode"] = target_mode
    if target_mode == "coarse":
        schema_metadata.update(los_binning_metadata_dict(breakdown=coarse_breakdown))

    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    train_loader = _make_loader(dataset, train_idx, batch_size, num_workers, shuffle=True)
    val_loader = (
        _make_loader(dataset, val_idx, batch_size, num_workers, shuffle=False)
        if val_idx is not None and len(val_idx) >= batch_size
        else None
    )
    if cfg["train"].get("optimizer", "adamw") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg["train"]["learning_rate"],
            weight_decay=cfg["train"].get("weight_decay", 0.0),
        )
    monitor_mode = str(cfg["train"].get("monitor_mode", "max")).lower()
    scheduler = ReduceLROnPlateau(
        optimizer, monitor_mode, patience=cfg["train"].get("lr_scheduler_patience", 5)
    )
    early_stopper = EarlyStopper(patience=cfg["train"].get("early_stopping_patience", 6))
    epochs = int(fixed_epochs or cfg["train"]["epochs"])
    ce_weight = float(cfg.get("loss", {}).get("ce_weight", 1.0))
    best_value = -float("inf")
    best_epoch = epochs

    if verbose:
        print(
            f"\n[{role}] start: train={len(train_idx)} "
            f"valid={0 if val_idx is None else len(val_idx)} epochs={epochs} "
            f"target_mode={target_mode} loss={loss_type} "
            f"coarse_breakdown={coarse_breakdown} run_dir={run_dir}"
        )

    for epoch in tqdm(
        range(1, epochs + 1),
        desc=role,
        leave=verbose,
        disable=not verbose,
    ):
        train_loss = _train_one_epoch_los(
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
        metrics = {"train_loss": float(train_loss)}
        cur_obj = -train_loss
        if val_loader is not None:
            val_metrics, _, _, _ = _evaluate_los(
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
            val_loss = float(val_metrics["loss"])
            monitored_value = float(
                val_metrics.get("macro_f1" if target_mode == "coarse" else "qwk", val_loss)
            )
            cur_obj = monitored_value if monitor_mode == "max" else -val_loss
            scheduler.step(monitored_value if monitor_mode == "max" else val_loss)
            metrics.update({"valid_loss": val_loss, **{f"valid_{k}": float(v) for k, v in val_metrics.items()}})
        logger.log_metrics(epoch, metrics)
        logger.maybe_save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=metrics,
            extra={"schema": schema_metadata},
        )
        is_best = cur_obj > best_value
        if is_best:
            best_value = cur_obj
            best_epoch = epoch
        if verbose:
            _print_los_epoch(
                role,
                epoch,
                epochs,
                metrics,
                float(optimizer.param_groups[0]["lr"]),
                target_mode,
                is_best,
            )
        if val_loader is not None and early_stopper(-cur_obj):
            if verbose:
                print(f"[{role}] early stopping at epoch={epoch}")
            break

    best_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if not os.path.exists(best_path):
        torch.save(
            {
                "epoch": best_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "metrics": {"train_loss": float(train_loss)},
                "cfg": cfg,
                "schema": schema_metadata,
            },
            best_path,
        )
    else:
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    calibration_path = None
    if val_loader is not None:
        calibration_path = _save_los_calibration(
            run_dir,
            model,
            val_loader,
            criterion,
            device,
            loss_type,
            ce_weight,
            ce_class_weight,
            target_mode,
            cfg,
            coarse_breakdown,
        )
        final_metrics, _, _, _ = _evaluate_los(
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
        final_metrics = {"split": "predictor_val_final", **_jsonable_metrics(final_metrics)}
        logger.log_metrics(best_epoch, final_metrics)
        _save_json(os.path.join(run_dir, "predictor_val_final_metrics.json"), final_metrics)
        if verbose:
            _print_final_metrics(role, "predictor_val", final_metrics)
            if calibration_path is not None:
                _print_los_calibration_summary(role, calibration_path)
    if verbose:
        print(f"[{role}] finished: best_epoch={best_epoch} best_metric={best_value:.4f}")
    return {
        "checkpoint_path": best_path,
        "calibration_path": calibration_path,
        "best_epoch": int(best_epoch),
        "run_dir": run_dir,
    }


def _train_one_epoch_los(
    model,
    loader,
    criterion,
    optimizer,
    device,
    loss_type,
    ce_weight,
    ce_class_weight,
    target_mode,
    coarse_breakdown: bool = False,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for x, y, raw_y in loader:
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


def _evaluate_los(
    model,
    loader,
    criterion,
    device,
    loss_type,
    ce_weight,
    ce_class_weight,
    target_mode,
    coarse_breakdown: bool = False,
):
    model.eval()
    all_logits = []
    all_targets_raw = []
    all_targets = []
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for x, y, raw_y in loader:
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


def _save_los_calibration(
    run_dir: str,
    model,
    val_loader,
    criterion,
    device,
    loss_type,
    ce_weight,
    ce_class_weight,
    target_mode,
    cfg,
    coarse_breakdown: bool = False,
) -> str:
    val_metrics, val_logits_np, val_raw_np, val_preds_np = _evaluate_los(
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
    if target_mode == "coarse":
        num_classes = get_los_coarse_num_classes(breakdown=coarse_breakdown)
        class_labels = get_los_coarse_class_labels(breakdown=coarse_breakdown)
        val_targets_np = map_los_array_to_coarse_bins(
            val_raw_np,
            breakdown=coarse_breakdown,
        ).astype(np.int64)
        payload: dict[str, Any] = {
            "target_mode": "coarse",
            "num_classes": num_classes,
            "breakdown": bool(coarse_breakdown),
            "los_coarse_breakdown": bool(coarse_breakdown),
            "los_bins": {str(idx): label for idx, label in enumerate(class_labels)},
            "raw": {"valid": {k: float(v) for k, v in val_metrics.items()}},
            "calibrated": {"valid": {k: float(v) for k, v in val_metrics.items()}},
        }
        if _is_ce_like_loss(loss_type):
            fitted_temperature = (
                _fit_temperature_scaling(val_logits_np, val_targets_np)
                if bool(cfg.get("calibration", {}).get("enabled", True))
                else 1.0
            )
            payload["temperature"] = {"fitted": float(fitted_temperature), "source": "validation_nll"}
        path = os.path.join(run_dir, "calibration.json")
        _save_json(path, payload)
        return path

    val_targets_np = val_raw_np.astype(int) - 1
    if loss_type == "ce":
        variants = {
            "baseline": {
                "thresholds": [],
                "validation_metrics": {k: float(v) for k, v in val_metrics.items()},
                "objective": "baseline",
                "constraints": {},
            }
        }
    else:
        baseline_thresholds = np.full(val_logits_np.shape[1], 0.5, dtype=np.float32)
        variants = {
            "baseline": {
                "thresholds": baseline_thresholds.tolist(),
                "validation_metrics": {k: float(v) for k, v in val_metrics.items()},
                "objective": "baseline",
                "constraints": {},
            }
        }
        if bool(cfg.get("calibration", {}).get("enabled", True)):
            calibration_cfg = cfg.get("calibration", {})
            for variant_name, objective_name, min_metrics in [
                ("qwk_first", "qwk", None),
                (
                    "qwk_constrained",
                    "qwk",
                    {
                        "within_1_acc": float(
                            val_metrics["within_1_acc"]
                            + calibration_cfg.get("constrained_qwk", {}).get("min_within_1_delta", 0.0)
                        ),
                        "within_2_acc": float(
                            val_metrics["within_2_acc"]
                            + calibration_cfg.get("constrained_qwk", {}).get("min_within_2_delta", 0.0)
                        ),
                    },
                ),
                ("mae_first", "mae", None),
            ]:
                result = fit_ordinal_thresholds(
                    logits_np=val_logits_np,
                    targets_np=val_targets_np,
                    objective=objective_name,
                    min_metrics=min_metrics,
                )
                variant_preds = result["preds"]
                variant_metrics = compute_ordinal_metrics(val_targets_np, variant_preds)
                variant_metrics["loss"] = val_metrics["loss"]
                variants[variant_name] = {
                    "thresholds": np.asarray(result["thresholds"]).tolist(),
                    "validation_metrics": {k: float(v) for k, v in variant_metrics.items()},
                    "objective": objective_name,
                    "constraints": min_metrics or {},
                    "best_score": float(result["best_score"]),
                }
    path = os.path.join(run_dir, "calibration.json")
    _save_json(path, {"enabled": bool(cfg.get("calibration", {}).get("enabled", True)), "variants": variants})
    return path


def _print_los_calibration_summary(role: str, calibration_path: str) -> None:
    if not os.path.exists(calibration_path):
        return
    with open(calibration_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if "temperature" in payload:
        temp = payload.get("temperature", {}).get("fitted")
        if temp is not None:
            print(f"[{role}] calibration: temperature={float(temp):.4f}")
        return
    variants = payload.get("variants", {})
    if variants:
        names = ", ".join(sorted(variants))
        print(f"[{role}] calibration variants: {names}")


def _evaluate_discharge_checkpoint(
    cfg: dict[str, Any],
    root: str,
    checkpoint_path: str,
    indices: np.ndarray,
    run_dir: str,
    device: torch.device,
    *,
    role: str,
    split_name: str,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = ckpt.get("cfg", cfg)
    schema = ckpt.get("schema", {})
    dataset = DischargePredictionDataset(
        root=root,
        do_preprocess=_dataset_do_preprocess(ckpt_cfg),
        include_los_in_targets=ckpt_cfg.get("targets", {}).get("include_los", False),
    )
    target_names = list(schema.get("target_col_names", dataset.target_col_names))
    target_dims = [int(v) for v in schema.get("target_col_dims", dataset.target_col_dims)]
    ad_col_dims = [int(v) for v in schema.get("admission_col_dims", dataset.ad_col_dims)]
    model = MultiTaskDischargePredictor(
        ad_col_dims=ad_col_dims,
        target_col_names=target_names,
        target_col_dims=target_dims,
        **ckpt_cfg.get("model", {}).get("params", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = MultiTaskCategoricalLoss()
    loader = _make_loader(
        dataset,
        indices,
        int(ckpt_cfg.get("train", {}).get("batch_size", cfg["train"]["batch_size"])),
        int(ckpt_cfg.get("train", {}).get("num_workers", 0)),
        shuffle=False,
    )
    metrics, _, _ = _evaluate_discharge(model, loader, criterion, device, target_names)
    payload = {"split": split_name, **_jsonable_metrics(metrics)}
    append_jsonl(os.path.join(run_dir, "metrics.jsonl"), {"epoch": -1, **payload})
    _save_json(os.path.join(run_dir, f"{split_name}_metrics.json"), payload)
    _print_final_metrics(role, split_name, payload)
    return payload


def _evaluate_los_checkpoint(
    cfg: dict[str, Any],
    root: str,
    checkpoint_path: str,
    indices: np.ndarray,
    run_dir: str,
    device: torch.device,
    *,
    role: str,
    split_name: str,
    calibration_path: str | None,
) -> dict[str, Any]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    ckpt_cfg = ckpt.get("cfg", cfg)
    dataset = LOSPredictionDataset(
        root=root,
        do_preprocess=_dataset_do_preprocess(ckpt_cfg),
    )
    (
        model,
        criterion,
        ce_class_weight,
        target_mode,
        loss_type,
        coarse_breakdown,
    ) = _build_los_model_and_loss(
        ckpt_cfg,
        dataset,
        np.asarray(indices, dtype=np.int64),
        device,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    ce_weight = float(ckpt_cfg.get("loss", {}).get("ce_weight", 1.0))
    loader = _make_loader(
        dataset,
        indices,
        int(ckpt_cfg.get("train", {}).get("batch_size", cfg["train"]["batch_size"])),
        int(ckpt_cfg.get("train", {}).get("num_workers", 0)),
        shuffle=False,
    )
    metrics, _, _, _ = _evaluate_los(
        model,
        loader,
        criterion,
        device,
        loss_type,
        ce_weight,
        ce_class_weight,
        target_mode,
        coarse_breakdown,
    )
    payload = {"split": split_name, **_jsonable_metrics(metrics)}
    if calibration_path is not None and os.path.exists(calibration_path):
        with open(calibration_path, "r", encoding="utf-8") as f:
            calibration_payload = json.load(f)
        temp = calibration_payload.get("temperature", {}).get("fitted")
        if temp is not None:
            payload["calibrated_temperature"] = float(temp)
    append_jsonl(os.path.join(run_dir, "metrics.jsonl"), {"epoch": -1, **payload})
    _save_json(os.path.join(run_dir, f"{split_name}_metrics.json"), payload)
    _print_final_metrics(role, split_name, payload)
    if calibration_path is not None:
        _print_los_calibration_summary(role, calibration_path)
    return payload


def _build_provider_cfg(base_cfg: dict[str, Any], checkpoint_path: str, calibration_path: str | None = None) -> dict[str, Any]:
    out = copy.deepcopy(base_cfg)
    out["enabled"] = True
    out["checkpoint_path"] = checkpoint_path
    if calibration_path is not None:
        out["calibration_path"] = calibration_path
    return out


def _init_soft_discharge_cache(
    n: int,
    discharge_provider: ForecastedDischargeProvider,
    input_metadata: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    soft_cfg = discharge_provider.describe_soft_config()
    if not bool(soft_cfg.get("enabled", False)):
        return None

    cache_heads: dict[str, Any] = {}
    target_dims = soft_cfg["target_dims"]
    target_to_col_idx = soft_cfg["target_to_col_idx"]
    for head_name in soft_cfg["soft_head_names"]:
        num_classes = int(target_dims[head_name])
        cache_heads[head_name] = {
            "hard": torch.zeros((n,), dtype=torch.long),
            "probs": torch.zeros((n, num_classes), dtype=torch.float32),
            "logits": torch.zeros((n, num_classes), dtype=torch.float32),
            "target_col_idx": torch.tensor(int(target_to_col_idx[head_name]), dtype=torch.long),
            "num_classes": torch.tensor(num_classes, dtype=torch.long),
            "class_to_embedding_idx": torch.arange(num_classes, dtype=torch.long),
            "mask": torch.ones((n,), dtype=torch.bool),
        }

    return {
        "head_names": list(soft_cfg["head_names"]),
        "soft_head_names": list(soft_cfg["soft_head_names"]),
        "heads": cache_heads,
        "metadata": {
            "mode": soft_cfg["mode"],
            "temperature": float(soft_cfg["temperature"]),
            "target_to_col_idx": dict(target_to_col_idx),
            "input_metadata": copy.deepcopy(input_metadata or {}),
        },
    }


def _assign_soft_discharge_chunk(
    soft_discharge_cache: dict[str, Any] | None,
    chunk: np.ndarray,
    soft_payload: dict[str, Any] | None,
) -> None:
    if soft_discharge_cache is None or soft_payload is None:
        return

    for head_name, src_head in soft_payload["heads"].items():
        dst_head = soft_discharge_cache["heads"][head_name]
        dst_head["hard"][chunk] = src_head["hard"].to(dtype=torch.long)
        dst_head["probs"][chunk] = src_head["probs"].to(dtype=torch.float32)
        if "logits" in src_head:
            dst_head["logits"][chunk] = src_head["logits"].to(dtype=torch.float32)


def _slice_soft_discharge_cache(
    soft_discharge_cache: dict[str, Any] | None,
    indices: np.ndarray,
) -> dict[str, Any] | None:
    if soft_discharge_cache is None:
        return None

    subset_heads: dict[str, Any] = {}
    for head_name, head_payload in soft_discharge_cache["heads"].items():
        subset_heads[head_name] = {
            "hard": head_payload["hard"][indices].clone(),
            "probs": head_payload["probs"][indices].clone(),
            "logits": head_payload["logits"][indices].clone(),
            "target_col_idx": head_payload["target_col_idx"].clone(),
            "num_classes": head_payload["num_classes"].clone(),
            "class_to_embedding_idx": head_payload["class_to_embedding_idx"].clone(),
            "mask": head_payload["mask"][indices].clone(),
        }
    return {
        "head_names": list(soft_discharge_cache["head_names"]),
        "soft_head_names": list(soft_discharge_cache["soft_head_names"]),
        "heads": subset_heads,
        "metadata": copy.deepcopy(soft_discharge_cache["metadata"]),
    }


def _log_soft_discharge_cache_summary(
    soft_discharge_cache: dict[str, Any] | None,
    base_dataset,
) -> None:
    if soft_discharge_cache is None:
        return

    col_dims = list(base_dataset.col_info[1])
    metadata = soft_discharge_cache["metadata"]
    print("[SOFT D CACHE]")
    print("enabled=True")
    print(f"mode={metadata['mode']}")
    print(f"heads={soft_discharge_cache['soft_head_names']}")
    print(f"temperature={metadata['temperature']}")
    print(f"num_heads_soft={len(soft_discharge_cache['soft_head_names'])}")
    print(f"num_heads_hard={len(soft_discharge_cache['head_names']) - len(soft_discharge_cache['soft_head_names'])}")
    for head_name in soft_discharge_cache["soft_head_names"]:
        head_payload = soft_discharge_cache["heads"][head_name]
        col_idx = int(head_payload["target_col_idx"].item())
        emb_rows = int(col_dims[col_idx])
        probs_shape = tuple(head_payload["probs"].shape)
        print(
            f"head={head_name} probs_shape={probs_shape} "
            f"emb_rows={emb_rows} col_idx={col_idx}"
        )


def _forecast_into_cache(
    base_dataset,
    indices: np.ndarray,
    discharge_provider: ForecastedDischargeProvider,
    los_provider: ForecastedLOSProvider,
    device: torch.device,
    batch_size: int,
    x_cache: torch.Tensor,
    los_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None = None,
) -> None:
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.stack([base_dataset[int(i)][0] for i in chunk], dim=0).to(device)
        x_pred, soft_payload = discharge_provider.predict_with_cache_payload(x)
        los_pred = los_provider(x_pred)
        x_cache[chunk] = x_pred.cpu()
        los_cache[chunk] = los_pred.cpu()
        _assign_soft_discharge_chunk(soft_discharge_cache, chunk, soft_payload)


def _init_caches(
    base_dataset,
    los_template: str,
    discharge_provider: ForecastedDischargeProvider | None = None,
    input_metadata: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
    n = len(base_dataset)
    x_cache = torch.stack([base_dataset[i][0] for i in range(n)], dim=0).clone()
    if los_template == "distribution":
        los_cache = torch.zeros((n, 37), dtype=torch.float32)
    else:
        los_cache = torch.zeros((n,), dtype=torch.long)
    soft_discharge_cache = None
    if discharge_provider is not None:
        soft_discharge_cache = _init_soft_discharge_cache(n, discharge_provider, input_metadata=input_metadata)
    return x_cache, los_cache, soft_discharge_cache


def _default_joint_stage1_cfg(parent_cfg: dict[str, Any]) -> dict[str, Any]:
    parent_train = parent_cfg.get("train", {})
    return {
        "device": parent_cfg.get("device"),
        "run_name": "joint_consistent_predictor",
        "model": {
            "name": "joint_consistent_predictor",
            "params": {},
        },
        "train": {
            "batch_size": 1024,
            "learning_rate": 1.0e-3,
            "epochs": 50,
            "seed": int(parent_train.get("seed", 1)),
            "fold": int(parent_cfg.get("fold", 0)),
            "num_folds": int(parent_train.get("n_folds", 5)),
            "num_workers": int(parent_train.get("num_workers", 0)),
            "test_ratio": float(parent_train.get("test_ratio", 0.15)),
            "lr_scheduler_patience": 5,
            "early_stopping_patience": 5,
            "optimizer": "adamw",
            "weight_decay": 1.0e-5,
            "monitor_metric": "valid_balanced_score",
            "monitor_mode": "max",
            "do_preprocess": bool(parent_train.get("do_preprocess", False)),
        },
        "joint_predictor": {
            "predictor_type": "joint_consistent",
            "joint_direction": "los_to_d",
            "condition_mode": "predicted",
            "detach_condition": True,
            "los_target_mode": "coarse",
            "lambda_los": 1.0,
            "lambda_aux": 0.3,
            "lambda_joint": 0.0,
            "prior_recon_weight": 0.5,
            "beta_kl_start": 0.0,
            "beta_kl_max": 0.001,
            "kl_anneal_epochs": 10,
            "joint_heads": ",".join(LEGACY_TOP3_HEADS),
            "save_cache": True,
            "cache_dir": None,
        },
        "joint_struct_loss": {
            "enabled": False,
            "lambda_struct": 0.0,
            "loss_type": "soft_js_d",
            "risk_head_set": "new_dvD_top3",
            "stopgrad_los": True,
            "min_los_support": 1.0e-6,
            "eps": 1.0e-8,
            "weight_by_los_support": True,
            "use_ema": False,
            "ema_momentum": 0.95,
        },
    }


def _build_joint_stage1_cfg(
    parent_cfg: dict[str, Any],
    stage1_override: dict[str, Any],
    *,
    fold: int,
) -> dict[str, Any]:
    cfg = _deep_merge_dict(_default_joint_stage1_cfg(parent_cfg), stage1_override)
    cfg["device"] = parent_cfg.get("device")
    cfg.setdefault("train", {})
    cfg["train"]["seed"] = int(cfg["train"].get("seed", parent_cfg.get("train", {}).get("seed", 1)))
    cfg["train"]["fold"] = int(fold)
    cfg["train"]["num_folds"] = int(
        cfg["train"].get("num_folds", parent_cfg.get("train", {}).get("n_folds", 5))
    )
    cfg.setdefault("joint_predictor", {})
    cfg["joint_predictor"]["save_cache"] = True
    return cfg


def _row_index_lookup(base_dataset) -> tuple[dict[int, int], np.ndarray]:
    raw_row_idx = base_dataset.raw_row_index.to_numpy(dtype=np.int64, copy=True)
    return {int(row_idx): int(pos) for pos, row_idx in enumerate(raw_row_idx)}, raw_row_idx


def _cache_positions_for_expected_indices(
    cache_payload: dict[str, Any],
    base_dataset,
    expected_indices: np.ndarray,
    *,
    split_name: str,
) -> np.ndarray:
    expected_indices = np.asarray(expected_indices, dtype=np.int64)
    lookup, raw_row_idx = _row_index_lookup(base_dataset)
    expected_rows = raw_row_idx[expected_indices]
    cache_rows = cache_payload["row_idx"].detach().cpu().numpy().astype(np.int64)
    if cache_rows.shape[0] != expected_indices.shape[0]:
        raise ValueError(
            f"{split_name}: cache row count mismatch cache={cache_rows.shape[0]} expected={expected_indices.shape[0]}"
        )
    if set(cache_rows.tolist()) != set(expected_rows.tolist()):
        raise ValueError(
            f"{split_name}: cache row_idx set does not match expected dataset rows."
        )
    positions = []
    for row_idx in cache_rows.tolist():
        if int(row_idx) not in lookup:
            raise ValueError(f"{split_name}: row_idx={row_idx} not present in active dataset.")
        positions.append(lookup[int(row_idx)])
    return np.asarray(positions, dtype=np.int64)


def _init_joint_soft_discharge_cache(
    base_dataset,
    n: int,
    cache_payload: dict[str, Any],
    input_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    target_names = [str(name) for name in cache_payload.get("metadata", {}).get("target_col_names", [])]
    if not target_names:
        target_names = [str(name) for name in cache_payload["final_d_probs"].keys()]
    contract = resolve_joint_forecast_contract(base_dataset.col_info, target_names)

    heads: dict[str, Any] = {}
    for head in contract.heads:
        head_name = head.name
        if head_name not in cache_payload["final_d_probs"]:
            continue
        probs = cache_payload["final_d_probs"][head_name].detach().cpu().to(dtype=torch.float32)
        num_classes = int(probs.shape[1])
        if int(head.num_classes) != num_classes:
            raise ValueError(
                f"Joint cache head {head_name} cardinality mismatch: cache={num_classes} dataset={int(head.num_classes)}"
            )
        head_payload = {
            "hard": torch.zeros((n,), dtype=torch.long),
            "probs": torch.zeros((n, num_classes), dtype=torch.float32),
            "target_col_idx": torch.tensor(int(head.target_col_idx), dtype=torch.long),
            "num_classes": torch.tensor(num_classes, dtype=torch.long),
            "class_to_embedding_idx": torch.arange(num_classes, dtype=torch.long),
            "mask": torch.zeros((n,), dtype=torch.bool),
        }
        logits = cache_payload.get("final_d_logits", {}).get(head_name)
        if logits is not None:
            head_payload["logits"] = torch.zeros((n, num_classes), dtype=torch.float32)
        else:
            head_payload["logits"] = torch.zeros((n, num_classes), dtype=torch.float32)
        heads[head_name] = head_payload

    return {
        "head_names": list(contract.head_names),
        "soft_head_names": list(contract.head_names),
        "heads": heads,
        "metadata": {
            "mode": "joint_distribution",
            "temperature": 1.0,
            "target_to_col_idx": contract.target_to_col_idx,
            "input_metadata": copy.deepcopy(input_metadata or {}),
            "source": "joint_forecast_input",
        },
    }


def _assign_joint_cache_split(
    *,
    base_dataset,
    split_name: str,
    cache_payload: dict[str, Any],
    expected_indices: np.ndarray,
    x_cache: torch.Tensor,
    los_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None,
    joint_mode: str,
) -> None:
    positions = _cache_positions_for_expected_indices(
        cache_payload,
        base_dataset,
        expected_indices,
        split_name=split_name,
    )
    col_list, _col_dims, _ad_col_index, dis_col_index = base_dataset.col_info
    discharge_cols = {str(col_list[idx]): int(idx) for idx in dis_col_index}

    for head_name, hard_values in cache_payload["final_d_pred"].items():
        col_idx = discharge_cols.get(str(head_name))
        if col_idx is None:
            raise ValueError(f"{split_name}: unknown discharge head {head_name!r} in joint cache.")
        x_cache[positions, col_idx] = hard_values.detach().cpu().to(dtype=x_cache.dtype)
        if soft_discharge_cache is not None and head_name in soft_discharge_cache["heads"]:
            probs = cache_payload["final_d_probs"][head_name].detach().cpu().to(dtype=torch.float32)
            hard = hard_values.detach().cpu().to(dtype=torch.long)
            dst_head = soft_discharge_cache["heads"][head_name]
            dst_head["hard"][positions] = hard
            dst_head["probs"][positions] = probs
            logits = cache_payload.get("final_d_logits", {}).get(head_name)
            if logits is not None:
                dst_head["logits"][positions] = logits.detach().cpu().to(dtype=torch.float32)
            else:
                dst_head["logits"][positions] = probs.clamp_min(1.0e-12).log()
            dst_head["mask"][positions] = True

    final_los_probs = cache_payload["final_los_probs"].detach().cpu().to(dtype=torch.float32)
    los_metadata = cache_payload.get("metadata", {})
    coarse_breakdown = bool(
        los_metadata.get("los_coarse_breakdown", los_metadata.get("breakdown", False))
    ) or (final_los_probs.ndim == 2 and final_los_probs.shape[1] == 9)
    if joint_mode == "distribution":
        if final_los_probs.ndim != 2:
            raise ValueError(f"{split_name}: joint LOS probs must be rank-2, got {tuple(final_los_probs.shape)}")
        if final_los_probs.shape[1] in {6, 9}:
            los_cache[positions] = expand_coarse_distribution_to_raw_los(
                final_los_probs,
                breakdown=coarse_breakdown,
            ).to(dtype=torch.float32)
        elif final_los_probs.shape[1] == 37:
            los_cache[positions] = final_los_probs
        else:
            raise ValueError(
                f"{split_name}: unsupported joint LOS probability width {final_los_probs.shape[1]} (expected 6, 9, or 37)."
            )
        return

    final_los_pred = cache_payload["final_los_pred"].detach().cpu().to(dtype=torch.long)
    pred_space = str(cache_payload.get("metadata", {}).get("final_los_pred_space", "")).lower()
    if pred_space == "coarse_class":
        los_cache[positions] = map_coarse_array_to_raw_los(
            final_los_pred,
            breakdown=coarse_breakdown,
        ).to(dtype=torch.long)
    else:
        los_cache[positions] = final_los_pred


def _resolve_joint_cache_paths(joint_cfg: dict[str, Any], run_dir: str | None) -> dict[str, str]:
    def _resolve_cache_path(path_value: Any, base_dir: str | None = None) -> str:
        path_str = str(path_value)
        if os.path.isabs(path_str):
            return path_str

        candidates = [path_str]
        if base_dir:
            candidates.append(os.path.join(base_dir, path_str))

        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate

        return os.path.join(base_dir, path_str) if base_dir else path_str

    input_cfg = joint_cfg.get("joint_forecast_input", {})
    source_run_dir = input_cfg.get("source_run_dir")
    if source_run_dir:
        manifest_path = os.path.join(str(source_run_dir), "joint_cache", "cache_manifest.json")
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(f"Joint cache manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        return {
            str(key): _resolve_cache_path(value, str(source_run_dir))
            for key, value in manifest.items()
        }
    if run_dir is not None:
        manifest_path = os.path.join(run_dir, "joint_cache", "cache_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            return {
                str(key): _resolve_cache_path(value, run_dir)
                for key, value in manifest.items()
            }

    explicit = {}
    for key in ("train_cache_path", "val_cache_path", "test_cache_path", "gnn_val_cache_path", "outer_test_cache_path"):
        value = input_cfg.get(key)
        if value:
            explicit[key.replace("_cache_path", "")] = str(value)
    if explicit:
        return explicit
    raise ValueError("joint_forecast_input requires source_run_dir or explicit cache paths.")


def prepare_joint_forecast_fold_data(
    cfg: dict[str, Any],
    root: str,
    base_dataset,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    fold_dir: str,
    device: torch.device,
) -> ForecastedFoldData:
    pipeline_cfg = cfg.get("joint_forecast_pipeline", {})
    input_metadata = resolve_model_forecast_input_metadata(cfg)
    seed = int(cfg["train"].get("seed", 42)) + int(cfg.get("fold", 0)) * 1000
    set_seed(seed)
    train_core_idx, predictor_val_idx, gnn_val_idx = split_outer_train_for_forecasted_pipeline(
        base_dataset, outer_train_idx, cfg, seed
    )
    outer_test_idx = np.asarray(outer_test_idx, dtype=np.int64)
    _assert_disjoint(
        {
            "train_core": train_core_idx,
            "predictor_val": predictor_val_idx,
            "gnn_val": gnn_val_idx,
            "outer_test": outer_test_idx,
        }
    )

    input_cfg = pipeline_cfg.get("joint_forecast_input", {})
    joint_run_dir = None
    has_explicit_cache_paths = any(
        input_cfg.get(key)
        for key in (
            "train_cache_path",
            "val_cache_path",
            "test_cache_path",
            "gnn_val_cache_path",
            "outer_test_cache_path",
        )
    )
    if not input_cfg.get("source_run_dir") and not has_explicit_cache_paths:
        stage1_cfg = _build_joint_stage1_cfg(
            cfg,
            dict(pipeline_cfg.get("stage1", {})),
            fold=int(cfg.get("fold", 0)),
        )
        joint_run_dir = os.path.join(fold_dir, "joint_predictor")
        stage1_cfg["joint_predictor"]["cache_dir"] = os.path.join(joint_run_dir, "joint_cache")
        run_joint_consistent_predictor(
            stage1_cfg,
            os.path.abspath(root),
            run_dir=joint_run_dir,
            split_indices={
                "train": train_core_idx,
                "val": predictor_val_idx,
                "test": outer_test_idx,
            },
            export_indices={
                "train": train_core_idx,
                "gnn_val": gnn_val_idx,
                "outer_test": outer_test_idx,
            },
        )

    cache_paths = _resolve_joint_cache_paths(pipeline_cfg, joint_run_dir)
    joint_mode = str(input_cfg.get("mode", "distribution")).lower()
    if joint_mode not in {"distribution", "hard"}:
        raise ValueError(f"Unsupported joint_forecast_input.mode: {joint_mode}")

    x_cache, los_cache, _ = _init_caches(
        base_dataset,
        "distribution" if joint_mode == "distribution" else "hard",
        discharge_provider=None,
        input_metadata=input_metadata,
    )

    cache_payloads = {
        "train": torch.load(str(cache_paths["train"]), map_location="cpu", weights_only=False),
        "gnn_val": torch.load(
            str(cache_paths.get("gnn_val", cache_paths.get("val"))),
            map_location="cpu",
            weights_only=False,
        ),
        "outer_test": torch.load(
            str(cache_paths.get("outer_test", cache_paths.get("test"))),
            map_location="cpu",
            weights_only=False,
        ),
    }
    soft_discharge_cache = None
    if joint_mode == "distribution":
        soft_discharge_cache = _init_joint_soft_discharge_cache(
            base_dataset,
            len(base_dataset),
            cache_payloads["train"],
            input_metadata,
        )

    _assign_joint_cache_split(
        base_dataset=base_dataset,
        split_name="train",
        cache_payload=cache_payloads["train"],
        expected_indices=train_core_idx,
        x_cache=x_cache,
        los_cache=los_cache,
        soft_discharge_cache=soft_discharge_cache,
        joint_mode=joint_mode,
    )
    _assign_joint_cache_split(
        base_dataset=base_dataset,
        split_name="gnn_val",
        cache_payload=cache_payloads["gnn_val"],
        expected_indices=gnn_val_idx,
        x_cache=x_cache,
        los_cache=los_cache,
        soft_discharge_cache=soft_discharge_cache,
        joint_mode=joint_mode,
    )
    _assign_joint_cache_split(
        base_dataset=base_dataset,
        split_name="outer_test",
        cache_payload=cache_payloads["outer_test"],
        expected_indices=outer_test_idx,
        x_cache=x_cache,
        los_cache=los_cache,
        soft_discharge_cache=soft_discharge_cache,
        joint_mode=joint_mode,
    )

    cached_dir = os.path.join(fold_dir, "cached_predictions")
    os.makedirs(cached_dir, exist_ok=True)
    torch.save(
        {
            "x": x_cache[train_core_idx],
            "los": los_cache[train_core_idx],
            "indices": train_core_idx,
            "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, train_core_idx),
        },
        os.path.join(cached_dir, "train_core_joint.pt"),
    )
    torch.save(
        {
            "x": x_cache[gnn_val_idx],
            "los": los_cache[gnn_val_idx],
            "indices": gnn_val_idx,
            "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, gnn_val_idx),
        },
        os.path.join(cached_dir, "gnn_val_joint.pt"),
    )
    torch.save(
        {
            "x": x_cache[outer_test_idx],
            "los": los_cache[outer_test_idx],
            "indices": outer_test_idx,
            "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, outer_test_idx),
        },
        os.path.join(cached_dir, "outer_test_joint.pt"),
    )

    split_payload = {
        "train_core_idx": train_core_idx.tolist(),
        "predictor_val_idx": predictor_val_idx.tolist(),
        "gnn_val_idx": gnn_val_idx.tolist(),
        "outer_test_idx": outer_test_idx.tolist(),
        "joint_mode": joint_mode,
        "joint_cache_paths": cache_paths,
        "joint_run_dir": joint_run_dir,
        "forecast_input_metadata": input_metadata,
    }
    _save_json(os.path.join(fold_dir, "joint_forecast_pipeline_splits.json"), split_payload)
    _save_json(os.path.join(fold_dir, "forecast_input_metadata.json"), input_metadata)

    cached_dataset = ForecastCacheDataset(base_dataset, x_cache, los_cache, soft_discharge_cache)
    batch_size = int(cfg["train"]["batch_size"])
    return ForecastedFoldData(
        train_idx=train_core_idx,
        val_idx=gnn_val_idx,
        test_idx=outer_test_idx,
        train_loader=_make_loader(cached_dataset, train_core_idx, batch_size, int(cfg["train"]["num_workers"]), shuffle=True),
        val_loader=_make_loader(cached_dataset, gnn_val_idx, batch_size, int(cfg["train"]["num_workers"]), shuffle=False),
        test_loader=_make_loader(cached_dataset, outer_test_idx, batch_size, int(cfg["train"]["num_workers"]), shuffle=False),
        split_payload=split_payload,
    )


def _train_inner_oof_predictors(
    cfg: dict[str, Any],
    root: str,
    train_core_idx: np.ndarray,
    fold_dir: str,
    device: torch.device,
    discharge_cfg: dict[str, Any],
    los_cfg: dict[str, Any],
    discharge_best_epoch: int,
    los_best_epoch: int,
    base_dataset,
    x_cache: torch.Tensor,
    los_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None,
) -> None:
    pipeline_cfg = cfg.get("forecasted_pipeline", {})
    n_inner = int(pipeline_cfg.get("oof", {}).get("n_inner_folds", 5))
    labels = _labels_for_indices(base_dataset, train_core_idx)
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=cfg["train"].get("seed", 42))
    batch_size = int(cfg["train"]["batch_size"])
    for inner_fold, (inner_train_pos, inner_holdout_pos) in enumerate(skf.split(np.zeros(len(train_core_idx)), labels)):
        inner_train_idx = train_core_idx[inner_train_pos]
        inner_holdout_idx = train_core_idx[inner_holdout_pos]
        inner_dir = os.path.join(fold_dir, "predictors", "oof", f"inner_{inner_fold}")
        print(
            f"[OOF inner {inner_fold}/{n_inner - 1}] start: "
            f"train={len(inner_train_idx)} holdout={len(inner_holdout_idx)} "
            f"fixed_epochs discharge={discharge_best_epoch} los={los_best_epoch}"
        )
        discharge_result = _train_discharge_predictor(
            discharge_cfg,
            root,
            inner_train_idx,
            None,
            os.path.join(inner_dir, "discharge"),
            device,
            fixed_epochs=discharge_best_epoch,
            role=f"OOF inner {inner_fold} discharge predictor",
            verbose=False,
        )
        los_result = _train_los_predictor(
            los_cfg,
            root,
            inner_train_idx,
            None,
            os.path.join(inner_dir, "los"),
            device,
            fixed_epochs=los_best_epoch,
            role=f"OOF inner {inner_fold} LOS predictor",
            verbose=False,
        )
        discharge_provider = ForecastedDischargeProvider(
            normalize_forecasted_discharge_cfg(
                cfg,
                _build_provider_cfg(cfg.get("forecasted_discharge", {}), discharge_result["checkpoint_path"]),
            ),
            base_dataset,
            device,
        )
        los_provider = ForecastedLOSProvider(
            normalize_forecasted_los_cfg(
                cfg,
                _build_provider_cfg(
                    cfg.get("forecasted_los", {}),
                    los_result["checkpoint_path"],
                    los_result.get("calibration_path"),
                ),
            ),
            base_dataset,
            device,
        )
        _forecast_into_cache(
            base_dataset,
            inner_holdout_idx,
            discharge_provider,
            los_provider,
            device,
            batch_size,
            x_cache,
            los_cache,
            soft_discharge_cache,
        )
        print(
            f"[OOF inner {inner_fold}/{n_inner - 1}] finished: "
            f"discharge_best_epoch={discharge_result['best_epoch']} "
            f"los_best_epoch={los_result['best_epoch']}"
        )


def prepare_forecasted_fold_data(
    cfg: dict[str, Any],
    root: str,
    base_dataset,
    outer_train_idx: np.ndarray,
    outer_test_idx: np.ndarray,
    fold_dir: str,
    device: torch.device,
) -> ForecastedFoldData:
    pipeline_cfg = cfg.get("forecasted_pipeline", {})
    input_metadata = resolve_model_forecast_input_metadata(cfg)
    seed = int(cfg["train"].get("seed", 42)) + int(cfg.get("fold", 0)) * 1000
    set_seed(seed)
    train_core_idx, predictor_val_idx, gnn_val_idx = split_outer_train_for_forecasted_pipeline(
        base_dataset, outer_train_idx, cfg, seed
    )
    outer_test_idx = np.asarray(outer_test_idx, dtype=np.int64)
    _assert_disjoint(
        {
            "train_core": train_core_idx,
            "predictor_val": predictor_val_idx,
            "gnn_val": gnn_val_idx,
            "outer_test": outer_test_idx,
        }
    )

    discharge_cfg_path = pipeline_cfg.get("discharge_config_path", "configs/discharge_predictor.yaml")
    los_cfg_path = pipeline_cfg.get("los_config_path", "configs/los_ce_predictor.yaml")
    discharge_cfg = _load_yaml(str(discharge_cfg_path))
    los_cfg = _load_yaml(str(los_cfg_path))
    _inherit_dataset_settings(cfg, discharge_cfg)
    _inherit_dataset_settings(cfg, los_cfg)
    discharge_cfg["device"] = cfg.get("device")
    los_cfg["device"] = cfg.get("device")

    selection_dir = os.path.join(fold_dir, "predictors", "selection")
    discharge_selection = _train_discharge_predictor(
        discharge_cfg,
        root,
        train_core_idx,
        predictor_val_idx,
        os.path.join(selection_dir, "discharge"),
        device,
        role="selection discharge predictor",
        verbose=True,
    )
    los_selection = _train_los_predictor(
        los_cfg,
        root,
        train_core_idx,
        predictor_val_idx,
        os.path.join(selection_dir, "los"),
        device,
        role="selection LOS predictor",
        verbose=True,
    )
    _evaluate_discharge_checkpoint(
        discharge_cfg,
        root,
        discharge_selection["checkpoint_path"],
        outer_test_idx,
        discharge_selection["run_dir"],
        device,
        role="fold-final discharge predictor",
        split_name="outer_test_final",
    )
    _evaluate_los_checkpoint(
        los_cfg,
        root,
        los_selection["checkpoint_path"],
        outer_test_idx,
        los_selection["run_dir"],
        device,
        role="fold-final LOS predictor",
        split_name="outer_test_final",
        calibration_path=los_selection.get("calibration_path"),
    )

    los_return_type = str(cfg.get("forecasted_los", {}).get("return_type", "hard")).lower()
    normalized_forecast_discharge_cfg = normalize_forecasted_discharge_cfg(
        cfg,
        _build_provider_cfg(cfg.get("forecasted_discharge", {}), discharge_selection["checkpoint_path"]),
    )
    final_discharge_provider = ForecastedDischargeProvider(
        normalized_forecast_discharge_cfg,
        base_dataset,
        device,
    )
    final_los_provider = ForecastedLOSProvider(
        normalize_forecasted_los_cfg(
            cfg,
            _build_provider_cfg(
                cfg.get("forecasted_los", {}),
                los_selection["checkpoint_path"],
                los_selection.get("calibration_path"),
            ),
        ),
        base_dataset,
        device,
    )
    x_cache, los_cache, soft_discharge_cache = _init_caches(
        base_dataset,
        los_return_type,
        final_discharge_provider,
        input_metadata=input_metadata,
    )
    batch_size = int(cfg["train"]["batch_size"])

    train_prediction_mode = str(pipeline_cfg.get("train_prediction_mode", "oof")).lower()
    if train_prediction_mode == "oof":
        _train_inner_oof_predictors(
            cfg,
            root,
            train_core_idx,
            fold_dir,
            device,
            discharge_cfg,
            los_cfg,
            int(discharge_selection["best_epoch"]),
            int(los_selection["best_epoch"]),
            base_dataset,
            x_cache,
            los_cache,
            soft_discharge_cache,
        )
    elif train_prediction_mode == "in_sample":
        _forecast_into_cache(
            base_dataset,
            train_core_idx,
            final_discharge_provider,
            final_los_provider,
            device,
            batch_size,
            x_cache,
            los_cache,
            soft_discharge_cache,
        )
    else:
        raise ValueError(f"Unsupported forecasted_pipeline.train_prediction_mode: {train_prediction_mode}")

    _forecast_into_cache(
        base_dataset,
        gnn_val_idx,
        final_discharge_provider,
        final_los_provider,
        device,
        batch_size,
        x_cache,
        los_cache,
        soft_discharge_cache,
    )
    _forecast_into_cache(
        base_dataset,
        outer_test_idx,
        final_discharge_provider,
        final_los_provider,
        device,
        batch_size,
        x_cache,
        los_cache,
        soft_discharge_cache,
    )
    _log_soft_discharge_cache_summary(soft_discharge_cache, base_dataset)

    cached_dir = os.path.join(fold_dir, "cached_predictions")
    os.makedirs(cached_dir, exist_ok=True)
    torch.save(
        {
            "x": x_cache[train_core_idx],
            "los": los_cache[train_core_idx],
            "indices": train_core_idx,
            "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, train_core_idx),
        },
        os.path.join(cached_dir, f"train_core_{train_prediction_mode}.pt"),
    )
    torch.save(
        {
            "x": x_cache[gnn_val_idx],
            "los": los_cache[gnn_val_idx],
            "indices": gnn_val_idx,
            "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, gnn_val_idx),
        },
        os.path.join(cached_dir, "gnn_val.pt"),
    )
    torch.save(
        {
            "x": x_cache[outer_test_idx],
            "los": los_cache[outer_test_idx],
            "indices": outer_test_idx,
            "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, outer_test_idx),
        },
        os.path.join(cached_dir, "outer_test.pt"),
    )

    split_payload = {
        "train_core_idx": train_core_idx.tolist(),
        "predictor_val_idx": predictor_val_idx.tolist(),
        "gnn_val_idx": gnn_val_idx.tolist(),
        "outer_test_idx": outer_test_idx.tolist(),
        "train_prediction_mode": train_prediction_mode,
        "discharge_selection": discharge_selection,
        "los_selection": los_selection,
        "forecast_input_metadata": input_metadata,
    }
    _save_json(os.path.join(fold_dir, "forecasted_pipeline_splits.json"), split_payload)
    _save_json(os.path.join(fold_dir, "forecast_input_metadata.json"), input_metadata)

    cached_dataset = ForecastCacheDataset(base_dataset, x_cache, los_cache, soft_discharge_cache)
    return ForecastedFoldData(
        train_idx=train_core_idx,
        val_idx=gnn_val_idx,
        test_idx=outer_test_idx,
        train_loader=_make_loader(cached_dataset, train_core_idx, batch_size, int(cfg["train"]["num_workers"]), shuffle=True),
        val_loader=_make_loader(cached_dataset, gnn_val_idx, batch_size, int(cfg["train"]["num_workers"]), shuffle=False),
        test_loader=_make_loader(cached_dataset, outer_test_idx, batch_size, int(cfg["train"]["num_workers"]), shuffle=False),
        split_payload=split_payload,
    )
