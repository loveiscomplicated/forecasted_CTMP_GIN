from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.discharge_predictor import (
    JointGenerativePredictor,
    expand_coarse_distribution_to_raw_los,
)
from src.models.forecasted_ctmp_gin import (
    CANONICAL_JOINT_FORECAST_HEADS,
    OutcomeAwareForecastedCTMPGIN,
    OutcomeAwareForecastedGIN,
    assert_soft_discharge_contract_matches_cached_metadata,
    resolve_joint_forecast_contract,
)
from src.trainers.run_joint_consistent_predictor import (
    JointPredictionDataset,
    _build_d_target_dict,
    _generate_joint_drift_reports,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.experiment import ExperimentLogger


@dataclass
class OutcomeAwareBatch:
    x: torch.Tensor
    y: torch.Tensor
    d_targets: torch.Tensor
    los_target: torch.Tensor
    los_raw: torch.Tensor
    row_idx: torch.Tensor


class OutcomeAwareStage2Dataset(Dataset):
    def __init__(self, base_dataset: TEDSTensorDataset, joint_dataset: JointPredictionDataset) -> None:
        if len(base_dataset) != len(joint_dataset):
            raise ValueError(
                "OutcomeAwareStage2Dataset requires TEDS and joint datasets with identical length."
            )
        if not np.array_equal(
            base_dataset.raw_row_index.to_numpy(dtype=np.int64, copy=True),
            joint_dataset.row_idx.numpy(),
        ):
            raise ValueError("OutcomeAwareStage2Dataset row_idx mismatch between datasets.")
        self.base_dataset = base_dataset
        self.joint_dataset = joint_dataset

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> OutcomeAwareBatch:
        x, y, _los_unused = self.base_dataset[index]
        joint_item = self.joint_dataset[index]
        return OutcomeAwareBatch(
            x=x.long(),
            y=torch.as_tensor(y).long(),
            d_targets=joint_item.d_targets.long(),
            los_target=joint_item.los_target.long(),
            los_raw=joint_item.los_raw.long(),
            row_idx=joint_item.row_idx.long(),
        )


def _collate_outcome_batch(
    batch: list[OutcomeAwareBatch],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([item.x for item in batch], dim=0),
        torch.stack([item.y for item in batch], dim=0),
        torch.stack([item.d_targets for item in batch], dim=0),
        torch.stack([item.los_target for item in batch], dim=0),
        torch.stack([item.los_raw for item in batch], dim=0),
        torch.stack([item.row_idx for item in batch], dim=0),
    )


def _make_loader(
    dataset: Dataset,
    indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    *,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, np.asarray(indices, dtype=np.int64).tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=_collate_outcome_batch,
    )


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def resolve_stage2_pretrained_paths(
    *,
    fold: int,
    stage2_cfg: dict[str, Any],
    fallback_predictor_checkpoint_path: str,
    fallback_baseline_checkpoint_path: str,
    fallback_baseline_metrics: dict[str, Any],
) -> tuple[str, str, dict[str, Any]]:
    source_run_dir = stage2_cfg.get("source_run_dir")
    if not source_run_dir:
        return (
            fallback_predictor_checkpoint_path,
            fallback_baseline_checkpoint_path,
            fallback_baseline_metrics,
        )

    source_run_dir = str(source_run_dir)
    single_run_cfg_path = os.path.join(source_run_dir, "config.final.yaml")
    single_run_split_path = os.path.join(source_run_dir, "single_run_splits.json")
    single_run = os.path.exists(single_run_cfg_path) and os.path.exists(single_run_split_path)

    if single_run:
        source_artifact_dir = source_run_dir
        result_path = os.path.join(source_artifact_dir, "single_run_result.json")
    else:
        source_artifact_dir = os.path.join(source_run_dir, "folds", f"fold_{int(fold)}")
        result_path = os.path.join(source_artifact_dir, "fold_result.json")

    predictor_checkpoint_path = os.path.join(
        source_artifact_dir,
        "joint_predictor",
        "checkpoints",
        "best.pt",
    )
    baseline_checkpoint_path = os.path.join(
        source_artifact_dir,
        "checkpoints",
        "best.pt",
    )
    if not os.path.exists(predictor_checkpoint_path):
        raise FileNotFoundError(
            f"Stage2 predictor checkpoint not found: {predictor_checkpoint_path}"
        )
    if not os.path.exists(baseline_checkpoint_path):
        raise FileNotFoundError(
            f"Stage2 baseline CTMP-GIN checkpoint not found: {baseline_checkpoint_path}"
        )

    baseline_metrics = dict(fallback_baseline_metrics)
    if os.path.exists(result_path):
        with open(result_path, "r", encoding="utf-8") as f:
            source_result = json.load(f)
        baseline_metrics = {
            "baseline_valid_auc": float(
                (source_result.get("best_valid_metrics") or {}).get("valid_auc", float("nan"))
            ),
            "baseline_test_auc": float(source_result.get("test_auc", float("nan"))),
            "baseline_valid_f1": float(
                (source_result.get("best_valid_metrics") or {}).get("valid_f1", float("nan"))
            ),
            "baseline_test_f1": float(source_result.get("test_f1", float("nan"))),
            "baseline_valid_acc": float(
                (source_result.get("best_valid_metrics") or {}).get("valid_acc", float("nan"))
            ),
            "baseline_test_acc": float(source_result.get("test_acc", float("nan"))),
        }
    return predictor_checkpoint_path, baseline_checkpoint_path, baseline_metrics


def _load_stage1_predictor(
    checkpoint_path: str,
    joint_dataset: JointPredictionDataset,
    device: torch.device,
) -> JointGenerativePredictor:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt["cfg"]
    model = JointGenerativePredictor(
        ad_col_dims=joint_dataset.ad_col_dims,
        target_col_names=joint_dataset.target_col_names,
        target_col_dims=joint_dataset.target_col_dims,
        los_num_classes=joint_dataset.los_num_classes,
        **cfg["model"]["params"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model


def _resolve_predictor_admission_col_indices(
    base_dataset: TEDSTensorDataset,
    joint_dataset: JointPredictionDataset,
) -> list[int]:
    col_list, col_dims, ad_col_index, _dis_col_index = base_dataset.col_info
    feature_idx_by_name: dict[str, int] = {}
    col_dim_by_name: dict[str, int] = {}
    feature_idx = 0
    for name, dim in zip(col_list, col_dims):
        name = str(name)
        if name == "LOS":
            continue
        if name in feature_idx_by_name:
            raise ValueError(f"Duplicate base dataset feature column name: {name}")
        feature_idx_by_name[name] = feature_idx
        col_dim_by_name[name] = int(dim)
        feature_idx += 1

    if hasattr(base_dataset, "processed_tensor"):
        input_width = int(base_dataset.processed_tensor.shape[1] - 1)
        if feature_idx != input_width:
            raise ValueError(
                "Base dataset feature width does not match non-LOS col_info columns: "
                f"feature_width={input_width} non_los_col_info={feature_idx}"
            )

    joint_ad_names = [str(name) for name in joint_dataset.ad_col_names]
    missing = [name for name in joint_ad_names if name not in feature_idx_by_name]
    if missing:
        raise ValueError(
            "Joint predictor admission columns are not present in base dataset features. "
            f"missing={missing[:10]}"
        )

    mismatched_dims = [
        (name, int(expected_dim), int(col_dim_by_name[name]))
        for name, expected_dim in zip(joint_ad_names, joint_dataset.ad_col_dims)
        if int(col_dim_by_name[name]) != int(expected_dim)
    ]
    if mismatched_dims:
        raise ValueError(
            "Joint predictor admission column cardinalities do not match base dataset. "
            f"mismatched={mismatched_dims[:10]}"
        )

    base_ad_names = [
        str(col_list[int(idx)])
        for idx in ad_col_index
        if idx is not None and 0 <= int(idx) < len(col_list)
    ]
    joint_ad_name_set = set(joint_ad_names)
    ignored_base_only = [name for name in base_ad_names if name not in joint_ad_name_set]
    if ignored_base_only:
        print(
            "[stage2] predictor admission schema aligned by joint predictor names: "
            f"predictor_cols={len(joint_ad_names)} base_admission_cols={len(base_ad_names)} "
            f"ignored_base_only={ignored_base_only[:10]}"
        )

    return [int(feature_idx_by_name[name]) for name in joint_ad_names]


def _count_trainable_params(module: nn.Module) -> int:
    return sum(param.numel() for param in module.parameters() if param.requires_grad)


def _set_requires_grad(module: nn.Module, enabled: bool) -> None:
    for param in module.parameters():
        param.requires_grad = bool(enabled)


def _stage2_backbone(wrapper: nn.Module) -> nn.Module:
    if hasattr(wrapper, "ctmp_gin"):
        return getattr(wrapper, "ctmp_gin")
    if hasattr(wrapper, "gin"):
        return getattr(wrapper, "gin")
    raise AttributeError("Outcome-aware wrapper has no known backbone attribute.")


def _stage2_classifier(wrapper: nn.Module) -> nn.Module:
    backbone = _stage2_backbone(wrapper)
    if hasattr(backbone, "classifier_b"):
        return getattr(backbone, "classifier_b")
    if hasattr(backbone, "classifier"):
        return getattr(backbone, "classifier")
    raise AttributeError("Outcome-aware backbone has no known classifier attribute.")


def _stage2_gated_fusion(wrapper: nn.Module) -> nn.Module | None:
    backbone = _stage2_backbone(wrapper)
    return getattr(backbone, "gated_fusion", None)


def _apply_ctmp_gin_freeze_policy(
    wrapper: nn.Module,
    stage2_cfg: dict[str, Any],
) -> None:
    freeze_ctmp_gin = bool(stage2_cfg.get("freeze_ctmp_gin", True))
    freeze_backbone = bool(stage2_cfg.get("freeze_ctmp_gin_backbone", True))
    train_gated_fusion = bool(stage2_cfg.get("train_gated_fusion", False))
    train_classifier = bool(stage2_cfg.get("train_classifier", False))
    train_predictor = bool(stage2_cfg.get("train_predictor", True))
    backbone = _stage2_backbone(wrapper)
    gated_fusion = _stage2_gated_fusion(wrapper)
    classifier = _stage2_classifier(wrapper)

    if isinstance(wrapper, OutcomeAwareForecastedGIN) and train_gated_fusion:
        raise ValueError("GIN outcome-aware stage2 does not support train_gated_fusion=true.")

    _set_requires_grad(wrapper.predictor, train_predictor)
    _set_requires_grad(backbone, not freeze_ctmp_gin)
    if freeze_ctmp_gin:
        return

    if freeze_backbone:
        _set_requires_grad(backbone, False)
        if train_gated_fusion and gated_fusion is not None:
            _set_requires_grad(gated_fusion, True)
        if train_classifier:
            _set_requires_grad(classifier, True)


def _set_stage2_module_modes(
    wrapper: nn.Module,
    stage2_cfg: dict[str, Any],
    *,
    is_training: bool,
) -> None:
    if bool(stage2_cfg.get("train_predictor", True)) and is_training:
        wrapper.predictor.train()
    else:
        wrapper.predictor.eval()

    freeze_ctmp_gin = bool(stage2_cfg.get("freeze_ctmp_gin", True))
    freeze_backbone = bool(stage2_cfg.get("freeze_ctmp_gin_backbone", True))
    train_gated_fusion = bool(stage2_cfg.get("train_gated_fusion", False))
    train_classifier = bool(stage2_cfg.get("train_classifier", False))
    backbone = _stage2_backbone(wrapper)
    gated_fusion = _stage2_gated_fusion(wrapper)
    classifier = _stage2_classifier(wrapper)
    if freeze_ctmp_gin or not is_training:
        backbone.eval()
        return

    if freeze_backbone:
        backbone.eval()
        if train_gated_fusion and gated_fusion is not None:
            gated_fusion.train()
        if train_classifier:
            classifier.train()
        return

    backbone.train()


def _build_optimizer(
    wrapper: nn.Module,
    stage2_cfg: dict[str, Any],
) -> torch.optim.Optimizer:
    params = [param for param in wrapper.parameters() if param.requires_grad]
    if not params:
        raise ValueError("Outcome-aware stage2 resolved zero trainable parameters.")
    return torch.optim.AdamW(
        params,
        lr=float(stage2_cfg["learning_rate"]),
        weight_decay=float(stage2_cfg.get("weight_decay", 0.0)),
    )


def _compute_aux_loss(
    output,
    d_targets: torch.Tensor,
    los_target: torch.Tensor,
    target_names: list[str],
) -> torch.Tensor:
    ce = nn.CrossEntropyLoss()
    d_target_dict = _build_d_target_dict(d_targets, target_names)
    aux = output.reasonb_logits.new_zeros(())
    for head_name, logits in output.predictor_output.prior_d_logits.items():
        aux = aux + ce(logits, d_target_dict[head_name].long())
    aux = aux + ce(output.predictor_output.prior_los_logits, los_target.long())
    return aux


def _label_count(values: np.ndarray) -> list[int]:
    return np.bincount(values.astype(int), minlength=2).tolist()


def _compute_binary_metrics(
    *,
    loss_mean: float,
    logits_chunks: list[np.ndarray],
    target_chunks: list[np.ndarray],
    aux_loss_mean: float | None = None,
    outcome_loss_mean: float | None = None,
    diagnostics: dict[str, float] | None = None,
) -> dict[str, Any]:
    logits = np.concatenate(logits_chunks, axis=0).reshape(-1)
    targets = np.concatenate(target_chunks, axis=0).reshape(-1)
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(np.int64)
    metrics = {
        "loss": float(loss_mean),
        "acc": float((preds == targets).mean()),
        "precision": float(precision_score(targets, preds, zero_division=0)),
        "recall": float(recall_score(targets, preds, zero_division=0)),
        "f1": float(f1_score(targets, preds, zero_division=0)),
        "auc": float(roc_auc_score(targets, probs)) if len(np.unique(targets)) > 1 else 0.0,
        "predicted_label_counts": _label_count(preds),
        "true_label_counts": _label_count(targets),
    }
    if aux_loss_mean is not None:
        metrics["aux_loss"] = float(aux_loss_mean)
    if outcome_loss_mean is not None:
        metrics["outcome_loss"] = float(outcome_loss_mean)
    if diagnostics:
        metrics.update({key: float(value) for key, value in diagnostics.items()})
    return metrics


def _evaluate_split(
    wrapper: nn.Module,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    target_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    criterion = nn.BCEWithLogitsLoss()
    wrapper.predictor.eval()
    _stage2_backbone(wrapper).eval()
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    row_idx_chunks: list[np.ndarray] = []
    total_loss = 0.0
    total_outcome = 0.0
    total_aux = 0.0
    n_batches = 0
    diag_sums: dict[str, float] = {}
    with torch.no_grad():
        for x, y, d_targets, los_target, _los_raw, row_idx in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d_targets = d_targets.to(device, non_blocking=True)
            los_target = los_target.to(device, non_blocking=True)
            output = wrapper(x, edge_index)
            logits = output.reasonb_logits.squeeze(1)
            outcome_loss = criterion(logits, y.float())
            aux_loss = _compute_aux_loss(output, d_targets, los_target, target_names)
            total_loss += float(outcome_loss.detach().cpu())
            total_outcome += float(outcome_loss.detach().cpu())
            total_aux += float(aux_loss.detach().cpu())
            n_batches += 1
            logits_chunks.append(logits.detach().cpu().numpy())
            target_chunks.append(y.detach().cpu().numpy())
            row_idx_chunks.append(row_idx.detach().cpu().numpy())
            for key, value in output.diagnostics.items():
                diag_sums[key] = diag_sums.get(key, 0.0) + float(value)
    diag_means = {key: value / max(n_batches, 1) for key, value in diag_sums.items()}
    metrics = _compute_binary_metrics(
        loss_mean=total_loss / max(n_batches, 1),
        logits_chunks=logits_chunks,
        target_chunks=target_chunks,
        aux_loss_mean=total_aux / max(n_batches, 1),
        outcome_loss_mean=total_outcome / max(n_batches, 1),
        diagnostics=diag_means,
    )
    payload = {
        "row_idx": np.concatenate(row_idx_chunks, axis=0).tolist(),
    }
    return metrics, payload


def _rows_from_cached_ctmp_payload(
    base_dataset: TEDSTensorDataset,
    cache_payload: dict[str, Any],
) -> torch.Tensor:
    indices = torch.as_tensor(cache_payload["indices"], dtype=torch.long)
    raw_rows = base_dataset.raw_row_index.to_numpy(dtype=np.int64, copy=True)
    return torch.as_tensor(raw_rows[indices.cpu().numpy()], dtype=torch.long)


def _align_cached_to_live_rows(
    *,
    cached_values: torch.Tensor,
    cached_rows: torch.Tensor,
    live_rows: torch.Tensor,
    split_name: str,
    value_name: str,
) -> torch.Tensor:
    cached_rows_list = [int(row) for row in cached_rows.detach().cpu().tolist()]
    live_rows_list = [int(row) for row in live_rows.detach().cpu().tolist()]
    row_to_pos = {row: pos for pos, row in enumerate(cached_rows_list)}
    missing = [row for row in live_rows_list if row not in row_to_pos]
    if missing:
        raise ValueError(
            f"{split_name}: {value_name} cache is missing live rows, first_missing={missing[:5]}"
        )
    positions = torch.as_tensor(
        [row_to_pos[row] for row in live_rows_list],
        dtype=torch.long,
        device=cached_values.device,
    )
    return cached_values.index_select(0, positions)


def _normalize_probs(probs: torch.Tensor) -> torch.Tensor:
    probs = probs.detach().cpu().to(dtype=torch.float32).clamp_min(1.0e-12)
    return probs / probs.sum(dim=1, keepdim=True).clamp_min(1.0e-12)


def _probability_diff_stats(
    cached_probs: torch.Tensor,
    live_probs: torch.Tensor,
) -> dict[str, float]:
    if tuple(cached_probs.shape) != tuple(live_probs.shape):
        raise ValueError(
            f"Probability shape mismatch cache={tuple(cached_probs.shape)} live={tuple(live_probs.shape)}"
        )
    cached_probs = _normalize_probs(cached_probs)
    live_probs = _normalize_probs(live_probs)
    abs_diff = (cached_probs - live_probs).abs()
    midpoint = 0.5 * (cached_probs + live_probs)
    js = 0.5 * (
        (cached_probs * (cached_probs / midpoint).log()).sum(dim=1)
        + (live_probs * (live_probs / midpoint).log()).sum(dim=1)
    )
    return {
        "mean_abs_diff": float(abs_diff.mean().item()),
        "max_abs_diff": float(abs_diff.max().item()),
        "mean_js_divergence": float(js.mean().item()),
        "max_js_divergence": float(js.max().item()),
    }


def _aggregate_head_diff_stats(
    head_stats: dict[str, dict[str, float]],
) -> dict[str, float]:
    if not head_stats:
        return {}
    keys = next(iter(head_stats.values())).keys()
    aggregated: dict[str, float] = {}
    for key in keys:
        values = [float(stats[key]) for stats in head_stats.values()]
        aggregated[f"d_prob_{key}_mean_over_heads"] = float(np.mean(values))
        aggregated[f"d_prob_{key}_max_over_heads"] = float(np.max(values))
    return aggregated


def _collect_live_prior_payload(
    *,
    wrapper: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    target_names: list[str],
) -> dict[str, Any]:
    final_d_probs: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    prior_los_probs_chunks: list[torch.Tensor] = []
    ctmp_los_probs_chunks: list[torch.Tensor] = []
    row_idx_chunks: list[torch.Tensor] = []

    wrapper.predictor.eval()
    backbone = _stage2_backbone(wrapper)
    with torch.no_grad():
        for x, _y, _d_targets, _los_target, _los_raw, row_idx in dataloader:
            x = x.to(device, non_blocking=True)
            ad_x = torch.index_select(x.long(), dim=1, index=wrapper.admission_idx_t)
            predictor_output = wrapper.predictor.forward_prior(ad_x, sample=False)
            for head_name in target_names:
                final_d_probs[head_name].append(
                    predictor_output.prior_d_probs[head_name].detach().cpu()
                )
            prior_los_probs = predictor_output.prior_los_probs.detach().cpu()
            ctmp_los_probs = prior_los_probs
            if ctmp_los_probs.shape[1] in {6, 9} and int(backbone.max_los) == 37:
                ctmp_los_probs = expand_coarse_distribution_to_raw_los(ctmp_los_probs)
            prior_los_probs_chunks.append(prior_los_probs)
            ctmp_los_probs_chunks.append(ctmp_los_probs.detach().cpu())
            row_idx_chunks.append(row_idx.detach().cpu().long())

    return {
        "row_idx": torch.cat(row_idx_chunks, dim=0),
        "final_d_probs": {
            name: torch.cat(values, dim=0) for name, values in final_d_probs.items()
        },
        "prior_los_probs": torch.cat(prior_los_probs_chunks, dim=0),
        "ctmp_los_probs": torch.cat(ctmp_los_probs_chunks, dim=0),
    }


def _compare_ctmp_cache_to_live_prior(
    *,
    split_name: str,
    base_dataset: TEDSTensorDataset,
    cache_payload: dict[str, Any],
    live_payload: dict[str, Any],
    target_names: list[str],
) -> dict[str, Any]:
    live_rows = live_payload["row_idx"]
    cached_rows = _rows_from_cached_ctmp_payload(base_dataset, cache_payload)
    head_stats: dict[str, dict[str, float]] = {}
    soft_discharge = cache_payload.get("soft_discharge")
    if soft_discharge is None:
        raise ValueError(f"{split_name}: cached CTMP payload has no soft_discharge block.")
    for head_name in target_names:
        cached_head = soft_discharge["heads"][head_name]["probs"]
        cached_head = _align_cached_to_live_rows(
            cached_values=cached_head,
            cached_rows=cached_rows,
            live_rows=live_rows,
            split_name=split_name,
            value_name=f"soft_discharge.{head_name}.probs",
        )
        head_stats[head_name] = _probability_diff_stats(
            cached_head,
            live_payload["final_d_probs"][head_name],
        )

    cached_los = _align_cached_to_live_rows(
        cached_values=cache_payload["los"],
        cached_rows=cached_rows,
        live_rows=live_rows,
        split_name=split_name,
        value_name="los",
    )
    los_stats = _probability_diff_stats(cached_los, live_payload["ctmp_los_probs"])
    return {
        "split": split_name,
        "cache_rows": int(cached_rows.numel()),
        "compared_rows": int(live_rows.numel()),
        "d_heads": head_stats,
        "d_aggregate": _aggregate_head_diff_stats(head_stats),
        "los": los_stats,
    }


def _compare_joint_cache_to_live_prior(
    *,
    split_name: str,
    cache_payload: dict[str, Any],
    live_payload: dict[str, Any],
    target_names: list[str],
) -> dict[str, Any]:
    live_rows = live_payload["row_idx"]
    cached_rows = torch.as_tensor(cache_payload["row_idx"], dtype=torch.long)
    head_stats: dict[str, dict[str, float]] = {}
    for head_name in target_names:
        cached_head = _align_cached_to_live_rows(
            cached_values=cache_payload["final_d_probs"][head_name],
            cached_rows=cached_rows,
            live_rows=live_rows,
            split_name=split_name,
            value_name=f"final_d_probs.{head_name}",
        )
        head_stats[head_name] = _probability_diff_stats(
            cached_head,
            live_payload["final_d_probs"][head_name],
        )

    cached_los = _align_cached_to_live_rows(
        cached_values=cache_payload["final_los_probs"],
        cached_rows=cached_rows,
        live_rows=live_rows,
        split_name=split_name,
        value_name="final_los_probs",
    )
    los_stats = _probability_diff_stats(cached_los, live_payload["prior_los_probs"])
    return {
        "split": split_name,
        "cache_rows": int(cached_rows.numel()),
        "compared_rows": int(live_rows.numel()),
        "d_heads": head_stats,
        "d_aggregate": _aggregate_head_diff_stats(head_stats),
        "los": los_stats,
    }


def _resolve_optional_cache_path(
    path_value: Any,
    *,
    fold_dir: str,
) -> str:
    path = str(path_value)
    if os.path.isabs(path):
        return path
    candidates = [
        path,
        os.path.join(fold_dir, path),
        os.path.join(fold_dir, "joint_predictor", path),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return os.path.join(fold_dir, path)


def _joint_cache_path_for_split(
    forecast_split_payload: dict[str, Any] | None,
    *,
    fold_dir: str,
    split_name: str,
) -> str | None:
    if not forecast_split_payload:
        return None
    cache_paths = forecast_split_payload.get("joint_cache_paths") or {}
    aliases = {
        "train_core": ("train",),
        "gnn_val": ("gnn_val", "val"),
        "outer_test": ("outer_test", "test"),
    }
    for key in aliases[split_name]:
        if key in cache_paths:
            path = _resolve_optional_cache_path(cache_paths[key], fold_dir=fold_dir)
            return path if os.path.exists(path) else None
    return None


def _flatten_preflight_for_metrics(
    diagnostics: dict[str, Any],
) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for split_name, split_payload in diagnostics.get("ctmp_cache_vs_live", {}).items():
        for key, value in split_payload.get("d_aggregate", {}).items():
            flattened[f"preflight_ctmp_{split_name}_{key}"] = float(value)
        for key, value in split_payload.get("los", {}).items():
            flattened[f"preflight_ctmp_{split_name}_los_{key}"] = float(value)
    for split_name, split_payload in diagnostics.get("joint_cache_vs_live", {}).items():
        for key, value in split_payload.get("d_aggregate", {}).items():
            flattened[f"preflight_joint_{split_name}_{key}"] = float(value)
        for key, value in split_payload.get("los", {}).items():
            flattened[f"preflight_joint_{split_name}_los_{key}"] = float(value)
    return flattened


def _run_stage2_preflight_diagnostics(
    *,
    wrapper: nn.Module,
    base_dataset: TEDSTensorDataset,
    artifact_dir: str,
    forecast_split_payload: dict[str, Any] | None,
    loaders: dict[str, DataLoader],
    edge_index: torch.Tensor,
    device: torch.device,
    target_names: list[str],
    baseline_metrics: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    epoch0_valid_metrics, _ = _evaluate_split(
        wrapper, loaders["gnn_val"], edge_index, device, target_names
    )
    epoch0_test_metrics, _ = _evaluate_split(
        wrapper, loaders["outer_test"], edge_index, device, target_names
    )

    live_payloads = {
        split_name: _collect_live_prior_payload(
            wrapper=wrapper,
            dataloader=loader,
            device=device,
            target_names=target_names,
        )
        for split_name, loader in loaders.items()
    }

    ctmp_cache_vs_live: dict[str, Any] = {}
    ctmp_cache_files = {
        "train_core": "train_core_joint.pt",
        "gnn_val": "gnn_val_joint.pt",
        "outer_test": "outer_test_joint.pt",
    }
    for split_name, file_name in ctmp_cache_files.items():
        path = os.path.join(artifact_dir, "cached_predictions", file_name)
        cache_payload = torch.load(path, map_location="cpu", weights_only=False)
        ctmp_cache_vs_live[split_name] = _compare_ctmp_cache_to_live_prior(
            split_name=split_name,
            base_dataset=base_dataset,
            cache_payload=cache_payload,
            live_payload=live_payloads[split_name],
            target_names=target_names,
        )

    joint_cache_vs_live: dict[str, Any] = {}
    for split_name in ("train_core", "gnn_val", "outer_test"):
        joint_cache_path = _joint_cache_path_for_split(
            forecast_split_payload,
            fold_dir=artifact_dir,
            split_name=split_name,
        )
        if joint_cache_path is None:
            continue
        cache_payload = torch.load(joint_cache_path, map_location="cpu", weights_only=False)
        joint_cache_vs_live[split_name] = _compare_joint_cache_to_live_prior(
            split_name=split_name,
            cache_payload=cache_payload,
            live_payload=live_payloads[split_name],
            target_names=target_names,
        )

    baseline_valid_auc = float(baseline_metrics["baseline_valid_auc"])
    baseline_test_auc = float(baseline_metrics["baseline_test_auc"])
    diagnostics = {
        "epoch0_live_eval": {
            "valid": epoch0_valid_metrics,
            "test": epoch0_test_metrics,
            "baseline_valid_auc": baseline_valid_auc,
            "baseline_test_auc": baseline_test_auc,
            "valid_auc_abs_diff_from_saved_cache_baseline": abs(
                float(epoch0_valid_metrics["auc"]) - baseline_valid_auc
            ),
            "test_auc_abs_diff_from_saved_cache_baseline": abs(
                float(epoch0_test_metrics["auc"]) - baseline_test_auc
            ),
        },
        "ctmp_cache_vs_live": ctmp_cache_vs_live,
        "joint_cache_vs_live": joint_cache_vs_live,
    }
    log_metrics = {
        "stage2_epoch0_valid_auc_before_update": float(epoch0_valid_metrics["auc"]),
        "stage2_epoch0_valid_acc_before_update": float(epoch0_valid_metrics["acc"]),
        "stage2_epoch0_valid_f1_before_update": float(epoch0_valid_metrics["f1"]),
        "stage2_epoch0_test_auc_before_update": float(epoch0_test_metrics["auc"]),
        "stage2_epoch0_test_acc_before_update": float(epoch0_test_metrics["acc"]),
        "stage2_epoch0_test_f1_before_update": float(epoch0_test_metrics["f1"]),
        "baseline_saved_cache_valid_auc": baseline_valid_auc,
        "baseline_saved_cache_test_auc": baseline_test_auc,
        "baseline_vs_stage2_epoch0_valid_auc_abs_diff": abs(
            float(epoch0_valid_metrics["auc"]) - baseline_valid_auc
        ),
        "baseline_vs_stage2_epoch0_test_auc_abs_diff": abs(
            float(epoch0_test_metrics["auc"]) - baseline_test_auc
        ),
    }
    log_metrics.update(_flatten_preflight_for_metrics(diagnostics))
    return diagnostics, log_metrics


def _fmt_metric(value: Any) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(value_float):
        return "nan"
    return f"{value_float:.4f}"


def _print_stage2_epoch0_metrics(metrics: dict[str, Any]) -> None:
    tqdm.write(
        "[stage2][epoch 0 preflight] "
        f"valid_auc={_fmt_metric(metrics['stage2_epoch0_valid_auc_before_update'])} "
        f"valid_acc={_fmt_metric(metrics['stage2_epoch0_valid_acc_before_update'])} "
        f"valid_f1={_fmt_metric(metrics['stage2_epoch0_valid_f1_before_update'])} "
        f"test_auc={_fmt_metric(metrics['stage2_epoch0_test_auc_before_update'])} "
        f"baseline_valid_auc={_fmt_metric(metrics['baseline_saved_cache_valid_auc'])} "
        f"valid_auc_diff={_fmt_metric(metrics['baseline_vs_stage2_epoch0_valid_auc_abs_diff'])}"
    )
    ctmp_valid_los = metrics.get("preflight_ctmp_gnn_val_los_mean_abs_diff")
    ctmp_valid_d = metrics.get(
        "preflight_ctmp_gnn_val_d_prob_mean_abs_diff_mean_over_heads"
    )
    if ctmp_valid_los is not None and ctmp_valid_d is not None:
        tqdm.write(
            "[stage2][cache vs live gnn_val] "
            f"d_prob_mae={_fmt_metric(ctmp_valid_d)} "
            f"los_prob_mae={_fmt_metric(ctmp_valid_los)}"
        )


def _print_stage2_epoch_metrics(
    *,
    epoch: int,
    max_epochs: int,
    metrics: dict[str, Any],
) -> None:
    tqdm.write(
        f"[stage2][epoch {epoch}/{max_epochs}] "
        f"train_outcome_loss={_fmt_metric(metrics['train_outcome_loss'])} "
        f"train_aux_loss={_fmt_metric(metrics['train_aux_loss'])} "
        f"train_total_loss={_fmt_metric(metrics['train_total_loss'])} "
        f"valid_auc={_fmt_metric(metrics['valid_auc'])} "
        f"valid_acc={_fmt_metric(metrics['valid_acc'])} "
        f"valid_f1={_fmt_metric(metrics['valid_f1'])} "
        f"valid_loss={_fmt_metric(metrics['valid_loss'])}"
    )


def _export_stage2_prior_cache(
    *,
    wrapper: nn.Module,
    dataloader: DataLoader,
    edge_index: torch.Tensor,
    device: torch.device,
    target_names: list[str],
    split_name: str,
    output_dir: str,
) -> str:
    del edge_index  # cache export is prior-only predictor output
    final_d_logits: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    final_d_probs: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    final_d_pred: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    los_logits_chunks: list[torch.Tensor] = []
    los_probs_chunks: list[torch.Tensor] = []
    los_pred_chunks: list[torch.Tensor] = []
    row_idx_chunks: list[torch.Tensor] = []
    target_d_chunks: dict[str, list[torch.Tensor]] = {name: [] for name in target_names}
    los_target_chunks: list[torch.Tensor] = []
    los_raw_chunks: list[torch.Tensor] = []

    wrapper.predictor.eval()
    with torch.no_grad():
        for x, _y, d_targets, los_target, los_raw, row_idx in dataloader:
            x = x.to(device, non_blocking=True)
            ad_x = torch.index_select(x.long(), dim=1, index=wrapper.admission_idx_t)
            predictor_output = wrapper.predictor.forward_prior(ad_x, sample=False)
            for idx, head_name in enumerate(target_names):
                logits = predictor_output.prior_d_logits[head_name].detach().cpu()
                probs = predictor_output.prior_d_probs[head_name].detach().cpu()
                final_d_logits[head_name].append(logits)
                final_d_probs[head_name].append(probs)
                final_d_pred[head_name].append(torch.argmax(probs, dim=1).long())
                target_d_chunks[head_name].append(d_targets[:, idx].long())
            los_logits_chunks.append(predictor_output.prior_los_logits.detach().cpu())
            los_probs_chunks.append(predictor_output.prior_los_probs.detach().cpu())
            los_pred_chunks.append(torch.argmax(predictor_output.prior_los_probs.detach().cpu(), dim=1).long())
            row_idx_chunks.append(row_idx.long())
            los_target_chunks.append(los_target.long())
            los_raw_chunks.append(los_raw.long())

    payload = {
        "split": split_name,
        "final_d_logits": {name: torch.cat(values, dim=0) for name, values in final_d_logits.items()},
        "final_d_probs": {name: torch.cat(values, dim=0) for name, values in final_d_probs.items()},
        "final_d_pred": {name: torch.cat(values, dim=0) for name, values in final_d_pred.items()},
        "final_los_logits": torch.cat(los_logits_chunks, dim=0),
        "final_los_probs": torch.cat(los_probs_chunks, dim=0),
        "final_los_pred": torch.cat(los_pred_chunks, dim=0),
        "row_idx": torch.cat(row_idx_chunks, dim=0),
        "targets": {
            "d": {name: torch.cat(values, dim=0) for name, values in target_d_chunks.items()},
            "los_target": torch.cat(los_target_chunks, dim=0),
            "los_raw": torch.cat(los_raw_chunks, dim=0),
        },
        "metadata": {
            "predictor_type": "joint_generative",
            "final_los_pred_space": "coarse_class",
            "target_col_names": list(target_names),
        },
    }
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{split_name}.pt")
    torch.save(payload, path)
    return path


def run_outcome_aware_stage2(
    *,
    cfg: dict[str, Any],
    root: str,
    fold_dir: str,
    base_dataset: TEDSTensorDataset,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    edge_index: torch.Tensor,
    ctmp_gin_model: nn.Module,
    predictor_checkpoint_path: str,
    baseline_checkpoint_path: str,
    baseline_metrics: dict[str, Any],
    forecast_split_payload: dict[str, Any] | None,
    device: torch.device,
    source_artifact_dir: str | None = None,
) -> dict[str, Any]:
    stage2_cfg = copy.deepcopy(cfg["joint_forecast_pipeline"]["stage2"])
    stage2_dir = os.path.join(fold_dir, "outcome_aware_stage2")
    os.makedirs(stage2_dir, exist_ok=False)
    os.makedirs(os.path.join(stage2_dir, "checkpoints"), exist_ok=True)
    artifact_dir = str(source_artifact_dir or fold_dir)

    if str(stage2_cfg.get("mode", "outcome_aware")).lower() != "outcome_aware":
        raise ValueError("Unsupported stage2 mode.")
    model_name = str(cfg["model"]["name"])
    if model_name not in {"ctmp_gin", "gin"}:
        raise ValueError("Outcome-aware stage2 currently supports only CTMP-GIN and GIN.")
    if str(cfg["joint_forecast_pipeline"]["joint_forecast_input"]["mode"]).lower() != "distribution":
        raise ValueError("Outcome-aware stage2 requires joint_forecast_input.mode=distribution.")
    if str(stage2_cfg.get("selection_metric", "valid_auc")).lower() != "valid_auc":
        raise ValueError("Outcome-aware stage2 checkpoint selection must use valid_auc.")

    joint_dataset = JointPredictionDataset(
        root=root,
        do_preprocess=bool(cfg["train"].get("do_preprocess", False)),
        los_target_mode=str(cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"].get("los_target_mode", "coarse")),
    )
    predictor = _load_stage1_predictor(predictor_checkpoint_path, joint_dataset, device)
    contract = resolve_joint_forecast_contract(
        base_dataset.col_info,
        list(joint_dataset.target_col_names),
    )
    if contract.head_names != CANONICAL_JOINT_FORECAST_HEADS:
        raise ValueError(
            "Outcome-aware stage2 requires the exact existing 12-head joint forecast contract."
        )
    cached_soft_path = os.path.join(artifact_dir, "cached_predictions", "train_core_joint.pt")
    if os.path.exists(cached_soft_path):
        cached_payload = torch.load(cached_soft_path, map_location="cpu", weights_only=False)
        cached_soft_discharge = cached_payload.get("soft_discharge")
        if cached_soft_discharge is not None:
            assert_soft_discharge_contract_matches_cached_metadata(
                contract,
                cached_soft_discharge,
            )

    predictor_admission_col_indices = _resolve_predictor_admission_col_indices(
        base_dataset,
        joint_dataset,
    )
    if model_name == "ctmp_gin":
        wrapper = OutcomeAwareForecastedCTMPGIN(
            predictor=predictor,
            ctmp_gin=ctmp_gin_model,
            contract=contract,
            admission_col_indices=predictor_admission_col_indices,
            discharge_col_indices=list(base_dataset.col_info[3]),
            sample_prior_in_train=bool(stage2_cfg.get("sample_prior_in_train", False)),
            discharge_placeholder_index=0,
        ).to(device)
    else:
        wrapper = OutcomeAwareForecastedGIN(
            predictor=predictor,
            gin=ctmp_gin_model,
            contract=contract,
            admission_col_indices=predictor_admission_col_indices,
            discharge_col_indices=list(base_dataset.col_info[3]),
            sample_prior_in_train=bool(stage2_cfg.get("sample_prior_in_train", False)),
            discharge_placeholder_index=0,
        ).to(device)
    _apply_ctmp_gin_freeze_policy(wrapper, stage2_cfg)

    predictor_trainable = _count_trainable_params(wrapper.predictor)
    backbone = _stage2_backbone(wrapper)
    ctmp_trainable = _count_trainable_params(backbone)
    backbone_label = model_name
    print(
        f"[stage2] trainable_params predictor={predictor_trainable:,} {backbone_label}={ctmp_trainable:,} total={predictor_trainable + ctmp_trainable:,}"
    )

    dataset = OutcomeAwareStage2Dataset(base_dataset, joint_dataset)
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    train_loader = _make_loader(dataset, train_idx, batch_size, num_workers, shuffle=True)
    val_loader = _make_loader(dataset, val_idx, batch_size, num_workers, shuffle=False)
    test_loader = _make_loader(dataset, test_idx, batch_size, num_workers, shuffle=False)
    eval_loaders = {
        "train_core": _make_loader(
            dataset,
            train_idx,
            batch_size,
            num_workers,
            shuffle=False,
        ),
        "gnn_val": val_loader,
        "outer_test": test_loader,
    }

    optimizer = _build_optimizer(wrapper, stage2_cfg)
    early_stopper = EarlyStopper(patience=int(stage2_cfg.get("early_stopping_patience", 5)))
    logger_cfg = copy.deepcopy(cfg)
    logger_cfg.setdefault("train", {})
    logger_cfg["train"]["monitor_metric"] = "valid_auc"
    logger_cfg["train"]["monitor_mode"] = "max"
    logger = ExperimentLogger(logger_cfg, stage2_dir)
    print("[stage2] running epoch-0 cache/live preflight diagnostics")
    preflight_diagnostics, preflight_metrics = _run_stage2_preflight_diagnostics(
        wrapper=wrapper,
        base_dataset=base_dataset,
        artifact_dir=artifact_dir,
        forecast_split_payload=forecast_split_payload,
        loaders=eval_loaders,
        edge_index=edge_index,
        device=device,
        target_names=joint_dataset.target_col_names,
        baseline_metrics=baseline_metrics,
    )
    preflight_path = os.path.join(stage2_dir, "outcome_aware_stage2_preflight.json")
    _save_json(preflight_path, preflight_diagnostics)
    epoch0_metrics = {
        "trainable_predictor_params": int(predictor_trainable),
        "trainable_ctmp_gin_params": int(ctmp_trainable),
        "trainable_total_params": int(predictor_trainable + ctmp_trainable),
        **preflight_metrics,
    }
    logger.log_metrics(0, epoch0_metrics)
    _print_stage2_epoch0_metrics(epoch0_metrics)
    criterion = nn.BCEWithLogitsLoss()
    lambda_aux = float(stage2_cfg.get("lambda_aux", 0.0))

    best_valid_auc = -float("inf")
    best_epoch = 0
    max_epochs = int(stage2_cfg["max_epochs"])
    for epoch in tqdm(range(1, max_epochs + 1), desc="outcome-aware stage2"):
        _set_stage2_module_modes(wrapper, stage2_cfg, is_training=True)
        total_outcome = 0.0
        total_aux = 0.0
        total_loss = 0.0
        n_batches = 0
        train_logits_chunks: list[np.ndarray] = []
        train_target_chunks: list[np.ndarray] = []
        diag_sums: dict[str, float] = {}
        for x, y, d_targets, los_target, _los_raw, _row_idx in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            d_targets = d_targets.to(device, non_blocking=True)
            los_target = los_target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            output = wrapper(x, edge_index)
            logits = output.reasonb_logits.squeeze(1)
            outcome_loss = criterion(logits, y.float())
            aux_loss = _compute_aux_loss(output, d_targets, los_target, joint_dataset.target_col_names)
            loss = outcome_loss + lambda_aux * aux_loss
            loss.backward()
            optimizer.step()
            total_outcome += float(outcome_loss.detach().cpu())
            total_aux += float(aux_loss.detach().cpu())
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            train_logits_chunks.append(logits.detach().cpu().numpy())
            train_target_chunks.append(y.detach().cpu().numpy())
            for key, value in output.diagnostics.items():
                diag_sums[key] = diag_sums.get(key, 0.0) + float(value)

        train_diag = {key: value / max(n_batches, 1) for key, value in diag_sums.items()}
        train_metrics = _compute_binary_metrics(
            loss_mean=total_loss / max(n_batches, 1),
            logits_chunks=train_logits_chunks,
            target_chunks=train_target_chunks,
            aux_loss_mean=total_aux / max(n_batches, 1),
            outcome_loss_mean=total_outcome / max(n_batches, 1),
            diagnostics=train_diag,
        )
        _set_stage2_module_modes(wrapper, stage2_cfg, is_training=False)
        valid_metrics, _ = _evaluate_split(
            wrapper, val_loader, edge_index, device, joint_dataset.target_col_names
        )
        log_metrics = {
            "train_outcome_loss": float(train_metrics["outcome_loss"]),
            "train_aux_loss": float(train_metrics["aux_loss"]),
            "train_total_loss": float(train_metrics["loss"]),
            "valid_loss": float(valid_metrics["loss"]),
            "valid_acc": float(valid_metrics["acc"]),
            "valid_precision": float(valid_metrics["precision"]),
            "valid_recall": float(valid_metrics["recall"]),
            "valid_f1": float(valid_metrics["f1"]),
            "valid_auc": float(valid_metrics["auc"]),
            "train_predicted_label_counts": train_metrics["predicted_label_counts"],
            "train_true_label_counts": train_metrics["true_label_counts"],
            "valid_predicted_label_counts": valid_metrics["predicted_label_counts"],
            "valid_true_label_counts": valid_metrics["true_label_counts"],
        }
        for prefix, metrics in (("train", train_metrics), ("valid", valid_metrics)):
            for key in ("d_entropy_mean", "los_entropy_mean", "mu_p_mean", "mu_p_std", "logstd_p_mean", "logstd_p_std"):
                if key in metrics:
                    log_metrics[f"{prefix}_{key}"] = float(metrics[key])
        logger.log_metrics(epoch, log_metrics)
        _print_stage2_epoch_metrics(
            epoch=epoch,
            max_epochs=max_epochs,
            metrics=log_metrics,
        )
        logger.maybe_save_checkpoint(
            epoch=epoch,
            model=wrapper,
            optimizer=optimizer,
            scheduler=None,
            metrics=log_metrics,
            extra={
                "predictor_state_dict": wrapper.predictor.state_dict(),
                "backbone_state_dict": backbone.state_dict(),
                "stage1_checkpoint_path": predictor_checkpoint_path,
                "baseline_checkpoint_path": baseline_checkpoint_path,
                "freeze_settings": stage2_cfg,
                "forecast_split_payload": forecast_split_payload,
            },
        )
        if float(valid_metrics["auc"]) > best_valid_auc:
            best_valid_auc = float(valid_metrics["auc"])
            best_epoch = epoch
        if early_stopper(-float(valid_metrics["auc"])):
            break

    best_ckpt_path = os.path.join(stage2_dir, "checkpoints", "best.pt")
    if os.path.exists(best_ckpt_path):
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        wrapper.load_state_dict(best_ckpt["model_state_dict"], strict=True)

    _set_stage2_module_modes(wrapper, stage2_cfg, is_training=False)
    valid_metrics, valid_payload = _evaluate_split(
        wrapper, val_loader, edge_index, device, joint_dataset.target_col_names
    )
    test_metrics, test_payload = _evaluate_split(
        wrapper, test_loader, edge_index, device, joint_dataset.target_col_names
    )

    cache_dir = os.path.join(stage2_dir, "joint_cache")
    train_cache_path = _export_stage2_prior_cache(
        wrapper=wrapper,
        dataloader=train_loader,
        edge_index=edge_index,
        device=device,
        target_names=joint_dataset.target_col_names,
        split_name="train",
        output_dir=cache_dir,
    )
    valid_cache_path = _export_stage2_prior_cache(
        wrapper=wrapper,
        dataloader=val_loader,
        edge_index=edge_index,
        device=device,
        target_names=joint_dataset.target_col_names,
        split_name="gnn_val",
        output_dir=cache_dir,
    )
    test_cache_path = _export_stage2_prior_cache(
        wrapper=wrapper,
        dataloader=test_loader,
        edge_index=edge_index,
        device=device,
        target_names=joint_dataset.target_col_names,
        split_name="outer_test",
        output_dir=cache_dir,
    )
    drift_payload = _generate_joint_drift_reports(
        run_dir=stage2_dir,
        train_cache_path=train_cache_path,
        eval_cache_paths={
            "gnn_val": valid_cache_path,
            "outer_test": test_cache_path,
        },
    )

    summary = {
        "baseline_valid_auc": float(baseline_metrics["baseline_valid_auc"]),
        "baseline_test_auc": float(baseline_metrics["baseline_test_auc"]),
        "stage2_valid_auc": float(valid_metrics["auc"]),
        "stage2_test_auc": float(test_metrics["auc"]),
        "baseline_valid_f1": float(baseline_metrics.get("baseline_valid_f1", float("nan"))),
        "baseline_test_f1": float(baseline_metrics.get("baseline_test_f1", float("nan"))),
        "stage2_valid_f1": float(valid_metrics["f1"]),
        "stage2_test_f1": float(test_metrics["f1"]),
        "baseline_valid_acc": float(baseline_metrics.get("baseline_valid_acc", float("nan"))),
        "baseline_test_acc": float(baseline_metrics.get("baseline_test_acc", float("nan"))),
        "stage2_valid_acc": float(valid_metrics["acc"]),
        "stage2_test_acc": float(test_metrics["acc"]),
        "train_cache_path": train_cache_path,
        "valid_cache_path": valid_cache_path,
        "test_cache_path": test_cache_path,
        "preflight_diagnostics_path": preflight_path,
        "preflight_epoch0": preflight_diagnostics["epoch0_live_eval"],
        "joint_drift_summary": drift_payload,
        "best_epoch": int(best_epoch),
    }
    _save_json(os.path.join(stage2_dir, "outcome_aware_stage2_summary.json"), summary)

    return {
        "best_epoch": int(best_epoch),
        "best_valid_metric": float(best_valid_auc),
        "best_valid_metrics": {
            "valid_auc": float(valid_metrics["auc"]),
            "valid_f1": float(valid_metrics["f1"]),
            "valid_acc": float(valid_metrics["acc"]),
        },
        "test_loss": float(test_metrics["loss"]),
        "test_acc": float(test_metrics["acc"]),
        "test_precision": float(test_metrics["precision"]),
        "test_recall": float(test_metrics["recall"]),
        "test_f1": float(test_metrics["f1"]),
        "test_auc": float(test_metrics["auc"]),
        "baseline_valid_auc": float(baseline_metrics["baseline_valid_auc"]),
        "baseline_test_auc": float(baseline_metrics["baseline_test_auc"]),
        "stage2_valid_auc": float(valid_metrics["auc"]),
        "stage2_test_auc": float(test_metrics["auc"]),
        "baseline_valid_f1": float(baseline_metrics.get("baseline_valid_f1", float("nan"))),
        "baseline_test_f1": float(baseline_metrics.get("baseline_test_f1", float("nan"))),
        "stage2_valid_f1": float(valid_metrics["f1"]),
        "stage2_test_f1": float(test_metrics["f1"]),
        "baseline_valid_acc": float(baseline_metrics.get("baseline_valid_acc", float("nan"))),
        "baseline_test_acc": float(baseline_metrics.get("baseline_test_acc", float("nan"))),
        "stage2_valid_acc": float(valid_metrics["acc"]),
        "stage2_test_acc": float(test_metrics["acc"]),
        "run_dir": stage2_dir,
        "valid_row_idx": valid_payload["row_idx"],
        "test_row_idx": test_payload["row_idx"],
    }
