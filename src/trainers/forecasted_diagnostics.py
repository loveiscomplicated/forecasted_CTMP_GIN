from __future__ import annotations

import copy
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.model_selection import StratifiedKFold
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data_processing.discharge_prediction_dataset import DischargePredictionDataset
from src.data_processing.los_prediction_dataset import LOSPredictionDataset
from src.data_processing.splits import holdout_test_split_stratified, kfold_stratified
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.discharge_predictor.los_utils import (
    LOS_COARSE_BINS,
    LOS_COARSE_BIN_REPRESENTATIVES,
    expand_coarse_distribution_to_raw_los,
    map_coarse_array_to_raw_los,
    map_los_array_to_coarse_bins,
)
from src.models.factory import build_edge, build_model
from src.trainers.forecasted_discharge import ForecastedDischargeProvider
from src.trainers.forecasted_los import ForecastedLOSProvider
from src.trainers.forecasted_pipeline import (
    ForecastCacheDataset,
    _build_provider_cfg,
    _inherit_dataset_settings,
    _train_discharge_predictor,
    _train_los_predictor,
    _make_loader,
    split_outer_train_for_forecasted_pipeline,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.experiment import ExperimentLogger, ensure_run_dir, make_run_id, save_yaml
from src.utils.seed_set import set_seed


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _base_x_tensor(base_dataset) -> torch.Tensor:
    return base_dataset.processed_tensor[:, :-1].long()


def _base_y_tensor(base_dataset) -> torch.Tensor:
    return base_dataset.processed_tensor[:, -1].long()


def _base_los_tensor(base_dataset) -> torch.Tensor:
    return base_dataset.LOS.long()


def _base_x_feature_names(base_dataset) -> list[str]:
    col_list = list(base_dataset.col_info[0])
    return [str(name) for name in col_list if str(name) != "LOS"]


def _base_x_name_to_pos(base_dataset) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(_base_x_feature_names(base_dataset))}


def _select_base_x_by_names(base_dataset, feature_names: list[str]) -> torch.Tensor:
    base_x = _base_x_tensor(base_dataset)
    name_to_pos = _base_x_name_to_pos(base_dataset)
    missing = [name for name in feature_names if name not in name_to_pos]
    if missing:
        raise RuntimeError(
            "Base dataset x tensor is missing expected feature columns: "
            + ", ".join(map(str, missing[:10]))
        )
    index = torch.tensor([name_to_pos[name] for name in feature_names], dtype=torch.long)
    return base_x[:, index]


def _as_index_array(indices: np.ndarray | list[int] | None, n: int) -> np.ndarray:
    if indices is None:
        return np.arange(n, dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _first_mismatch(lhs: torch.Tensor, rhs: torch.Tensor) -> tuple[int, Any, Any] | None:
    lhs_cpu = lhs.detach().cpu()
    rhs_cpu = rhs.detach().cpu()
    neq = lhs_cpu != rhs_cpu
    if neq.ndim == 0:
        return None if not bool(neq.item()) else (0, lhs_cpu.item(), rhs_cpu.item())
    flat = torch.nonzero(neq.reshape(-1), as_tuple=False)
    if flat.numel() == 0:
        return None
    flat_idx = int(flat[0, 0])
    return flat_idx, lhs_cpu.reshape(-1)[flat_idx].item(), rhs_cpu.reshape(-1)[flat_idx].item()


def _exact_match_rate(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    if lhs.numel() == 0:
        return 1.0
    return float((lhs.detach().cpu() == rhs.detach().cpu()).float().mean().item())


def _raise_value_mismatch(name: str, lhs: torch.Tensor, rhs: torch.Tensor, message: str) -> None:
    mismatch = _first_mismatch(lhs, rhs)
    detail = ""
    if mismatch is not None:
        flat_idx, lhs_value, rhs_value = mismatch
        detail = f" first_flat_mismatch={flat_idx} lhs={lhs_value} rhs={rhs_value}"
    raise RuntimeError(f"{message} for {name}.{detail}")


def _coarse_to_raw_mapping() -> dict[int, int]:
    return {i: int(v) for i, v in enumerate(LOS_COARSE_BIN_REPRESENTATIVES)}


def _raw_to_coarse_table() -> dict[str, int]:
    raw = torch.arange(1, 38, dtype=torch.long)
    coarse = map_los_array_to_coarse_bins(raw).long()
    return {str(int(r)): int(c) for r, c in zip(raw.tolist(), coarse.tolist())}


def _los_mapping_payload() -> dict[str, Any]:
    return {
        "coarse_to_raw": _coarse_to_raw_mapping(),
        "raw_to_coarse": _raw_to_coarse_table(),
        "coarse_bins": [list(pair) for pair in LOS_COARSE_BINS],
    }


def _assert_raw_los_valid(raw_los: torch.Tensor, context: str) -> None:
    raw_los = raw_los.long()
    if raw_los.numel() == 0:
        return
    if int(raw_los.min().item()) < 1 or int(raw_los.max().item()) > 37:
        raise RuntimeError(
            f"{context}: LOS raw values must be in 1..37, got "
            f"min={int(raw_los.min())} max={int(raw_los.max())}."
        )
    if torch.any(raw_los == 0):
        raise RuntimeError(f"{context}: LOS 0 is not a valid raw LOS code.")


def build_oracle_forecast_cache(
    base_dataset,
    indices: np.ndarray | list[int] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a ForecastCacheDataset-compatible cache from oracle CTMP-GIN inputs."""
    n = len(base_dataset)
    idx = _as_index_array(indices, n)
    x_source = _base_x_tensor(base_dataset)
    los_source = _base_los_tensor(base_dataset)
    x_cache = x_source.clone()
    los_cache = torch.zeros((n,), dtype=torch.long)
    los_cache[idx] = los_source[idx].cpu()

    for row in idx[: min(len(idx), 1024)]:
        row_i = int(row)
        x_item, _, los_item = base_dataset[row_i]
        if not torch.equal(x_cache[row_i], x_item.long().cpu()):
            _raise_value_mismatch(
                str(row_i),
                x_cache[row_i],
                x_item.long().cpu(),
                "Oracle x_cache does not match base_dataset x",
            )
        if int(los_cache[row_i].item()) != int(los_item):
            raise RuntimeError(
                f"Oracle los_cache does not match base_dataset LOS at row={row_i}: "
                f"cache={int(los_cache[row_i])} base={int(los_item)}"
            )
    return x_cache.to(device), los_cache.to(device)


def _discharge_mapping(base_dataset, discharge_dataset: DischargePredictionDataset) -> list[tuple[str, int, int]]:
    col_list, _, _, dis_col_index = base_dataset.col_info
    discharge_names = [str(col_list[idx]) for idx in dis_col_index if idx is not None]
    base_x_name_to_pos = _base_x_name_to_pos(base_dataset)
    discharge_name_to_x_pos = {
        name: int(base_x_name_to_pos[name]) for name in discharge_names if name in base_x_name_to_pos
    }
    mapping: list[tuple[str, int, int]] = []
    for target_pos, target_name in enumerate(discharge_dataset.target_col_names):
        if target_name not in discharge_name_to_x_pos:
            raise RuntimeError(
                f"Discharge target {target_name} is absent from CTMP-GIN x tensor columns."
            )
        mapping.append((str(target_name), int(target_pos), int(discharge_name_to_x_pos[target_name])))
    return mapping


def _los_predictor_target(los_dataset: LOSPredictionDataset, los_cfg: dict[str, Any] | None) -> torch.Tensor:
    mode = str((los_cfg or {}).get("los_target_mode", (los_cfg or {}).get("target_mode", "fine"))).lower()
    if mode == "coarse":
        return map_los_array_to_coarse_bins(los_dataset.los_raw).long()
    return los_dataset.y.long()


def build_predictor_target_forecast_cache(
    base_dataset,
    discharge_dataset: DischargePredictionDataset,
    los_dataset: LOSPredictionDataset,
    indices: np.ndarray | list[int] | None = None,
    device: str | torch.device = "cpu",
    los_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build cache from predictor training targets and fail on value-space mismatch."""
    n = len(base_dataset)
    idx = _as_index_array(indices, n)
    x_cache = _base_x_tensor(base_dataset).clone()
    base_x = _base_x_tensor(base_dataset)
    mapping = _discharge_mapping(base_dataset, discharge_dataset)

    for target_name, target_pos, ctmp_col_idx in mapping:
        target_values = discharge_dataset.y[:, target_pos].long()
        ctmp_values = base_x[:, ctmp_col_idx].long()
        rate = _exact_match_rate(target_values[idx], ctmp_values[idx])
        if rate < 1.0:
            _raise_value_mismatch(
                target_name,
                target_values[idx],
                ctmp_values[idx],
                "Value-space mismatch: predictor target encoding does not match CTMP-GIN input encoding",
            )
        x_cache[idx, ctmp_col_idx] = target_values[idx].cpu()

    los_target = _los_predictor_target(los_dataset, los_cfg)
    base_los = _base_los_tensor(base_dataset)
    los_rate = _exact_match_rate(los_target[idx], base_los[idx])
    if los_rate < 1.0:
        _raise_value_mismatch(
            "LOS",
            los_target[idx],
            base_los[idx],
            "Value-space mismatch: LOS predictor target encoding does not match CTMP-GIN LOS input encoding",
        )

    los_cache = torch.zeros((n,), dtype=torch.long)
    los_cache[idx] = los_target[idx].cpu()
    return x_cache.to(device), los_cache.to(device)


def build_predictor_target_transformed_forecast_cache(
    base_dataset,
    discharge_dataset: DischargePredictionDataset,
    los_dataset: LOSPredictionDataset,
    indices: np.ndarray | list[int] | None = None,
    device: str | torch.device = "cpu",
    los_cfg: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build cache from predictor targets, converting coarse LOS targets to raw representative LOS."""
    n = len(base_dataset)
    idx = _as_index_array(indices, n)
    x_cache = _base_x_tensor(base_dataset).clone()
    base_x = _base_x_tensor(base_dataset)
    mapping = _discharge_mapping(base_dataset, discharge_dataset)

    for target_name, target_pos, ctmp_col_idx in mapping:
        target_values = discharge_dataset.y[:, target_pos].long()
        ctmp_values = base_x[:, ctmp_col_idx].long()
        rate = _exact_match_rate(target_values[idx], ctmp_values[idx])
        if rate < 1.0:
            _raise_value_mismatch(
                target_name,
                target_values[idx],
                ctmp_values[idx],
                "Value-space mismatch: predictor target encoding does not match CTMP-GIN input encoding",
            )
        x_cache[idx, ctmp_col_idx] = target_values[idx].cpu()

    mode = str((los_cfg or {}).get("los_target_mode", (los_cfg or {}).get("target_mode", "fine"))).lower()
    los_target = _los_predictor_target(los_dataset, los_cfg).long()
    transformed_raw = map_coarse_array_to_raw_los(los_target).long() if mode == "coarse" else los_target.long()
    _assert_raw_los_valid(transformed_raw[idx], "predictor_target_cache_transformed")

    los_cache = torch.zeros((n,), dtype=torch.long)
    los_cache[idx] = transformed_raw[idx].cpu()
    payload = {
        **_los_mapping_payload(),
        "los_target_mode": mode,
        "coarse_min": int(los_target[idx].min().item()),
        "coarse_max": int(los_target[idx].max().item()),
        "transformed_raw_min": int(transformed_raw[idx].min().item()),
        "transformed_raw_max": int(transformed_raw[idx].max().item()),
        "uses_los_zero": bool(torch.any(transformed_raw[idx] == 0).item()),
        "los_cache_dtype": str(los_cache.dtype),
    }
    return x_cache.to(device), los_cache.to(device), payload


def build_oracle_coarse_los_forecast_cache(
    base_dataset,
    indices: np.ndarray | list[int] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build oracle cache while compressing true raw LOS through coarse representatives."""
    n = len(base_dataset)
    idx = _as_index_array(indices, n)
    x_cache = _base_x_tensor(base_dataset).clone()
    true_raw = _base_los_tensor(base_dataset).long()
    coarse = map_los_array_to_coarse_bins(true_raw).long()
    repr_raw = map_coarse_array_to_raw_los(coarse).long()
    _assert_raw_los_valid(repr_raw[idx], "oracle_cache_coarse_los")

    los_cache = torch.zeros((n,), dtype=torch.long)
    los_cache[idx] = repr_raw[idx].cpu()
    abs_err = torch.abs(true_raw[idx].cpu() - repr_raw[idx].cpu())
    true_counts = torch.bincount(true_raw[idx].cpu(), minlength=38)[1:].tolist()
    coarse_counts = torch.bincount(coarse[idx].cpu(), minlength=6).tolist()
    repr_counts_raw = torch.bincount(repr_raw[idx].cpu(), minlength=38)
    repr_counts = {str(i): int(v) for i, v in enumerate(repr_counts_raw.tolist()) if i > 0 and v}
    confusion_summary = {}
    for raw in range(1, 38):
        raw_t = torch.tensor([raw], dtype=torch.long)
        coarse_value = int(map_los_array_to_coarse_bins(raw_t)[0])
        repr_value = int(map_coarse_array_to_raw_los(torch.tensor([coarse_value]))[0])
        confusion_summary[f"raw_{raw}"] = {
            "coarse": coarse_value,
            "representative_raw": repr_value,
        }
    payload = {
        **_los_mapping_payload(),
        "true_raw_min": int(true_raw[idx].min().item()),
        "true_raw_max": int(true_raw[idx].max().item()),
        "coarse_min": int(coarse[idx].min().item()),
        "coarse_max": int(coarse[idx].max().item()),
        "repr_raw_min": int(repr_raw[idx].min().item()),
        "repr_raw_max": int(repr_raw[idx].max().item()),
        "los_zero_used": bool(torch.any(repr_raw[idx] == 0).item()),
        "coarse_compression_mae": float(abs_err.float().mean().item()),
        "coarse_compression_within_1": float((abs_err <= 1).float().mean().item()),
        "coarse_compression_within_2": float((abs_err <= 2).float().mean().item()),
        "true_raw_distribution": {str(i + 1): int(v) for i, v in enumerate(true_counts) if v},
        "coarse_distribution": {str(i): int(v) for i, v in enumerate(coarse_counts) if v},
        "representative_raw_distribution": repr_counts,
        "confusion_summary": confusion_summary,
    }
    return x_cache.to(device), los_cache.to(device), payload


def _labels_for_indices(dataset, indices: np.ndarray) -> np.ndarray:
    return np.asarray([int(dataset[int(i)][1]) for i in indices], dtype=np.int64)


def _load_child_predictor_cfgs(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    pipeline_cfg = cfg.get("forecasted_pipeline", {})
    discharge_cfg = _load_yaml(str(pipeline_cfg.get("discharge_config_path", "configs/discharge_predictor.yaml")))
    los_cfg = _load_yaml(str(pipeline_cfg.get("los_config_path", "configs/los_ce_predictor.yaml")))
    _inherit_dataset_settings(cfg, discharge_cfg)
    _inherit_dataset_settings(cfg, los_cfg)
    discharge_cfg["device"] = cfg.get("device")
    los_cfg["device"] = cfg.get("device")
    return discharge_cfg, los_cfg


def _normalized_mode(mode: str) -> str:
    aliases = {
        "oracle_D_predicted_LOS_hard": "oracle_d_predicted_los_hard",
        "oracle_D_predicted_LOS_distribution": "oracle_d_predicted_los_distribution",
        "predicted_D_oracle_LOS": "predicted_d_oracle_los",
        "predicted_D_predicted_LOS": "predicted_d_predicted_los",
        "predicted_D_predicted_LOS_oracle_head_ablation": "predicted_d_predicted_los_oracle_head_ablation",
        "oracle_D_predicted_LOS_predicted_head_ablation": "oracle_d_predicted_los_predicted_head_ablation",
        "oracle_d_predicted_los": "oracle_d_predicted_los_hard",
    }
    return aliases.get(mode, mode)


def _los_return_type_for_mode(cfg: dict[str, Any], mode: str) -> str:
    if mode == "oracle_d_predicted_los_hard":
        return "hard"
    if mode == "oracle_d_predicted_los_distribution":
        return "distribution"
    return str(cfg.get("forecasted_los", {}).get("return_type", "hard")).lower()


def _provider_cfg_with_los_return_type(
    cfg: dict[str, Any],
    checkpoint_path: str,
    calibration_path: str | None,
    return_type: str,
) -> dict[str, Any]:
    forecast_cfg = _build_provider_cfg(cfg.get("forecasted_los", {}), checkpoint_path, calibration_path)
    forecast_cfg["return_type"] = return_type
    return forecast_cfg


def _forecast_discharge_only_into_cache(
    base_dataset,
    indices: np.ndarray,
    discharge_provider: ForecastedDischargeProvider,
    device: torch.device,
    batch_size: int,
    x_cache: torch.Tensor,
) -> None:
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.stack([base_dataset[int(i)][0] for i in chunk], dim=0).to(device)
        x_pred = discharge_provider(x)
        x_cache[chunk] = x_pred.cpu()


def _forecast_los_only_into_cache(
    base_dataset,
    indices: np.ndarray,
    los_provider: ForecastedLOSProvider,
    device: torch.device,
    batch_size: int,
    los_cache: torch.Tensor,
) -> None:
    indices = np.asarray(indices, dtype=np.int64)
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.stack([base_dataset[int(i)][0] for i in chunk], dim=0).to(device)
        los_pred = los_provider(x)
        los_cache[chunk] = los_pred.cpu()


def _train_oof_discharge_predictions(
    data: "DiagnosticData",
    root: str,
    run_dir: str,
    device: torch.device,
    discharge_cfg: dict[str, Any],
    fixed_epochs: int,
    x_cache: torch.Tensor,
) -> None:
    pipeline_cfg = data.cfg.get("forecasted_pipeline", {})
    n_inner = int(pipeline_cfg.get("oof", {}).get("n_inner_folds", 5))
    labels = _labels_for_indices(data.base_dataset, data.train_idx)
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=int(data.cfg["train"].get("seed", 42)))
    batch_size = int(data.cfg["train"]["batch_size"])
    for inner_fold, (inner_train_pos, inner_holdout_pos) in enumerate(skf.split(np.zeros(len(data.train_idx)), labels)):
        inner_train_idx = data.train_idx[inner_train_pos]
        inner_holdout_idx = data.train_idx[inner_holdout_pos]
        inner_dir = os.path.join(run_dir, "predictors", "oof", f"inner_{inner_fold}", "discharge")
        print(
            f"[mixed OOF discharge {inner_fold}/{n_inner - 1}] "
            f"train={len(inner_train_idx)} holdout={len(inner_holdout_idx)} fixed_epochs={fixed_epochs}"
        )
        result = _train_discharge_predictor(
            discharge_cfg,
            root,
            inner_train_idx,
            None,
            inner_dir,
            device,
            fixed_epochs=fixed_epochs,
            role=f"mixed OOF discharge predictor {inner_fold}",
            verbose=False,
        )
        provider = ForecastedDischargeProvider(
            _build_provider_cfg(data.cfg.get("forecasted_discharge", {}), result["checkpoint_path"]),
            data.base_dataset,
            device,
        )
        _forecast_discharge_only_into_cache(
            data.base_dataset, inner_holdout_idx, provider, device, batch_size, x_cache
        )


def _train_oof_los_predictions(
    data: "DiagnosticData",
    root: str,
    run_dir: str,
    device: torch.device,
    los_cfg: dict[str, Any],
    fixed_epochs: int,
    los_cache: torch.Tensor,
) -> None:
    pipeline_cfg = data.cfg.get("forecasted_pipeline", {})
    n_inner = int(pipeline_cfg.get("oof", {}).get("n_inner_folds", 5))
    labels = _labels_for_indices(data.base_dataset, data.train_idx)
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=int(data.cfg["train"].get("seed", 42)))
    batch_size = int(data.cfg["train"]["batch_size"])
    for inner_fold, (inner_train_pos, inner_holdout_pos) in enumerate(skf.split(np.zeros(len(data.train_idx)), labels)):
        inner_train_idx = data.train_idx[inner_train_pos]
        inner_holdout_idx = data.train_idx[inner_holdout_pos]
        inner_dir = os.path.join(run_dir, "predictors", "oof", f"inner_{inner_fold}", "los")
        print(
            f"[mixed OOF LOS {inner_fold}/{n_inner - 1}] "
            f"train={len(inner_train_idx)} holdout={len(inner_holdout_idx)} fixed_epochs={fixed_epochs}"
        )
        result = _train_los_predictor(
            los_cfg,
            root,
            inner_train_idx,
            None,
            inner_dir,
            device,
            fixed_epochs=fixed_epochs,
            role=f"mixed OOF LOS predictor {inner_fold}",
            verbose=False,
        )
        provider = ForecastedLOSProvider(
            _build_provider_cfg(
                data.cfg.get("forecasted_los", {}),
                result["checkpoint_path"],
                result.get("calibration_path"),
            ),
            data.base_dataset,
            device,
        )
        _forecast_los_only_into_cache(
            data.base_dataset, inner_holdout_idx, provider, device, batch_size, los_cache
        )


def build_mixed_actual_forecast_cache(
    data: "DiagnosticData",
    root: str,
    diagnostic_dir: str,
    device: torch.device,
    *,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Train predictors now, then replace exactly one forecasted component with oracle values."""
    if mode not in {
        "oracle_d_predicted_los_hard",
        "oracle_d_predicted_los_distribution",
        "predicted_d_oracle_los",
        "predicted_d_predicted_los",
    }:
        raise ValueError(f"Unsupported mixed actual forecast mode: {mode}")

    cfg = data.cfg
    discharge_cfg, los_cfg = _load_child_predictor_cfgs(cfg)
    batch_size = int(cfg["train"]["batch_size"])
    train_prediction_mode = str(cfg.get("forecasted_pipeline", {}).get("train_prediction_mode", "oof")).lower()
    x_cache = _base_x_tensor(data.base_dataset).clone()
    los_return_type = _los_return_type_for_mode(cfg, mode)
    if mode in {"oracle_d_predicted_los_hard", "oracle_d_predicted_los_distribution", "predicted_d_predicted_los"} and los_return_type == "distribution":
        los_cache = torch.zeros((len(data.base_dataset), 37), dtype=torch.float32)
    else:
        los_cache = _base_los_tensor(data.base_dataset).clone()

    selection_dir = os.path.join(diagnostic_dir, "predictors", "selection")
    payload: dict[str, Any] = {
        "mode": mode,
        "train_prediction_mode": train_prediction_mode,
        "train_core_size": int(len(data.train_idx)),
        "predictor_val_size": int(len(data.predictor_val_idx)),
        "gnn_val_size": int(len(data.val_idx)),
        "test_size": int(len(data.test_idx)),
    }

    needs_los_prediction = mode in {
        "oracle_d_predicted_los_hard",
        "oracle_d_predicted_los_distribution",
        "predicted_d_predicted_los",
    }
    needs_discharge_prediction = mode in {
        "predicted_d_oracle_los",
        "predicted_d_predicted_los",
    }

    if needs_los_prediction:
        los_selection = _train_los_predictor(
            los_cfg,
            root,
            data.train_idx,
            data.predictor_val_idx if len(data.predictor_val_idx) else None,
            os.path.join(selection_dir, "los"),
            device,
            role="mixed selection LOS predictor",
            verbose=True,
        )
        final_los_provider = ForecastedLOSProvider(
            _provider_cfg_with_los_return_type(
                cfg,
                los_selection["checkpoint_path"],
                los_selection.get("calibration_path"),
                los_return_type,
            ),
            data.base_dataset,
            device,
        )
        if train_prediction_mode == "oof":
            _train_oof_los_predictions(
                data,
                root,
                diagnostic_dir,
                device,
                los_cfg,
                int(los_selection["best_epoch"]),
                los_cache,
            )
        elif train_prediction_mode == "in_sample":
            _forecast_los_only_into_cache(
                data.base_dataset, data.train_idx, final_los_provider, device, batch_size, los_cache
            )
        else:
            raise ValueError(f"Unsupported train_prediction_mode: {train_prediction_mode}")
        _forecast_los_only_into_cache(data.base_dataset, data.val_idx, final_los_provider, device, batch_size, los_cache)
        _forecast_los_only_into_cache(data.base_dataset, data.test_idx, final_los_provider, device, batch_size, los_cache)
        payload.update(
            {
                "los_source": "predicted",
                "los_selection": los_selection,
                "los_return_type": los_return_type,
            }
        )

    if needs_discharge_prediction:
        discharge_selection = _train_discharge_predictor(
            discharge_cfg,
            root,
            data.train_idx,
            data.predictor_val_idx if len(data.predictor_val_idx) else None,
            os.path.join(selection_dir, "discharge"),
            device,
            role="mixed selection discharge predictor",
            verbose=True,
        )
        final_discharge_provider = ForecastedDischargeProvider(
            _build_provider_cfg(cfg.get("forecasted_discharge", {}), discharge_selection["checkpoint_path"]),
            data.base_dataset,
            device,
        )
        if train_prediction_mode == "oof":
            _train_oof_discharge_predictions(
                data,
                root,
                diagnostic_dir,
                device,
                discharge_cfg,
                int(discharge_selection["best_epoch"]),
                x_cache,
            )
        elif train_prediction_mode == "in_sample":
            _forecast_discharge_only_into_cache(
                data.base_dataset, data.train_idx, final_discharge_provider, device, batch_size, x_cache
            )
        else:
            raise ValueError(f"Unsupported train_prediction_mode: {train_prediction_mode}")
        _forecast_discharge_only_into_cache(
            data.base_dataset, data.val_idx, final_discharge_provider, device, batch_size, x_cache
        )
        _forecast_discharge_only_into_cache(
            data.base_dataset, data.test_idx, final_discharge_provider, device, batch_size, x_cache
        )
        payload.update(
            {
                "discharge_source": "predicted",
                "discharge_selection": discharge_selection,
            }
        )
    else:
        payload["discharge_source"] = "oracle"

    if not needs_los_prediction:
        payload["los_source"] = "oracle_raw"
        payload["los_return_type"] = "raw"

    if los_cache.ndim == 1:
        _assert_raw_los_valid(los_cache[np.concatenate([data.train_idx, data.val_idx, data.test_idx])], mode)
    payload["los_cache_shape"] = tuple(los_cache.shape)
    payload["x_cache_shape"] = tuple(x_cache.shape)
    payload["cache_roundtrip_match"] = _assert_cache_roundtrip(
        data.base_dataset,
        x_cache,
        los_cache,
        np.concatenate([data.train_idx, data.val_idx, data.test_idx]),
    )
    return x_cache, los_cache, payload


def _discharge_target_lookup(
    base_dataset,
    discharge_dataset: DischargePredictionDataset,
) -> dict[str, tuple[int, int]]:
    lookup: dict[str, tuple[int, int]] = {}
    for target_name, target_pos, ctmp_col_idx in _discharge_mapping(base_dataset, discharge_dataset):
        lookup[str(target_name)] = (int(target_pos), int(ctmp_col_idx))
    return lookup


def _resolve_override_heads(
    base_dataset,
    discharge_dataset: DischargePredictionDataset,
    override_head: str | None,
) -> list[tuple[str, int, int]]:
    if not override_head:
        raise RuntimeError("Head ablation modes require --override-head.")
    lookup = _discharge_target_lookup(base_dataset, discharge_dataset)
    head_names = [part.strip() for part in str(override_head).split(",") if part.strip()]
    if not head_names:
        raise RuntimeError("Head ablation modes require at least one non-empty override head.")
    missing = [name for name in head_names if name not in lookup]
    if missing:
        available = ", ".join(sorted(lookup))
        raise RuntimeError(
            f"Unknown override head(s): {', '.join(missing)}. Available discharge heads: {available}"
        )
    resolved: list[tuple[str, int, int]] = []
    seen: set[str] = set()
    for name in head_names:
        if name in seen:
            continue
        seen.add(name)
        target_pos, x_col_idx = lookup[name]
        resolved.append((name, int(target_pos), int(x_col_idx)))
    return resolved


def _apply_head_overrides(
    *,
    oracle_x: torch.Tensor,
    predicted_x: torch.Tensor,
    override_x_col_idx_list: list[int],
    indices: np.ndarray,
    base_source: str,
    override_source: str,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if base_source not in {"oracle", "predicted"}:
        raise ValueError(f"Unsupported base_source: {base_source}")
    if override_source not in {"oracle", "predicted"}:
        raise ValueError(f"Unsupported override_source: {override_source}")
    if base_source == override_source:
        raise ValueError("base_source and override_source must differ for head override ablation.")

    x_cache = oracle_x.clone() if base_source == "oracle" else predicted_x.clone()
    source_x = oracle_x if override_source == "oracle" else predicted_x
    idx = np.asarray(indices, dtype=np.int64)
    changed_any_mask = torch.zeros((len(idx),), dtype=torch.bool)
    per_head_summary: list[dict[str, Any]] = []
    for override_x_col_idx in override_x_col_idx_list:
        before_col = x_cache[idx, override_x_col_idx].clone()
        source_col = source_x[idx, override_x_col_idx].clone()
        x_cache[idx, override_x_col_idx] = source_col
        changed_mask = before_col.long() != source_col.long()
        changed_any_mask |= changed_mask
        per_head_summary.append(
            {
                "override_x_col_idx": int(override_x_col_idx),
                "num_changed_rows": int(changed_mask.long().sum().item()),
                "changed_rate": float(changed_mask.float().mean().item()) if len(idx) else 0.0,
                "predicted_oracle_match_rate": _exact_match_rate(
                    predicted_x[idx, override_x_col_idx].long(),
                    oracle_x[idx, override_x_col_idx].long(),
                ),
            }
        )
    payload = {
        "num_override_rows": int(len(idx)),
        "num_override_heads": int(len(override_x_col_idx_list)),
        "num_changed_rows": int(changed_any_mask.long().sum().item()),
        "changed_rate": float(changed_any_mask.float().mean().item()) if len(idx) else 0.0,
        "per_head_summary": per_head_summary,
    }
    return x_cache, payload


def build_single_head_ablation_forecast_cache(
    data: "DiagnosticData",
    root: str,
    diagnostic_dir: str,
    device: torch.device,
    *,
    mode: str,
    override_head: str,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if mode not in {
        "predicted_d_predicted_los_oracle_head_ablation",
        "oracle_d_predicted_los_predicted_head_ablation",
    }:
        raise ValueError(f"Unsupported single-head ablation mode: {mode}")

    resolved_heads = _resolve_override_heads(
        data.base_dataset,
        data.discharge_dataset,
        override_head,
    )
    resolved_head_names = [name for name, _, _ in resolved_heads]
    resolved_target_positions = [target_pos for _, target_pos, _ in resolved_heads]
    resolved_x_col_indices = [x_col_idx for _, _, x_col_idx in resolved_heads]
    predicted_x_cache, los_cache, predicted_payload = build_mixed_actual_forecast_cache(
        data,
        root,
        diagnostic_dir,
        device,
        mode="predicted_d_predicted_los",
    )
    oracle_x = _base_x_tensor(data.base_dataset).clone()
    eval_indices = np.concatenate([data.train_idx, data.val_idx, data.test_idx]).astype(np.int64)
    if mode == "predicted_d_predicted_los_oracle_head_ablation":
        base_source = "predicted"
        override_source = "oracle"
    else:
        base_source = "oracle"
        override_source = "predicted"

    x_cache, override_summary = _apply_head_overrides(
        oracle_x=oracle_x,
        predicted_x=predicted_x_cache,
        override_x_col_idx_list=resolved_x_col_indices,
        indices=eval_indices,
        base_source=base_source,
        override_source=override_source,
    )

    split_summaries: dict[str, Any] = {}
    for split_name, split_idx in {
        "train": data.train_idx,
        "valid": data.val_idx,
        "test": data.test_idx,
    }.items():
        _, split_payload = _apply_head_overrides(
            oracle_x=oracle_x,
            predicted_x=predicted_x_cache,
            override_x_col_idx_list=resolved_x_col_indices,
            indices=np.asarray(split_idx, dtype=np.int64),
            base_source=base_source,
            override_source=override_source,
        )
        split_summaries[split_name] = split_payload

    payload = {
        "mode": mode,
        "override_head": ",".join(resolved_head_names),
        "override_heads": resolved_head_names,
        "override_target_positions": resolved_target_positions,
        "override_x_col_indices": resolved_x_col_indices,
        "base_d_source": base_source,
        "override_head_source": override_source,
        "los_source": "predicted",
        "predicted_joint_payload": predicted_payload,
        "overall_override_summary": override_summary,
        "split_override_summary": split_summaries,
        "x_cache_shape": tuple(x_cache.shape),
        "los_cache_shape": tuple(los_cache.shape),
    }
    if los_cache.ndim == 1:
        _assert_raw_los_valid(los_cache[eval_indices], mode)
    payload["cache_roundtrip_match"] = _assert_cache_roundtrip(
        data.base_dataset,
        x_cache,
        los_cache,
        eval_indices,
    )
    return x_cache, los_cache, payload


def _range_payload(values: torch.Tensor) -> dict[str, int | None]:
    if values.numel() == 0:
        return {"min": None, "max": None}
    values = values.detach().cpu().long()
    return {"min": int(values.min().item()), "max": int(values.max().item())}


def _assert_cache_roundtrip(base_dataset, x_cache: torch.Tensor, los_cache: torch.Tensor, indices: np.ndarray) -> bool:
    cached_dataset = ForecastCacheDataset(base_dataset, x_cache.cpu(), los_cache.cpu())
    for row in indices[: min(len(indices), 1024)]:
        row_i = int(row)
        x_item, y_item, los_item = cached_dataset[row_i]
        if not torch.equal(x_item.long(), x_cache.cpu()[row_i].long()):
            return False
        if int(y_item) != int(_base_y_tensor(base_dataset)[row_i]):
            return False
        if los_cache.ndim == 1 and int(los_item) != int(los_cache.cpu()[row_i]):
            return False
    return True


def audit_forecast_value_space(
    base_dataset,
    discharge_dataset: DischargePredictionDataset,
    los_dataset: LOSPredictionDataset,
    cfg: dict[str, Any],
    *,
    los_cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_ad_x = _select_base_x_by_names(base_dataset, [str(name) for name in discharge_dataset.ad_col_names])

    admission_x_match_discharge = torch.equal(base_ad_x, discharge_dataset.x.long())
    admission_x_match_los = torch.equal(base_ad_x, los_dataset.x.long())
    if not admission_x_match_discharge:
        _raise_value_mismatch(
            "admission_x_discharge_dataset",
            base_ad_x,
            discharge_dataset.x.long(),
            "Admission x mismatch",
        )
    if not admission_x_match_los:
        _raise_value_mismatch(
            "admission_x_los_dataset",
            base_ad_x,
            los_dataset.x.long(),
            "Admission x mismatch",
        )

    mapping = _discharge_mapping(base_dataset, discharge_dataset)
    value_checks: list[dict[str, Any]] = []
    all_discharge_value_match = True
    base_x = _base_x_tensor(base_dataset)
    for target_name, target_pos, ctmp_col_idx in mapping:
        target_values = discharge_dataset.y[:, target_pos].long()
        ctmp_values = base_x[:, ctmp_col_idx].long()
        rate = _exact_match_rate(target_values, ctmp_values)
        target_range = _range_payload(target_values)
        ctmp_range = _range_payload(ctmp_values)
        value_checks.append(
            {
                "target_name": target_name,
                "target_pos": target_pos,
                "discharge_col_idx": ctmp_col_idx,
                "target_min": target_range["min"],
                "target_max": target_range["max"],
                "ctmp_min": ctmp_range["min"],
                "ctmp_max": ctmp_range["max"],
                "exact_match_rate": rate,
            }
        )
        if rate < 1.0:
            all_discharge_value_match = False

    los_target = _los_predictor_target(los_dataset, los_cfg)
    base_los = _base_los_tensor(base_dataset)
    los_rate = _exact_match_rate(los_target, base_los)
    los_target_range = _range_payload(los_target)
    los_base_range = _range_payload(base_los)
    model_params = cfg.get("model", {}).get("params", {})
    los_num_embeddings = int(model_params.get("max_los", 37)) + 1
    los_min = int(base_los.min().item())
    los_max = int(base_los.max().item())
    los_index_valid = los_min >= 0 and los_max < los_num_embeddings

    x_oracle, los_oracle = build_oracle_forecast_cache(base_dataset)
    cache_roundtrip_match = _assert_cache_roundtrip(
        base_dataset, x_oracle, los_oracle, np.arange(len(base_dataset), dtype=np.int64)
    )
    non_discharge_cols = sorted(set(range(base_x.shape[1])) - {m[2] for m in mapping})
    non_discharge_match = torch.equal(x_oracle[:, non_discharge_cols].cpu(), base_x[:, non_discharge_cols].cpu())

    if not los_index_valid:
        raise RuntimeError(
            f"LOS index range [{los_min}, {los_max}] is outside embedding range [0, {los_num_embeddings - 1}]"
        )
    if not cache_roundtrip_match:
        raise RuntimeError("ForecastCacheDataset roundtrip failed for oracle cache.")
    if not non_discharge_match:
        raise RuntimeError("Oracle cache changed non-discharge columns.")

    return {
        "num_rows": len(base_dataset),
        "num_admission_cols": len(base_dataset.col_info[2]),
        "num_discharge_targets": len(discharge_dataset.target_col_names),
        "admission_x_match_discharge_dataset": admission_x_match_discharge,
        "admission_x_match_los_dataset": admission_x_match_los,
        "all_discharge_value_match": all_discharge_value_match,
        "los_value_match": los_rate == 1.0,
        "los_min": los_min,
        "los_max": los_max,
        "los_embedding_num_embeddings": los_num_embeddings,
        "los_index_valid": los_index_valid,
        "cache_roundtrip_match": cache_roundtrip_match,
        "non_discharge_cols_unchanged": non_discharge_match,
        "discharge_mapping": value_checks,
        "los_value_check": {
            "target_min": los_target_range["min"],
            "target_max": los_target_range["max"],
            "ctmp_min": los_base_range["min"],
            "ctmp_max": los_base_range["max"],
            "exact_match_rate": los_rate,
            "target_mode": str((los_cfg or {}).get("los_target_mode", (los_cfg or {}).get("target_mode", "fine"))),
        },
    }


@dataclass
class DiagnosticData:
    cfg: dict[str, Any]
    los_cfg: dict[str, Any] | None
    base_dataset: TEDSTensorDataset
    discharge_dataset: DischargePredictionDataset
    los_dataset: LOSPredictionDataset
    outer_train_idx: np.ndarray
    predictor_val_idx: np.ndarray
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray


@dataclass
class JointPredictionArrays:
    target_names: list[str]
    target_dims: list[int]
    discharge_pred: np.ndarray
    discharge_conf: np.ndarray
    los_pred: np.ndarray
    los_conf: np.ndarray
    provenance: dict[str, Any]


def _build_base_dataset(cfg: dict[str, Any], root: str) -> TEDSTensorDataset:
    remove_los = cfg["model"]["name"] not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]
    if not remove_los:
        cfg.setdefault("edge", {})["remove_los"] = False
    return TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=cfg["train"].get("do_preprocess", True),
    )


def _dataset_do_preprocess(cfg: dict[str, Any]) -> bool:
    return bool(cfg.get("train", {}).get("do_preprocess", True))


def _load_los_cfg(cfg: dict[str, Any]) -> dict[str, Any] | None:
    path = cfg.get("forecasted_pipeline", {}).get("los_config_path")
    if path in {None, ""}:
        return None
    return _load_yaml(str(path))


def _prepare_data(cfg: dict[str, Any], root: str, fold: int, seed: int) -> DiagnosticData:
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("train", {})["seed"] = seed
    set_seed(seed)
    base_dataset = _build_base_dataset(copy.deepcopy(cfg), root)
    labels = np.asarray([int(base_dataset[i][1]) for i in range(len(base_dataset))], dtype=np.int64)

    if bool(cfg.get("train", {}).get("cv", False)):
        all_idx = np.arange(len(base_dataset), dtype=np.int64)
        folds = list(
            kfold_stratified(
                trainval_idx=all_idx,
                labels=labels,
                n_folds=int(cfg["train"].get("n_folds", 5)),
                seed=seed,
            )
        )
        _, outer_train_idx, outer_test_idx = folds[int(fold)]
    else:
        outer_train_idx, outer_test_idx = holdout_test_split_stratified(
            dataset=base_dataset,
            test_ratio=float(cfg["train"]["test_ratio"]),
            seed=seed,
            labels=labels,
        )

    if bool(cfg.get("forecasted_pipeline", {}).get("enabled", False)):
        train_idx, predictor_val_idx, val_idx = split_outer_train_for_forecasted_pipeline(
            base_dataset, outer_train_idx, cfg, seed + int(fold) * 1000
        )
    else:
        train_idx = outer_train_idx
        predictor_val_idx = np.asarray([], dtype=np.int64)
        val_idx = outer_test_idx
    test_idx = outer_test_idx

    do_preprocess = _dataset_do_preprocess(cfg)
    discharge_dataset = DischargePredictionDataset(
        root=root,
        do_preprocess=do_preprocess,
        include_los_in_targets=False,
    )
    los_dataset = LOSPredictionDataset(root=root, do_preprocess=do_preprocess)
    return DiagnosticData(
        cfg=cfg,
        los_cfg=_load_los_cfg(cfg),
        base_dataset=base_dataset,
        discharge_dataset=discharge_dataset,
        los_dataset=los_dataset,
        outer_train_idx=outer_train_idx.astype(np.int64),
        predictor_val_idx=predictor_val_idx.astype(np.int64),
        train_idx=train_idx.astype(np.int64),
        val_idx=val_idx.astype(np.int64),
        test_idx=test_idx.astype(np.int64),
    )


def _resolve_los_calibration_path(checkpoint_path: str | None) -> str | None:
    if checkpoint_path in {None, ""}:
        return None
    run_dir = os.path.dirname(os.path.dirname(str(checkpoint_path)))
    calibration_path = os.path.join(run_dir, "calibration.json")
    return calibration_path if os.path.exists(calibration_path) else None


def _checkpoint_epoch(checkpoint_path: str) -> int:
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    epoch = ckpt.get("epoch")
    if epoch is None:
        raise RuntimeError(f"Checkpoint is missing epoch metadata: {checkpoint_path}")
    return int(epoch)


def _predictor_run_root(checkpoint_path: str | None) -> str | None:
    if checkpoint_path in {None, ""}:
        return None
    for parent in Path(str(checkpoint_path)).resolve().parents:
        if parent.name == "predictors":
            return str(parent.parent)
    return None


def _existing_oof_checkpoint(
    selection_checkpoint_path: str | None,
    *,
    inner_fold: int,
    predictor_kind: str,
) -> tuple[str | None, str | None]:
    run_root = _predictor_run_root(selection_checkpoint_path)
    if run_root is None:
        return None, None
    checkpoint_path = os.path.join(
        run_root,
        "predictors",
        "oof",
        f"inner_{inner_fold}",
        predictor_kind,
        "checkpoints",
        "best.pt",
    )
    if not os.path.exists(checkpoint_path):
        return None, None
    calibration_path = None
    if predictor_kind == "los":
        calibration_path = _resolve_los_calibration_path(checkpoint_path)
    return checkpoint_path, calibration_path


def _discharge_target_specs(
    base_dataset,
    discharge_dataset: DischargePredictionDataset,
) -> tuple[list[str], list[int], list[int]]:
    mapping = _discharge_mapping(base_dataset, discharge_dataset)
    target_names = [name for name, _, _ in mapping]
    target_positions = [target_pos for _, target_pos, _ in mapping]
    target_name_to_dim = {
        str(name): int(dim)
        for name, dim in zip(discharge_dataset.target_col_names, discharge_dataset.target_col_dims)
    }
    target_dims = [int(target_name_to_dim[name]) for name in target_names]
    return target_names, target_positions, target_dims


def _oracle_joint_targets(data: DiagnosticData) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[int]]:
    target_names, target_positions, target_dims = _discharge_target_specs(data.base_dataset, data.discharge_dataset)
    discharge_actual = data.discharge_dataset.y[:, target_positions].detach().cpu().numpy().astype(np.int64)
    los_actual = map_los_array_to_coarse_bins(data.los_dataset.los_raw.long()).detach().cpu().numpy().astype(np.int64)
    labels = _base_y_tensor(data.base_dataset).detach().cpu().numpy().astype(np.int64)
    return discharge_actual, los_actual, labels, target_names, target_dims


def _discharge_predictions_for_indices(
    base_dataset,
    indices: np.ndarray,
    provider: ForecastedDischargeProvider,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    num_targets = len(provider.target_names)
    pred = np.zeros((len(indices), num_targets), dtype=np.int64)
    conf = np.zeros((len(indices), num_targets), dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    write_pos = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.stack([base_dataset[int(i)][0] for i in chunk], dim=0).to(device)
        with torch.no_grad():
            ad_x = torch.index_select(x.long(), dim=1, index=provider.ad_idx_t)
            outputs = provider.model(ad_x)
        for target_idx, name in enumerate(provider.target_names):
            probs = torch.softmax(outputs[name], dim=1)
            pred[write_pos : write_pos + len(chunk), target_idx] = (
                torch.argmax(probs, dim=1).detach().cpu().numpy().astype(np.int64)
            )
            conf[write_pos : write_pos + len(chunk), target_idx] = (
                torch.max(probs, dim=1).values.detach().cpu().numpy().astype(np.float32)
            )
        write_pos += len(chunk)
    return pred, conf


def _coarse_probs_from_raw_probs(raw_probs: torch.Tensor, output_offset: int) -> torch.Tensor:
    coarse_probs = torch.zeros((raw_probs.shape[0], 6), dtype=raw_probs.dtype, device=raw_probs.device)
    raw_codes = torch.arange(raw_probs.shape[1], device=raw_probs.device, dtype=torch.long) + int(output_offset)
    coarse_codes = map_los_array_to_coarse_bins(raw_codes.cpu()).to(device=raw_probs.device)
    for coarse_bin in range(6):
        mask = coarse_codes == coarse_bin
        if torch.any(mask):
            coarse_probs[:, coarse_bin] = raw_probs[:, mask].sum(dim=1)
    return coarse_probs


def _los_outputs_to_coarse_probs(provider: ForecastedLOSProvider, outputs) -> torch.Tensor:
    ce_like = {"ce", "focal", "focal_alpha", "cb_focal"}
    if provider.los_target_mode == "coarse":
        if provider.loss_type in ce_like:
            source = provider.probability_source
            if source == "auto":
                source = "calibrated" if provider.return_type == "distribution" else "raw"
            temperature = 1.0 if source in {"ce", "raw"} else float(provider.temperature)
            return torch.softmax(outputs / temperature, dim=1)
        if provider.loss_type == "hybrid_ce_ordinal":
            source = provider.probability_source
            if source == "auto":
                source = "ce" if provider.return_type == "hard" else "calibrated"
            if source in {"ce", "raw"}:
                return torch.softmax(outputs["ce"], dim=1)
            if source == "calibrated":
                return torch.softmax(outputs["ce"] / float(provider.temperature), dim=1)
            if source == "ordinal":
                return provider._ordinal_logits_to_distribution(outputs["ordinal"])
            raise ValueError(
                f"Unsupported forecasted_los.probability_source for coarse hybrid: {provider.probability_source}"
            )
        return provider._ordinal_logits_to_distribution(outputs)

    raw_probs = provider._outputs_to_distribution(outputs)
    return _coarse_probs_from_raw_probs(raw_probs, provider.output_offset)


def _los_predictions_for_indices(
    base_dataset,
    indices: np.ndarray,
    provider: ForecastedLOSProvider,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = np.zeros((len(indices),), dtype=np.int64)
    conf = np.zeros((len(indices),), dtype=np.float32)
    indices = np.asarray(indices, dtype=np.int64)
    write_pos = 0
    for start in range(0, len(indices), batch_size):
        chunk = indices[start : start + batch_size]
        x = torch.stack([base_dataset[int(i)][0] for i in chunk], dim=0).to(device)
        with torch.no_grad():
            ad_x = torch.index_select(x.long(), dim=1, index=provider.ad_idx_t)
            outputs = provider.model(ad_x)
            coarse_probs = _los_outputs_to_coarse_probs(provider, outputs)
        pred[write_pos : write_pos + len(chunk)] = (
            torch.argmax(coarse_probs, dim=1).detach().cpu().numpy().astype(np.int64)
        )
        conf[write_pos : write_pos + len(chunk)] = (
            torch.max(coarse_probs, dim=1).values.detach().cpu().numpy().astype(np.float32)
        )
        write_pos += len(chunk)
    return pred, conf


def _inner_oof_splits(indices: np.ndarray, labels: np.ndarray, seed: int, n_inner: int):
    skf = StratifiedKFold(n_splits=n_inner, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(indices)), labels))


def _collect_joint_predictions(
    data: DiagnosticData,
    *,
    root: str,
    device: torch.device,
    batch_size: int,
    discharge_checkpoint_path: str,
    los_checkpoint_path: str,
    diag_dir: str,
) -> JointPredictionArrays:
    discharge_ckpt = str(discharge_checkpoint_path)
    los_ckpt = str(los_checkpoint_path)
    discharge_cfg, los_cfg = _load_child_predictor_cfgs(data.cfg)
    target_names, _, target_dims = _discharge_target_specs(data.base_dataset, data.discharge_dataset)
    n_rows = len(data.base_dataset)
    discharge_pred = np.full((n_rows, len(target_names)), -1, dtype=np.int64)
    discharge_conf = np.full((n_rows, len(target_names)), np.nan, dtype=np.float32)
    los_pred = np.full((n_rows,), -1, dtype=np.int64)
    los_conf = np.full((n_rows,), np.nan, dtype=np.float32)

    selection_discharge = ForecastedDischargeProvider(
        _build_provider_cfg(data.cfg.get("forecasted_discharge", {}), discharge_ckpt),
        data.base_dataset,
        device,
    )
    selection_los = ForecastedLOSProvider(
        _build_provider_cfg(
            data.cfg.get("forecasted_los", {}),
            los_ckpt,
            _resolve_los_calibration_path(los_ckpt),
        ),
        data.base_dataset,
        device,
    )

    provenance: dict[str, Any] = {
        "train_prediction_mode": str(
            data.cfg.get("forecasted_pipeline", {}).get("train_prediction_mode", "oof")
        ).lower(),
        "selection_discharge_checkpoint": discharge_ckpt,
        "selection_los_checkpoint": los_ckpt,
        "selection_los_calibration_path": _resolve_los_calibration_path(los_ckpt),
        "oof_reused": {"discharge": [], "los": []},
        "oof_trained": {"discharge": [], "los": []},
    }

    selection_splits = {
        "predictor_val": data.predictor_val_idx,
        "valid": data.val_idx,
        "test": data.test_idx,
    }
    for split_name, split_idx in selection_splits.items():
        if len(split_idx) == 0:
            provenance[f"{split_name}_selection_rows"] = 0
            continue
        split_pred, split_conf = _discharge_predictions_for_indices(
            data.base_dataset, split_idx, selection_discharge, device, batch_size
        )
        discharge_pred[split_idx] = split_pred
        discharge_conf[split_idx] = split_conf
        split_los_pred, split_los_conf = _los_predictions_for_indices(
            data.base_dataset, split_idx, selection_los, device, batch_size
        )
        los_pred[split_idx] = split_los_pred
        los_conf[split_idx] = split_los_conf
        provenance[f"{split_name}_selection_rows"] = int(len(split_idx))

    train_prediction_mode = provenance["train_prediction_mode"]
    if train_prediction_mode == "in_sample":
        train_pred, train_conf = _discharge_predictions_for_indices(
            data.base_dataset, data.train_idx, selection_discharge, device, batch_size
        )
        discharge_pred[data.train_idx] = train_pred
        discharge_conf[data.train_idx] = train_conf
        train_los_pred, train_los_conf = _los_predictions_for_indices(
            data.base_dataset, data.train_idx, selection_los, device, batch_size
        )
        los_pred[data.train_idx] = train_los_pred
        los_conf[data.train_idx] = train_los_conf
    elif train_prediction_mode == "oof":
        n_inner = int(data.cfg.get("forecasted_pipeline", {}).get("oof", {}).get("n_inner_folds", 5))
        labels = _labels_for_indices(data.base_dataset, data.train_idx)
        inner_splits = _inner_oof_splits(
            data.train_idx,
            labels,
            int(data.cfg["train"].get("seed", 42)),
            n_inner,
        )
        discharge_fixed_epochs = _checkpoint_epoch(discharge_ckpt)
        los_fixed_epochs = _checkpoint_epoch(los_ckpt)
        for inner_fold, (_, inner_holdout_pos) in enumerate(inner_splits):
            inner_train_pos = np.setdiff1d(
                np.arange(len(data.train_idx), dtype=np.int64),
                np.asarray(inner_holdout_pos, dtype=np.int64),
                assume_unique=True,
            )
            inner_train_idx = data.train_idx[inner_train_pos]
            inner_holdout_idx = data.train_idx[np.asarray(inner_holdout_pos, dtype=np.int64)]

            discharge_inner_ckpt, _ = _existing_oof_checkpoint(
                discharge_ckpt,
                inner_fold=inner_fold,
                predictor_kind="discharge",
            )
            if discharge_inner_ckpt is None:
                discharge_result = _train_discharge_predictor(
                    discharge_cfg,
                    root,
                    inner_train_idx,
                    None,
                    os.path.join(diag_dir, "predictors", "oof", f"inner_{inner_fold}", "discharge"),
                    device,
                    fixed_epochs=discharge_fixed_epochs,
                    role=f"joint plausibility OOF discharge predictor {inner_fold}",
                    verbose=False,
                )
                discharge_inner_ckpt = str(discharge_result["checkpoint_path"])
                provenance["oof_trained"]["discharge"].append(
                    {"inner_fold": inner_fold, "checkpoint_path": discharge_inner_ckpt}
                )
            else:
                provenance["oof_reused"]["discharge"].append(
                    {"inner_fold": inner_fold, "checkpoint_path": discharge_inner_ckpt}
                )
            discharge_provider = ForecastedDischargeProvider(
                _build_provider_cfg(data.cfg.get("forecasted_discharge", {}), discharge_inner_ckpt),
                data.base_dataset,
                device,
            )
            holdout_pred, holdout_conf = _discharge_predictions_for_indices(
                data.base_dataset, inner_holdout_idx, discharge_provider, device, batch_size
            )
            discharge_pred[inner_holdout_idx] = holdout_pred
            discharge_conf[inner_holdout_idx] = holdout_conf

            los_inner_ckpt, los_inner_calibration = _existing_oof_checkpoint(
                los_ckpt,
                inner_fold=inner_fold,
                predictor_kind="los",
            )
            if los_inner_ckpt is None:
                los_result = _train_los_predictor(
                    los_cfg,
                    root,
                    inner_train_idx,
                    None,
                    os.path.join(diag_dir, "predictors", "oof", f"inner_{inner_fold}", "los"),
                    device,
                    fixed_epochs=los_fixed_epochs,
                    role=f"joint plausibility OOF LOS predictor {inner_fold}",
                    verbose=False,
                )
                los_inner_ckpt = str(los_result["checkpoint_path"])
                los_inner_calibration = los_result.get("calibration_path")
                provenance["oof_trained"]["los"].append(
                    {
                        "inner_fold": inner_fold,
                        "checkpoint_path": los_inner_ckpt,
                        "calibration_path": los_inner_calibration,
                    }
                )
            else:
                provenance["oof_reused"]["los"].append(
                    {
                        "inner_fold": inner_fold,
                        "checkpoint_path": los_inner_ckpt,
                        "calibration_path": los_inner_calibration,
                    }
                )
            los_provider = ForecastedLOSProvider(
                _build_provider_cfg(
                    data.cfg.get("forecasted_los", {}),
                    los_inner_ckpt,
                    los_inner_calibration,
                ),
                data.base_dataset,
                device,
            )
            holdout_los_pred, holdout_los_conf = _los_predictions_for_indices(
                data.base_dataset, inner_holdout_idx, los_provider, device, batch_size
            )
            los_pred[inner_holdout_idx] = holdout_los_pred
            los_conf[inner_holdout_idx] = holdout_los_conf
    else:
        raise ValueError(f"Unsupported train_prediction_mode: {train_prediction_mode}")

    missing_discharge = int(np.sum(discharge_pred[:, 0] < 0)) if discharge_pred.size else 0
    missing_los = int(np.sum(los_pred < 0))
    if missing_discharge > 0 or missing_los > 0:
        raise RuntimeError(
            "Joint plausibility audit failed to populate predicted arrays for every row. "
            f"missing_discharge_rows={missing_discharge} missing_los_rows={missing_los}"
        )
    return JointPredictionArrays(
        target_names=target_names,
        target_dims=target_dims,
        discharge_pred=discharge_pred,
        discharge_conf=discharge_conf,
        los_pred=los_pred,
        los_conf=los_conf,
        provenance=provenance,
    )


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _contingency_table(d_values: np.ndarray, los_bins: np.ndarray, d_dim: int) -> np.ndarray:
    table = np.zeros((6, int(d_dim)), dtype=np.float64)
    for los_bin, d_value in zip(los_bins.tolist(), d_values.tolist()):
        if 0 <= int(los_bin) < 6 and 0 <= int(d_value) < int(d_dim):
            table[int(los_bin), int(d_value)] += 1.0
    return table


def _conditional_probs_from_table(table: np.ndarray) -> np.ndarray:
    row_sums = table.sum(axis=1, keepdims=True)
    out = np.zeros_like(table, dtype=np.float64)
    nonzero = row_sums.squeeze(1) > 0
    out[nonzero] = table[nonzero] / row_sums[nonzero]
    return out


def _js_divergence(prob_p: np.ndarray, prob_q: np.ndarray) -> float:
    prob_p = np.asarray(prob_p, dtype=np.float64)
    prob_q = np.asarray(prob_q, dtype=np.float64)
    prob_p = np.clip(prob_p, 1.0e-12, 1.0)
    prob_q = np.clip(prob_q, 1.0e-12, 1.0)
    prob_p = prob_p / prob_p.sum()
    prob_q = prob_q / prob_q.sum()
    mixture = 0.5 * (prob_p + prob_q)
    kl_pm = np.sum(prob_p * np.log(prob_p / mixture))
    kl_qm = np.sum(prob_q * np.log(prob_q / mixture))
    return float(0.5 * (kl_pm + kl_qm))


def _conditional_js_divergence(table_oracle: np.ndarray, table_other: np.ndarray) -> float:
    cond_oracle = _conditional_probs_from_table(table_oracle)
    cond_other = _conditional_probs_from_table(table_other)
    weights = table_oracle.sum(axis=1)
    weight_total = float(weights.sum())
    if weight_total <= 0.0:
        return 0.0
    score = 0.0
    for los_bin in range(table_oracle.shape[0]):
        if weights[los_bin] <= 0.0:
            continue
        score += float(weights[los_bin] / weight_total) * _js_divergence(
            cond_oracle[los_bin],
            cond_other[los_bin],
        )
    return float(score)


def _cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=np.float64)
    n = float(table.sum())
    if n <= 0.0:
        return 0.0
    row_sums = table.sum(axis=1, keepdims=True)
    col_sums = table.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    mask = expected > 0.0
    if not np.any(mask):
        return 0.0
    chi2 = float(np.sum(((table - expected) ** 2)[mask] / expected[mask]))
    min_dim = min(table.shape[0] - 1, table.shape[1] - 1)
    if min_dim <= 0:
        return 0.0
    return float(np.sqrt((chi2 / n) / float(min_dim)))


def _positive_rate_by_los(labels: np.ndarray, los_bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rates = np.full((6,), np.nan, dtype=np.float64)
    counts = np.zeros((6,), dtype=np.int64)
    labels = labels.astype(np.int64)
    los_bins = los_bins.astype(np.int64)
    for los_bin in range(6):
        mask = los_bins == los_bin
        counts[los_bin] = int(np.sum(mask))
        if counts[los_bin] > 0:
            rates[los_bin] = float(np.mean(labels[mask]))
    return rates, counts


def _mean_abs_rate_drift(lhs: np.ndarray, rhs: np.ndarray) -> float:
    mask = ~(np.isnan(lhs) | np.isnan(rhs))
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs(lhs[mask] - rhs[mask])))


def _rare_combo_map(
    d_values: np.ndarray,
    los_bins: np.ndarray,
    *,
    d_dim: int,
    threshold: float,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    total = max(int(len(d_values)), 1)
    table = _contingency_table(d_values, los_bins, d_dim)
    rare = table / float(total) < float(threshold)
    probability_map: dict[tuple[int, int], float] = {}
    for los_bin in range(table.shape[0]):
        for d_value in range(table.shape[1]):
            probability_map[(int(d_value), int(los_bin))] = float(table[los_bin, d_value] / float(total))
    return rare.astype(bool), probability_map


def _rare_rate_for_rows(d_values: np.ndarray, los_bins: np.ndarray, rare_map: np.ndarray) -> float:
    hits = [
        bool(rare_map[int(los_bin), int(d_value)])
        for d_value, los_bin in zip(d_values.tolist(), los_bins.tolist())
        if 0 <= int(los_bin) < rare_map.shape[0] and 0 <= int(d_value) < rare_map.shape[1]
    ]
    return _safe_mean([1.0 if hit else 0.0 for hit in hits])


def _print_audit_summary(summary: dict[str, Any]) -> None:
    print("[AUDIT SUMMARY]")
    keys = [
        "num_rows",
        "num_admission_cols",
        "num_discharge_targets",
        "admission_x_match_discharge_dataset",
        "admission_x_match_los_dataset",
        "all_discharge_value_match",
        "los_value_match",
        "los_min",
        "los_max",
        "los_embedding_num_embeddings",
        "los_index_valid",
        "cache_roundtrip_match",
        "non_discharge_cols_unchanged",
    ]
    for key in keys:
        print(f"{key}={summary[key]}")
    print("[Discharge target mapping]")
    for row in summary["discharge_mapping"]:
        print(f"target_name={row['target_name']} -> discharge_col_idx={row['discharge_col_idx']}")
    print("[Value-space check]")
    for row in summary["discharge_mapping"]:
        print(
            f"{row['target_name']}: target_min={row['target_min']}, target_max={row['target_max']}, "
            f"ctmp_min={row['ctmp_min']}, ctmp_max={row['ctmp_max']}, "
            f"exact_match_rate={row['exact_match_rate']:.6f}"
        )
    los = summary["los_value_check"]
    print(
        f"LOS: target_min={los['target_min']}, target_max={los['target_max']}, "
        f"ctmp_min={los['ctmp_min']}, ctmp_max={los['ctmp_max']}, "
        f"exact_match_rate={los['exact_match_rate']:.6f}, target_mode={los['target_mode']}"
    )


def _diagnostic_dir() -> str:
    path = os.path.join(
        "runs",
        "diagnostics",
        "forecast_cache_alignment",
        time.strftime("%Y%m%d-%H%M%S"),
    )
    os.makedirs(path, exist_ok=False)
    return path


def _write_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def _write_metrics_summary(path: str, row: dict[str, Any]) -> None:
    columns = [
        "mode",
        "fold",
        "seed",
        "acc",
        "f1",
        "precision",
        "recall",
        "auc",
        "loss",
        "los_source",
        "los_target_mode",
        "los_input_mode",
        "los_min",
        "los_max",
        "uses_los_zero",
        "basis_valid",
        "notes",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerow({key: row.get(key) for key in columns})


def _metrics_row(
    *,
    mode: str,
    fold: int,
    seed: int,
    results: dict[str, Any] | None,
    los_source: str,
    los_target_mode: str,
    los_input_mode: str,
    los_values: torch.Tensor | None,
    basis_valid: bool | None,
    notes: str,
) -> dict[str, Any]:
    los_min = int(los_values.min().item()) if los_values is not None and los_values.numel() else None
    los_max = int(los_values.max().item()) if los_values is not None and los_values.numel() else None
    uses_los_zero = bool(torch.any(los_values == 0).item()) if los_values is not None and los_values.numel() else None
    results = results or {}
    return {
        "mode": mode,
        "fold": fold,
        "seed": seed,
        "acc": results.get("test_acc"),
        "f1": results.get("test_f1"),
        "precision": results.get("test_precision"),
        "recall": results.get("test_recall"),
        "auc": results.get("test_auc"),
        "loss": results.get("test_loss"),
        "los_source": los_source,
        "los_target_mode": los_target_mode,
        "los_input_mode": los_input_mode,
        "los_min": los_min,
        "los_max": los_max,
        "uses_los_zero": uses_los_zero,
        "basis_valid": basis_valid,
        "notes": notes,
    }


def _print_los_mapping() -> None:
    print("[Coarse-to-raw LOS mapping]")
    for coarse, raw in _coarse_to_raw_mapping().items():
        print(f"coarse={coarse} -> raw={raw}")


def _print_transformed_payload(payload: dict[str, Any]) -> None:
    _print_los_mapping()
    print("[Predictor target transformed cache]")
    print(f"LOS target coarse_min={payload['coarse_min']}")
    print(f"LOS target coarse_max={payload['coarse_max']}")
    print(f"LOS transformed raw_min={payload['transformed_raw_min']}")
    print(f"LOS transformed raw_max={payload['transformed_raw_max']}")
    print("los_embedding_num_embeddings=38")
    print(f"los_index_valid={payload['transformed_raw_min'] >= 1 and payload['transformed_raw_max'] <= 37}")
    print(f"uses_los_zero={payload['uses_los_zero']}")


def _print_oracle_coarse_payload(payload: dict[str, Any]) -> None:
    print("[Oracle coarse LOS ablation]")
    for key in [
        "true_raw_min",
        "true_raw_max",
        "coarse_min",
        "coarse_max",
        "repr_raw_min",
        "repr_raw_max",
        "los_zero_used",
        "coarse_compression_mae",
        "coarse_compression_within_1",
        "coarse_compression_within_2",
    ]:
        print(f"{key}={payload[key]}")
    print(f"raw_to_coarse_table={payload['raw_to_coarse']}")
    print(f"coarse_to_raw_table={payload['coarse_to_raw']}")


def audit_los_distribution_basis(cfg: dict[str, Any], base_dataset, device: torch.device) -> dict[str, Any]:
    model_params = cfg.get("model", {}).get("params", {})
    los_embedding_num_embeddings = int(model_params.get("max_los", 37)) + 1
    coarse_probs = torch.eye(6, dtype=torch.float32)
    raw_probs_37 = expand_coarse_distribution_to_raw_los(coarse_probs)
    p_raw = torch.zeros((coarse_probs.shape[0], los_embedding_num_embeddings), dtype=torch.float32)
    p_raw[:, 1:38] = raw_probs_37
    used_embedding_indices = [idx for idx in range(los_embedding_num_embeddings) if float(p_raw[:, idx].sum()) > 0.0]
    uses_embedding_zero = 0 in used_embedding_indices
    uses_rows_0_to_5_directly = used_embedding_indices == list(range(6))
    basis_valid = (
        raw_probs_37.shape[-1] == 37
        and p_raw.shape[-1] == los_embedding_num_embeddings
        and torch.allclose(p_raw[:, 0], torch.zeros_like(p_raw[:, 0]))
        and torch.allclose(p_raw.sum(dim=-1), torch.ones_like(p_raw[:, 0]), atol=1.0e-5)
        and not uses_rows_0_to_5_directly
    )
    payload = {
        "los_target_mode": str((cfg.get("forecasted_los", {}) or {}).get("target_mode", "coarse")),
        "los_input_mode": "distribution",
        "prob_shape": tuple(coarse_probs.shape),
        "p_raw_shape": tuple(p_raw.shape),
        "los_embedding_shape": (los_embedding_num_embeddings, int(model_params.get("los_embedding_dim", 8))),
        "distribution_handling": "expanded_raw_distribution",
        "coarse_to_raw": _coarse_to_raw_mapping(),
        "used_embedding_indices": used_embedding_indices,
        "uses_embedding_zero": uses_embedding_zero,
        "uses_rows_0_to_5_directly": uses_rows_0_to_5_directly,
        "p_raw_zero_mass_max": float(p_raw[:, 0].max().item()),
        "basis_valid": basis_valid,
    }
    if not basis_valid:
        raise RuntimeError(
            "Invalid LOS distribution basis: coarse probability is not expanded onto raw LOS embedding rows 1..37."
        )
    return payload


def _build_provider_cfg_for_audit(cfg: dict[str, Any], checkpoint_path: str, return_type: str) -> dict[str, Any]:
    forecast_cfg = copy.deepcopy(cfg.get("forecasted_los", {}))
    forecast_cfg["enabled"] = True
    forecast_cfg["checkpoint_path"] = checkpoint_path
    forecast_cfg["return_type"] = return_type
    return forecast_cfg


def audit_los_hard_runtime(
    cfg: dict[str, Any],
    base_dataset,
    device: torch.device,
    *,
    los_checkpoint_path: str | None = None,
) -> dict[str, Any]:
    coarse = torch.arange(6, dtype=torch.long)
    injected_from_mapping = map_coarse_array_to_raw_los(coarse).long()
    _assert_raw_los_valid(injected_from_mapping, "los_hard_runtime_audit")
    payload: dict[str, Any] = {
        **_los_mapping_payload(),
        "coarse_pred_min": int(coarse.min().item()),
        "coarse_pred_max": int(coarse.max().item()),
        "injected_los_min": int(injected_from_mapping.min().item()),
        "injected_los_max": int(injected_from_mapping.max().item()),
        "uses_los_zero": bool(torch.any(injected_from_mapping == 0).item()),
        "injected_los_unique": sorted(set(int(v) for v in injected_from_mapping.tolist())),
        "runtime_provider_checked": False,
    }
    checkpoint_path = los_checkpoint_path or cfg.get("forecasted_los", {}).get("checkpoint_path")
    if checkpoint_path:
        provider = ForecastedLOSProvider(
            _build_provider_cfg_for_audit(cfg, str(checkpoint_path), "hard"),
            base_dataset,
            device,
        )
        x_batch = torch.stack([base_dataset[i][0] for i in range(min(1024, len(base_dataset)))], dim=0).to(device)
        injected = provider(x_batch).detach().cpu().long()
        _assert_raw_los_valid(injected, "los_hard_runtime_audit provider output")
        payload.update(
            {
                "runtime_provider_checked": True,
                "provider_injected_los_min": int(injected.min().item()),
                "provider_injected_los_max": int(injected.max().item()),
                "provider_uses_los_zero": bool(torch.any(injected == 0).item()),
                "provider_injected_los_unique": sorted(set(int(v) for v in injected.unique().tolist())),
            }
        )
        if not set(payload["provider_injected_los_unique"]).issubset(set(payload["injected_los_unique"])):
            raise RuntimeError(
                "LOS hard runtime audit failed: provider output contains values outside coarse-to-raw representatives."
            )
    return payload


def _print_distribution_basis(payload: dict[str, Any]) -> None:
    print("[LOS distribution basis audit]")
    for key in [
        "los_target_mode",
        "los_input_mode",
        "prob_shape",
        "p_raw_shape",
        "los_embedding_shape",
        "distribution_handling",
        "coarse_to_raw",
        "used_embedding_indices",
        "uses_embedding_zero",
        "uses_rows_0_to_5_directly",
        "p_raw_zero_mass_max",
        "basis_valid",
    ]:
        print(f"{key}={payload[key]}")


def _print_hard_runtime(payload: dict[str, Any]) -> None:
    print("[LOS hard runtime audit]")
    for key in [
        "coarse_pred_min",
        "coarse_pred_max",
        "injected_los_min",
        "injected_los_max",
        "uses_los_zero",
        "injected_los_unique",
        "coarse_to_raw",
        "runtime_provider_checked",
    ]:
        print(f"{key}={payload[key]}")


def _split_index_map(data: DiagnosticData) -> dict[str, np.ndarray]:
    return {
        "train": np.asarray(data.train_idx, dtype=np.int64),
        "valid": np.asarray(data.val_idx, dtype=np.int64),
        "test": np.asarray(data.test_idx, dtype=np.int64),
    }


def _trace_positions(indices: np.ndarray, batch_size: int) -> list[tuple[int, int]]:
    trace: list[tuple[int, int]] = []
    position = 0
    for start in range(0, len(indices), batch_size):
        chunk = np.asarray(indices[start : start + batch_size], dtype=np.int64)
        for row_idx in chunk.tolist():
            trace.append((position, int(row_idx)))
            position += 1
    return trace


def _sidecar_value(series, row_idx: int, *, fallback_row_idx: int) -> str:
    if series is None:
        return str(fallback_row_idx)
    value = series.iloc[row_idx]
    if value is None or (hasattr(value, "isna") and value.isna()):
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value)


def _joint_alignment_rows(
    data: DiagnosticData,
    *,
    batch_size: int,
    device: torch.device,
    discharge_provider: ForecastedDischargeProvider,
    los_provider: ForecastedLOSProvider,
) -> tuple[list[dict[str, Any]], bool, int, int]:
    split_indices = _split_index_map(data)
    base_caseid = getattr(data.base_dataset, "caseid_series", None)
    discharge_caseid = getattr(data.discharge_dataset, "caseid_series", None)
    los_caseid = getattr(data.los_dataset, "caseid_series", None)
    raw_row_index = getattr(data.base_dataset, "raw_row_index", None)

    caseid_available = base_caseid is not None and discharge_caseid is not None and los_caseid is not None
    if caseid_available:
        duplicate_caseid_count = int(base_caseid.duplicated().sum())
        missing_caseid_count = int(base_caseid.isna().sum())
    else:
        duplicate_caseid_count = 0
        missing_caseid_count = 0

    rows: list[dict[str, Any]] = []
    y_tensor = _base_y_tensor(data.base_dataset)
    for split_name, indices in split_indices.items():
        d_trace = _trace_positions(indices, batch_size)
        los_trace = _trace_positions(indices, batch_size)
        if len(d_trace) != len(los_trace) or len(d_trace) != len(indices):
            raise RuntimeError(f"Trace length mismatch for split={split_name}.")

        for start in range(0, len(indices), batch_size):
            chunk = np.asarray(indices[start : start + batch_size], dtype=np.int64)
            x = torch.stack([data.base_dataset[int(i)][0] for i in chunk], dim=0).to(device)
            _ = discharge_provider(x)
            _ = los_provider(x)

        for (position_in_loader, d_cache_row_idx), (_, los_cache_row_idx) in zip(d_trace, los_trace):
            gnn_row_idx = int(indices[position_in_loader])
            base_row_key = int(raw_row_index.iloc[gnn_row_idx]) if raw_row_index is not None else gnn_row_idx
            gnn_caseid = _sidecar_value(base_caseid, gnn_row_idx, fallback_row_idx=base_row_key)
            d_caseid = _sidecar_value(discharge_caseid, d_cache_row_idx, fallback_row_idx=d_cache_row_idx)
            los_caseid_value = _sidecar_value(los_caseid, los_cache_row_idx, fallback_row_idx=los_cache_row_idx)
            match_row_idx = gnn_row_idx == d_cache_row_idx == los_cache_row_idx
            match_caseid = gnn_caseid == d_caseid == los_caseid_value
            rows.append(
                {
                    "split": split_name,
                    "position_in_loader": int(position_in_loader),
                    "gnn_row_idx": int(gnn_row_idx),
                    "d_cache_row_idx": int(d_cache_row_idx),
                    "los_cache_row_idx": int(los_cache_row_idx),
                    "gnn_caseid": gnn_caseid,
                    "d_cache_caseid": d_caseid,
                    "los_cache_caseid": los_caseid_value,
                    "y": int(y_tensor[gnn_row_idx].item()),
                    "match_row_idx": bool(match_row_idx),
                    "match_caseid": bool(match_caseid),
                    "split_name_gnn": split_name,
                    "split_name_d_cache": split_name,
                    "split_name_los_cache": split_name,
                    "match_split": True,
                }
            )
    return rows, caseid_available, duplicate_caseid_count, missing_caseid_count


def _summarize_joint_alignment(
    rows: list[dict[str, Any]],
    *,
    caseid_available: bool,
    duplicate_caseid_count: int,
    missing_caseid_count: int,
) -> dict[str, Any]:
    split_names = ["train", "valid", "test"]
    mismatches: list[dict[str, Any]] = []
    split_summary: dict[str, dict[str, Any]] = {}

    for split_name in split_names:
        split_rows = [row for row in rows if row["split"] == split_name]
        gnn_row_id_match_d_cache = all(row["gnn_row_idx"] == row["d_cache_row_idx"] for row in split_rows)
        gnn_row_id_match_los_cache = all(row["gnn_row_idx"] == row["los_cache_row_idx"] for row in split_rows)
        d_cache_match_los_cache = all(row["d_cache_row_idx"] == row["los_cache_row_idx"] for row in split_rows)
        split_caseid_match = all(row["match_caseid"] for row in split_rows) if split_rows else True
        split_summary[split_name] = {
            "num_rows": len(split_rows),
            "gnn_row_id_match_d_cache": gnn_row_id_match_d_cache,
            "gnn_row_id_match_los_cache": gnn_row_id_match_los_cache,
            "d_cache_match_los_cache": d_cache_match_los_cache,
            "caseid_match": split_caseid_match,
            "row_idx_match_rate": float(np.mean([row["match_row_idx"] for row in split_rows])) if split_rows else 1.0,
            "caseid_match_rate": float(np.mean([row["match_caseid"] for row in split_rows])) if split_rows else 1.0,
        }

    for row in rows:
        split_match = row["split_name_gnn"] == row["split_name_d_cache"] == row["split_name_los_cache"]
        row_idx_match = row["gnn_row_idx"] == row["d_cache_row_idx"] == row["los_cache_row_idx"]
        caseid_match = row["match_caseid"]
        if not (split_match and row_idx_match and caseid_match):
            mismatches.append(
                {
                    "split": row["split"],
                    "position_in_loader": row["position_in_loader"],
                    "gnn_row_idx": row["gnn_row_idx"],
                    "d_cache_row_idx": row["d_cache_row_idx"],
                    "los_cache_row_idx": row["los_cache_row_idx"],
                    "gnn_caseid": row["gnn_caseid"],
                    "d_cache_caseid": row["d_cache_caseid"],
                    "los_cache_caseid": row["los_cache_caseid"],
                    "y": row["y"],
                    "match_row_idx": row["match_row_idx"],
                    "match_caseid": row["match_caseid"],
                }
            )

    row_idx_match_rate = float(np.mean([row["match_row_idx"] for row in rows])) if rows else 1.0
    caseid_match_rate = float(np.mean([row["match_caseid"] for row in rows])) if rows else 1.0
    return {
        "num_rows": len(rows),
        "splits": split_summary,
        "gnn_row_id_match_d_cache": all(row["gnn_row_idx"] == row["d_cache_row_idx"] for row in rows),
        "gnn_row_id_match_los_cache": all(row["gnn_row_idx"] == row["los_cache_row_idx"] for row in rows),
        "d_cache_match_los_cache": all(row["d_cache_row_idx"] == row["los_cache_row_idx"] for row in rows),
        "caseid_available": caseid_available,
        "caseid_match": all(row["match_caseid"] for row in rows),
        "row_idx_match_rate": row_idx_match_rate,
        "caseid_match_rate": caseid_match_rate,
        "num_mismatches": len(mismatches),
        "first_20_mismatches": mismatches[:20],
        "duplicate_caseid_count": duplicate_caseid_count,
        "missing_caseid_count": missing_caseid_count,
    }


def _print_joint_alignment_summary(summary: dict[str, Any]) -> None:
    print("[JOINT CACHE ALIGNMENT AUDIT]")
    print(f"num_rows={summary['num_rows']}")
    for split_name in ["train", "valid", "test"]:
        split_summary = summary["splits"][split_name]
        print(f"split={split_name}")
        print(f"gnn_row_id_match_d_cache={split_summary['gnn_row_id_match_d_cache']}")
        print(f"gnn_row_id_match_los_cache={split_summary['gnn_row_id_match_los_cache']}")
        print(f"d_cache_match_los_cache={split_summary['d_cache_match_los_cache']}")
        print(f"caseid_available={summary['caseid_available']}")
        print(f"caseid_match={split_summary['caseid_match']}")
    print(f"num_mismatches={summary['num_mismatches']}")
    print(f"first_20_mismatches={summary['first_20_mismatches']}")
    print(f"duplicate_caseid_count={summary['duplicate_caseid_count']}")
    print(f"missing_caseid_count={summary['missing_caseid_count']}")


def _write_joint_alignment_csv(path: str, rows: list[dict[str, Any]]) -> None:
    columns = [
        "split",
        "position_in_loader",
        "gnn_row_idx",
        "d_cache_row_idx",
        "los_cache_row_idx",
        "gnn_caseid",
        "d_cache_caseid",
        "los_cache_caseid",
        "y",
        "match_row_idx",
        "match_caseid",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def audit_joint_cache_alignment(
    data: DiagnosticData,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    discharge_checkpoint_path: str | None,
    los_checkpoint_path: str | None,
    diag_dir: str,
) -> dict[str, Any]:
    discharge_ckpt = discharge_checkpoint_path or cfg.get("forecasted_discharge", {}).get("checkpoint_path")
    los_ckpt = los_checkpoint_path or cfg.get("forecasted_los", {}).get("checkpoint_path")
    if not discharge_ckpt:
        raise RuntimeError("joint_cache_alignment_audit requires a discharge checkpoint path.")
    if not los_ckpt:
        raise RuntimeError("joint_cache_alignment_audit requires a LOS checkpoint path.")

    batch_size = int(cfg["train"]["batch_size"])
    discharge_provider = ForecastedDischargeProvider(
        _build_provider_cfg(cfg.get("forecasted_discharge", {}), str(discharge_ckpt)),
        data.base_dataset,
        device,
    )
    los_provider = ForecastedLOSProvider(
        _build_provider_cfg(cfg.get("forecasted_los", {}), str(los_ckpt)),
        data.base_dataset,
        device,
    )
    rows, caseid_available, duplicate_caseid_count, missing_caseid_count = _joint_alignment_rows(
        data,
        batch_size=batch_size,
        device=device,
        discharge_provider=discharge_provider,
        los_provider=los_provider,
    )
    summary = _summarize_joint_alignment(
        rows,
        caseid_available=caseid_available,
        duplicate_caseid_count=duplicate_caseid_count,
        missing_caseid_count=missing_caseid_count,
    )
    _write_joint_alignment_csv(os.path.join(diag_dir, "joint_cache_alignment_audit.csv"), rows)
    _write_json(os.path.join(diag_dir, "joint_cache_alignment_audit.json"), summary)
    _print_joint_alignment_summary(summary)
    if summary["num_mismatches"] > 0:
        raise RuntimeError(
            f"Joint cache alignment audit failed with {summary['num_mismatches']} mismatches."
        )
    return {"summary": summary, "rows_written": len(rows)}


def _write_csv_rows(path: str, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in columns})


def _problematic_head_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row["js_divergence_P_D_given_LOS"]),
        float(row["rare_combo_rate_predicted"]),
        abs(float(row["delta_cramers_v"])),
        str(row["target_name"]),
    )


def _print_joint_plausibility_summary(
    summary_rows: list[dict[str, Any]],
    top_problematic_heads: list[dict[str, Any]],
    *,
    rare_threshold: float,
) -> None:
    overall_row = next((row for row in summary_rows if row["split"] == "overall"), summary_rows[0])
    print("[JOINT PLAUSIBILITY AUDIT]")
    print(f"num_eval_rows={overall_row['num_eval_rows']}")
    print(f"rare_threshold={rare_threshold}")
    print(f"overall_rare_combo_rate_oracle={overall_row['overall_rare_combo_rate_oracle']:.6f}")
    print(f"overall_rare_combo_rate_predicted={overall_row['overall_rare_combo_rate_predicted']:.6f}")
    print(f"overall_rare_combo_rate_mixed_D_pred={overall_row['overall_rare_combo_rate_mixed_D_pred']:.6f}")
    print(f"overall_rare_combo_rate_mixed_LOS_pred={overall_row['overall_rare_combo_rate_mixed_LOS_pred']:.6f}")
    print("[Top problematic heads]")
    for rank, row in enumerate(top_problematic_heads[:10], start=1):
        print(
            f"{rank}. {row['target_name']}: "
            f"JS={float(row['js_divergence_P_D_given_LOS']):.6f}, "
            f"rare_rate_pred={float(row['rare_combo_rate_predicted']):.6f}, "
            f"delta_cramers_v={float(row['delta_cramers_v']):.6f}"
        )


def audit_joint_plausibility(
    data: DiagnosticData,
    cfg: dict[str, Any],
    device: torch.device,
    *,
    discharge_checkpoint_path: str | None,
    los_checkpoint_path: str | None,
    diag_dir: str,
    rare_threshold: float,
) -> dict[str, Any]:
    discharge_ckpt = discharge_checkpoint_path or cfg.get("forecasted_discharge", {}).get("checkpoint_path")
    los_ckpt = los_checkpoint_path or cfg.get("forecasted_los", {}).get("checkpoint_path")
    if not discharge_ckpt:
        raise RuntimeError("joint_plausibility_audit requires a discharge checkpoint path.")
    if not los_ckpt:
        raise RuntimeError("joint_plausibility_audit requires a LOS checkpoint path.")

    alignment = audit_joint_cache_alignment(
        data,
        cfg,
        device,
        discharge_checkpoint_path=str(discharge_ckpt),
        los_checkpoint_path=str(los_ckpt),
        diag_dir=diag_dir,
    )

    root = os.path.join(os.path.dirname(__file__), "..", "data")
    batch_size = int(cfg["train"]["batch_size"])
    oracle_d, oracle_los, labels, target_names, target_dims = _oracle_joint_targets(data)
    predicted = _collect_joint_predictions(
        data,
        root=root,
        device=device,
        batch_size=batch_size,
        discharge_checkpoint_path=str(discharge_ckpt),
        los_checkpoint_path=str(los_ckpt),
        diag_dir=diag_dir,
    )
    if predicted.target_names != target_names:
        raise RuntimeError(
            "Discharge target-name mismatch between oracle dataset and predicted provider outputs. "
            f"oracle={target_names} predicted={predicted.target_names}"
        )

    split_indices = _split_index_map(data)
    split_indices["overall"] = np.concatenate([data.train_idx, data.val_idx, data.test_idx]).astype(np.int64)
    base_caseid = getattr(data.base_dataset, "caseid_series", None)
    raw_row_index = getattr(data.base_dataset, "raw_row_index", None)
    train_oracle_idx = np.asarray(data.train_idx, dtype=np.int64)

    summary_rows: list[dict[str, Any]] = []
    conditional_rows: list[dict[str, Any]] = []
    rare_rows: list[dict[str, Any]] = []
    positive_rate_rows: list[dict[str, Any]] = []
    confidence_rows: list[dict[str, Any]] = []
    per_head_summary_rows: list[dict[str, Any]] = []

    train_oracle_rates, train_oracle_counts = _positive_rate_by_los(labels[train_oracle_idx], oracle_los[train_oracle_idx])

    rare_maps: list[np.ndarray] = []
    rare_probability_maps: list[dict[tuple[int, int], float]] = []
    for head_idx, target_name in enumerate(target_names):
        rare_map, probability_map = _rare_combo_map(
            oracle_d[train_oracle_idx, head_idx],
            oracle_los[train_oracle_idx],
            d_dim=target_dims[head_idx],
            threshold=rare_threshold,
        )
        rare_maps.append(rare_map)
        rare_probability_maps.append(probability_map)
        for los_bin in range(6):
            for d_value in range(target_dims[head_idx]):
                probability = float(probability_map[(d_value, los_bin)])
                rare_rows.append(
                    {
                        "target_name": target_name,
                        "los_bin": los_bin,
                        "d_value": d_value,
                        "train_oracle_count": int(round(probability * len(train_oracle_idx))),
                        "train_oracle_probability": probability,
                        "is_rare": probability < float(rare_threshold),
                    }
                )

    variant_los_by_split: dict[str, dict[str, np.ndarray]] = {}
    for split_name, indices in split_indices.items():
        split_labels = labels[indices]
        oracle_rates, oracle_counts = _positive_rate_by_los(split_labels, oracle_los[indices])
        predicted_rates, predicted_counts = _positive_rate_by_los(split_labels, predicted.los_pred[indices])
        variant_los_by_split[split_name] = {
            "oracle_rates": oracle_rates,
            "predicted_rates": predicted_rates,
        }
        for source_variant, los_values in {
            "oracle": oracle_los[indices],
            "predicted": predicted.los_pred[indices],
            "mixed_D_pred": oracle_los[indices],
            "mixed_LOS_pred": predicted.los_pred[indices],
        }.items():
            rates, counts = _positive_rate_by_los(split_labels, los_values)
            for los_bin in range(6):
                positive_rate_rows.append(
                    {
                        "split": split_name,
                        "source_variant": source_variant,
                        "los_bin": los_bin,
                        "positive_rate": None if np.isnan(rates[los_bin]) else float(rates[los_bin]),
                        "count": int(counts[los_bin]),
                        "positive_count": int(round(float(counts[los_bin]) * float(0.0 if np.isnan(rates[los_bin]) else rates[los_bin]))),
                    }
                )

    for split_name, indices in split_indices.items():
        split_oracle_rates = variant_los_by_split[split_name]["oracle_rates"]
        split_predicted_rates = variant_los_by_split[split_name]["predicted_rates"]
        split_positive_rate_drift = _mean_abs_rate_drift(split_oracle_rates, split_predicted_rates)
        split_head_rates_oracle: list[float] = []
        split_head_rates_predicted: list[float] = []
        split_head_rates_mixed_d: list[float] = []
        split_head_rates_mixed_los: list[float] = []

        for head_idx, target_name in enumerate(target_names):
            head_dim = target_dims[head_idx]
            oracle_table = _contingency_table(oracle_d[indices, head_idx], oracle_los[indices], head_dim)
            predicted_table = _contingency_table(predicted.discharge_pred[indices, head_idx], predicted.los_pred[indices], head_dim)
            mixed_d_pred_table = _contingency_table(predicted.discharge_pred[indices, head_idx], oracle_los[indices], head_dim)
            mixed_los_pred_table = _contingency_table(oracle_d[indices, head_idx], predicted.los_pred[indices], head_dim)

            js_divergence = _conditional_js_divergence(oracle_table, predicted_table)
            cramers_v_oracle = _cramers_v(oracle_table)
            cramers_v_predicted = _cramers_v(predicted_table)
            delta_cramers_v = float(cramers_v_predicted - cramers_v_oracle)

            rare_map = rare_maps[head_idx]
            rare_rate_oracle = _rare_rate_for_rows(oracle_d[indices, head_idx], oracle_los[indices], rare_map)
            rare_rate_predicted = _rare_rate_for_rows(
                predicted.discharge_pred[indices, head_idx],
                predicted.los_pred[indices],
                rare_map,
            )
            rare_rate_mixed_d = _rare_rate_for_rows(
                predicted.discharge_pred[indices, head_idx],
                oracle_los[indices],
                rare_map,
            )
            rare_rate_mixed_los = _rare_rate_for_rows(
                oracle_d[indices, head_idx],
                predicted.los_pred[indices],
                rare_map,
            )
            split_head_rates_oracle.append(rare_rate_oracle)
            split_head_rates_predicted.append(rare_rate_predicted)
            split_head_rates_mixed_d.append(rare_rate_mixed_d)
            split_head_rates_mixed_los.append(rare_rate_mixed_los)

            head_summary = {
                "split": split_name,
                "target_name": target_name,
                "cramers_v_oracle": cramers_v_oracle,
                "cramers_v_predicted": cramers_v_predicted,
                "delta_cramers_v": delta_cramers_v,
                "js_divergence_P_D_given_LOS": js_divergence,
                "rare_combo_rate_oracle": rare_rate_oracle,
                "rare_combo_rate_predicted": rare_rate_predicted,
                "rare_combo_rate_mixed_D_pred": rare_rate_mixed_d,
                "rare_combo_rate_mixed_LOS_pred": rare_rate_mixed_los,
                "positive_rate_drift_by_los": split_positive_rate_drift,
            }
            per_head_summary_rows.append(head_summary)

            oracle_conditional = _conditional_probs_from_table(oracle_table)
            predicted_conditional = _conditional_probs_from_table(predicted_table)
            mixed_d_conditional = _conditional_probs_from_table(mixed_d_pred_table)
            mixed_los_conditional = _conditional_probs_from_table(mixed_los_pred_table)
            for los_bin in range(6):
                for d_value in range(head_dim):
                    conditional_rows.append(
                        {
                            "split": split_name,
                            "target_name": target_name,
                            "los_bin": los_bin,
                            "d_value": d_value,
                            "oracle_count": int(oracle_table[los_bin, d_value]),
                            "oracle_prob": float(oracle_conditional[los_bin, d_value]),
                            "predicted_count": int(predicted_table[los_bin, d_value]),
                            "predicted_prob": float(predicted_conditional[los_bin, d_value]),
                            "mixed_D_pred_count": int(mixed_d_pred_table[los_bin, d_value]),
                            "mixed_D_pred_prob": float(mixed_d_conditional[los_bin, d_value]),
                            "mixed_LOS_pred_count": int(mixed_los_pred_table[los_bin, d_value]),
                            "mixed_LOS_pred_prob": float(mixed_los_conditional[los_bin, d_value]),
                            "train_oracle_probability": float(rare_probability_maps[head_idx][(d_value, los_bin)]),
                            "train_oracle_is_rare": bool(rare_maps[head_idx][los_bin, d_value]),
                            **head_summary,
                        }
                    )

            for row_idx in indices.tolist():
                base_row_key = int(raw_row_index.iloc[row_idx]) if raw_row_index is not None else int(row_idx)
                caseid = _sidecar_value(base_caseid, int(row_idx), fallback_row_idx=base_row_key)
                oracle_pair = (
                    int(oracle_d[row_idx, head_idx]),
                    int(oracle_los[row_idx]),
                )
                predicted_pair = (
                    int(predicted.discharge_pred[row_idx, head_idx]),
                    int(predicted.los_pred[row_idx]),
                )
                confidence_rows.append(
                    {
                        "split": split_name,
                        "row_idx": int(row_idx),
                        "caseid": caseid,
                        "target_name": target_name,
                        "y": int(labels[row_idx]),
                        "oracle_d": oracle_pair[0],
                        "predicted_d": predicted_pair[0],
                        "oracle_los_bin": oracle_pair[1],
                        "predicted_los_bin": predicted_pair[1],
                        "rare_oracle": bool(rare_maps[head_idx][oracle_pair[1], oracle_pair[0]]),
                        "rare_predicted": bool(rare_maps[head_idx][predicted_pair[1], predicted_pair[0]]),
                        "rare_mixed_D_pred": bool(
                            rare_maps[head_idx][int(oracle_los[row_idx]), int(predicted.discharge_pred[row_idx, head_idx])]
                        ),
                        "rare_mixed_LOS_pred": bool(
                            rare_maps[head_idx][int(predicted.los_pred[row_idx]), int(oracle_d[row_idx, head_idx])]
                        ),
                        "joint_mismatch_predicted": bool(predicted_pair != oracle_pair),
                        "discharge_confidence": float(predicted.discharge_conf[row_idx, head_idx]),
                        "los_confidence": float(predicted.los_conf[row_idx]),
                    }
                )

        summary_rows.append(
            {
                "split": split_name,
                "num_eval_rows": int(len(indices)),
                "rare_threshold": float(rare_threshold),
                "overall_rare_combo_rate_oracle": _safe_mean(split_head_rates_oracle),
                "overall_rare_combo_rate_predicted": _safe_mean(split_head_rates_predicted),
                "overall_rare_combo_rate_mixed_D_pred": _safe_mean(split_head_rates_mixed_d),
                "overall_rare_combo_rate_mixed_LOS_pred": _safe_mean(split_head_rates_mixed_los),
            }
        )

    top_problematic_heads = sorted(
        [row for row in per_head_summary_rows if row["split"] == "overall"],
        key=_problematic_head_sort_key,
        reverse=True,
    )[:10]

    _write_csv_rows(
        os.path.join(diag_dir, "joint_plausibility_summary.csv"),
        summary_rows,
        [
            "split",
            "num_eval_rows",
            "rare_threshold",
            "overall_rare_combo_rate_oracle",
            "overall_rare_combo_rate_predicted",
            "overall_rare_combo_rate_mixed_D_pred",
            "overall_rare_combo_rate_mixed_LOS_pred",
        ],
    )
    _write_csv_rows(
        os.path.join(diag_dir, "per_head_conditional_distribution.csv"),
        conditional_rows,
        [
            "split",
            "target_name",
            "los_bin",
            "d_value",
            "oracle_count",
            "oracle_prob",
            "predicted_count",
            "predicted_prob",
            "mixed_D_pred_count",
            "mixed_D_pred_prob",
            "mixed_LOS_pred_count",
            "mixed_LOS_pred_prob",
            "train_oracle_probability",
            "train_oracle_is_rare",
            "cramers_v_oracle",
            "cramers_v_predicted",
            "delta_cramers_v",
            "js_divergence_P_D_given_LOS",
            "rare_combo_rate_oracle",
            "rare_combo_rate_predicted",
            "rare_combo_rate_mixed_D_pred",
            "rare_combo_rate_mixed_LOS_pred",
            "positive_rate_drift_by_los",
        ],
    )
    _write_csv_rows(
        os.path.join(diag_dir, "rare_combinations.csv"),
        rare_rows,
        [
            "target_name",
            "los_bin",
            "d_value",
            "train_oracle_count",
            "train_oracle_probability",
            "is_rare",
        ],
    )
    _write_csv_rows(
        os.path.join(diag_dir, "los_bin_positive_rate.csv"),
        positive_rate_rows,
        ["split", "source_variant", "los_bin", "positive_rate", "count", "positive_count"],
    )
    _write_csv_rows(
        os.path.join(diag_dir, "confidence_vs_joint_mismatch.csv"),
        confidence_rows,
        [
            "split",
            "row_idx",
            "caseid",
            "target_name",
            "y",
            "oracle_d",
            "predicted_d",
            "oracle_los_bin",
            "predicted_los_bin",
            "rare_oracle",
            "rare_predicted",
            "rare_mixed_D_pred",
            "rare_mixed_LOS_pred",
            "joint_mismatch_predicted",
            "discharge_confidence",
            "los_confidence",
        ],
    )

    payload = {
        "summary_rows": summary_rows,
        "per_head_summary": per_head_summary_rows,
        "top_problematic_heads": top_problematic_heads,
        "rare_threshold": float(rare_threshold),
        "num_eval_rows": int(len(split_indices["overall"])),
        "provenance": predicted.provenance,
        "alignment": alignment,
        "positive_rate_reference_train": {
            "rates": train_oracle_rates.tolist(),
            "counts": train_oracle_counts.tolist(),
        },
    }
    _write_json(os.path.join(diag_dir, "joint_plausibility_summary.json"), payload)
    _print_joint_plausibility_summary(summary_rows, top_problematic_heads, rare_threshold=float(rare_threshold))
    return payload


def _build_dataloaders_from_cache(data: DiagnosticData, x_cache: torch.Tensor, los_cache: torch.Tensor):
    cfg = data.cfg
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    cached_dataset = ForecastCacheDataset(data.base_dataset, x_cache.cpu(), los_cache.cpu())
    return (
        _make_loader(cached_dataset, data.train_idx, batch_size, num_workers, shuffle=True),
        _make_loader(cached_dataset, data.val_idx, batch_size, num_workers, shuffle=False),
        _make_loader(cached_dataset, data.test_idx, batch_size, num_workers, shuffle=False),
    )


def _train_with_cache(data: DiagnosticData, x_cache: torch.Tensor, los_cache: torch.Tensor, device: torch.device, mode: str) -> dict[str, Any]:
    from src.trainers import base as train_base

    cfg = copy.deepcopy(data.cfg)
    cfg["run_name"] = f"diagnose_{mode}__{cfg.get('run_name', cfg.get('model', {}).get('name', 'model'))}"
    cfg["device"] = str(device)
    cfg["model"]["params"]["col_info"] = data.base_dataset.col_info
    cfg["model"]["params"]["num_classes"] = data.base_dataset.num_classes
    cfg["model"]["params"]["device"] = str(device)
    cfg["forecasted_los"] = {"enabled": False}
    cfg["forecasted_discharge"] = {"enabled": False}

    run_dir = ensure_run_dir("runs", make_run_id(cfg))
    logger = ExperimentLogger(cfg, run_dir)
    save_yaml(os.path.join(run_dir, "diagnostic_config.yaml"), cfg)

    train_loader, val_loader, test_loader = _build_dataloaders_from_cache(data, x_cache, los_cache)
    train_df = data.base_dataset.processed_df.iloc[data.train_idx]

    num_nodes = len(data.base_dataset.col_info[2])
    edge_index = build_edge(
        model_name=cfg["model"]["name"],
        root=os.path.join(os.path.dirname(__file__), "..", "data"),
        seed=int(cfg["train"].get("seed", 42)),
        train_df=train_df,
        num_nodes=num_nodes,
        batch_size=int(cfg["train"]["batch_size"]),
        **cfg.get("edge", {}),
    ).to(device)

    model = build_model(model_name=cfg["model"]["name"], **cfg["model"].get("params", {})).to(device)
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, int(cfg["train"]["batch_size"]))

    criterion = nn.BCEWithLogitsLoss() if cfg["train"]["binary"] else nn.CrossEntropyLoss()
    if cfg["train"].get("optimizer", "adam") == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["train"]["learning_rate"]),
            weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["train"]["learning_rate"]),
            weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
        )
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=int(cfg["train"]["lr_scheduler_patience"]))
    early_stopper = EarlyStopper(patience=int(cfg["train"]["early_stopping_patience"]))

    original_send = train_base.send_discord_message
    train_base.send_discord_message = lambda message: print(f"[diagnostic] Discord message skipped: {message}")
    try:
        results = train_base.run_train_loop(
            model=model,
            edge_index=edge_index,
            binary=bool(cfg["train"]["binary"]),
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            device=device,
            logger=logger,
            epochs=int(cfg["train"]["epochs"]),
            decision_threshold=float(cfg["train"]["decision_threshold"]),
            model_name=cfg["model"]["name"],
            trial=None,
            los_provider=None,
            discharge_provider=None,
        )
    finally:
        train_base.send_discord_message = original_send

    results["run_dir"] = run_dir
    with open(os.path.join(run_dir, "diagnostic_summary.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=_json_default)
    return results


def run_diagnostic(
    config_path: str,
    mode: str,
    fold: int,
    seed: int,
    device_name: str | None = None,
    discharge_checkpoint_path: str | None = None,
    los_checkpoint_path: str | None = None,
    override_head: str | None = None,
    rare_threshold: float = 0.0001,
    dry_run: bool = False,
) -> dict[str, Any]:
    requested_mode = mode
    mode = _normalized_mode(mode)
    cfg = _load_yaml(config_path)
    if device_name is not None:
        cfg["device"] = device_name
    device = torch.device(cfg.get("device") or "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is not available.")

    root = os.path.join(os.path.dirname(__file__), "..", "data")
    data = _prepare_data(cfg, root, fold, seed)
    diag_dir = _diagnostic_dir()
    audit = audit_forecast_value_space(
        data.base_dataset,
        data.discharge_dataset,
        data.los_dataset,
        data.cfg,
        los_cfg=data.los_cfg,
    )
    _print_audit_summary(audit)
    _write_json(os.path.join(diag_dir, "audit_summary.json"), audit)
    _write_json(os.path.join(diag_dir, "los_mapping.json"), _los_mapping_payload())

    if mode == "joint_cache_alignment_audit":
        result = audit_joint_cache_alignment(
            data,
            data.cfg,
            device,
            discharge_checkpoint_path=discharge_checkpoint_path,
            los_checkpoint_path=los_checkpoint_path,
            diag_dir=diag_dir,
        )
        return {"audit": audit, "joint_cache_alignment": result, "diagnostic_dir": diag_dir}

    if mode == "joint_plausibility_audit":
        result = audit_joint_plausibility(
            data,
            data.cfg,
            device,
            discharge_checkpoint_path=discharge_checkpoint_path,
            los_checkpoint_path=los_checkpoint_path,
            diag_dir=diag_dir,
            rare_threshold=float(rare_threshold),
        )
        return {"audit": audit, "joint_plausibility": result, "diagnostic_dir": diag_dir}

    if mode in {"audit_only", "predictor_target_cache"} and (
        not audit["all_discharge_value_match"] or not audit["los_value_match"]
    ):
        _write_metrics_summary(
            os.path.join(diag_dir, "metrics_summary.csv"),
            _metrics_row(
                mode=mode,
                fold=fold,
                seed=seed,
                results=None,
                los_source="predictor_target",
                los_target_mode=audit["los_value_check"]["target_mode"],
                los_input_mode="raw",
                los_values=None,
                basis_valid=False,
                notes="value-space audit failed",
            ),
        )
        failed = []
        if not audit["all_discharge_value_match"]:
            failed.append("discharge")
        if not audit["los_value_match"]:
            failed.append("LOS")
        raise RuntimeError(
            "Value-space audit failed before cache training: "
            f"{', '.join(failed)} predictor target encoding does not match CTMP-GIN input encoding."
        )

    if mode == "audit_only":
        _write_metrics_summary(
            os.path.join(diag_dir, "metrics_summary.csv"),
            _metrics_row(
                mode=mode,
                fold=fold,
                seed=seed,
                results=None,
                los_source="audit",
                los_target_mode=audit["los_value_check"]["target_mode"],
                los_input_mode="raw",
                los_values=None,
                basis_valid=bool(audit["los_index_valid"]),
                notes="audit only",
            ),
        )
        return audit
    if mode == "oracle_cache":
        x_cache, los_cache = build_oracle_forecast_cache(data.base_dataset, device="cpu")
        mode_payload: dict[str, Any] = {}
        los_source = "oracle_raw"
        los_input_mode = "raw"
    elif mode == "predictor_target_cache":
        x_cache, los_cache = build_predictor_target_forecast_cache(
            data.base_dataset,
            data.discharge_dataset,
            data.los_dataset,
            los_cfg=data.los_cfg,
            device="cpu",
        )
        mode_payload = {}
        los_source = "predictor_target"
        los_input_mode = "raw"
    elif mode == "predictor_target_cache_transformed":
        x_cache, los_cache, mode_payload = build_predictor_target_transformed_forecast_cache(
            data.base_dataset,
            data.discharge_dataset,
            data.los_dataset,
            los_cfg=data.los_cfg,
            device="cpu",
        )
        _print_transformed_payload(mode_payload)
        _write_json(os.path.join(diag_dir, "predictor_target_cache_transformed.json"), mode_payload)
        los_source = "predictor_target_transformed"
        los_input_mode = "representative_raw"
    elif mode == "oracle_cache_coarse_los":
        x_cache, los_cache, mode_payload = build_oracle_coarse_los_forecast_cache(
            data.base_dataset,
            device="cpu",
        )
        _print_oracle_coarse_payload(mode_payload)
        _write_json(os.path.join(diag_dir, "oracle_cache_coarse_los.json"), mode_payload)
        los_source = "oracle_coarse_representative"
        los_input_mode = "representative_raw"
    elif mode in {
        "oracle_d_predicted_los_hard",
        "oracle_d_predicted_los_distribution",
        "predicted_d_oracle_los",
        "predicted_d_predicted_los",
    }:
        x_cache, los_cache, mode_payload = build_mixed_actual_forecast_cache(
            data,
            root,
            diag_dir,
            device,
            mode=mode,
        )
        mode_payload["requested_mode"] = requested_mode
        print("[Mixed actual forecast cache]")
        for key in [
            "mode",
            "discharge_source",
            "los_source",
            "train_prediction_mode",
            "train_core_size",
            "predictor_val_size",
            "gnn_val_size",
            "test_size",
            "los_return_type",
            "cache_roundtrip_match",
        ]:
            print(f"{key}={mode_payload[key]}")
        _write_json(os.path.join(diag_dir, f"{mode}.json"), mode_payload)
        los_source = str(mode_payload["los_source"])
        los_input_mode = "distribution" if los_cache.ndim == 2 else "raw"
    elif mode in {
        "predicted_d_predicted_los_oracle_head_ablation",
        "oracle_d_predicted_los_predicted_head_ablation",
    }:
        x_cache, los_cache, mode_payload = build_single_head_ablation_forecast_cache(
            data,
            root,
            diag_dir,
            device,
            mode=mode,
            override_head=str(override_head) if override_head is not None else None,
        )
        mode_payload["requested_mode"] = requested_mode
        print("[Head override ablation cache]")
        for key in [
            "mode",
            "override_head",
            "base_d_source",
            "override_head_source",
            "los_source",
            "cache_roundtrip_match",
        ]:
            print(f"{key}={mode_payload[key]}")
        print(
            "overall_changed_rows="
            f"{mode_payload['overall_override_summary']['num_changed_rows']}/"
            f"{mode_payload['overall_override_summary']['num_override_rows']}"
        )
        _write_json(os.path.join(diag_dir, f"{mode}.json"), mode_payload)
        los_source = str(mode_payload["los_source"])
        los_input_mode = "distribution" if los_cache.ndim == 2 else "raw"
    elif mode == "los_distribution_basis_audit":
        basis_payload = audit_los_distribution_basis(data.cfg, data.base_dataset, device)
        _print_distribution_basis(basis_payload)
        _write_json(os.path.join(diag_dir, "los_distribution_basis_audit.json"), basis_payload)
        _write_metrics_summary(
            os.path.join(diag_dir, "metrics_summary.csv"),
            _metrics_row(
                mode=mode,
                fold=fold,
                seed=seed,
                results=None,
                los_source="distribution_audit",
                los_target_mode=str(basis_payload["los_target_mode"]),
                los_input_mode="distribution",
                los_values=None,
                basis_valid=bool(basis_payload["basis_valid"]),
                notes="expanded raw distribution basis",
            ),
        )
        return {"audit": audit, "basis": basis_payload, "diagnostic_dir": diag_dir}
    elif mode == "los_hard_runtime_audit":
        hard_payload = audit_los_hard_runtime(
            data.cfg,
            data.base_dataset,
            device,
            los_checkpoint_path=los_checkpoint_path,
        )
        _print_hard_runtime(hard_payload)
        _write_json(os.path.join(diag_dir, "los_hard_runtime_audit.json"), hard_payload)
        _write_metrics_summary(
            os.path.join(diag_dir, "metrics_summary.csv"),
            _metrics_row(
                mode=mode,
                fold=fold,
                seed=seed,
                results=None,
                los_source="hard_runtime_audit",
                los_target_mode=str(data.cfg.get("forecasted_los", {}).get("target_mode", "coarse")),
                los_input_mode="hard",
                los_values=torch.tensor(hard_payload["injected_los_unique"], dtype=torch.long),
                basis_valid=not bool(hard_payload["uses_los_zero"]),
                notes="provider checked" if hard_payload["runtime_provider_checked"] else "mapping contract only",
            ),
        )
        return {"audit": audit, "hard_runtime": hard_payload, "diagnostic_dir": diag_dir}
    else:
        raise ValueError(f"Unsupported diagnostic mode: {mode}")

    for row in data.train_idx[: min(len(data.train_idx), 1024)]:
        row_i = int(row)
        x_item, _, los_item = data.base_dataset[row_i]
        if mode == "oracle_cache":
            if not torch.equal(x_cache[row_i].long(), x_item.long().cpu()):
                _raise_value_mismatch(str(row_i), x_cache[row_i], x_item.long().cpu(), "Oracle cache x mismatch")
            if int(los_cache[row_i]) != int(los_item):
                raise RuntimeError(f"Oracle cache LOS mismatch at row={row_i}")

    if dry_run:
        notes = f"dry_run cache_valid mode_payload_keys={sorted(mode_payload.keys())}"
        _write_json(os.path.join(diag_dir, f"{mode}.json"), {**mode_payload, "dry_run": True})
        _write_metrics_summary(
            os.path.join(diag_dir, "metrics_summary.csv"),
            _metrics_row(
                mode=mode,
                fold=fold,
                seed=seed,
                results=None,
                los_source=los_source,
                los_target_mode=audit["los_value_check"]["target_mode"],
                los_input_mode=los_input_mode,
                los_values=los_cache,
                basis_valid=True,
                notes=notes,
            ),
        )
        print(f"[DIAGNOSTIC DRY RUN] mode={mode} diagnostic_dir={diag_dir}")
        return {"audit": audit, "mode_payload": mode_payload, "diagnostic_dir": diag_dir, "dry_run": True}

    results = _train_with_cache(data, x_cache, los_cache, device, mode)
    _write_json(os.path.join(diag_dir, f"{mode}.json"), {**mode_payload, "results": results})
    _write_metrics_summary(
        os.path.join(diag_dir, "metrics_summary.csv"),
        _metrics_row(
            mode=mode,
            fold=fold,
            seed=seed,
            results=results,
            los_source=los_source,
            los_target_mode=audit["los_value_check"]["target_mode"],
            los_input_mode=los_input_mode,
            los_values=los_cache,
            basis_valid=True,
            notes=f"training_run_dir={results['run_dir']}",
        ),
    )
    print(f"[DIAGNOSTIC RESULT] mode={mode} run_dir={results['run_dir']}")
    for key in ["test_loss", "test_acc", "test_precision", "test_recall", "test_f1", "test_auc"]:
        if key in results:
            print(f"{key}={results[key]}")
    return {"audit": audit, "results": results, "diagnostic_dir": diag_dir}
