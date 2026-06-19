from __future__ import annotations

import copy
import json
import os
import shutil
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from src.data_processing.splits import (
    holdout_test_split_stratified,
    kfold_stratified,
    make_loaders,
)
from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.forecast_inputs import (
    ensure_model_forecast_defaults,
    resolve_model_forecast_input_metadata,
)
from src.models.factory import build_edge, build_model
from src.trainers.base import evaluate, run_train_loop
from src.trainers.forecasted_discharge import build_forecasted_discharge_provider
from src.trainers.forecasted_los import build_forecasted_los_provider
from src.trainers.forecasted_pipeline import (
    ForecastCacheDataset,
    ForecastedFoldData,
    joint_forecast_pipeline_enabled,
    prepare_forecasted_fold_data,
    prepare_joint_forecast_fold_data,
)
from src.trainers.outcome_aware_stage2 import (
    resolve_stage2_pretrained_paths,
    run_outcome_aware_stage2,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.device_set import device_set
from src.utils.experiment import (
    ExperimentLogger,
    _get_command_line,
    _get_git_info,
    ensure_run_dir,
    make_run_id,
    save_text,
    save_yaml,
)
from src.utils.seed_set import set_seed


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=False)


def _cv_run_id(cfg: dict[str, Any]) -> str:
    k = cfg["train"]["n_folds"]
    test_ratio = cfg["train"]["test_ratio"]
    return make_run_id(cfg) + f"__cv={k}__test={test_ratio}"


def _cv_dir_from_cfg(cfg: dict[str, Any], cv_run_dir: str | None = None) -> str:
    if cv_run_dir is not None:
        return cv_run_dir
    return os.path.join("runs", _cv_run_id(cfg))


def _splits_path(cv_dir: str) -> str:
    return os.path.join(cv_dir, "kfold_splits.json")


def _fold_dir(cv_dir: str, fold: int) -> str:
    return os.path.join(cv_dir, "folds", f"fold_{fold}")


def _source_fold_dir(source_run_dir: str, fold: int) -> str:
    return os.path.join(str(source_run_dir), "folds", f"fold_{int(fold)}")


def _resolve_stage2_source_artifacts(
    source_run_dir: str,
    fold: int,
) -> dict[str, str]:
    source_run_dir = str(source_run_dir)
    single_run_cfg_path = os.path.join(source_run_dir, "config.final.yaml")
    single_run_split_path = os.path.join(source_run_dir, "single_run_splits.json")
    if os.path.exists(single_run_cfg_path) and os.path.exists(single_run_split_path):
        return {
            "source_kind": "single_run",
            "source_artifact_dir": source_run_dir,
            "source_cfg_path": single_run_cfg_path,
            "split_path": single_run_split_path,
        }

    source_fold_dir = _source_fold_dir(source_run_dir, fold)
    return {
        "source_kind": "kfold_fold",
        "source_artifact_dir": source_fold_dir,
        "source_cfg_path": os.path.join(source_fold_dir, "config.final.yaml"),
        "split_path": os.path.join(source_fold_dir, "joint_forecast_pipeline_splits.json"),
    }


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _load_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_yaml_file(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _single_run_status_path(run_dir: str) -> str:
    return os.path.join(run_dir, "single_run_status.json")


def _single_run_result_path(run_dir: str) -> str:
    return os.path.join(run_dir, "single_run_result.json")


def _write_single_run_status(
    run_dir: str,
    *,
    status: str,
    current_stage: str | None = None,
    last_completed_stage: str | None = None,
    error: str | None = None,
    traceback_text: str | None = None,
) -> None:
    payload: dict[str, Any] = {
        "status": status,
        "run_dir": run_dir,
    }
    if current_stage is not None:
        payload["current_stage"] = current_stage
    if last_completed_stage is not None:
        payload["last_completed_stage"] = last_completed_stage
    if error is not None:
        payload["error"] = error
    if traceback_text is not None:
        payload["traceback"] = traceback_text
    _save_json(_single_run_status_path(run_dir), payload)


def _single_run_paths(run_dir: str) -> dict[str, str]:
    return {
        "config": os.path.join(run_dir, "config.final.yaml"),
        "single_run_splits": os.path.join(run_dir, "single_run_splits.json"),
        "joint_splits": os.path.join(run_dir, "joint_forecast_pipeline_splits.json"),
        "joint_predictor_ckpt": os.path.join(
            run_dir, "joint_predictor", "checkpoints", "best.pt"
        ),
        "cache_train": os.path.join(run_dir, "cached_predictions", "train_core_joint.pt"),
        "cache_val": os.path.join(run_dir, "cached_predictions", "gnn_val_joint.pt"),
        "cache_test": os.path.join(run_dir, "cached_predictions", "outer_test_joint.pt"),
        "baseline_ckpt": os.path.join(run_dir, "checkpoints", "best.pt"),
        "edge_index": os.path.join(run_dir, "edge_index.pt"),
        "metrics": os.path.join(run_dir, "metrics.jsonl"),
        "best_txt": os.path.join(run_dir, "best.txt"),
        "stage2_dir": os.path.join(run_dir, "outcome_aware_stage2"),
        "result": _single_run_result_path(run_dir),
        "status": _single_run_status_path(run_dir),
    }


def _load_single_run_cfg_for_resume(
    cfg: dict[str, Any],
    resume_run_dir: str | None,
) -> tuple[dict[str, Any], str | None]:
    if not resume_run_dir:
        return copy.deepcopy(cfg), None

    run_dir = str(resume_run_dir)
    paths = _single_run_paths(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Resume run dir not found: {run_dir}")
    if not os.path.exists(paths["config"]):
        raise FileNotFoundError(f"Resume single-run config not found: {paths['config']}")

    resumed_cfg = _load_yaml_file(paths["config"])
    ensure_model_forecast_defaults(resumed_cfg)
    if cfg.get("device") is not None and resumed_cfg.get("device") is None:
        resumed_cfg["device"] = cfg["device"]
    return resumed_cfg, run_dir


def _single_run_stage_state(run_dir: str) -> dict[str, bool]:
    paths = _single_run_paths(run_dir)
    stage1_complete = all(
        os.path.exists(paths[key])
        for key in (
            "joint_predictor_ckpt",
            "joint_splits",
            "cache_train",
            "cache_val",
            "cache_test",
        )
    )
    baseline_complete = stage1_complete and os.path.exists(paths["baseline_ckpt"])
    result_complete = False
    if os.path.exists(paths["result"]):
        try:
            result_payload = _load_json(paths["result"])
            result_complete = str(result_payload.get("status", "")).lower() == "completed"
        except Exception:
            result_complete = False
    stage2_complete = baseline_complete and result_complete
    return {
        "stage1_complete": stage1_complete,
        "baseline_complete": baseline_complete,
        "stage2_complete": stage2_complete,
    }


def _backup_path(path: str, suffix: str) -> str | None:
    if not os.path.exists(path):
        return None
    backup_path = f"{path}_{suffix}_{time.strftime('%Y%m%d-%H%M%S')}"
    shutil.move(path, backup_path)
    return backup_path


def _backup_incomplete_single_run_stage(run_dir: str, stage_name: str) -> None:
    paths = _single_run_paths(run_dir)
    if stage_name == "stage1":
        for path in (
            os.path.join(run_dir, "joint_predictor"),
            os.path.join(run_dir, "cached_predictions"),
            paths["joint_splits"],
            paths["single_run_splits"],
            os.path.join(run_dir, "forecast_input_metadata.json"),
        ):
            _backup_path(path, "stage1_incomplete")
    elif stage_name == "baseline":
        for path in (
            os.path.join(run_dir, "checkpoints"),
            paths["metrics"],
            paths["best_txt"],
        ):
            _backup_path(path, "baseline_incomplete")
    elif stage_name == "stage2":
        _backup_path(paths["stage2_dir"], "stage2_incomplete")


class _CachedSplitDataset(Dataset):
    def __init__(
        self,
        base_dataset: TEDSTensorDataset,
        split_payload: dict[str, Any],
    ) -> None:
        self.base_dataset = base_dataset
        self.x = split_payload["x"].long()
        los = split_payload["los"]
        self.los = los.float() if los.ndim == 2 else los.long()
        self.indices = torch.as_tensor(split_payload["indices"], dtype=torch.long)
        self.soft_discharge_cache = split_payload.get("soft_discharge")

    def __len__(self) -> int:
        return int(self.indices.numel())

    def __getitem__(self, index: int):
        row_idx = int(self.indices[index].item())
        _, y, _ = self.base_dataset[row_idx]
        if self.soft_discharge_cache is None:
            return self.x[index], y, self.los[index]

        soft_discharge: dict[str, dict[str, torch.Tensor | int]] = {}
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
        return self.x[index], y, self.los[index], forecast_meta


def _build_loader_from_cached_payload(
    *,
    base_dataset: TEDSTensorDataset,
    split_payload: dict[str, Any],
    expected_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    actual_indices = torch.as_tensor(split_payload["indices"], dtype=torch.long).cpu().numpy()
    expected = np.asarray(expected_indices, dtype=np.int64)
    if not np.array_equal(actual_indices, expected):
        raise ValueError("Cached split indices do not match saved split metadata.")
    dataset = _CachedSplitDataset(base_dataset, split_payload)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def _load_fold_forecasted_data_from_cache(
    cfg: dict[str, Any],
    *,
    fold_dir: str,
    base_dataset: TEDSTensorDataset,
) -> ForecastedFoldData:
    paths = _single_run_paths(fold_dir)
    split_payload = _load_json(paths["joint_splits"])
    train_idx = np.asarray(split_payload["train_core_idx"], dtype=np.int64)
    val_idx = np.asarray(split_payload["gnn_val_idx"], dtype=np.int64)
    test_idx = np.asarray(split_payload["outer_test_idx"], dtype=np.int64)
    train_cache = torch.load(paths["cache_train"], map_location="cpu", weights_only=False)
    val_cache = torch.load(paths["cache_val"], map_location="cpu", weights_only=False)
    test_cache = torch.load(paths["cache_test"], map_location="cpu", weights_only=False)
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    return ForecastedFoldData(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_loader=_build_loader_from_cached_payload(
            base_dataset=base_dataset,
            split_payload=train_cache,
            expected_indices=train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        val_loader=_build_loader_from_cached_payload(
            base_dataset=base_dataset,
            split_payload=val_cache,
            expected_indices=val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        test_loader=_build_loader_from_cached_payload(
            base_dataset=base_dataset,
            split_payload=test_cache,
            expected_indices=test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        split_payload=split_payload,
    )


def _load_single_run_forecasted_data(
    cfg: dict[str, Any],
    *,
    run_dir: str,
    base_dataset: TEDSTensorDataset,
) -> ForecastedFoldData:
    paths = _single_run_paths(run_dir)
    single_splits = _load_json(paths["single_run_splits"])
    split_payload = _load_json(paths["joint_splits"])
    train_idx = np.asarray(single_splits["train_core_idx"], dtype=np.int64)
    val_idx = np.asarray(single_splits["gnn_val_idx"], dtype=np.int64)
    test_idx = np.asarray(single_splits["stage2_test_idx"], dtype=np.int64)
    train_cache = torch.load(paths["cache_train"], map_location="cpu", weights_only=False)
    val_cache = torch.load(paths["cache_val"], map_location="cpu", weights_only=False)
    test_cache = torch.load(paths["cache_test"], map_location="cpu", weights_only=False)
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    return ForecastedFoldData(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        train_loader=_build_loader_from_cached_payload(
            base_dataset=base_dataset,
            split_payload=train_cache,
            expected_indices=train_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        ),
        val_loader=_build_loader_from_cached_payload(
            base_dataset=base_dataset,
            split_payload=val_cache,
            expected_indices=val_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        test_loader=_build_loader_from_cached_payload(
            base_dataset=base_dataset,
            split_payload=test_cache,
            expected_indices=test_idx,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        ),
        split_payload=split_payload,
    )


def _restore_logger_best_from_checkpoint(
    logger: ExperimentLogger,
    *,
    map_location: torch.device,
) -> None:
    best_path = os.path.join(logger.ckpt_dir, "best.pt")
    if not os.path.exists(best_path):
        return
    state = torch.load(best_path, map_location=map_location, weights_only=False)
    metrics = state.get("metrics", {})
    monitor = str(logger.policy.monitor)
    if monitor not in metrics:
        return
    try:
        logger.best_value = float(metrics[monitor])
        logger.best_epoch = int(state.get("epoch"))
    except Exception:
        logger.best_value = None
        logger.best_epoch = None


def _restore_training_state_from_last_checkpoint(
    *,
    fold_dir: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    logger: ExperimentLogger,
    device: torch.device,
) -> int:
    last_path = os.path.join(fold_dir, "checkpoints", "last.pt")
    if not os.path.exists(last_path):
        raise FileNotFoundError(f"Cannot resume fold: checkpoint not found: {last_path}")
    state = torch.load(last_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    if state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    _restore_logger_best_from_checkpoint(logger, map_location=device)
    if logger.best_value is None:
        metrics = state.get("metrics", {})
        monitor = str(logger.policy.monitor)
        if monitor in metrics:
            try:
                logger.best_value = float(metrics[monitor])
                logger.best_epoch = int(state.get("epoch"))
            except Exception:
                logger.best_value = None
                logger.best_epoch = None
    last_epoch = int(state.get("epoch", 0))
    start_epoch = last_epoch + 1
    print(f"[resume] loaded {last_path}; continuing from epoch {start_epoch}")
    return start_epoch


def _load_edge_index_for_single_run(
    *,
    cfg: dict[str, Any],
    root: str,
    run_dir: str,
    dataset: TEDSTensorDataset,
    train_idx: np.ndarray,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    paths = _single_run_paths(run_dir)
    if os.path.exists(paths["edge_index"]):
        return torch.load(paths["edge_index"], map_location=device).to(device)

    train_df = dataset.processed_df.iloc[train_idx]
    edge_index = build_edge(
        model_name=cfg["model"]["name"],
        root=root,
        seed=int(cfg["train"].get("seed", 42)),
        train_df=train_df,
        num_nodes=num_nodes,
        batch_size=int(cfg["train"]["batch_size"]),
        **cfg.get("edge", {}),
    ).to(device)
    torch.save(edge_index.cpu(), paths["edge_index"])
    return edge_index


def _reconstruct_baseline_results(
    *,
    cfg: dict[str, Any],
    run_dir: str,
    model: nn.Module,
    edge_index: torch.Tensor,
    forecasted_data: ForecastedFoldData,
    device: torch.device,
) -> dict[str, Any]:
    paths = _single_run_paths(run_dir)
    checkpoint = torch.load(paths["baseline_ckpt"], map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    criterion = nn.BCEWithLogitsLoss() if cfg["train"]["binary"] else nn.CrossEntropyLoss()
    valid_loss, valid_acc, valid_precision, valid_recall, valid_f1, valid_auc = evaluate(
        model,
        forecasted_data.val_loader,
        criterion,
        float(cfg["train"]["decision_threshold"]),
        device,
        bool(cfg["train"]["binary"]),
        edge_index,
    )
    test_loss, test_acc, test_precision, test_recall, test_f1, test_auc = evaluate(
        model,
        forecasted_data.test_loader,
        criterion,
        float(cfg["train"]["decision_threshold"]),
        device,
        bool(cfg["train"]["binary"]),
        edge_index,
    )
    best_epoch = checkpoint.get("epoch")
    return {
        "best_epoch": int(best_epoch) if best_epoch is not None else None,
        "best_valid_metric": float(valid_auc),
        "best_valid_metrics": {
            "valid_loss": float(valid_loss),
            "valid_acc": float(valid_acc),
            "valid_precision": float(valid_precision),
            "valid_recall": float(valid_recall),
            "valid_f1": float(valid_f1),
            "valid_auc": float(valid_auc),
        },
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "test_auc": float(test_auc),
    }


def _stage2_baseline_metrics_from_results(
    baseline_results: dict[str, Any],
) -> dict[str, float]:
    best_valid_metrics = baseline_results.get("best_valid_metrics") or {}
    return {
        "baseline_valid_auc": float(best_valid_metrics.get("valid_auc", float("nan"))),
        "baseline_test_auc": float(baseline_results.get("test_auc", float("nan"))),
        "baseline_valid_f1": float(best_valid_metrics.get("valid_f1", float("nan"))),
        "baseline_test_f1": float(baseline_results.get("test_f1", float("nan"))),
        "baseline_valid_acc": float(best_valid_metrics.get("valid_acc", float("nan"))),
        "baseline_test_acc": float(baseline_results.get("test_acc", float("nan"))),
    }


def _has_stage2_baseline_metrics(metrics: dict[str, Any]) -> bool:
    return all(
        key in metrics
        for key in (
            "baseline_valid_auc",
            "baseline_test_auc",
            "baseline_valid_f1",
            "baseline_test_f1",
            "baseline_valid_acc",
            "baseline_test_acc",
        )
    )


def _resume_completed_single_run(run_dir: str) -> dict[str, Any]:
    result = _load_json(_single_run_result_path(run_dir))
    _write_single_run_status(
        run_dir,
        status="completed",
        current_stage=None,
        last_completed_stage="stage2",
    )
    return result


def _build_dataset(cfg: dict[str, Any], root: str) -> TEDSTensorDataset:
    ensure_model_forecast_defaults(cfg)
    remove_los = cfg["model"]["name"] not in [
        "gin",
        "a3tgcn_2_points",
        "gin_gru_2_points",
    ]
    if not remove_los:
        cfg.setdefault("edge", {})["remove_los"] = False

    return TEDSTensorDataset(
        root=root,
        binary=cfg["train"].get("binary", True),
        ig_label=cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=cfg["train"].get("do_preprocess", True),
    )


def _labels_from_dataset(dataset: TEDSTensorDataset) -> np.ndarray:
    return np.array([dataset[i][1] for i in range(len(dataset))])


def _set_model_params(
    cfg: dict[str, Any], dataset: TEDSTensorDataset, device: torch.device
) -> int:
    cfg["model"]["params"]["col_info"] = dataset.col_info
    cfg["model"]["params"]["num_classes"] = dataset.num_classes
    cfg["model"]["params"]["device"] = str(device)

    if cfg["model"]["name"] == "gin":
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2])
    print(f"num_nodes set to {num_nodes}")
    return num_nodes


def _prepare_fold_dirs(cv_dir: str, fold: int) -> str:
    fold_dir = _fold_dir(cv_dir, fold)
    os.makedirs(fold_dir, exist_ok=False)
    os.makedirs(os.path.join(fold_dir, "checkpoints"), exist_ok=True)
    return fold_dir


def _write_fold_status(cv_dir: str, fold: int, payload: dict[str, Any]) -> None:
    status_path = os.path.join(_fold_dir(cv_dir, fold), "fold_status.json")
    _save_json(status_path, payload)


def _save_fold_result(cv_dir: str, fold: int, result: dict[str, Any]) -> None:
    result_path = os.path.join(_fold_dir(cv_dir, fold), "fold_result.json")
    _save_json(result_path, result)


def _load_fold_splits(cv_dir: str) -> dict[str, Any]:
    return _load_json(_splits_path(cv_dir))


def _stage2_source_run_dir(cfg: dict[str, Any], source_run_dir: str | None) -> str:
    if source_run_dir:
        return str(source_run_dir)
    stage2_cfg = cfg.get("joint_forecast_pipeline", {}).get("stage2", {})
    configured = stage2_cfg.get("source_run_dir")
    if not configured:
        raise ValueError(
            "Outcome-aware stage2-only requires --source_run_dir or "
            "joint_forecast_pipeline.stage2.source_run_dir."
        )
    return str(configured)


def _build_stage2_only_cfg(
    requested_cfg: dict[str, Any],
    source_cfg: dict[str, Any],
    source_run_dir: str,
) -> dict[str, Any]:
    fold_cfg = copy.deepcopy(source_cfg)
    fold_cfg["run_name"] = requested_cfg.get(
        "run_name",
        f"{source_cfg.get('run_name', 'ctmp_gin')}_outcome_aware_stage2",
    )
    fold_cfg["device"] = requested_cfg.get("device", source_cfg.get("device"))

    requested_joint = requested_cfg.get("joint_forecast_pipeline", {})
    source_joint = fold_cfg.setdefault("joint_forecast_pipeline", {})
    source_joint["enabled"] = True
    source_joint["stage2"] = copy.deepcopy(requested_joint.get("stage2", {}))
    source_joint["stage2"]["enabled"] = True
    source_joint["stage2"]["mode"] = "outcome_aware"
    source_joint["stage2"]["source_run_dir"] = source_run_dir

    source_joint["joint_forecast_input"] = copy.deepcopy(
        requested_joint.get(
            "joint_forecast_input",
            source_joint.get("joint_forecast_input", {}),
        )
    )
    source_joint["joint_forecast_input"]["mode"] = "distribution"
    source_joint["joint_forecast_input"]["source_run_dir"] = source_run_dir
    return fold_cfg


def _stage2_only_run_dir(
    cfg: dict[str, Any],
    source_run_dir: str,
    fold: int,
    stage2_run_dir: str | None,
) -> str:
    if stage2_run_dir:
        ensure_dir(stage2_run_dir)
        return stage2_run_dir

    run_cfg = copy.deepcopy(cfg)
    run_cfg["run_name"] = f"{cfg.get('run_name', 'ctmp_gin')}_stage2_only"
    source_id = os.path.basename(os.path.normpath(source_run_dir)).split("__")[0]
    run_id = make_run_id(run_cfg) + f"__source={source_id}__fold={int(fold)}"
    run_dir = os.path.join("runs", run_id)
    ensure_dir(run_dir)
    return run_dir


def prepare_kfold_run(
    cfg: dict[str, Any], root: str, cv_run_dir: str | None = None
) -> dict[str, Any]:
    ensure_model_forecast_defaults(cfg)
    cv_dir = _cv_dir_from_cfg(cfg, cv_run_dir)
    ensure_dir(cv_dir)
    ensure_dir(os.path.join(cv_dir, "folds"))

    save_yaml(os.path.join(cv_dir, "config.final.yaml"), cfg)
    save_text(os.path.join(cv_dir, "command.txt"), _get_command_line() + "\n")
    save_text(os.path.join(cv_dir, "git.txt"), _get_git_info())

    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    dataset = _build_dataset(copy.deepcopy(cfg), root)
    labels = _labels_from_dataset(dataset)

    folds: list[dict[str, Any]] = []
    if bool(cfg.get("forecasted_pipeline", {}).get("enabled", False)) or joint_forecast_pipeline_enabled(cfg):
        all_idx = np.arange(len(dataset), dtype=np.int64)
        for fold, train_idx, outer_test_idx in kfold_stratified(
            trainval_idx=all_idx,
            labels=labels,
            n_folds=cfg["train"]["n_folds"],
            seed=seed,
        ):
            folds.append(
                {
                    "fold": int(fold),
                    "train_idx": train_idx.tolist(),
                    "val_idx": outer_test_idx.tolist(),
                }
            )
        trainval_idx = all_idx
        test_idx = np.array([], dtype=np.int64)
        split_mode = "forecasted_outer_kfold"
    else:
        trainval_idx, test_idx = holdout_test_split_stratified(
            dataset=dataset,
            test_ratio=cfg["train"]["test_ratio"],
            seed=seed,
            labels=labels,
        )
        for fold, train_idx, val_idx in kfold_stratified(
            trainval_idx=trainval_idx,
            labels=labels,
            n_folds=cfg["train"]["n_folds"],
            seed=seed,
        ):
            folds.append(
                {
                    "fold": int(fold),
                    "train_idx": train_idx.tolist(),
                    "val_idx": val_idx.tolist(),
                }
            )
        split_mode = "holdout_test_plus_kfold_val"

    splits = {
        "cv_id": os.path.basename(cv_dir),
        "split_mode": split_mode,
        "seed": int(seed),
        "test_ratio": float(cfg["train"]["test_ratio"]),
        "n_folds": int(cfg["train"]["n_folds"]),
        "trainval_idx": trainval_idx.tolist(),
        "test_idx": test_idx.tolist(),
        "folds": folds,
    }
    _save_json(_splits_path(cv_dir), splits)
    return {"cv_dir": cv_dir, "splits_path": _splits_path(cv_dir), "splits": splits}


def _find_fold_split(splits: dict[str, Any], fold: int) -> dict[str, Any]:
    for fold_info in splits["folds"]:
        if int(fold_info["fold"]) == fold:
            return fold_info
    raise ValueError(f"Fold {fold} not found in saved splits")


def run_single_fold(
    cfg: dict[str, Any],
    root: str,
    fold: int,
    cv_run_dir: str,
    *,
    resume_from_last: bool = False,
) -> dict[str, Any]:
    splits = _load_fold_splits(cv_run_dir)
    fold_split = _find_fold_split(splits, fold)
    if resume_from_last:
        fold_dir = _fold_dir(cv_run_dir, fold)
        if not os.path.isdir(fold_dir):
            raise FileNotFoundError(f"Cannot resume fold: fold dir not found: {fold_dir}")
    else:
        fold_dir = _prepare_fold_dirs(cv_run_dir, fold)

    _write_fold_status(
        cv_run_dir,
        fold,
        {
            "fold": fold,
            "status": "running",
            "run_dir": fold_dir,
            "resume_from_last": bool(resume_from_last),
        },
    )

    try:
        fold_cfg = copy.deepcopy(cfg)
        fold_cfg["fold"] = fold

        seed = fold_cfg["train"].get("seed", 42)
        set_seed(seed)
        device = device_set(fold_cfg["device"])

        dataset = _build_dataset(fold_cfg, root)
        num_nodes = _set_model_params(fold_cfg, dataset, device)
        forecast_input_metadata = resolve_model_forecast_input_metadata(fold_cfg)
        fold_logger = ExperimentLogger(fold_cfg, fold_dir)
        if forecast_input_metadata:
            _save_json(
                os.path.join(fold_dir, "forecast_input_metadata.json"),
                forecast_input_metadata,
            )

        raw_train_idx = np.array(fold_split["train_idx"], dtype=np.int64)
        raw_val_idx = np.array(fold_split["val_idx"], dtype=np.int64)
        raw_test_idx = np.array(splits["test_idx"], dtype=np.int64)

        discharge_provider = None
        los_provider = None
        if joint_forecast_pipeline_enabled(fold_cfg):
            if resume_from_last:
                forecasted_data = _load_fold_forecasted_data_from_cache(
                    cfg=fold_cfg,
                    fold_dir=fold_dir,
                    base_dataset=dataset,
                )
            else:
                forecasted_data = prepare_joint_forecast_fold_data(
                    cfg=fold_cfg,
                    root=root,
                    base_dataset=dataset,
                    outer_train_idx=raw_train_idx,
                    outer_test_idx=raw_val_idx,
                    fold_dir=fold_dir,
                    device=device,
                )
            train_idx = forecasted_data.train_idx
            val_idx = forecasted_data.val_idx
            test_idx = forecasted_data.test_idx
            train_loader = forecasted_data.train_loader
            val_loader = forecasted_data.val_loader
            test_loader = forecasted_data.test_loader
        elif bool(fold_cfg.get("forecasted_pipeline", {}).get("enabled", False)):
            if resume_from_last:
                raise ValueError(
                    "--resume_fold_from_last currently supports joint_forecast_pipeline cached folds only."
                )
            forecasted_data = prepare_forecasted_fold_data(
                cfg=fold_cfg,
                root=root,
                base_dataset=dataset,
                outer_train_idx=raw_train_idx,
                outer_test_idx=raw_val_idx,
                fold_dir=fold_dir,
                device=device,
            )
            train_idx = forecasted_data.train_idx
            val_idx = forecasted_data.val_idx
            test_idx = forecasted_data.test_idx
            train_loader = forecasted_data.train_loader
            val_loader = forecasted_data.val_loader
            test_loader = forecasted_data.test_loader
        else:
            discharge_provider = build_forecasted_discharge_provider(
                fold_cfg, dataset, device
            )
            los_provider = build_forecasted_los_provider(fold_cfg, dataset, device)
            train_idx = raw_train_idx
            val_idx = raw_val_idx
            test_idx = raw_test_idx
            train_loader, val_loader, test_loader = make_loaders(
                dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                batch_size=fold_cfg["train"]["batch_size"],
                num_workers=fold_cfg["train"]["num_workers"],
                drop_last=True,
            )

        train_df = dataset.processed_df.iloc[train_idx]

        if fold_cfg["model"]["name"] == "xgboost":
            from src.models.xgboost import train_xgboost

            result = train_xgboost(
                train_idx,
                val_idx,
                test_idx,
                dataset.processed_df,
                fold_logger,
                fold_cfg,
            )
            result["fold"] = fold
            result["run_dir"] = fold_dir
            result["status"] = "completed"
            _save_fold_result(cv_run_dir, fold, result)
            _write_fold_status(
                cv_run_dir,
                fold,
                {"fold": fold, "status": "completed", "run_dir": fold_dir},
            )
            return result

        if fold_cfg["model"]["name"] in ["a3tgcn", "a3tgcn_2_points"]:
            fold_cfg["model"]["params"]["batch_size"] = fold_cfg["train"].get(
                "batch_size", 32
            )

        model = build_model(
            model_name=fold_cfg["model"]["name"], **fold_cfg["model"].get("params", {})
        )
        model = model.to(device)

        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(model)
        print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

        if resume_from_last:
            edge_index = _load_edge_index_for_single_run(
                cfg=fold_cfg,
                root=root,
                run_dir=fold_dir,
                dataset=dataset,
                train_idx=train_idx,
                num_nodes=num_nodes,
                device=device,
            )
            print(f"  edge_index loaded/reused for resume: {os.path.join(fold_dir, 'edge_index.pt')}")
        else:
            edge_index = build_edge(
                model_name=fold_cfg["model"]["name"],
                root=root,
                seed=seed,
                train_df=train_df,
                num_nodes=num_nodes,
                batch_size=fold_cfg["train"]["batch_size"],
                **fold_cfg.get("edge", {}),
            )
            edge_index = edge_index.to(device)  # type: ignore

        if hasattr(model, "precompute_edge_index_2"):
            model.precompute_edge_index_2(edge_index, fold_cfg["train"]["batch_size"])

        if resume_from_last:
            pass
        elif fold_cfg.get("edge", {}).get("is_mi_based", False):
            edge_index_save_path = os.path.join(fold_dir, "edge_index.pt")
            torch.save(edge_index.cpu(), edge_index_save_path)
            print(f"  edge_index saved: {edge_index_save_path}")
        else:
            print(
                "  edge_index save skipped (not MI-based, fully connected is trivially reconstructable)"
            )

        print(f"edge index: \n{edge_index}")
        print(f"edge index shape: \n{edge_index.shape}")

        if fold_cfg["train"]["binary"]:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        if fold_cfg["train"].get("optimizer", "adam") == "adamw":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=fold_cfg["train"]["learning_rate"],
                weight_decay=fold_cfg["train"].get("weight_decay", 0.0),
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=fold_cfg["train"]["learning_rate"],
                weight_decay=fold_cfg["train"].get("weight_decay", 0.0),
            )

        scheduler = ReduceLROnPlateau(
            optimizer, "min", patience=fold_cfg["train"]["lr_scheduler_patience"]
        )
        early_stopper = EarlyStopper(
            patience=fold_cfg["train"]["early_stopping_patience"]
        )
        start_epoch = 1
        if resume_from_last:
            start_epoch = _restore_training_state_from_last_checkpoint(
                fold_dir=fold_dir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                logger=fold_logger,
                device=device,
            )

        results = run_train_loop(
            model=model,
            edge_index=edge_index,
            binary=fold_cfg["train"]["binary"],
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            test_dataloader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            early_stopper=early_stopper,
            device=device,
            logger=fold_logger,
            start_epoch=start_epoch,
            epochs=fold_cfg["train"]["epochs"],
            decision_threshold=fold_cfg["train"]["decision_threshold"],
            model_name=fold_cfg["model"].get("name", "Unknown"),
            trial=None,
            los_provider=los_provider,
            discharge_provider=discharge_provider,
            checkpoint_extra=(
                {"forecast_input_metadata": forecast_input_metadata}
                if forecast_input_metadata
                else None
            ),
        )

        if (
            joint_forecast_pipeline_enabled(fold_cfg)
            and bool(
                fold_cfg.get("joint_forecast_pipeline", {})
                .get("stage2", {})
                .get("enabled", False)
            )
        ):
            predictor_checkpoint_path = os.path.join(
                forecasted_data.split_payload["joint_run_dir"],
                "checkpoints",
                "best.pt",
            )
            baseline_checkpoint_path = os.path.join(
                fold_dir,
                "checkpoints",
                "best.pt",
            )
            baseline_metrics = {
                "baseline_valid_auc": float(
                    (results.get("best_valid_metrics") or {}).get("valid_auc", float("nan"))
                ),
                "baseline_test_auc": float(results.get("test_auc", float("nan"))),
                "baseline_valid_f1": float(
                    (results.get("best_valid_metrics") or {}).get("valid_f1", float("nan"))
                ),
                "baseline_test_f1": float(results.get("test_f1", float("nan"))),
                "baseline_valid_acc": float(
                    (results.get("best_valid_metrics") or {}).get("valid_acc", float("nan"))
                ),
                "baseline_test_acc": float(results.get("test_acc", float("nan"))),
            }
            (
                predictor_checkpoint_path,
                baseline_checkpoint_path,
                baseline_metrics,
            ) = resolve_stage2_pretrained_paths(
                fold=fold,
                stage2_cfg=fold_cfg.get("joint_forecast_pipeline", {}).get("stage2", {}),
                fallback_predictor_checkpoint_path=predictor_checkpoint_path,
                fallback_baseline_checkpoint_path=baseline_checkpoint_path,
                fallback_baseline_metrics=baseline_metrics,
            )
            stage2_results = run_outcome_aware_stage2(
                cfg=fold_cfg,
                root=root,
                fold_dir=fold_dir,
                base_dataset=dataset,
                train_idx=train_idx,
                val_idx=val_idx,
                test_idx=test_idx,
                edge_index=edge_index,
                ctmp_gin_model=model,
                predictor_checkpoint_path=predictor_checkpoint_path,
                baseline_checkpoint_path=baseline_checkpoint_path,
                baseline_metrics=baseline_metrics,
                forecast_split_payload=forecasted_data.split_payload,
                device=device,
            )
            results = {
                **results,
                **stage2_results,
                "best_epoch": int(stage2_results["best_epoch"]),
                "best_valid_metric": float(stage2_results["best_valid_metric"]),
                "best_valid_metrics": dict(stage2_results["best_valid_metrics"]),
                "test_loss": float(stage2_results["test_loss"]),
                "test_acc": float(stage2_results["test_acc"]),
                "test_precision": float(stage2_results["test_precision"]),
                "test_recall": float(stage2_results["test_recall"]),
                "test_f1": float(stage2_results["test_f1"]),
                "test_auc": float(stage2_results["test_auc"]),
            }

        results["fold"] = fold
        results["run_dir"] = fold_dir
        results["status"] = "completed"

        _save_fold_result(cv_run_dir, fold, results)
        _write_fold_status(
            cv_run_dir,
            fold,
            {"fold": fold, "status": "completed", "run_dir": fold_dir},
        )
        return results

    except Exception as exc:
        error_payload = {
            "fold": fold,
            "status": "failed",
            "run_dir": fold_dir,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_fold_status(cv_run_dir, fold, error_payload)
        raise


def run_outcome_aware_stage2_only(
    cfg: dict[str, Any],
    root: str,
    *,
    fold: int = 0,
    source_run_dir: str | None = None,
    stage2_run_dir: str | None = None,
) -> dict[str, Any]:
    source_run_dir = _stage2_source_run_dir(cfg, source_run_dir)
    source_info = _resolve_stage2_source_artifacts(source_run_dir, fold)
    source_cfg_path = source_info["source_cfg_path"]
    split_path = source_info["split_path"]
    if not os.path.exists(source_cfg_path):
        raise FileNotFoundError(f"Source fold config not found: {source_cfg_path}")
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Source fold split file not found: {split_path}")

    source_cfg = _load_yaml_file(source_cfg_path)
    fold_cfg = _build_stage2_only_cfg(cfg, source_cfg, source_run_dir)
    fold_cfg["fold"] = int(fold)

    run_dir = _stage2_only_run_dir(fold_cfg, source_run_dir, fold, stage2_run_dir)
    save_yaml(os.path.join(run_dir, "config.final.yaml"), fold_cfg)
    save_text(os.path.join(run_dir, "command.txt"), _get_command_line() + "\n")
    save_text(os.path.join(run_dir, "git.txt"), _get_git_info())

    split_payload = _load_json(split_path)
    train_idx = np.asarray(split_payload["train_core_idx"], dtype=np.int64)
    val_idx = np.asarray(split_payload["gnn_val_idx"], dtype=np.int64)
    test_idx = np.asarray(split_payload["outer_test_idx"], dtype=np.int64)

    seed = int(fold_cfg["train"].get("seed", 42))
    set_seed(seed)
    device = device_set(fold_cfg["device"])

    dataset = _build_dataset(fold_cfg, root)
    _set_model_params(fold_cfg, dataset, device)

    model = build_model(
        model_name=fold_cfg["model"]["name"],
        **fold_cfg["model"].get("params", {}),
    ).to(device)

    edge_path = os.path.join(source_info["source_artifact_dir"], "edge_index.pt")
    if os.path.exists(edge_path):
        edge_index = torch.load(edge_path, map_location=device).to(device)
    else:
        train_df = dataset.processed_df.iloc[train_idx]
        num_nodes = len(dataset.col_info[2])
        edge_index = build_edge(
            model_name=fold_cfg["model"]["name"],
            root=root,
            seed=seed,
            train_df=train_df,
            num_nodes=num_nodes,
            batch_size=int(fold_cfg["train"]["batch_size"]),
            **fold_cfg.get("edge", {}),
        ).to(device)

    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, int(fold_cfg["train"]["batch_size"]))

    stage2_cfg = fold_cfg.get("joint_forecast_pipeline", {}).get("stage2", {})
    predictor_checkpoint_path, baseline_checkpoint_path, baseline_metrics = (
        resolve_stage2_pretrained_paths(
            fold=fold,
            stage2_cfg=stage2_cfg,
            fallback_predictor_checkpoint_path="",
            fallback_baseline_checkpoint_path="",
            fallback_baseline_metrics={},
        )
    )
    baseline_ckpt = torch.load(baseline_checkpoint_path, map_location=device)
    model.load_state_dict(baseline_ckpt["model_state_dict"], strict=True)
    if not _has_stage2_baseline_metrics(baseline_metrics):
        if source_info["source_kind"] != "single_run":
            raise ValueError(
                "Stage2-only source is missing baseline metrics; expected fold_result.json "
                "for k-fold sources."
            )
        print("[stage2-only] baseline metrics missing; reconstructing from source cache")
        source_forecasted_data = _load_single_run_forecasted_data(
            fold_cfg,
            run_dir=source_info["source_artifact_dir"],
            base_dataset=dataset,
        )
        baseline_results = _reconstruct_baseline_results(
            cfg=fold_cfg,
            run_dir=source_info["source_artifact_dir"],
            model=model,
            edge_index=edge_index,
            forecasted_data=source_forecasted_data,
            device=device,
        )
        baseline_metrics = _stage2_baseline_metrics_from_results(baseline_results)

    print(f"[stage2-only] source_run_dir={source_run_dir}")
    print(f"[stage2-only] source_kind={source_info['source_kind']}")
    print(f"[stage2-only] source_artifact_dir={source_info['source_artifact_dir']}")
    print(f"[stage2-only] predictor_checkpoint={predictor_checkpoint_path}")
    print(f"[stage2-only] ctmp_gin_checkpoint={baseline_checkpoint_path}")
    print(f"[stage2-only] output_dir={run_dir}")
    print(
        "[stage2-only] split sizes "
        f"train={len(train_idx)} valid={len(val_idx)} test={len(test_idx)}"
    )

    result = run_outcome_aware_stage2(
        cfg=fold_cfg,
        root=root,
        fold_dir=run_dir,
        base_dataset=dataset,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        edge_index=edge_index,
        ctmp_gin_model=model,
        predictor_checkpoint_path=predictor_checkpoint_path,
        baseline_checkpoint_path=baseline_checkpoint_path,
        baseline_metrics=baseline_metrics,
        forecast_split_payload=split_payload,
        device=device,
        source_artifact_dir=source_info["source_artifact_dir"],
    )
    result = {
        **result,
        "fold": int(fold),
        "run_dir": run_dir,
        "stage2_dir": result["run_dir"],
        "source_run_dir": source_run_dir,
        "source_artifact_dir": source_info["source_artifact_dir"],
        "source_kind": source_info["source_kind"],
        "status": "completed",
    }
    _save_json(os.path.join(run_dir, "stage2_only_result.json"), result)
    return result


def _validate_outcome_aware_single_run_cfg(cfg: dict[str, Any]) -> None:
    if not joint_forecast_pipeline_enabled(cfg):
        raise ValueError("outcome-aware single run requires joint_forecast_pipeline.enabled=true.")
    if str(cfg.get("model", {}).get("name")) not in {"ctmp_gin", "gin"}:
        raise ValueError(
            "outcome-aware single run currently supports only model.name in {'ctmp_gin', 'gin'}."
        )

    joint_cfg = cfg.get("joint_forecast_pipeline", {})
    stage2_cfg = joint_cfg.get("stage2", {})
    if not bool(stage2_cfg.get("enabled", False)):
        raise ValueError("outcome-aware single run requires joint_forecast_pipeline.stage2.enabled=true.")
    if stage2_cfg.get("source_run_dir"):
        raise ValueError(
            "outcome-aware single run trains fresh stage1 and baseline checkpoints; "
            "remove joint_forecast_pipeline.stage2.source_run_dir."
        )

    input_cfg = joint_cfg.get("joint_forecast_input", {})
    if str(input_cfg.get("mode", "distribution")).lower() != "distribution":
        raise ValueError("outcome-aware single run requires joint_forecast_input.mode=distribution.")
    cache_keys = (
        "train_cache_path",
        "val_cache_path",
        "test_cache_path",
        "gnn_val_cache_path",
        "outer_test_cache_path",
    )
    if input_cfg.get("source_run_dir") or any(input_cfg.get(key) for key in cache_keys):
        raise ValueError(
            "outcome-aware single run generates a fresh forecast cache; remove "
            "joint_forecast_input.source_run_dir and explicit cache paths."
        )


def run_outcome_aware_single_run(
    cfg: dict[str, Any],
    root: str,
    *,
    resume_run_dir: str | None = None,
) -> dict[str, Any]:
    single_cfg, resumed_run_dir = _load_single_run_cfg_for_resume(cfg, resume_run_dir)
    single_cfg.setdefault("train", {})["cv"] = False
    single_cfg["fold"] = 0
    ensure_model_forecast_defaults(single_cfg)
    _validate_outcome_aware_single_run_cfg(single_cfg)

    if resumed_run_dir is None:
        run_id = make_run_id(single_cfg) + "__single"
        run_dir = ensure_run_dir("runs", run_id)
        save_yaml(os.path.join(run_dir, "config.final.yaml"), single_cfg)
        save_text(os.path.join(run_dir, "command.txt"), _get_command_line() + "\n")
        save_text(os.path.join(run_dir, "git.txt"), _get_git_info())
    else:
        run_dir = resumed_run_dir
        if _single_run_stage_state(run_dir)["stage2_complete"]:
            return _resume_completed_single_run(run_dir)
    _write_single_run_status(run_dir, status="running", current_stage="initializing")

    try:
        seed = int(single_cfg["train"].get("seed", 42))
        split_seed = int(single_cfg["train"].get("split_seed", seed))
        set_seed(split_seed)
        device = device_set(single_cfg["device"])

        dataset = _build_dataset(single_cfg, root)
        labels = _labels_from_dataset(dataset)
        outer_train_idx, outer_test_idx = holdout_test_split_stratified(
            dataset=dataset,
            test_ratio=float(single_cfg["train"]["test_ratio"]),
            seed=split_seed,
            labels=labels,
        )

        num_nodes = _set_model_params(single_cfg, dataset, device)
        forecast_input_metadata = resolve_model_forecast_input_metadata(single_cfg)
        if forecast_input_metadata:
            _save_json(
                os.path.join(run_dir, "forecast_input_metadata.json"),
                forecast_input_metadata,
            )
        stage_state = _single_run_stage_state(run_dir)

        if stage_state["stage1_complete"]:
            forecasted_data = _load_single_run_forecasted_data(
                single_cfg,
                run_dir=run_dir,
                base_dataset=dataset,
            )
            predictor_checkpoint_path = _single_run_paths(run_dir)["joint_predictor_ckpt"]
        else:
            if resumed_run_dir is not None:
                _backup_incomplete_single_run_stage(run_dir, "stage1")
            _write_single_run_status(
                run_dir,
                status="running",
                current_stage="stage1",
                last_completed_stage=None,
            )
            forecasted_data = prepare_joint_forecast_fold_data(
                cfg=single_cfg,
                root=root,
                base_dataset=dataset,
                outer_train_idx=np.asarray(outer_train_idx, dtype=np.int64),
                outer_test_idx=np.asarray(outer_test_idx, dtype=np.int64),
                fold_dir=run_dir,
                device=device,
            )
            predictor_checkpoint_path = os.path.join(
                str(forecasted_data.split_payload["joint_run_dir"]),
                "checkpoints",
                "best.pt",
            )
            if not os.path.exists(predictor_checkpoint_path):
                raise FileNotFoundError(
                    f"Stage1 predictor checkpoint missing after training: {predictor_checkpoint_path}"
                )
            _save_json(
                os.path.join(run_dir, "single_run_splits.json"),
                {
                    "split_mode": "holdout_test_plus_forecasted_inner_val",
                    "seed": int(seed),
                    "split_seed": int(split_seed),
                    "outer_train_idx": np.asarray(outer_train_idx, dtype=np.int64).tolist(),
                    "outer_test_idx": np.asarray(outer_test_idx, dtype=np.int64).tolist(),
                    "train_core_idx": np.asarray(
                        forecasted_data.train_idx, dtype=np.int64
                    ).tolist(),
                    "predictor_val_idx": forecasted_data.split_payload["predictor_val_idx"],
                    "gnn_val_idx": np.asarray(
                        forecasted_data.val_idx, dtype=np.int64
                    ).tolist(),
                    "stage2_test_idx": np.asarray(
                        forecasted_data.test_idx, dtype=np.int64
                    ).tolist(),
                    "forecast_split_payload": forecasted_data.split_payload,
                },
            )
            _write_single_run_status(
                run_dir,
                status="running",
                current_stage="baseline",
                last_completed_stage="stage1",
            )

        train_idx = forecasted_data.train_idx
        val_idx = forecasted_data.val_idx
        test_idx = forecasted_data.test_idx

        set_seed(seed)
        model = build_model(
            model_name=single_cfg["model"]["name"],
            **single_cfg["model"].get("params", {}),
        ).to(device)
        total_trainable_params = sum(
            param.numel() for param in model.parameters() if param.requires_grad
        )
        print(model)
        print(f"학습 가능한 파라미터 개수: {total_trainable_params:,}")

        edge_index = _load_edge_index_for_single_run(
            cfg=single_cfg,
            root=root,
            run_dir=run_dir,
            dataset=dataset,
            train_idx=train_idx,
            num_nodes=num_nodes,
            device=device,
        )
        if hasattr(model, "precompute_edge_index_2"):
            model.precompute_edge_index_2(edge_index, single_cfg["train"]["batch_size"])

        baseline_checkpoint_path = _single_run_paths(run_dir)["baseline_ckpt"]
        if stage_state["baseline_complete"]:
            baseline_results = _reconstruct_baseline_results(
                cfg=single_cfg,
                run_dir=run_dir,
                model=model,
                edge_index=edge_index,
                forecasted_data=forecasted_data,
                device=device,
            )
        else:
            if resumed_run_dir is not None:
                _backup_incomplete_single_run_stage(run_dir, "baseline")
            _write_single_run_status(
                run_dir,
                status="running",
                current_stage="baseline",
                last_completed_stage="stage1",
            )
            baseline_logger = ExperimentLogger(single_cfg, run_dir)
            criterion = (
                nn.BCEWithLogitsLoss()
                if single_cfg["train"]["binary"]
                else nn.CrossEntropyLoss()
            )
            if single_cfg["train"].get("optimizer", "adam") == "adamw":
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=single_cfg["train"]["learning_rate"],
                    weight_decay=single_cfg["train"].get("weight_decay", 0.0),
                )
            else:
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=single_cfg["train"]["learning_rate"],
                    weight_decay=single_cfg["train"].get("weight_decay", 0.0),
                )
            scheduler = ReduceLROnPlateau(
                optimizer,
                "min",
                patience=single_cfg["train"]["lr_scheduler_patience"],
            )
            early_stopper = EarlyStopper(
                patience=single_cfg["train"]["early_stopping_patience"]
            )
            baseline_results = run_train_loop(
                model=model,
                edge_index=edge_index,
                binary=single_cfg["train"]["binary"],
                train_dataloader=forecasted_data.train_loader,
                val_dataloader=forecasted_data.val_loader,
                test_dataloader=forecasted_data.test_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                early_stopper=early_stopper,
                device=device,
                logger=baseline_logger,
                epochs=single_cfg["train"]["epochs"],
                decision_threshold=single_cfg["train"]["decision_threshold"],
                model_name=single_cfg["model"].get("name", "Unknown"),
                trial=None,
                checkpoint_extra=(
                    {"forecast_input_metadata": forecast_input_metadata}
                    if forecast_input_metadata
                    else None
                ),
            )
            if not os.path.exists(baseline_checkpoint_path):
                raise FileNotFoundError(
                    f"Baseline CTMP-GIN checkpoint missing after training: {baseline_checkpoint_path}"
                )
            _write_single_run_status(
                run_dir,
                status="running",
                current_stage="stage2",
                last_completed_stage="baseline",
            )

        baseline_ckpt = torch.load(baseline_checkpoint_path, map_location=device)
        model.load_state_dict(baseline_ckpt["model_state_dict"], strict=True)

        baseline_metrics = {
            "baseline_valid_auc": float(
                (baseline_results.get("best_valid_metrics") or {}).get("valid_auc", float("nan"))
            ),
            "baseline_test_auc": float(baseline_results.get("test_auc", float("nan"))),
            "baseline_valid_f1": float(
                (baseline_results.get("best_valid_metrics") or {}).get("valid_f1", float("nan"))
            ),
            "baseline_test_f1": float(baseline_results.get("test_f1", float("nan"))),
            "baseline_valid_acc": float(
                (baseline_results.get("best_valid_metrics") or {}).get("valid_acc", float("nan"))
            ),
            "baseline_test_acc": float(baseline_results.get("test_acc", float("nan"))),
        }
        if resumed_run_dir is not None and not stage_state["stage2_complete"]:
            _backup_incomplete_single_run_stage(run_dir, "stage2")
        _write_single_run_status(
            run_dir,
            status="running",
            current_stage="stage2",
            last_completed_stage="baseline",
        )
        stage2_results = run_outcome_aware_stage2(
            cfg=single_cfg,
            root=root,
            fold_dir=run_dir,
            base_dataset=dataset,
            train_idx=train_idx,
            val_idx=val_idx,
            test_idx=test_idx,
            edge_index=edge_index,
            ctmp_gin_model=model,
            predictor_checkpoint_path=predictor_checkpoint_path,
            baseline_checkpoint_path=baseline_checkpoint_path,
            baseline_metrics=baseline_metrics,
            forecast_split_payload=forecasted_data.split_payload,
            device=device,
        )
        results = {
            **baseline_results,
            **stage2_results,
            "best_epoch": int(stage2_results["best_epoch"]),
            "best_valid_metric": float(stage2_results["best_valid_metric"]),
            "best_valid_metrics": dict(stage2_results["best_valid_metrics"]),
            "test_loss": float(stage2_results["test_loss"]),
            "test_acc": float(stage2_results["test_acc"]),
            "test_precision": float(stage2_results["test_precision"]),
            "test_recall": float(stage2_results["test_recall"]),
            "test_f1": float(stage2_results["test_f1"]),
            "test_auc": float(stage2_results["test_auc"]),
            "run_dir": run_dir,
            "stage2_dir": stage2_results["run_dir"],
            "stage1_predictor_checkpoint_path": predictor_checkpoint_path,
            "baseline_checkpoint_path": baseline_checkpoint_path,
            "status": "completed",
        }
        _save_json(os.path.join(run_dir, "single_run_result.json"), results)
        _write_single_run_status(
            run_dir,
            status="completed",
            current_stage=None,
            last_completed_stage="stage2",
        )
        return results
    except Exception as exc:
        _write_single_run_status(
            run_dir,
            status="failed",
            error=str(exc),
            traceback_text=traceback.format_exc(),
        )
        raise


def _aggregate_numeric_metrics(
    fold_results: list[dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    metric_values: dict[str, list[float]] = {}

    for result in fold_results:
        for key, value in result.items():
            if key in {"fold"}:
                continue
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))

    aggregates: dict[str, dict[str, float | None]] = {}
    for key, values in metric_values.items():
        arr = np.asarray(values, dtype=float)
        valid = arr[~np.isnan(arr)]
        if valid.size == 0:
            aggregates[key] = {"mean": None, "std": None}
            continue
        aggregates[key] = {
            "mean": float(valid.mean()),
            "std": float(valid.std()),
        }
    return aggregates


def finalize_kfold_summary(cv_run_dir: str) -> dict[str, Any]:
    root_cfg_path = os.path.join(cv_run_dir, "config.final.yaml")
    with open(root_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    splits = _load_fold_splits(cv_run_dir)
    expected_folds = list(range(int(splits["n_folds"])))

    fold_results: list[dict[str, Any]] = []
    failed_folds: list[dict[str, Any]] = []

    for fold in expected_folds:
        fold_dir = _fold_dir(cv_run_dir, fold)
        result_path = os.path.join(fold_dir, "fold_result.json")
        status_path = os.path.join(fold_dir, "fold_status.json")

        if os.path.exists(result_path):
            fold_results.append(_load_json(result_path))
            continue

        failure_entry = {
            "fold": fold,
            "status": "missing",
            "run_dir": fold_dir,
        }
        if os.path.exists(status_path):
            failure_entry.update(_load_json(status_path))
        failed_folds.append(failure_entry)

    summary = {
        "cv_id": os.path.basename(cv_run_dir),
        "K": int(cfg["train"]["n_folds"]),
        "test_ratio": float(cfg["train"]["test_ratio"]),
        "status": "completed" if not failed_folds else "failed",
        "completed_folds": len(fold_results),
        "failed_folds": failed_folds,
        "fold_results": sorted(fold_results, key=lambda x: x["fold"]),
        "aggregates": _aggregate_numeric_metrics(fold_results),
    }

    _save_json(os.path.join(cv_run_dir, "cv_summary.json"), summary)
    return summary


def run_kfold_experiment(cfg: dict[str, Any], root: str) -> dict[str, Any]:
    prepared = prepare_kfold_run(cfg, root)
    cv_dir = prepared["cv_dir"]

    for fold_info in prepared["splits"]["folds"]:
        run_single_fold(cfg, root, int(fold_info["fold"]), cv_dir)

    return finalize_kfold_summary(cv_dir)
