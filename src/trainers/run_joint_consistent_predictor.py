from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from src.data_processing.discharge_prediction_dataset import DischargePredictionDataset
from src.data_processing.los_prediction_dataset import LOSPredictionDataset
from src.data_processing.splits import holdout_test_split_stratified, kfold_stratified
from src.diagnostics.diagnose_joint_predictor_joint_stats import compute_joint_stats
from src.analysis.joint_los_given_d_report import generate_joint_los_given_d_report
from src.models.discharge_predictor.conditioners import (
    parse_bool_flag,
    resolve_joint_heads,
)
from src.models.discharge_predictor.joint_consistent_predictor import (
    JointConsistentPredictor,
)
from src.models.discharge_predictor.joint_consistency_loss import JointConsistencyLoss
from src.models.discharge_predictor.joint_generative_predictor import (
    JointGenerativeLoss,
    JointGenerativePredictor,
    kl_beta_for_epoch,
)
from src.models.discharge_predictor.risk_heads import LEGACY_TOP3_HEADS
from src.models.discharge_predictor.soft_joint_drift_loss import SoftJointDriftLoss
from src.models.discharge_predictor.los_utils import (
    get_los_coarse_num_classes,
    infer_los_coarse_breakdown_from_cfg,
    infer_los_target_from_cfg,
    los_binning_metadata_dict,
    map_los_array_to_coarse_bins,
)
from src.models.discharge_predictor.metrics import (
    compute_discharge_metrics,
    compute_ordinal_metrics,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.device_set import device_set
from src.utils.experiment import ExperimentLogger, ensure_run_dir, make_run_id
from src.utils.seed_set import set_seed

DEFAULT_JOINT_HEADS = ",".join(LEGACY_TOP3_HEADS)
DEFAULT_DRIFT_HEADS = LEGACY_TOP3_HEADS
DEFAULT_JOINT_STRUCT_LOSS = {
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train joint-consistent discharge and LOS predictor"
    )
    parser.add_argument("--root", type=str, default="src/data")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=1.0e-3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--joint_direction", type=str, default="los_to_d")
    parser.add_argument("--condition_mode", type=str, default="predicted")
    parser.add_argument("--detach_condition", type=str, default="true")
    parser.add_argument("--predictor_type", type=str, default="joint_consistent")
    parser.add_argument("--los_target_mode", type=str, default="coarse")
    parser.add_argument("--los_coarse_breakdown", type=str, default="false")
    parser.add_argument("--lambda_los", type=float, default=1.0)
    parser.add_argument("--lambda_aux", type=float, default=0.3)
    parser.add_argument("--lambda_joint", type=float, default=0.0)
    parser.add_argument("--prior_recon_weight", type=float, default=0.5)
    parser.add_argument("--beta_kl_start", type=float, default=0.0)
    parser.add_argument("--beta_kl_max", type=float, default=0.001)
    parser.add_argument("--kl_anneal_epochs", type=int, default=10)
    parser.add_argument("--joint_heads", type=str, default=DEFAULT_JOINT_HEADS)
    parser.add_argument("--save_cache", action="store_true")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument(
        "--input_encoding", type=str, choices=["onehot", "embedding"], default="onehot"
    )
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--los_context_dim", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--conditioner_embedding_dim", type=int, default=None)
    parser.add_argument("--z_sampling_at_eval", type=str, default="false")
    parser.add_argument("--num_eval_samples", type=int, default=1)
    parser.add_argument("--weight_decay", type=float, default=1.0e-5)
    parser.add_argument("--train_ratio", type=float, default=0.85)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    return parser.parse_args()


@dataclass
class JointPredictionBatch:
    x: torch.Tensor
    d_targets: torch.Tensor
    los_target: torch.Tensor
    los_raw: torch.Tensor
    row_idx: torch.Tensor


class JointPredictionDataset(Dataset):
    def __init__(
        self,
        *,
        root: str,
        do_preprocess: bool,
        los_target_mode: str,
        los_coarse_breakdown: bool = False,
    ) -> None:
        discharge_ds = DischargePredictionDataset(
            root=root,
            do_preprocess=do_preprocess,
            include_los_in_targets=False,
        )
        los_ds = LOSPredictionDataset(root=root, do_preprocess=do_preprocess)
        if discharge_ds.x.shape[0] != los_ds.x.shape[0]:
            raise RuntimeError(
                "JointPredictionDataset: discharge and LOS datasets differ in size."
            )
        if not torch.equal(discharge_ds.x.long(), los_ds.x.long()):
            raise RuntimeError(
                "JointPredictionDataset: discharge and LOS admission tensors are not aligned."
            )
        if not discharge_ds.raw_row_index.equals(los_ds.raw_row_index):
            raise RuntimeError(
                "JointPredictionDataset: raw_row_index differs between discharge and LOS datasets."
            )

        self.x = discharge_ds.x.long()
        self.d_targets = discharge_ds.y.long()
        self.target_col_names = list(discharge_ds.target_col_names)
        self.target_col_dims = [int(v) for v in discharge_ds.target_col_dims]
        self.ad_col_names = list(discharge_ds.ad_col_names)
        self.ad_col_dims = list(discharge_ds.ad_col_dims)
        self.row_idx = torch.tensor(
            discharge_ds.raw_row_index.to_numpy(), dtype=torch.long
        )
        self.caseid = None
        if discharge_ds.caseid_series is not None:
            self.caseid = discharge_ds.caseid_series.astype(str).tolist()

        self.los_raw = los_ds.los_raw.long()
        self.los_encoded = los_ds.y.long()
        self.los_target_mode = str(los_target_mode).lower()
        self.los_coarse_breakdown = bool(los_coarse_breakdown)
        if self.los_target_mode == "coarse":
            self.los_target = map_los_array_to_coarse_bins(
                self.los_raw,
                breakdown=self.los_coarse_breakdown,
            ).long()
            self.los_num_classes = get_los_coarse_num_classes(
                breakdown=self.los_coarse_breakdown
            )
        elif self.los_target_mode == "raw37":
            self.los_target = self.los_encoded.long()
            self.los_num_classes = 37
        else:
            raise ValueError(f"Unsupported los_target_mode: {los_target_mode}")
        self.schema_metadata = {
            "admission_col_names": list(self.ad_col_names),
            "admission_col_dims": list(self.ad_col_dims),
            "target_col_names": list(self.target_col_names),
            "target_col_dims": list(self.target_col_dims),
            "los_target_mode": self.los_target_mode,
            "los_num_classes": int(self.los_num_classes),
        }
        if self.los_target_mode == "coarse":
            self.schema_metadata.update(
                los_binning_metadata_dict(breakdown=self.los_coarse_breakdown)
            )

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> JointPredictionBatch:
        return JointPredictionBatch(
            x=self.x[idx],
            d_targets=self.d_targets[idx],
            los_target=self.los_target[idx],
            los_raw=self.los_raw[idx],
            row_idx=self.row_idx[idx],
        )


def _collate_joint_batch(
    batch: list[JointPredictionBatch],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.stack([item.x for item in batch], dim=0),
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
    shuffle: bool,
    pin_memory: bool = True,
) -> DataLoader:
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=_collate_joint_batch,
    )


def _pin_memory_for_device(device: torch.device) -> bool:
    return device.type == "cuda"


def _save_json(path: str, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _split_indices(
    dataset: JointPredictionDataset, cfg: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = map_los_array_to_coarse_bins(
        dataset.los_raw,
        breakdown=dataset.los_coarse_breakdown,
    ).long().cpu().numpy()
    trainval_idx, test_idx = holdout_test_split_stratified(
        dataset,
        test_ratio=float(cfg["train"]["test_ratio"]),
        seed=int(cfg["train"]["seed"]),
        labels=labels,
    )
    selected_fold = int(cfg["train"]["fold"])
    n_folds = int(cfg["train"]["num_folds"])
    for fold, train_idx, val_idx in kfold_stratified(
        trainval_idx=trainval_idx,
        labels=labels,
        n_folds=n_folds,
        seed=int(cfg["train"]["seed"]),
    ):
        if int(fold) == selected_fold:
            return train_idx, val_idx, test_idx
    raise ValueError(f"Fold {selected_fold} not found for num_folds={n_folds}")


def _balanced_validation_score(
    metrics: dict[str, float], *, los_target_mode: str
) -> float:
    los_key = "los_macro_f1" if los_target_mode == "coarse" else "los_qwk"
    return 0.5 * float(metrics["discharge_mean_macro_f1"]) + 0.5 * float(
        metrics[los_key]
    )


def _build_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "device": args.device,
        "run_name": "joint_consistent_predictor",
        "model": {
            "name": "joint_consistent_predictor",
            "params": {
                "embedding_dim": int(args.embedding_dim),
                "input_encoding": str(args.input_encoding),
                "hidden_dim": int(args.hidden_dim),
                "latent_dim": int(args.latent_dim),
                "los_context_dim": int(args.los_context_dim),
                "num_layers": int(args.num_layers),
                "dropout": float(args.dropout),
                "conditioner_embedding_dim": args.conditioner_embedding_dim,
                "z_sampling_at_eval": parse_bool_flag(args.z_sampling_at_eval),
                "num_eval_samples": int(args.num_eval_samples),
            },
        },
        "train": {
            "batch_size": int(args.batch_size),
            "learning_rate": float(args.lr),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
            "fold": int(args.fold),
            "num_folds": int(args.num_folds),
            "num_workers": int(args.num_workers),
            "test_ratio": float(args.test_ratio),
            "lr_scheduler_patience": 5,
            "early_stopping_patience": 5,
            "optimizer": "adamw",
            "weight_decay": float(args.weight_decay),
            "monitor_metric": "valid_balanced_score",
            "monitor_mode": "max",
            "do_preprocess": False,
        },
        "joint_predictor": {
            "predictor_type": str(args.predictor_type),
            "joint_direction": str(args.joint_direction),
            "condition_mode": str(args.condition_mode),
            "detach_condition": parse_bool_flag(args.detach_condition),
            "los_target_mode": str(args.los_target_mode),
            "los_coarse_breakdown": parse_bool_flag(args.los_coarse_breakdown),
            "lambda_los": float(args.lambda_los),
            "lambda_aux": float(args.lambda_aux),
            "lambda_joint": float(args.lambda_joint),
            "prior_recon_weight": float(args.prior_recon_weight),
            "beta_kl_start": float(args.beta_kl_start),
            "beta_kl_max": float(args.beta_kl_max),
            "kl_anneal_epochs": int(args.kl_anneal_epochs),
            "joint_heads": str(args.joint_heads),
            "save_cache": bool(args.save_cache),
            "cache_dir": args.cache_dir,
        },
        "joint_struct_loss": dict(DEFAULT_JOINT_STRUCT_LOSS),
    }


def _normalize_joint_struct_loss_cfg(
    cfg: dict[str, Any],
    *,
    predictor_type: str,
    target_head_names: list[str],
) -> tuple[dict[str, Any], SoftJointDriftLoss | None]:
    merged = dict(DEFAULT_JOINT_STRUCT_LOSS)
    merged.update(cfg.get("joint_struct_loss") or {})
    cfg["joint_struct_loss"] = merged
    enabled = bool(merged.get("enabled", False))
    lambda_struct = float(merged.get("lambda_struct", 0.0))
    if not enabled or lambda_struct <= 0.0:
        return merged, None
    if predictor_type != "joint_consistent":
        raise ValueError(
            "joint_struct_loss is currently supported only for "
            "joint_predictor.predictor_type=joint_consistent"
        )
    if str(merged.get("loss_type", "soft_js_d")).lower() != "soft_js_d":
        raise ValueError(
            "Unsupported joint_struct_loss.loss_type. v1 supports only soft_js_d."
        )
    module = SoftJointDriftLoss(
        risk_head_set=str(merged["risk_head_set"]),
        available_heads=target_head_names,
        stopgrad_los=bool(merged.get("stopgrad_los", True)),
        min_los_support=float(merged.get("min_los_support", 1.0e-6)),
        eps=float(merged.get("eps", 1.0e-8)),
        weight_by_los_support=bool(merged.get("weight_by_los_support", True)),
        use_ema=bool(merged.get("use_ema", False)),
        ema_momentum=float(merged.get("ema_momentum", 0.95)),
    )
    merged["enabled"] = True
    merged["lambda_struct"] = lambda_struct
    merged["loss_type"] = "soft_js_d"
    merged["resolved_risk_heads"] = list(module.resolved_risk_heads)
    return merged, module


def _build_d_target_dict(
    y: torch.Tensor, target_names: list[str]
) -> dict[str, torch.Tensor]:
    return {name: y[:, idx].contiguous() for idx, name in enumerate(target_names)}


def _teacher_ratio(cfg: dict[str, Any], epoch: int) -> float:
    mode = str(cfg["joint_predictor"]["condition_mode"]).lower()
    if mode != "scheduled":
        return 0.0
    epochs = max(int(cfg["train"]["epochs"]), 1)
    if epochs == 1:
        return 0.0
    progress = float(epoch - 1) / float(epochs - 1)
    return max(0.0, 1.0 - progress)


def _predict_los_labels(probs: torch.Tensor, target_mode: str) -> torch.Tensor:
    pred = torch.argmax(probs, dim=1).long()
    if target_mode == "raw37":
        return pred + 1
    return pred


def _evaluate_generative_prior(
    model: JointGenerativePredictor,
    loader: DataLoader,
    criterion: JointGenerativeLoss,
    device: torch.device,
    dataset: JointPredictionDataset,
    *,
    beta_kl: float,
    posterior_diagnostics: bool,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    total_prior_loss = 0.0
    n_batches = 0
    d_logits: dict[str, list[np.ndarray]] = {
        name: [] for name in dataset.target_col_names
    }
    d_targets: dict[str, list[np.ndarray]] = {
        name: [] for name in dataset.target_col_names
    }
    los_logits_chunks: list[torch.Tensor] = []
    los_probs_chunks: list[torch.Tensor] = []
    los_targets_chunks: list[torch.Tensor] = []
    los_raw_chunks: list[torch.Tensor] = []
    row_idx_chunks: list[torch.Tensor] = []
    latent_mu_chunks: list[torch.Tensor] = []
    latent_logstd_chunks: list[torch.Tensor] = []
    posterior_mu_chunks: list[torch.Tensor] = []
    posterior_logstd_chunks: list[torch.Tensor] = []
    loss_sums: dict[str, float] = {
        "recon_q_D": 0.0,
        "recon_q_LOS": 0.0,
        "recon_p_D": 0.0,
        "recon_p_LOS": 0.0,
        "KL": 0.0,
        "total_loss": 0.0,
    }
    diagnostic_batches = 0

    with torch.no_grad():
        for x, y_d_cpu, y_los_cpu, los_raw, row_idx in loader:
            x = x.to(device, non_blocking=True)
            y_d = y_d_cpu.to(device, non_blocking=True)
            y_los = y_los_cpu.to(device, non_blocking=True)
            d_target_dict = _build_d_target_dict(y_d, dataset.target_col_names)
            if posterior_diagnostics:
                output = model(x, d_targets=d_target_dict, los_targets=y_los)
            else:
                # Cache/export must match deployment: no true future targets enter the model.
                output = model(x)

            recon_p_d, recon_p_los = criterion.reconstruction_terms(
                output.prior_d_logits,
                output.prior_los_logits,
                d_targets=d_target_dict,
                los_targets=y_los,
            )
            prior_loss = recon_p_d + recon_p_los
            total_prior_loss += float(prior_loss.detach().cpu())
            n_batches += 1

            if posterior_diagnostics and output.kl is not None:
                _, loss_metrics = criterion(
                    output,
                    d_targets=d_target_dict,
                    los_targets=y_los,
                    beta_kl=float(beta_kl),
                )
                for key in loss_sums:
                    loss_sums[key] += float(loss_metrics[key])
                diagnostic_batches += 1
            else:
                loss_sums["recon_p_D"] += float(recon_p_d.detach().cpu())
                loss_sums["recon_p_LOS"] += float(recon_p_los.detach().cpu())

            for idx, head_name in enumerate(dataset.target_col_names):
                d_logits[head_name].append(
                    output.prior_d_logits[head_name].detach().cpu().numpy()
                )
                d_targets[head_name].append(y_d_cpu[:, idx].detach().cpu().numpy())
            los_logits_chunks.append(output.prior_los_logits.detach().cpu())
            los_probs_chunks.append(output.prior_los_probs.detach().cpu())
            los_targets_chunks.append(y_los_cpu.detach().cpu())
            los_raw_chunks.append(los_raw.detach().cpu())
            row_idx_chunks.append(row_idx.detach().cpu())
            latent_mu_chunks.append(output.mu_p.detach().cpu())
            latent_logstd_chunks.append(output.logstd_p.detach().cpu())
            if output.mu_q is not None:
                posterior_mu_chunks.append(output.mu_q.detach().cpu())
            if output.logstd_q is not None:
                posterior_logstd_chunks.append(output.logstd_q.detach().cpu())

    d_logits_np = {
        name: np.concatenate(parts, axis=0) for name, parts in d_logits.items()
    }
    d_targets_np = {
        name: np.concatenate(parts, axis=0) for name, parts in d_targets.items()
    }
    discharge_metrics = compute_discharge_metrics(d_logits_np, d_targets_np)
    los_probs = torch.cat(los_probs_chunks, dim=0)
    los_targets = torch.cat(los_targets_chunks, dim=0)
    los_raw = torch.cat(los_raw_chunks, dim=0)
    if dataset.los_target_mode == "coarse":
        los_pred_metric = torch.argmax(los_probs, dim=1).cpu().numpy()
        los_true_metric = los_targets.cpu().numpy()
    else:
        los_pred_metric = (torch.argmax(los_probs, dim=1) + 1).cpu().numpy()
        los_true_metric = los_raw.cpu().numpy()
    los_metrics = compute_ordinal_metrics(los_true_metric, los_pred_metric)
    mu_p = torch.cat(latent_mu_chunks, dim=0)
    logstd_p = torch.cat(latent_logstd_chunks, dim=0)
    mu_q = torch.cat(posterior_mu_chunks, dim=0) if posterior_mu_chunks else None
    logstd_q = (
        torch.cat(posterior_logstd_chunks, dim=0) if posterior_logstd_chunks else None
    )
    loss_denominator = max(
        diagnostic_batches if posterior_diagnostics else n_batches, 1
    )
    metrics = {
        "loss": total_prior_loss / max(n_batches, 1),
        "discharge_mean_accuracy": float(discharge_metrics["mean_accuracy"]),
        "discharge_mean_macro_f1": float(discharge_metrics["mean_macro_f1"]),
        "los_acc": float(los_metrics["acc"]),
        "los_macro_f1": float(los_metrics["macro_f1"]),
        "los_mae": float(los_metrics["mae"]),
        "los_within_1_acc": float(los_metrics["within_1_acc"]),
        "los_within_2_acc": float(los_metrics["within_2_acc"]),
        "los_qwk": float(los_metrics["qwk"]),
        "recon_q_D": float(loss_sums["recon_q_D"] / loss_denominator),
        "recon_q_LOS": float(loss_sums["recon_q_LOS"] / loss_denominator),
        "recon_p_D": float(loss_sums["recon_p_D"] / max(n_batches, 1)),
        "recon_p_LOS": float(loss_sums["recon_p_LOS"] / max(n_batches, 1)),
        "KL": float(loss_sums["KL"] / loss_denominator),
        "beta_kl": float(beta_kl),
        "total_loss": float(loss_sums["total_loss"] / loss_denominator),
        "mu_p_mean": float(mu_p.mean().item()),
        "mu_p_std": float(mu_p.std(unbiased=False).item()),
        "logstd_p_mean": float(logstd_p.mean().item()),
        "logstd_p_std": float(logstd_p.std(unbiased=False).item()),
    }
    if mu_q is not None:
        metrics["mu_q_mean"] = float(mu_q.mean().item())
        metrics["mu_q_std"] = float(mu_q.std(unbiased=False).item())
    if logstd_q is not None:
        metrics["logstd_q_mean"] = float(logstd_q.mean().item())
        metrics["logstd_q_std"] = float(logstd_q.std(unbiased=False).item())
    for key, value in discharge_metrics.items():
        if key.startswith(("acc_", "f1_")):
            metrics[f"discharge_{key}"] = float(value)
    metrics["balanced_score"] = _balanced_validation_score(
        {
            "discharge_mean_macro_f1": metrics["discharge_mean_macro_f1"],
            "los_macro_f1": metrics["los_macro_f1"],
            "los_qwk": metrics["los_qwk"],
        },
        los_target_mode=dataset.los_target_mode,
    )
    payload = {
        "d_logits_np": d_logits_np,
        "d_targets_np": d_targets_np,
        "los_logits": torch.cat(los_logits_chunks, dim=0),
        "los_probs": los_probs,
        "los_targets": los_targets,
        "los_raw": los_raw,
        "row_idx": torch.cat(row_idx_chunks, dim=0),
    }
    return metrics, payload


def _evaluate(
    model: JointConsistentPredictor | JointGenerativePredictor,
    loader: DataLoader,
    criterion: JointConsistencyLoss | JointGenerativeLoss,
    device: torch.device,
    dataset: JointPredictionDataset,
    *,
    beta_kl: float = 0.0,
) -> tuple[dict[str, float], dict[str, Any]]:
    if isinstance(model, JointGenerativePredictor):
        if not isinstance(criterion, JointGenerativeLoss):
            raise TypeError("JointGenerativePredictor requires JointGenerativeLoss")
        return _evaluate_generative_prior(
            model,
            loader,
            criterion,
            device,
            dataset,
            beta_kl=float(beta_kl),
            posterior_diagnostics=True,
        )

    model.eval()
    total_loss = 0.0
    n_batches = 0
    d_logits: dict[str, list[np.ndarray]] = {
        name: [] for name in dataset.target_col_names
    }
    d_targets: dict[str, list[np.ndarray]] = {
        name: [] for name in dataset.target_col_names
    }
    los_logits_chunks: list[torch.Tensor] = []
    los_probs_chunks: list[torch.Tensor] = []
    los_targets_chunks: list[torch.Tensor] = []
    los_raw_chunks: list[torch.Tensor] = []
    row_idx_chunks: list[torch.Tensor] = []
    loss_metric_sums: dict[str, float] = {}
    with torch.no_grad():
        for x, y_d_cpu, y_los_cpu, los_raw, row_idx in loader:
            x = x.to(device, non_blocking=True)
            y_d = y_d_cpu.to(device, non_blocking=True)
            y_los = y_los_cpu.to(device, non_blocking=True)
            d_target_dict = _build_d_target_dict(y_d, dataset.target_col_names)
            d_target_dict_cpu = _build_d_target_dict(y_d_cpu, dataset.target_col_names)
            output = model(
                x,
                d_targets=d_target_dict,
                los_targets=y_los,
            )
            loss, loss_metrics = criterion(
                output,
                d_targets=d_target_dict,
                los_targets=y_los,
                d_targets_for_joint=d_target_dict_cpu,
                los_targets_for_joint=y_los_cpu,
            )
            total_loss += float(loss.detach().cpu())
            n_batches += 1
            for key, value in loss_metrics.items():
                if key.startswith("loss_struct"):
                    loss_metric_sums[key] = loss_metric_sums.get(key, 0.0) + float(value)

            for idx, head_name in enumerate(dataset.target_col_names):
                d_logits[head_name].append(
                    output.final_d_logits[head_name].detach().cpu().numpy()
                )
                d_targets[head_name].append(y_d_cpu[:, idx].detach().cpu().numpy())
            los_logits_chunks.append(output.final_los_logits.detach().cpu())
            los_probs_chunks.append(output.final_los_probs.detach().cpu())
            los_targets_chunks.append(y_los_cpu.detach().cpu())
            los_raw_chunks.append(los_raw.detach().cpu())
            row_idx_chunks.append(row_idx.detach().cpu())

    d_logits_np = {
        name: np.concatenate(parts, axis=0) for name, parts in d_logits.items()
    }
    d_targets_np = {
        name: np.concatenate(parts, axis=0) for name, parts in d_targets.items()
    }
    discharge_metrics = compute_discharge_metrics(d_logits_np, d_targets_np)
    los_probs = torch.cat(los_probs_chunks, dim=0)
    los_targets = torch.cat(los_targets_chunks, dim=0)
    los_raw = torch.cat(los_raw_chunks, dim=0)
    if dataset.los_target_mode == "coarse":
        los_pred_metric = torch.argmax(los_probs, dim=1).cpu().numpy()
        los_true_metric = los_targets.cpu().numpy()
    else:
        los_pred_metric = (torch.argmax(los_probs, dim=1) + 1).cpu().numpy()
        los_true_metric = los_raw.cpu().numpy()
    los_metrics = compute_ordinal_metrics(los_true_metric, los_pred_metric)
    metrics = {
        "loss": total_loss / max(n_batches, 1),
        "discharge_mean_accuracy": float(discharge_metrics["mean_accuracy"]),
        "discharge_mean_macro_f1": float(discharge_metrics["mean_macro_f1"]),
        "los_acc": float(los_metrics["acc"]),
        "los_macro_f1": float(los_metrics["macro_f1"]),
        "los_mae": float(los_metrics["mae"]),
        "los_within_1_acc": float(los_metrics["within_1_acc"]),
        "los_within_2_acc": float(los_metrics["within_2_acc"]),
        "los_qwk": float(los_metrics["qwk"]),
    }
    for key, value in loss_metric_sums.items():
        metrics[key] = float(value / max(n_batches, 1))
    for key, value in discharge_metrics.items():
        if key.startswith(("acc_", "f1_")):
            metrics[f"discharge_{key}"] = float(value)
    metrics["balanced_score"] = _balanced_validation_score(
        {
            "discharge_mean_macro_f1": metrics["discharge_mean_macro_f1"],
            "los_macro_f1": metrics["los_macro_f1"],
            "los_qwk": metrics["los_qwk"],
        },
        los_target_mode=dataset.los_target_mode,
    )
    payload = {
        "d_logits_np": d_logits_np,
        "d_targets_np": d_targets_np,
        "los_logits": torch.cat(los_logits_chunks, dim=0),
        "los_probs": los_probs,
        "los_targets": los_targets,
        "los_raw": los_raw,
        "row_idx": torch.cat(row_idx_chunks, dim=0),
    }
    return metrics, payload


def _print_epoch(
    epoch: int, epochs: int, train_loss: float, metrics: dict[str, float]
) -> None:
    parts = [
        f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f}",
        f"val_loss={metrics['loss']:.4f}",
        f"d_macro_f1={metrics['discharge_mean_macro_f1']:.4f}",
        f"los_macro_f1={metrics['los_macro_f1']:.4f}",
        f"los_qwk={metrics['los_qwk']:.4f}",
        f"balanced={metrics['balanced_score']:.4f}",
    ]
    if "KL" in metrics:
        parts.extend(
            [
                f"KL={metrics['KL']:.4f}",
                f"beta_kl={metrics['beta_kl']:.6f}",
                f"recon_q={metrics['recon_q_D'] + metrics['recon_q_LOS']:.4f}",
                f"recon_p={metrics['recon_p_D'] + metrics['recon_p_LOS']:.4f}",
            ]
        )
    for key in ("mu_p_std", "logstd_p_std", "mu_q_std", "logstd_q_std"):
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.4f}")
    print(" | ".join(parts))


def _write_joint_stats_artifacts(output_dir: str, summary: dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    _save_json(os.path.join(output_dir, "joint_stats_summary.json"), summary)
    pd.DataFrame(summary["per_head"]).to_csv(
        os.path.join(output_dir, "joint_stats_per_head.csv"), index=False
    )
    pd.DataFrame(summary["los_given_d_rows"]).to_csv(
        os.path.join(output_dir, "per_head_conditional_los_given_d.csv"), index=False
    )


def _generate_joint_drift_reports(
    *,
    run_dir: str,
    train_cache_path: str,
    eval_cache_paths: dict[str, str],
    focus_heads: tuple[str, ...] = DEFAULT_DRIFT_HEADS,
    rare_threshold: float = 0.0025,
) -> dict[str, Any]:
    diagnostics_root = os.path.join(run_dir, "joint_drift")
    os.makedirs(diagnostics_root, exist_ok=True)

    train_cache = torch.load(train_cache_path, map_location="cpu", weights_only=False)
    train_cache["_path"] = str(train_cache_path)
    available_heads = set(train_cache.get("final_d_pred", {}).keys())
    selected_heads = [head for head in focus_heads if head in available_heads]

    payload: dict[str, Any] = {}
    for split_name, eval_cache_path in eval_cache_paths.items():
        eval_cache = torch.load(eval_cache_path, map_location="cpu", weights_only=False)
        eval_cache["_path"] = str(eval_cache_path)
        split_dir = os.path.join(diagnostics_root, str(split_name))
        summary = compute_joint_stats(
            train_cache,
            eval_cache,
            rare_threshold=float(rare_threshold),
        )
        _write_joint_stats_artifacts(split_dir, summary)
        report = generate_joint_los_given_d_report(
            split_dir,
            heads=selected_heads or None,
            top_k=max(len(selected_heads), 3),
            limit=200,
            script_path=Path("scripts/analyze_joint_los_given_d_report.py"),
        )
        payload[str(split_name)] = {
            "summary_path": os.path.join(split_dir, "joint_stats_summary.json"),
            "per_head_path": os.path.join(split_dir, "joint_stats_per_head.csv"),
            "los_given_d_path": os.path.join(
                split_dir, "per_head_conditional_los_given_d.csv"
            ),
            "report_summary_path": report["artifacts"]["summary_json"],
            "mean_js_d_given_los": float(summary["mean_js_d_given_los"]),
            "mean_js_los_given_d": float(summary["mean_js_los_given_d"]),
            "mean_rare_combo_rate_predicted": float(
                summary["mean_rare_combo_rate_predicted"]
            ),
            "focused_heads": list(report["heads"]),
        }
    _save_json(os.path.join(diagnostics_root, "summary.json"), payload)
    return payload


def _save_predictions_csv(
    run_dir: str,
    payload: dict[str, Any],
    dataset: JointPredictionDataset,
    split_name: str,
) -> None:
    rows: dict[str, Any] = {"row_idx": payload["row_idx"].cpu().numpy().astype(int)}
    for head_name in dataset.target_col_names:
        rows[f"true_{head_name}"] = payload["d_targets_np"][head_name].astype(int)
        rows[f"pred_{head_name}"] = np.argmax(
            payload["d_logits_np"][head_name], axis=1
        ).astype(int)
    los_probs = payload["los_probs"]
    rows["true_los"] = payload["los_raw"].cpu().numpy().astype(int)
    if dataset.los_target_mode == "coarse":
        rows["pred_los"] = torch.argmax(los_probs, dim=1).cpu().numpy().astype(int)
    else:
        rows["pred_los"] = (
            (torch.argmax(los_probs, dim=1) + 1).cpu().numpy().astype(int)
        )
    pd.DataFrame(rows).to_csv(
        os.path.join(run_dir, f"{split_name}_predictions.csv"), index=False
    )


def _export_cache(
    *,
    output_dir: str,
    split_name: str,
    payload: dict[str, Any],
    dataset: JointPredictionDataset,
    cfg: dict[str, Any],
    caseid_lookup: dict[int, str] | None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    joint_cfg = cfg.get("joint_predictor", {})
    predictor_type = str(joint_cfg.get("predictor_type", "joint_consistent")).lower()
    final_d_logits = {
        head_name: torch.tensor(values, dtype=torch.float32)
        for head_name, values in payload["d_logits_np"].items()
    }
    final_d_probs = {
        head_name: torch.softmax(logits, dim=1)
        for head_name, logits in final_d_logits.items()
    }
    final_d_pred = {
        head_name: torch.argmax(probs, dim=1).long()
        for head_name, probs in final_d_probs.items()
    }
    final_los_probs = payload["los_probs"].to(dtype=torch.float32)
    if dataset.los_target_mode == "coarse":
        final_los_pred = torch.argmax(final_los_probs, dim=1).long()
    else:
        final_los_pred = (torch.argmax(final_los_probs, dim=1) + 1).long()
    cache_payload = {
        "split": split_name,
        "final_d_logits": final_d_logits,
        "final_d_probs": final_d_probs,
        "final_d_pred": final_d_pred,
        "final_los_logits": payload["los_logits"].to(dtype=torch.float32),
        "final_los_probs": final_los_probs,
        "final_los_pred": final_los_pred,
        "row_idx": payload["row_idx"].long(),
        "targets": {
            "d": {
                head_name: torch.tensor(values, dtype=torch.long)
                for head_name, values in payload["d_targets_np"].items()
            },
            "los_target": payload["los_targets"].long(),
            "los_raw": payload["los_raw"].long(),
        },
        "caseid": (
            None
            if caseid_lookup is None
            else [
                caseid_lookup.get(int(idx), str(int(idx)))
                for idx in payload["row_idx"].tolist()
            ]
        ),
        "metadata": {
            **dataset.schema_metadata,
            "predictor_type": predictor_type,
            "joint_direction": joint_cfg.get("joint_direction"),
            "condition_mode": joint_cfg.get("condition_mode"),
            "detach_condition": bool(joint_cfg.get("detach_condition", False)),
            "joint_heads": resolve_joint_heads(
                dataset.target_col_names,
                joint_cfg.get("joint_heads", "all"),
            ),
            "latent_dim": cfg.get("model", {}).get("params", {}).get("latent_dim"),
            "prior_recon_weight": joint_cfg.get("prior_recon_weight"),
            "beta_kl_start": joint_cfg.get("beta_kl_start"),
            "beta_kl_max": joint_cfg.get("beta_kl_max"),
            "kl_anneal_epochs": joint_cfg.get("kl_anneal_epochs"),
            "z_sampling_at_eval": cfg.get("model", {})
            .get("params", {})
            .get("z_sampling_at_eval"),
            "num_eval_samples": cfg.get("model", {})
            .get("params", {})
            .get("num_eval_samples"),
            "seed": int(cfg["train"]["seed"]),
            "fold": int(cfg["train"]["fold"]),
            "split": split_name,
            "default_discharge_representation": "hard_argmax",
            "default_los_representation": "distribution",
            "final_los_pred_space": (
                "coarse_class" if dataset.los_target_mode == "coarse" else "raw_los"
            ),
        },
    }
    joint_struct_cfg = cfg.get("joint_struct_loss") or {}
    if bool(joint_struct_cfg.get("enabled", False)) and float(joint_struct_cfg.get("lambda_struct", 0.0)) > 0.0:
        cache_payload["metadata"]["joint_struct_loss"] = {
            **joint_struct_cfg,
            "resolved_risk_heads": list(joint_struct_cfg.get("resolved_risk_heads", [])),
        }
    path = os.path.join(output_dir, f"{split_name}.pt")
    torch.save(cache_payload, path)
    return path


def run_joint_consistent_predictor(
    cfg: dict[str, Any],
    root: str,
    *,
    run_dir: str | None = None,
    split_indices: dict[str, np.ndarray] | None = None,
    export_indices: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = device_set(cfg.get("device"))
    joint_cfg = cfg["joint_predictor"]
    los_target_mode = infer_los_target_from_cfg(joint_cfg)
    los_coarse_breakdown = (
        infer_los_coarse_breakdown_from_cfg(joint_cfg)
        if los_target_mode == "coarse"
        else False
    )
    if los_target_mode == "coarse":
        joint_cfg["los_coarse_breakdown"] = bool(los_coarse_breakdown)
        joint_cfg["num_classes"] = get_los_coarse_num_classes(
            breakdown=los_coarse_breakdown
        )
    dataset = JointPredictionDataset(
        root=root,
        do_preprocess=bool(cfg["train"].get("do_preprocess", False)),
        los_target_mode=los_target_mode,
        los_coarse_breakdown=los_coarse_breakdown,
    )
    if run_dir is None:
        run_id = make_run_id(cfg)
        run_dir = ensure_run_dir("runs", run_id)
    else:
        os.makedirs(run_dir, exist_ok=False)
        os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    logger = ExperimentLogger(cfg, run_dir)

    if split_indices is None:
        train_idx, val_idx, test_idx = _split_indices(dataset, cfg)
    else:
        train_idx = np.asarray(split_indices["train"], dtype=np.int64)
        val_idx = np.asarray(split_indices["val"], dtype=np.int64)
        test_idx = np.asarray(split_indices["test"], dtype=np.int64)
    pin_memory = _pin_memory_for_device(device)
    _save_json(
        os.path.join(run_dir, "split_indices.json"),
        {
            "train_idx": train_idx.tolist(),
            "val_idx": val_idx.tolist(),
            "test_idx": test_idx.tolist(),
            "custom_split_indices": bool(split_indices is not None),
            "fold": int(cfg["train"]["fold"]),
            "num_folds": int(cfg["train"]["num_folds"]),
            "stratification": "coarse_los",
        },
    )

    train_loader = _make_loader(
        dataset,
        train_idx,
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        True,
        pin_memory,
    )
    val_loader = _make_loader(
        dataset,
        val_idx,
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        False,
        pin_memory,
    )
    test_loader = _make_loader(
        dataset,
        test_idx,
        int(cfg["train"]["batch_size"]),
        int(cfg["train"]["num_workers"]),
        False,
        pin_memory,
    )

    predictor_type = str(
        cfg["joint_predictor"].get("predictor_type", "joint_consistent")
    ).lower()
    if predictor_type == "joint_generative":
        model = JointGenerativePredictor(
            ad_col_dims=dataset.ad_col_dims,
            target_col_names=dataset.target_col_names,
            target_col_dims=dataset.target_col_dims,
            los_num_classes=dataset.los_num_classes,
            **cfg["model"]["params"],
        ).to(device)
        criterion = JointGenerativeLoss(
            lambda_los=float(cfg["joint_predictor"].get("lambda_los", 1.0)),
            prior_recon_weight=float(
                cfg["joint_predictor"].get("prior_recon_weight", 0.5)
            ),
        )
    elif predictor_type == "joint_consistent":
        struct_cfg, struct_loss_module = _normalize_joint_struct_loss_cfg(
            cfg,
            predictor_type=predictor_type,
            target_head_names=dataset.target_col_names,
        )
        model = JointConsistentPredictor(
            ad_col_dims=dataset.ad_col_dims,
            target_col_names=dataset.target_col_names,
            target_col_dims=dataset.target_col_dims,
            los_num_classes=dataset.los_num_classes,
            joint_direction=cfg["joint_predictor"]["joint_direction"],
            condition_mode=cfg["joint_predictor"]["condition_mode"],
            detach_condition=bool(cfg["joint_predictor"]["detach_condition"]),
            joint_heads=cfg["joint_predictor"]["joint_heads"],
            **cfg["model"]["params"],
        ).to(device)
        criterion = JointConsistencyLoss(
            lambda_los=float(cfg["joint_predictor"]["lambda_los"]),
            lambda_aux=float(cfg["joint_predictor"]["lambda_aux"]),
            lambda_joint=float(cfg["joint_predictor"]["lambda_joint"]),
            joint_head_names=model.selected_joint_heads,
            lambda_struct=float(struct_cfg.get("lambda_struct", 0.0)),
            struct_loss_module=struct_loss_module,
        )
    else:
        raise ValueError(
            f"Unsupported joint_predictor.predictor_type: {predictor_type}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["learning_rate"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, "max", patience=int(cfg["train"]["lr_scheduler_patience"])
    )
    early_stopper = EarlyStopper(patience=int(cfg["train"]["early_stopping_patience"]))
    epochs = int(cfg["train"]["epochs"])

    best_score = -float("inf")
    for epoch in tqdm(range(1, epochs + 1), desc="joint predictor"):
        model.train()
        total_loss = 0.0
        train_loss_sums: dict[str, float] = {
            "recon_q_D": 0.0,
            "recon_q_LOS": 0.0,
            "recon_p_D": 0.0,
            "recon_p_LOS": 0.0,
            "KL": 0.0,
            "beta_kl": 0.0,
            "total_loss": 0.0,
        }
        train_struct_sums: dict[str, float] = {}
        n_batches = 0
        oracle_ratio = _teacher_ratio(cfg, epoch)
        beta_kl = 0.0
        if predictor_type == "joint_generative":
            beta_kl = kl_beta_for_epoch(
                epoch,
                beta_start=float(cfg["joint_predictor"].get("beta_kl_start", 0.0)),
                beta_max=float(cfg["joint_predictor"].get("beta_kl_max", 0.001)),
                anneal_epochs=int(cfg["joint_predictor"].get("kl_anneal_epochs", 10)),
            )
        for x, y_d_cpu, y_los_cpu, _los_raw, _row_idx in train_loader:
            x = x.to(device, non_blocking=True)
            y_d = y_d_cpu.to(device, non_blocking=True)
            y_los = y_los_cpu.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            d_target_dict = _build_d_target_dict(y_d, dataset.target_col_names)
            if predictor_type == "joint_generative":
                output = model(x, d_targets=d_target_dict, los_targets=y_los)
                loss, loss_metrics = criterion(
                    output,
                    d_targets=d_target_dict,
                    los_targets=y_los,
                    beta_kl=float(beta_kl),
                )
                for key in train_loss_sums:
                    train_loss_sums[key] += float(loss_metrics[key])
            else:
                d_target_dict_cpu = _build_d_target_dict(
                    y_d_cpu, dataset.target_col_names
                )
                output = model(
                    x,
                    d_targets=d_target_dict,
                    los_targets=y_los,
                    oracle_ratio=oracle_ratio,
                )
                loss, loss_metrics = criterion(
                    output,
                    d_targets=d_target_dict,
                    los_targets=y_los,
                    d_targets_for_joint=d_target_dict_cpu,
                    los_targets_for_joint=y_los_cpu,
                )
                for key, value in loss_metrics.items():
                    if key.startswith("loss_struct"):
                        train_struct_sums[key] = train_struct_sums.get(key, 0.0) + float(value)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        val_metrics, _ = _evaluate(
            model,
            val_loader,
            criterion,
            device,
            dataset,
            beta_kl=float(beta_kl),
        )
        scheduler.step(float(val_metrics["balanced_score"]))
        current_lr = float(optimizer.param_groups[0]["lr"])
        log_metrics = {
            "lr": current_lr,
            "train_loss": float(train_loss),
            "oracle_ratio": float(oracle_ratio),
            **{f"valid_{k}": float(v) for k, v in val_metrics.items()},
        }
        if predictor_type == "joint_generative":
            for key, value in train_loss_sums.items():
                log_metrics[f"train_{key}"] = float(value / max(n_batches, 1))
        else:
            for key, value in train_struct_sums.items():
                if key == "loss_struct":
                    log_metrics["train_struct_loss"] = float(value / max(n_batches, 1))
                else:
                    suffix = key.removeprefix("loss_")
                    log_metrics[f"train_{suffix}"] = float(value / max(n_batches, 1))
            if "loss_struct" in val_metrics:
                log_metrics["valid_struct_loss"] = float(val_metrics["loss_struct"])
            for key, value in val_metrics.items():
                if key.startswith("loss_struct_"):
                    suffix = key.removeprefix("loss_")
                    log_metrics[f"valid_{suffix}"] = float(value)
            joint_struct_cfg = cfg.get("joint_struct_loss") or {}
            if bool(joint_struct_cfg.get("enabled", False)) and float(joint_struct_cfg.get("lambda_struct", 0.0)) > 0.0:
                log_metrics["lambda_struct"] = float(joint_struct_cfg["lambda_struct"])
        checkpoint_extra = {"schema": dataset.schema_metadata}
        joint_struct_cfg = cfg.get("joint_struct_loss") or {}
        if bool(joint_struct_cfg.get("enabled", False)) and float(joint_struct_cfg.get("lambda_struct", 0.0)) > 0.0:
            checkpoint_extra["joint_struct_loss"] = {
                **joint_struct_cfg,
                "resolved_risk_heads": list(
                    getattr(criterion.struct_loss_module, "resolved_risk_heads", [])
                ),
            }
        logger.log_metrics(epoch, log_metrics)
        logger.maybe_save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=log_metrics,
            extra=checkpoint_extra,
        )
        if logger.best_epoch == epoch:
            monitor_name = str(logger.policy.monitor)
            monitor_value = log_metrics.get(monitor_name)
            if monitor_value is not None:
                print(f"  ✅ New best saved: {monitor_name}={float(monitor_value):.4f}")
        _print_epoch(epoch, epochs, train_loss, val_metrics)
        best_score = max(best_score, float(val_metrics["balanced_score"]))
        if early_stopper(-float(val_metrics["balanced_score"])):
            print("--- Early Stopping activated ---")
            break

    best_ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    final_beta_kl = (
        kl_beta_for_epoch(
            epochs,
            beta_start=float(cfg["joint_predictor"].get("beta_kl_start", 0.0)),
            beta_max=float(cfg["joint_predictor"].get("beta_kl_max", 0.001)),
            anneal_epochs=int(cfg["joint_predictor"].get("kl_anneal_epochs", 10)),
        )
        if predictor_type == "joint_generative"
        else 0.0
    )
    val_metrics, val_payload = _evaluate(
        model,
        val_loader,
        criterion,
        device,
        dataset,
        beta_kl=float(final_beta_kl),
    )
    test_metrics, test_payload = _evaluate(
        model,
        test_loader,
        criterion,
        device,
        dataset,
        beta_kl=float(final_beta_kl),
    )
    logger.log_metrics(
        epochs,
        {
            "split": "val_final",
            **{f"val_final_{k}": float(v) for k, v in val_metrics.items()},
        },
    )
    logger.log_metrics(
        epochs,
        {"split": "test", **{f"test_{k}": float(v) for k, v in test_metrics.items()}},
    )
    _save_json(os.path.join(run_dir, "val_metrics.json"), val_metrics)
    _save_json(os.path.join(run_dir, "test_metrics.json"), test_metrics)
    _save_predictions_csv(run_dir, val_payload, dataset, "val")
    _save_predictions_csv(run_dir, test_payload, dataset, "test")

    cache_paths: dict[str, str] = {}
    if bool(cfg["joint_predictor"].get("save_cache", False)):
        cache_dir = cfg["joint_predictor"].get("cache_dir") or os.path.join(
            run_dir, "joint_cache"
        )
        caseid_lookup = None
        if dataset.caseid is not None:
            caseid_lookup = {
                int(row_idx): caseid
                for row_idx, caseid in zip(dataset.row_idx.tolist(), dataset.caseid)
            }
        export_plan = export_indices or {
            "train": train_idx,
            "val": val_idx,
            "test": test_idx,
        }
        metrics_by_split: dict[str, dict[str, float]] = {}
        payload_by_split: dict[str, dict[str, Any]] = {}
        if predictor_type != "joint_generative":
            payload_by_split.update({"val": val_payload, "test": test_payload})
            metrics_by_split.update({"val": val_metrics, "test": test_metrics})
            train_metrics, train_payload = _evaluate(
                model, train_loader, criterion, device, dataset
            )
            metrics_by_split["train"] = train_metrics
            payload_by_split["train"] = train_payload

        for split_name, indices in export_plan.items():
            indices = np.asarray(indices, dtype=np.int64)
            payload = payload_by_split.get(split_name)
            metrics = metrics_by_split.get(split_name)
            if payload is None or metrics is None:
                loader = _make_loader(
                    dataset,
                    indices,
                    int(cfg["train"]["batch_size"]),
                    int(cfg["train"]["num_workers"]),
                    False,
                    pin_memory,
                )
                if predictor_type == "joint_generative":
                    metrics, payload = _evaluate_generative_prior(
                        model,
                        loader,
                        criterion,
                        device,
                        dataset,
                        beta_kl=float(final_beta_kl),
                        posterior_diagnostics=False,
                    )
                else:
                    metrics, payload = _evaluate(
                        model, loader, criterion, device, dataset
                    )
                metrics_by_split[split_name] = metrics
                payload_by_split[split_name] = payload

            _save_json(os.path.join(run_dir, f"{split_name}_metrics.json"), metrics)
            cache_paths[str(split_name)] = _export_cache(
                output_dir=cache_dir,
                split_name=str(split_name),
                payload=payload,
                dataset=dataset,
                cfg=cfg,
                caseid_lookup=caseid_lookup,
            )
        _save_json(os.path.join(cache_dir, "cache_manifest.json"), cache_paths)
        if "train" in cache_paths:
            diagnostic_paths = {
                split_name: path
                for split_name, path in cache_paths.items()
                if split_name != "train"
            }
            if diagnostic_paths:
                drift_payload = _generate_joint_drift_reports(
                    run_dir=run_dir,
                    train_cache_path=cache_paths["train"],
                    eval_cache_paths=diagnostic_paths,
                )
                _save_json(
                    os.path.join(run_dir, "joint_drift_summary.json"),
                    drift_payload,
                )

    print(
        f"[joint predictor] best_valid_balanced={best_score:.4f} "
        f"test_d_macro_f1={test_metrics['discharge_mean_macro_f1']:.4f} "
        f"test_los_qwk={test_metrics['los_qwk']:.4f}"
    )
    return {
        "run_dir": run_dir,
        "best_valid_balanced": float(best_score),
        "test_discharge_mean_macro_f1": float(test_metrics["discharge_mean_macro_f1"]),
        "test_los_qwk": float(test_metrics["los_qwk"]),
        "cache_paths": cache_paths,
        "resolved_risk_heads": list(
            getattr(getattr(criterion, "struct_loss_module", None), "resolved_risk_heads", [])
        ),
    }


def main() -> None:
    args = parse_args()
    cfg = _build_config_from_args(args)
    run_joint_consistent_predictor(cfg, os.path.abspath(args.root))


if __name__ == "__main__":
    main()
