from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.models.discharge_predictor.los_utils import LOS_COARSE_BINS, map_los_array_to_coarse_bins
from src.models.factory import build_model
from src.models.forecast_inputs import ensure_model_forecast_defaults
from src.trainers.base import _move_soft_discharge_to_device, _unpack_batch
from src.utils.device_set import device_set


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _resolve_run_dir(run_name: str) -> Path:
    return _PROJECT_ROOT / "runs" / run_name


def _fold_dir(run_name: str, fold: int) -> Path:
    return _resolve_run_dir(run_name) / "folds" / f"fold_{int(fold)}"


def _safe_device(requested: str | None) -> torch.device:
    device = device_set(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA device is unavailable; falling back to cpu.")
        return torch.device("cpu")
    if device.type == "mps" and not torch.mps.is_available():
        print("Requested MPS device is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return device


def _build_dataset(cfg: dict[str, Any], dataset_root: Path) -> TEDSTensorDataset:
    ensure_model_forecast_defaults(cfg)
    remove_los = cfg["model"]["name"] not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]
    return TEDSTensorDataset(
        root=str(dataset_root),
        binary=bool(cfg["train"].get("binary", True)),
        ig_label=bool(cfg["train"].get("ig_label", False)),
        remove_los=remove_los,
        do_preprocess=bool(cfg["train"].get("do_preprocess", True)),
    )


class _CachedSplitDataset(Dataset):
    def __init__(self, base_dataset: TEDSTensorDataset, split_payload: dict[str, Any]) -> None:
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
        _, y, los_raw = self.base_dataset[row_idx]
        if self.soft_discharge_cache is None:
            return self.x[index], y, self.los[index], {"row_idx": row_idx, "los_raw": int(los_raw)}

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
            "row_idx": row_idx,
            "los_raw": int(los_raw),
        }
        return self.x[index], y, self.los[index], forecast_meta


def _build_cached_loader(
    *,
    base_dataset: TEDSTensorDataset,
    split_payload: dict[str, Any],
    expected_indices: np.ndarray,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    drop_last: bool,
) -> DataLoader:
    actual_indices = torch.as_tensor(split_payload["indices"], dtype=torch.long).cpu().numpy()
    expected = np.asarray(expected_indices, dtype=np.int64)
    if not np.array_equal(actual_indices, expected):
        raise ValueError("Cached split indices do not match saved split indices.")
    dataset = _CachedSplitDataset(base_dataset, split_payload)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
    )


def _load_baseline_artifacts(
    *,
    run_name: str,
    fold: int,
    dataset_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    fold_dir = _fold_dir(run_name, fold)
    cfg = _load_yaml(fold_dir / "config.final.yaml")
    split_payload = _load_json(fold_dir / "joint_forecast_pipeline_splits.json")
    ensure_model_forecast_defaults(cfg)
    cfg["model"]["params"]["device"] = str(device)

    dataset = _build_dataset(cfg, dataset_root)
    model = build_model(model_name=cfg["model"]["name"], **cfg["model"].get("params", {})).to(device)
    ckpt = torch.load(fold_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    edge_index = torch.load(fold_dir / "edge_index.pt", map_location=device, weights_only=False)
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, int(cfg["train"]["batch_size"]))

    cache_dir = fold_dir / "cached_predictions"
    cache_payloads = {
        "gnn_val": torch.load(cache_dir / "gnn_val_joint.pt", map_location="cpu", weights_only=False),
        "outer_test": torch.load(cache_dir / "outer_test_joint.pt", map_location="cpu", weights_only=False),
    }
    split_indices = {
        "gnn_val": np.asarray(split_payload["gnn_val_idx"], dtype=np.int64),
        "outer_test": np.asarray(split_payload["outer_test_idx"], dtype=np.int64),
    }
    loaders = {
        split_name: _build_cached_loader(
            base_dataset=dataset,
            split_payload=payload,
            expected_indices=split_indices[split_name],
            batch_size=int(cfg["train"]["batch_size"]),
            num_workers=int(cfg["train"].get("num_workers", 0)),
            shuffle=False,
            drop_last=True,
        )
        for split_name, payload in cache_payloads.items()
    }
    return {
        "cfg": cfg,
        "fold_dir": fold_dir,
        "dataset": dataset,
        "model": model,
        "ckpt": ckpt,
        "edge_index": edge_index,
        "cache_payloads": cache_payloads,
        "split_indices": split_indices,
        "loaders": loaders,
    }


def _safe_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return None
    try:
        return float(roc_auc_score(y_true, scores))
    except ValueError:
        return None


def _to_float_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def _binary_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray,
) -> dict[str, float | int | None]:
    if y_true.size == 0:
        return {
            "support": 0,
            "positive_count": 0,
            "negative_count": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "auc": None,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "tp": 0,
        }
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "support": int(y_true.size),
        "positive_count": int(np.sum(y_true == 1)),
        "negative_count": int(np.sum(y_true == 0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "auc": _to_float_or_none(_safe_auc(y_true, y_score)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _class_metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray, class_label: int) -> dict[str, float | int | None]:
    true_pos_mask = y_true == class_label
    pred_pos_mask = y_pred == class_label
    support = int(np.sum(true_pos_mask))
    correct = int(np.sum(true_pos_mask & pred_pos_mask))
    tp = correct
    fp = int(np.sum(~true_pos_mask & pred_pos_mask))
    fn = int(np.sum(true_pos_mask & ~pred_pos_mask))
    precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
    recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
    f1 = float((2.0 * precision * recall) / (precision + recall)) if precision + recall > 0 else 0.0
    return {
        "true_class": int(class_label),
        "support": support,
        "correct": correct,
        "class_accuracy": float(correct / support) if support > 0 else None,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _coarse_label_name(bin_idx: int) -> str:
    lo, hi = LOS_COARSE_BINS[int(bin_idx)]
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _group_value_sets(raw_los: np.ndarray) -> dict[str, np.ndarray]:
    coarse = np.asarray(map_los_array_to_coarse_bins(raw_los), dtype=np.int64)
    return {
        "coarse6": coarse,
        "raw37": np.asarray(raw_los, dtype=np.int64),
    }


def _all_group_labels(space: str) -> list[int]:
    if space == "coarse6":
        return list(range(6))
    if space == "raw37":
        return list(range(1, 38))
    raise ValueError(f"Unsupported LOS space: {space}")


def _group_label_name(space: str, label: int) -> str:
    if space == "coarse6":
        return _coarse_label_name(label)
    return str(label)


def _collect_eval_outputs(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    edge_index: torch.Tensor,
    device: torch.device,
    decision_threshold: float,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_seen = 0
    y_true_chunks: list[np.ndarray] = []
    y_pred_chunks: list[np.ndarray] = []
    y_score_chunks: list[np.ndarray] = []
    row_idx_chunks: list[np.ndarray] = []
    los_raw_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval_process", leave=False):
            x_batch, y_batch, los_batch, forecast_meta = _unpack_batch(batch)
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            los_batch = los_batch.to(device, non_blocking=True)
            soft_discharge = None
            if forecast_meta is not None:
                soft_discharge = _move_soft_discharge_to_device(forecast_meta.get("soft_discharge"), device)

            logits = model(
                x_batch,
                los_batch,
                edge_index,
                device=device,
                soft_discharge=soft_discharge,
            )
            if logits.ndim == 2 and logits.size(1) == 1:
                logits = logits.squeeze(1)
            loss = criterion(logits, y_batch.float())
            probs = torch.sigmoid(logits)
            preds = (probs >= decision_threshold).long()

            batch_size = int(y_batch.size(0))
            total_loss += float(loss.detach().cpu()) * batch_size
            total_seen += batch_size

            y_true_chunks.append(y_batch.detach().cpu().numpy().astype(np.int64))
            y_pred_chunks.append(preds.detach().cpu().numpy().astype(np.int64))
            y_score_chunks.append(probs.detach().cpu().numpy().astype(np.float64))
            row_idx_chunks.append(np.asarray(forecast_meta["row_idx"], dtype=np.int64))
            los_raw_chunks.append(np.asarray(forecast_meta["los_raw"], dtype=np.int64))

    y_true = np.concatenate(y_true_chunks, axis=0)
    y_pred = np.concatenate(y_pred_chunks, axis=0)
    y_score = np.concatenate(y_score_chunks, axis=0)
    row_idx = np.concatenate(row_idx_chunks, axis=0)
    los_raw = np.concatenate(los_raw_chunks, axis=0)
    overall = _binary_metrics_from_arrays(y_true, y_pred, y_score)
    overall["loss"] = float(total_loss / max(total_seen, 1))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_score": y_score,
        "row_idx": row_idx,
        "los_raw": los_raw,
        "overall": overall,
    }


def _subgroup_reports(
    *,
    split_name: str,
    eval_payload: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    y_true = eval_payload["y_true"]
    y_pred = eval_payload["y_pred"]
    y_score = eval_payload["y_score"]
    los_groups = _group_value_sets(eval_payload["los_raw"])
    group_metric_rows: list[dict[str, Any]] = []
    class_metric_rows: list[dict[str, Any]] = []
    confusion_rows: list[dict[str, Any]] = []

    for los_space, group_values in los_groups.items():
        for label in _all_group_labels(los_space):
            mask = group_values == label
            metrics = _binary_metrics_from_arrays(y_true[mask], y_pred[mask], y_score[mask])
            row = {
                "split": split_name,
                "los_space": los_space,
                "los_label": int(label),
                "los_label_name": _group_label_name(los_space, label),
            }
            row.update(metrics)
            group_metric_rows.append(row)

            for class_label in (0, 1):
                class_row = {
                    "split": split_name,
                    "los_space": los_space,
                    "los_label": int(label),
                    "los_label_name": _group_label_name(los_space, label),
                }
                class_row.update(_class_metrics_from_arrays(y_true[mask], y_pred[mask], class_label))
                class_metric_rows.append(class_row)

            conf = confusion_matrix(y_true[mask], y_pred[mask], labels=[0, 1])
            for true_class in (0, 1):
                for pred_class in (0, 1):
                    confusion_rows.append(
                        {
                            "split": split_name,
                            "los_space": los_space,
                            "los_label": int(label),
                            "los_label_name": _group_label_name(los_space, label),
                            "true_class": int(true_class),
                            "pred_class": int(pred_class),
                            "count": int(conf[true_class, pred_class]),
                        }
                    )
    return group_metric_rows, class_metric_rows, confusion_rows


def _load_stored_test_metrics(metrics_path: Path) -> dict[str, Any] | None:
    if not metrics_path.exists():
        return None
    last_test = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row.get("split") == "test":
                last_test = row
    return last_test


def _compare_metrics(current: dict[str, Any], stored: dict[str, Any] | None, mapping: dict[str, str]) -> dict[str, Any]:
    if stored is None:
        return {"stored_metrics_found": False}
    out: dict[str, Any] = {"stored_metrics_found": True}
    for current_key, stored_key in mapping.items():
        current_value = current.get(current_key)
        stored_value = stored.get(stored_key)
        out[current_key] = {
            "current": None if current_value is None else float(current_value),
            "stored": None if stored_value is None else float(stored_value),
            "diff": None
            if current_value is None or stored_value is None
            else float(current_value) - float(stored_value),
        }
    return out


def _print_summary(split_name: str, overall: dict[str, Any], subgroup_rows: list[dict[str, Any]]) -> None:
    print(
        f"[{split_name}] loss={overall['loss']:.6f} "
        f"acc={overall['accuracy']:.6f} prec={overall['precision']:.6f} "
        f"recall={overall['recall']:.6f} f1={overall['f1']:.6f} "
        f"auc={(overall['auc'] if overall['auc'] is not None else float('nan')):.6f}"
    )
    coarse_rows = [row for row in subgroup_rows if row["los_space"] == "coarse6" and row["support"] > 0]
    coarse_rows = sorted(coarse_rows, key=lambda row: row["accuracy"])
    print(f"[{split_name}] worst coarse6 LOS groups by accuracy:")
    for row in coarse_rows[:3]:
        print(
            f"  - {row['los_label_name']}: support={row['support']} "
            f"acc={row['accuracy']:.6f} f1={row['f1']:.6f}"
        )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    device = _safe_device(args.device)
    artifacts = _load_baseline_artifacts(
        run_name=args.run_name,
        fold=args.fold,
        dataset_root=Path(args.dataset_root),
        device=device,
    )
    cfg = artifacts["cfg"]
    fold_dir = artifacts["fold_dir"]
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else fold_dir / "diagnostics" / "forecasted_gnn_performance"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    criterion = nn.BCEWithLogitsLoss()
    decision_threshold = float(cfg["train"]["decision_threshold"])
    batch_size = int(cfg["train"]["batch_size"])

    all_group_rows: list[dict[str, Any]] = []
    all_class_rows: list[dict[str, Any]] = []
    all_confusion_rows: list[dict[str, Any]] = []
    split_payloads: dict[str, Any] = {}

    for split_name, loader in artifacts["loaders"].items():
        eval_payload = _collect_eval_outputs(
            model=artifacts["model"],
            loader=loader,
            criterion=criterion,
            edge_index=artifacts["edge_index"],
            device=device,
            decision_threshold=decision_threshold,
        )
        group_rows, class_rows, confusion_rows = _subgroup_reports(
            split_name=split_name,
            eval_payload=eval_payload,
        )
        all_group_rows.extend(group_rows)
        all_class_rows.extend(class_rows)
        all_confusion_rows.extend(confusion_rows)

        expected_rows = int(artifacts["split_indices"][split_name].shape[0])
        evaluated_rows = int(eval_payload["y_true"].shape[0])
        dropped_rows = expected_rows - evaluated_rows
        overall = dict(eval_payload["overall"])
        overall["evaluated_rows"] = evaluated_rows
        overall["expected_rows"] = expected_rows
        overall["dropped_tail_rows"] = dropped_rows
        overall["batch_size"] = batch_size
        overall["drop_last"] = True

        if split_name == "gnn_val":
            stored_metrics = artifacts["ckpt"].get("metrics") or {}
            compare_map = {
                "loss": "valid_loss",
                "accuracy": "valid_acc",
                "precision": "valid_precision",
                "recall": "valid_recall",
                "f1": "valid_f1",
                "auc": "valid_auc",
            }
        else:
            stored_metrics = _load_stored_test_metrics(fold_dir / "metrics.jsonl")
            compare_map = {
                "loss": "test_loss",
                "accuracy": "test_acc",
                "precision": "test_precision",
                "recall": "test_recall",
                "f1": "test_f1",
                "auc": "test_auc",
            }
        comparison = _compare_metrics(overall, stored_metrics, compare_map)
        split_payloads[split_name] = {
            "overall_metrics": overall,
            "stored_metric_comparison": comparison,
        }
        _print_summary(split_name, overall, group_rows)

    pd.DataFrame(all_group_rows).to_csv(output_dir / "los_group_metrics.csv", index=False)
    pd.DataFrame(all_class_rows).to_csv(output_dir / "los_group_class_metrics.csv", index=False)
    pd.DataFrame(all_confusion_rows).to_csv(output_dir / "los_group_confusion.csv", index=False)

    overall_payload = {
        "run_name": args.run_name,
        "fold": int(args.fold),
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "device": str(device),
        "batch_policy": {
            "drop_last": True,
            "note": "Last incomplete batch is excluded to match the fixed-batch edge_index used during training.",
        },
        "splits": split_payloads,
    }
    _save_json(output_dir / "overall_metrics.json", overall_payload)
    return {
        "output_dir": str(output_dir),
        "overall_metrics_path": str(output_dir / "overall_metrics.json"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay baseline CTMP-GIN eval from cached forecast inputs and export LOS-conditional metrics."
    )
    parser.add_argument("--run_name", required=True, help="Run directory name under runs/.")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate.")
    parser.add_argument(
        "--dataset_root",
        default=str(_PROJECT_ROOT / "src" / "data"),
        help="Dataset root containing raw/TEDS_Discharge.csv.",
    )
    parser.add_argument("--device", default=None, help="Torch device string, e.g. cpu, cuda, mps.")
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional custom output directory. Defaults to runs/<run>/folds/fold_<k>/diagnostics/forecasted_gnn_performance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = _run(args)
    print(f"Saved outputs to {result['output_dir']}")


if __name__ == "__main__":
    main()
