from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.discharge_predictor import resolve_risk_head_selection  # noqa: E402
from src.models.discharge_predictor.joint_consistent_predictor import (  # noqa: E402
    JointConsistentPredictor,
)
from src.models.discharge_predictor.joint_consistency_loss import (  # noqa: E402
    JointConsistencyLoss,
)
from src.models.discharge_predictor.los_utils import (  # noqa: E402
    infer_los_coarse_breakdown_from_cfg,
    infer_los_target_from_cfg,
)
from src.models.forecast_inputs import resolve_model_forecast_input_metadata  # noqa: E402
from src.trainers.base import evaluate  # noqa: E402
from src.trainers.forecasted_pipeline import (  # noqa: E402
    _assign_joint_cache_split,
    _init_caches,
    _init_joint_soft_discharge_cache,
    _slice_soft_discharge_cache,
)
from src.trainers.run_joint_consistent_predictor import (  # noqa: E402
    JointPredictionDataset,
    _evaluate,
    _make_loader,
    _pin_memory_for_device,
)
from src.trainers.run_kfold_cv import _build_dataset  # noqa: E402
from src.utils.device_set import device_set  # noqa: E402


SPLIT_TO_CACHE_FILE = {"valid": "gnn_val_joint.pt", "test": "outer_test_joint.pt"}
SPLIT_TO_JOINT_SPLIT_NAME = {"valid": "gnn_val", "test": "outer_test"}
SPLIT_TO_JOINT_CACHE_FILES = {
    "valid": ("gnn_val.pt", "val.pt"),
    "test": ("outer_test.pt", "test.pt"),
}


def _nan() -> float:
    return float("nan")


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _safe_device(requested: torch.device) -> torch.device:
    if requested.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    if requested.type == "mps" and not torch.mps.is_available():
        return torch.device("cpu")
    return requested


def _parse_registry_run_name(registry_path: Path, experiment_id: int) -> str | None:
    if not registry_path.exists():
        return None
    pattern = re.compile(
        rf"^\|\s*{experiment_id}\s*\|\s*[^|]+\|\s*`([^`]+)`\s*\|",
        re.MULTILINE,
    )
    text = registry_path.read_text(encoding="utf-8")
    match = pattern.search(text)
    return match.group(1) if match is not None else None


def _load_run_map_csv(path: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            raw_id = row.get("run_id") or row.get("id")
            if raw_id is None:
                continue
            try:
                run_id = int(str(raw_id).strip())
            except ValueError:
                continue
            run_dir = str(row.get("run_dir") or row.get("downstream_run_dir") or "").strip()
            if run_dir:
                rows[run_id] = run_dir
    return rows


def _find_run_dir(runs_root: Path, run_id: int, registry_path: Path, run_map: dict[int, str]) -> Path | None:
    mapped = run_map.get(run_id)
    if mapped:
        path = Path(mapped)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path if path.exists() else None

    downstream_patterns = [
        f"*ctmp_gin_joint_fresh_id{run_id}*",
        f"*ctmp_gin_joint_fresh_id{run_id}_breakdown*",
    ]
    downstream_candidates: list[Path] = []
    for pattern in downstream_patterns:
        downstream_candidates.extend(
            sorted(
                path
                for path in runs_root.glob(pattern)
                if path.is_dir() and (path / "folds").exists()
            )
        )
    if downstream_candidates:
        return downstream_candidates[0]

    registry_name = _parse_registry_run_name(registry_path, run_id)
    candidates: list[Path] = []
    if registry_name:
        direct = runs_root / registry_name
        if direct.exists():
            candidates.append(direct)
    candidates.extend(
        sorted(
            path
            for path in runs_root.glob(f"*id{run_id}*")
            if path.is_dir() and (path / "folds").exists()
        )
    )
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve())
        if key in seen:
            continue
        seen.add(key)
        return candidate
    return None


def _read_metrics_jsonl(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if not path.exists():
        return metrics
    best_valid_auc = -float("inf")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            valid_auc = row.get("valid_auc")
            try:
                valid_auc_f = float(valid_auc)
            except Exception:
                valid_auc_f = None
            if valid_auc_f is not None and valid_auc_f > best_valid_auc:
                best_valid_auc = valid_auc_f
                metrics["baseline_valid_auc"] = valid_auc_f
            for key in ("test_auc", "test_acc", "test_f1", "test_precision", "test_recall", "test_loss"):
                if key in row:
                    try:
                        metrics[f"baseline_{key}"] = float(row[key])
                    except Exception:
                        pass
    return metrics


def _load_downstream_model_and_edge(fold_dir: Path, cfg: dict[str, Any], device: torch.device) -> tuple[torch.nn.Module, torch.Tensor]:
    from src.models.factory import build_model

    checkpoint = torch.load(fold_dir / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    model_cfg = checkpoint.get("cfg", cfg)
    model_cfg["model"]["params"]["device"] = str(device)
    model = build_model(
        model_name=model_cfg["model"]["name"],
        **model_cfg["model"].get("params", {}),
    ).to(device)
    state_dict = dict(checkpoint["model_state_dict"])
    state_dict.pop("_cached_edge_index_2", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    edge_index = torch.load(fold_dir / "edge_index.pt", map_location=device, weights_only=False)
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, int(model_cfg["train"]["batch_size"]))
    return model, edge_index


class _SplitForecastCacheDataset(Dataset):
    def __init__(
        self,
        base_dataset,
        x_cache: torch.Tensor,
        los_cache: torch.Tensor,
        indices: Any,
        soft_discharge_cache: dict[str, Any] | None,
    ) -> None:
        self.base_dataset = base_dataset
        self.x_cache = x_cache
        self.los_cache = los_cache
        self.indices = torch.as_tensor(indices, dtype=torch.long)
        self.soft_discharge_cache = soft_discharge_cache

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    def __getitem__(self, index: int):
        dataset_index = int(self.indices[index].item())
        _, y, _ = self.base_dataset[dataset_index]
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
            if "logits" in head_payload:
                soft_discharge[head_name]["logits"] = head_payload["logits"][index]
            soft_discharge_mask[head_name] = head_payload["mask"][index]
        forecast_meta = {
            "soft_discharge": soft_discharge,
            "soft_discharge_mask": soft_discharge_mask,
            "metadata": dict(self.soft_discharge_cache.get("metadata", {})),
        }
        return self.x_cache[index], y, self.los_cache[index], forecast_meta


def _inflate_soft_discharge_cache(slice_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if slice_payload is None:
        return None
    inflated: dict[str, Any] = {
        "metadata": dict(slice_payload.get("metadata", {})),
        "head_names": list(slice_payload.get("head_names", [])),
        "soft_head_names": list(slice_payload.get("soft_head_names", [])),
        "heads": {},
    }
    for head_name, head_payload in slice_payload.get("heads", {}).items():
        inflated["heads"][head_name] = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in head_payload.items()
        }
    return inflated


def _evaluate_cached_split(
    fold_dir: Path,
    cfg: dict[str, Any],
    split_payload: dict[str, Any],
    device: torch.device,
    *,
    model: torch.nn.Module | None = None,
    edge_index: torch.Tensor | None = None,
    base_dataset: Any | None = None,
) -> dict[str, float]:
    if model is None or edge_index is None:
        model, edge_index = _load_downstream_model_and_edge(fold_dir, cfg, device)
    if base_dataset is None:
        base_dataset = _build_dataset(cfg, str(PROJECT_ROOT / "src" / "data"))
    criterion = (
        torch.nn.BCEWithLogitsLoss()
        if bool(cfg["train"].get("binary", True))
        else torch.nn.CrossEntropyLoss()
    )
    dataloader = DataLoader(
        _SplitForecastCacheDataset(
            base_dataset=base_dataset,
            x_cache=split_payload["x"],
            los_cache=split_payload["los"],
            indices=split_payload["indices"],
            soft_discharge_cache=_inflate_soft_discharge_cache(split_payload.get("soft_discharge")),
        ),
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 0)),
        drop_last=True,
    )
    loss, acc, precision, recall, f1, auc = evaluate(
        model=model,
        val_dataloader=dataloader,
        criterion=criterion,
        decision_threshold=float(cfg["train"].get("decision_threshold", 0.5)),
        device=device,
        binary=bool(cfg["train"].get("binary", True)),
        edge_index=edge_index,
    )
    return {
        "loss": float(loss),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
    }


def _resolve_split_indices(split_payload: dict[str, Any], split: str) -> list[int]:
    key = "gnn_val_idx" if split == "valid" else "outer_test_idx"
    return [int(idx) for idx in split_payload[key]]


def _joint_payload_to_cache_payload(
    payload: dict[str, Any],
    dataset: JointPredictionDataset,
    joint_cfg: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    final_d_logits = {
        head_name: torch.tensor(values, dtype=torch.float32)
        for head_name, values in payload["d_logits_np"].items()
    }
    final_d_probs = {head_name: torch.softmax(logits, dim=1) for head_name, logits in final_d_logits.items()}
    return {
        "split": split_name,
        "final_d_logits": final_d_logits,
        "final_d_probs": final_d_probs,
        "final_d_pred": {
            head_name: torch.argmax(probs, dim=1).long()
            for head_name, probs in final_d_probs.items()
        },
        "final_los_logits": payload["los_logits"].to(dtype=torch.float32),
        "final_los_probs": payload["los_probs"].to(dtype=torch.float32),
        "final_los_pred": (
            torch.argmax(payload["los_probs"], dim=1).long()
            if dataset.los_target_mode == "coarse"
            else (torch.argmax(payload["los_probs"], dim=1) + 1).long()
        ),
        "row_idx": payload["row_idx"].long(),
        "targets": {
            "d": {
                head_name: torch.tensor(values, dtype=torch.long)
                for head_name, values in payload["d_targets_np"].items()
            },
            "los_target": payload["los_targets"].long(),
            "los_raw": payload["los_raw"].long(),
        },
        "metadata": {
            **dataset.schema_metadata,
            "joint_direction": joint_cfg["joint_predictor"]["joint_direction"],
            "condition_mode": joint_cfg["joint_predictor"]["condition_mode"],
            "detach_condition": bool(joint_cfg["joint_predictor"]["detach_condition"]),
        },
    }


def _reconstruct_joint_cache_from_checkpoint(fold_dir: Path, split: str, device: torch.device) -> dict[str, Any] | None:
    joint_run_dir = fold_dir / "joint_predictor"
    config_path = joint_run_dir / "config.final.yaml"
    checkpoint_path = joint_run_dir / "checkpoints" / "best.pt"
    split_path = fold_dir / "joint_forecast_pipeline_splits.json"
    if not config_path.exists() or not checkpoint_path.exists() or not split_path.exists():
        return None
    joint_cfg = _load_yaml(config_path)
    predictor_type = str(joint_cfg.get("joint_predictor", {}).get("predictor_type", "joint_consistent")).lower()
    if predictor_type != "joint_consistent":
        return None

    dataset = JointPredictionDataset(
        root=str(PROJECT_ROOT / "src" / "data"),
        do_preprocess=bool(joint_cfg["train"].get("do_preprocess", False)),
        los_target_mode=infer_los_target_from_cfg(joint_cfg["joint_predictor"]),
        los_coarse_breakdown=infer_los_coarse_breakdown_from_cfg(joint_cfg["joint_predictor"]),
    )
    model = JointConsistentPredictor(
        ad_col_dims=dataset.ad_col_dims,
        target_col_names=dataset.target_col_names,
        target_col_dims=dataset.target_col_dims,
        los_num_classes=dataset.los_num_classes,
        joint_direction=joint_cfg["joint_predictor"]["joint_direction"],
        condition_mode=joint_cfg["joint_predictor"]["condition_mode"],
        detach_condition=bool(joint_cfg["joint_predictor"]["detach_condition"]),
        joint_heads=joint_cfg["joint_predictor"]["joint_heads"],
        **joint_cfg["model"]["params"],
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    criterion = JointConsistencyLoss(
        lambda_los=float(joint_cfg["joint_predictor"]["lambda_los"]),
        lambda_aux=float(joint_cfg["joint_predictor"]["lambda_aux"]),
        lambda_joint=float(joint_cfg["joint_predictor"]["lambda_joint"]),
        joint_head_names=model.selected_joint_heads,
    )
    split_indices = _resolve_split_indices(_load_json(split_path), split)
    loader = _make_loader(
        dataset,
        torch.tensor(split_indices, dtype=torch.long).numpy(),
        int(joint_cfg["train"]["batch_size"]),
        int(joint_cfg["train"].get("num_workers", 0)),
        False,
        _pin_memory_for_device(device),
    )
    _metrics, payload = _evaluate(model, loader, criterion, device, dataset)
    return _joint_payload_to_cache_payload(payload, dataset, joint_cfg, "gnn_val" if split == "valid" else "outer_test")


def _load_joint_cache_payload(fold_dir: Path, split: str, device: torch.device) -> tuple[dict[str, Any] | None, str | None]:
    for file_name in SPLIT_TO_JOINT_CACHE_FILES[split]:
        direct_path = fold_dir / "joint_predictor" / "joint_cache" / file_name
        if direct_path.exists():
            return torch.load(direct_path, map_location="cpu", weights_only=False), None
    reconstructed = _reconstruct_joint_cache_from_checkpoint(fold_dir, split, device)
    if reconstructed is not None:
        return reconstructed, "reconstructed_from_checkpoint"
    return None, "missing_cache"


def _load_cached_predictions_payload(fold_dir: Path, split: str) -> dict[str, Any] | None:
    path = fold_dir / "cached_predictions" / SPLIT_TO_CACHE_FILE[split]
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def _rebuild_cached_predictions_payload_from_joint_cache(
    *,
    fold_dir: Path,
    cfg: dict[str, Any],
    base_dataset,
    split: str,
    joint_cache_payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    split_path = fold_dir / "joint_forecast_pipeline_splits.json"
    if not split_path.exists():
        return None, "missing joint_forecast_pipeline_splits.json"

    split_payload = _load_json(split_path)
    try:
        indices = np.asarray(_resolve_split_indices(split_payload, split), dtype=np.int64)
    except KeyError as exc:
        return None, f"missing split indices in joint_forecast_pipeline_splits.json: {exc}"

    input_cfg = cfg.get("joint_forecast_pipeline", {}).get("joint_forecast_input", {})
    joint_mode = str(split_payload.get("joint_mode", input_cfg.get("mode", "distribution"))).lower()
    if joint_mode not in {"distribution", "hard"}:
        return None, f"unsupported joint_forecast_input.mode={joint_mode!r}"

    input_metadata = split_payload.get("forecast_input_metadata")
    if not isinstance(input_metadata, dict) or not input_metadata:
        input_metadata = resolve_model_forecast_input_metadata(cfg)

    x_cache, los_cache, _ = _init_caches(
        base_dataset,
        "distribution" if joint_mode == "distribution" else "hard",
        discharge_provider=None,
        input_metadata=input_metadata,
    )
    soft_discharge_cache = None
    if joint_mode == "distribution":
        soft_discharge_cache = _init_joint_soft_discharge_cache(
            base_dataset,
            len(base_dataset),
            joint_cache_payload,
            input_metadata,
        )

    split_name = SPLIT_TO_JOINT_SPLIT_NAME[split]
    try:
        _assign_joint_cache_split(
            base_dataset=base_dataset,
            split_name=split_name,
            cache_payload=joint_cache_payload,
            expected_indices=indices,
            x_cache=x_cache,
            los_cache=los_cache,
            soft_discharge_cache=soft_discharge_cache,
            joint_mode=joint_mode,
        )
    except Exception as exc:
        return None, f"failed to rebuild cached_predictions/{SPLIT_TO_CACHE_FILE[split]} from joint cache: {exc}"

    payload = {
        "x": x_cache[indices].clone(),
        "los": los_cache[indices].clone(),
        "indices": indices,
        "soft_discharge": _slice_soft_discharge_cache(soft_discharge_cache, indices),
    }
    return (
        payload,
        f"rebuilt cached_predictions/{SPLIT_TO_CACHE_FILE[split]} from joint predictor cache payload",
    )


def _oracle_values_by_row(
    base_dataset,
    indices: list[int],
    joint_cache_payload: dict[str, Any],
    head_names: list[str],
) -> dict[str, torch.Tensor]:
    raw_row_index = base_dataset.raw_row_index.to_numpy(dtype="int64", copy=True)
    cache_rows = joint_cache_payload["row_idx"].detach().cpu().numpy().astype("int64")
    row_to_pos = {int(row): pos for pos, row in enumerate(cache_rows.tolist())}
    oracle: dict[str, list[int]] = {head_name: [] for head_name in head_names}
    for dataset_idx in indices:
        raw_row = int(raw_row_index[int(dataset_idx)])
        if raw_row not in row_to_pos:
            raise KeyError(f"row_idx={raw_row} not found in joint cache payload")
        pos = row_to_pos[raw_row]
        for head_name in head_names:
            oracle[head_name].append(int(joint_cache_payload["targets"]["d"][head_name][pos].item()))
    return {
        head_name: torch.tensor(values, dtype=torch.long)
        for head_name, values in oracle.items()
    }


def apply_oracle_head_override(
    *,
    base_dataset,
    cached_predictions_payload: dict[str, Any],
    joint_cache_payload: dict[str, Any],
    override_heads: list[str],
) -> dict[str, Any]:
    x = cached_predictions_payload["x"].clone()
    los = cached_predictions_payload["los"].clone()
    indices = torch.as_tensor(cached_predictions_payload["indices"], dtype=torch.long)
    soft = _inflate_soft_discharge_cache(cached_predictions_payload.get("soft_discharge"))
    col_list, col_dims, _ad_idx, dis_idx = base_dataset.col_info
    discharge_cols = {str(col_list[idx]): int(idx) for idx in dis_idx}
    oracle_by_head = _oracle_values_by_row(
        base_dataset,
        [int(idx) for idx in indices.tolist()],
        joint_cache_payload,
        override_heads,
    )

    for head_name in override_heads:
        if head_name not in discharge_cols:
            raise KeyError(f"Unknown discharge head in base dataset: {head_name}")
        col_idx = discharge_cols[head_name]
        x[:, col_idx] = oracle_by_head[head_name].to(dtype=x.dtype)
        if soft is None or head_name not in soft.get("heads", {}):
            continue
        head_payload = soft["heads"][head_name]
        num_classes = int(head_payload["num_classes"].item() if torch.is_tensor(head_payload["num_classes"]) else head_payload["num_classes"])
        hard = oracle_by_head[head_name].clone()
        probs = F.one_hot(hard, num_classes=num_classes).to(dtype=torch.float32)
        head_payload["hard"] = hard
        head_payload["probs"] = probs
        head_payload["logits"] = probs.clamp_min(1.0e-12).log()
        head_payload["mask"] = torch.ones_like(hard, dtype=torch.bool)

    return {
        "x": x,
        "los": los,
        "indices": indices,
        "soft_discharge": soft,
    }


@dataclass
class OverrideResult:
    row: dict[str, Any]
    manifest_entry: dict[str, Any]


_FOLD_EVAL_CONTEXT: dict[tuple[str, str, str], dict[str, Any]] = {}


def _fold_eval_context(
    *,
    fold_dir: Path,
    split: str,
    device: torch.device,
) -> dict[str, Any]:
    key = (str(fold_dir.resolve()), str(split), str(device))
    if key in _FOLD_EVAL_CONTEXT:
        return _FOLD_EVAL_CONTEXT[key]

    cfg = _load_yaml(fold_dir / "config.final.yaml")
    base_dataset = _build_dataset(cfg, str(PROJECT_ROOT / "src" / "data"))
    baseline_metrics = _read_metrics_jsonl(fold_dir / "metrics.jsonl")
    cached_predictions_payload = _load_cached_predictions_payload(fold_dir, split)
    status = "ok"
    warnings: list[str] = []
    joint_cache_payload, joint_cache_warning = _load_joint_cache_payload(fold_dir, split, device)
    if cached_predictions_payload is None:
        warnings.append(f"missing cached_predictions/{SPLIT_TO_CACHE_FILE[split]}")

    if joint_cache_warning == "missing_cache" or joint_cache_payload is None:
        status = "missing_cache"
        warnings.append(f"missing joint cache for split={split}")
    else:
        if joint_cache_warning:
            warnings.append(joint_cache_warning)
        if cached_predictions_payload is None:
            cached_predictions_payload, rebuild_warning = _rebuild_cached_predictions_payload_from_joint_cache(
                fold_dir=fold_dir,
                cfg=cfg,
                base_dataset=base_dataset,
                split=split,
                joint_cache_payload=joint_cache_payload,
            )
            warnings.append(rebuild_warning)
            if cached_predictions_payload is None:
                status = "missing_cache"

    model = None
    edge_index = None
    baseline_eval: dict[str, float] = {}
    if status == "ok":
        model, edge_index = _load_downstream_model_and_edge(fold_dir, cfg, device)
        baseline_eval = _evaluate_cached_split(
            fold_dir,
            cfg,
            cached_predictions_payload,
            device,
            model=model,
            edge_index=edge_index,
            base_dataset=base_dataset,
        )

    context = {
        "cfg": cfg,
        "base_dataset": base_dataset,
        "available_heads": [str(base_dataset.col_info[0][idx]) for idx in base_dataset.col_info[3]],
        "baseline_metrics": baseline_metrics,
        "cached_predictions_payload": cached_predictions_payload,
        "joint_cache_payload": joint_cache_payload,
        "status": status,
        "warnings": warnings,
        "model": model,
        "edge_index": edge_index,
        "baseline_eval": baseline_eval,
    }
    _FOLD_EVAL_CONTEXT[key] = context
    return context


def run_override_for_fold(
    *,
    run_id: int,
    run_dir: Path,
    fold_dir: Path,
    head_set_name: str,
    split: str,
    device: torch.device,
) -> OverrideResult:
    context = _fold_eval_context(fold_dir=fold_dir, split=split, device=device)
    cfg = context["cfg"]
    base_dataset = context["base_dataset"]
    available_heads = context["available_heads"]
    override_heads = resolve_risk_head_selection(
        head_set_name,
        available_heads=available_heads,
        mode="strict_named_set",
        field_name="head_set_name",
    )

    baseline_metrics = context["baseline_metrics"]
    cached_predictions_payload = context["cached_predictions_payload"]
    joint_cache_payload = context["joint_cache_payload"]
    status = str(context["status"])
    warnings: list[str] = list(context["warnings"])
    split_metrics: dict[str, float] = {}
    if status == "ok":
        baseline_eval = context["baseline_eval"]
        override_payload = apply_oracle_head_override(
            base_dataset=base_dataset,
            cached_predictions_payload=cached_predictions_payload,
            joint_cache_payload=joint_cache_payload,
            override_heads=override_heads,
        )
        override_eval = _evaluate_cached_split(
            fold_dir,
            cfg,
            override_payload,
            device,
            model=context["model"],
            edge_index=context["edge_index"],
            base_dataset=base_dataset,
        )
        split_metrics = {
            "baseline_auc": baseline_eval["auc"],
            "override_auc": override_eval["auc"],
            "override_acc": override_eval["acc"],
            "override_precision": override_eval["precision"],
            "override_recall": override_eval["recall"],
            "override_f1": override_eval["f1"],
            "override_loss": override_eval["loss"],
        }

    fold_num = int(fold_dir.name.replace("fold_", ""))
    row = {
        "run_id": int(run_id),
        "run_name": run_dir.name,
        "fold": fold_num,
        "baseline_valid_auc": _nan(),
        "baseline_test_auc": float(baseline_metrics.get("baseline_test_auc", _nan())),
        "head_set_name": head_set_name,
        "override_heads": ",".join(override_heads),
        "override_valid_auc": _nan(),
        "override_test_auc": _nan(),
        "delta_valid_auc": _nan(),
        "delta_test_auc": _nan(),
        "valid_acc": _nan(),
        "valid_f1": _nan(),
        "valid_precision": _nan(),
        "valid_recall": _nan(),
        "status": status,
        "warnings": "; ".join(warnings),
    }
    if split == "valid":
        row["baseline_valid_auc"] = float(
            split_metrics.get("baseline_auc", baseline_metrics.get("baseline_valid_auc", _nan()))
        )
        row["override_valid_auc"] = float(split_metrics.get("override_auc", _nan()))
        row["delta_valid_auc"] = float(
            row["override_valid_auc"] - row["baseline_valid_auc"]
        ) if not math.isnan(row["override_valid_auc"]) and not math.isnan(row["baseline_valid_auc"]) else _nan()
        row["valid_acc"] = float(split_metrics.get("override_acc", _nan()))
        row["valid_f1"] = float(split_metrics.get("override_f1", _nan()))
        row["valid_precision"] = float(split_metrics.get("override_precision", _nan()))
        row["valid_recall"] = float(split_metrics.get("override_recall", _nan()))
    else:
        row["baseline_test_auc"] = float(
            split_metrics.get("baseline_auc", baseline_metrics.get("baseline_test_auc", _nan()))
        )
        row["override_test_auc"] = float(split_metrics.get("override_auc", _nan()))
        row["delta_test_auc"] = float(
            row["override_test_auc"] - row["baseline_test_auc"]
        ) if not math.isnan(row["override_test_auc"]) and not math.isnan(row["baseline_test_auc"]) else _nan()
    manifest_entry = {
        "run_id": int(run_id),
        "run_name": run_dir.name,
        "fold": fold_num,
        "head_set_name": head_set_name,
        "override_heads": override_heads,
        "split": split,
        "status": status,
        "warnings": warnings,
    }
    return OverrideResult(row=row, manifest_entry=manifest_entry)


def _write_summary_md(path: Path, rows: list[dict[str, Any]], split: str) -> None:
    lines = [
        "# Risk Head Override Summary",
        "",
        f"Split: `{split}`",
        "",
        "Interpretation rules:",
        "- If `new_dvD_top3 > old_total_drift_top3`, dV_D ranking is downstream-relevant.",
        "- If `new_dvD_top3 ~= old_total_drift_top3`, the gain is probably dominated by `FREQ_ATND_SELF_HELP_D` and `SUB1_D`.",
        "- If `new_dvD_top6` improves further, D-side mismatch is distributed across multiple heads.",
        "- If `new_dvD_top6` still remains far below the single-anchor 0.93 AUC region, D-side override alone is insufficient.",
        "",
        "| run_id | run_name | fold | head_set_name | baseline_valid_auc | baseline_test_auc | override_valid_auc | override_test_auc | delta_valid_auc | delta_test_auc | status | warnings |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {run_id} | {run_name} | {fold} | {head_set_name} | {baseline_valid_auc} | {baseline_test_auc} | {override_valid_auc} | {override_test_auc} | {delta_valid_auc} | {delta_test_auc} | {status} | {warnings} |".format(
                **{k: row.get(k, "") for k in row}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run eval-only oracle override checks for selected discharge risk-head sets.")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--registry-path", required=True)
    parser.add_argument("--run-ids", nargs="+", required=True, type=int)
    parser.add_argument("--run-map-csv", default=None)
    parser.add_argument("--head-sets", nargs="+", required=True)
    parser.add_argument("--override-mode", choices=["oracle"], default="oracle")
    parser.add_argument("--split", choices=["valid", "test"], required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root)
    registry_path = Path(args.registry_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _safe_device(device_set(None))
    run_map = _load_run_map_csv(Path(args.run_map_csv)) if args.run_map_csv else {}

    rows: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {
        "runs_root": str(runs_root),
        "registry_path": str(registry_path),
        "run_ids": list(args.run_ids),
        "head_sets": list(args.head_sets),
        "override_mode": args.override_mode,
        "split": args.split,
        "results": [],
    }
    for run_id in args.run_ids:
        run_dir = _find_run_dir(runs_root, int(run_id), registry_path, run_map)
        if run_dir is None:
            if args.strict:
                raise FileNotFoundError(f"Could not resolve downstream run dir for id{run_id}")
            for head_set_name in args.head_sets:
                rows.append(
                    {
                        "run_id": int(run_id),
                        "run_name": "",
                        "fold": 0,
                        "baseline_valid_auc": _nan(),
                        "baseline_test_auc": _nan(),
                        "head_set_name": head_set_name,
                        "override_heads": "",
                        "override_valid_auc": _nan(),
                        "override_test_auc": _nan(),
                        "delta_valid_auc": _nan(),
                        "delta_test_auc": _nan(),
                        "valid_acc": _nan(),
                        "valid_f1": _nan(),
                        "valid_precision": _nan(),
                        "valid_recall": _nan(),
                        "status": "invalid_run",
                        "warnings": f"could not resolve downstream run dir for id{run_id}",
                    }
                )
            continue

        fold_root = run_dir / "folds"
        if not fold_root.exists() or not fold_root.is_dir():
            if args.strict:
                raise FileNotFoundError(f"Downstream run dir has no folds/: {run_dir}")
            for head_set_name in args.head_sets:
                rows.append(
                    {
                        "run_id": int(run_id),
                        "run_name": run_dir.name,
                        "fold": 0,
                        "baseline_valid_auc": _nan(),
                        "baseline_test_auc": _nan(),
                        "head_set_name": head_set_name,
                        "override_heads": "",
                        "override_valid_auc": _nan(),
                        "override_test_auc": _nan(),
                        "delta_valid_auc": _nan(),
                        "delta_test_auc": _nan(),
                        "valid_acc": _nan(),
                        "valid_f1": _nan(),
                        "valid_precision": _nan(),
                        "valid_recall": _nan(),
                        "status": "invalid_run",
                        "warnings": f"resolved run dir has no folds/: {run_dir}",
                    }
                )
            continue
        fold_dirs = sorted(path for path in fold_root.iterdir() if path.is_dir() and path.name.startswith("fold_"))
        for fold_dir in fold_dirs:
            if not (fold_dir / "checkpoints" / "best.pt").exists():
                continue
            for head_set_name in args.head_sets:
                result = run_override_for_fold(
                    run_id=int(run_id),
                    run_dir=run_dir,
                    fold_dir=fold_dir,
                    head_set_name=head_set_name,
                    split=args.split,
                    device=device,
                )
                rows.append(result.row)
                manifest["results"].append(result.manifest_entry)

    csv_path = out_dir / "override_summary.csv"
    md_path = out_dir / "override_summary.md"
    manifest_path = out_dir / "manifest.json"
    fieldnames = [
        "run_id",
        "run_name",
        "fold",
        "baseline_valid_auc",
        "baseline_test_auc",
        "head_set_name",
        "override_heads",
        "override_valid_auc",
        "override_test_auc",
        "delta_valid_auc",
        "delta_test_auc",
        "valid_acc",
        "valid_f1",
        "valid_precision",
        "valid_recall",
        "status",
        "warnings",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    _write_summary_md(md_path, rows, args.split)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(csv_path)
    print(md_path)
    print(manifest_path)


if __name__ == "__main__":
    main()
