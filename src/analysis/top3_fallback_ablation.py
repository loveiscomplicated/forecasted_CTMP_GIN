from __future__ import annotations

import argparse
import copy
import csv
import json
import numpy as np
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml

from src.models.discharge_predictor.risk_heads import LEGACY_TOP3_HEADS

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))


REGISTRY_PATH = _PROJECT_ROOT / "forecasted_ctmp_gin_joint_predictor_experiments_registry_filled.md"
RUNS_DIR = _PROJECT_ROOT / "runs"
DIAGNOSTICS_DIR = RUNS_DIR / "diagnostics" / "fallback_ablation"
FALLBACK_RESULT_FILENAME = "fallback_ablation_eval.json"
FOCUSED_HEADS_PATH = (
    RUNS_DIR / "diagnostics" / "forecast_cache_alignment" / "distribution_diagnosis" / "focused_heads_summary.csv"
)
PER_HEAD_CONDITIONAL_PATH = (
    RUNS_DIR
    / "diagnostics"
    / "forecast_cache_alignment"
    / "distribution_diagnosis"
    / "per_head_conditional_distribution.csv"
)
TOP3_FALLBACK_HEADS = tuple((head_name, head_name[:-2]) for head_name in LEGACY_TOP3_HEADS)


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


def _safe_device(requested: torch.device) -> torch.device:
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA device is unavailable; falling back to cpu.")
        return torch.device("cpu")
    if requested.type == "mps" and not torch.mps.is_available():
        print("Requested MPS device is unavailable; falling back to cpu.")
        return torch.device("cpu")
    return requested


def _extract_experiment_id(name: str) -> int:
    match = re.search(r"id(\d+)(?:\D|$)", name)
    if match is None:
        raise ValueError(f"Could not extract experiment id from: {name}")
    return int(match.group(1))


def _parse_registry_run_id(registry_path: Path, experiment_id: int) -> str:
    pattern = re.compile(
        rf"^\|\s*{experiment_id}\s*\|\s*[^|]+\|\s*`([^`]+)`\s*\|",
        re.MULTILINE,
    )
    text = registry_path.read_text(encoding="utf-8")
    match = pattern.search(text)
    if match is None:
        raise ValueError(f"Experiment ID {experiment_id} not found in {registry_path}")
    return match.group(1)


def _resolve_original_joint_cache_paths(experiment_id: int) -> dict[str, Path]:
    run_id = _parse_registry_run_id(REGISTRY_PATH, experiment_id)
    joint_run_dir = RUNS_DIR / run_id
    manifest_path = joint_run_dir / "joint_cache" / "cache_manifest.json"
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        resolved = {
            "train": _PROJECT_ROOT / manifest["train"],
            "gnn_val": _PROJECT_ROOT / manifest.get("val", manifest.get("gnn_val")),
            "outer_test": _PROJECT_ROOT / manifest.get("test", manifest.get("outer_test")),
        }
    else:
        resolved = {
            "train": joint_run_dir / "joint_cache" / "train.pt",
            "gnn_val": joint_run_dir / "joint_cache" / "val.pt",
            "outer_test": joint_run_dir / "joint_cache" / "test.pt",
        }
    missing = [str(path) for path in resolved.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing original joint cache files for id{experiment_id}: {missing}"
        )
    return resolved


def _load_stored_test_metrics(metrics_path: Path) -> dict[str, float] | None:
    if not metrics_path.exists():
        return None
    last_test = None
    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            if record.get("split") == "test":
                last_test = record
    if last_test is None:
        return None
    return {
        "test_loss": float(last_test["test_loss"]),
        "test_acc": float(last_test["test_acc"]),
        "test_precision": float(last_test["test_precision"]),
        "test_recall": float(last_test["test_recall"]),
        "test_f1": float(last_test["test_f1"]),
        "test_auc": float(last_test["test_auc"]),
    }


def _default_admission_head_name(discharge_head: str) -> str:
    if not discharge_head.endswith("_D"):
        raise ValueError(f"Unsupported discharge head name: {discharge_head}")
    return discharge_head[:-2]


def _head_pairs_from_discharge_heads(discharge_heads: list[str]) -> list[tuple[str, str]]:
    return [(head, _default_admission_head_name(head)) for head in discharge_heads]


def _load_ranked_drift_heads() -> list[str]:
    ranked: list[str] = []
    if FOCUSED_HEADS_PATH.exists():
        with FOCUSED_HEADS_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            ranked.extend(str(row["target_name"]) for row in reader if row.get("target_name"))
    if len(ranked) >= 5:
        return ranked

    supplemental_scores: dict[str, tuple[float, float]] = {}
    if PER_HEAD_CONDITIONAL_PATH.exists():
        with PER_HEAD_CONDITIONAL_PATH.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                head_name = str(row["target_name"])
                js_score = float(row["js_divergence_P_D_given_LOS"])
                delta_score = float(row["delta_cramers_v"])
                existing = supplemental_scores.get(head_name)
                if existing is None or (js_score, delta_score) > existing:
                    supplemental_scores[head_name] = (js_score, delta_score)
    for head_name, _score in sorted(
        supplemental_scores.items(),
        key=lambda item: (item[1][0], item[1][1]),
        reverse=True,
    ):
        if head_name not in ranked:
            ranked.append(head_name)
    if len(ranked) < 5:
        raise ValueError(f"Could not resolve top5 drift heads from diagnostics files under {PER_HEAD_CONDITIONAL_PATH.parent}")
    return ranked


def _resolve_variant_head_pairs(variant_set: str) -> list[tuple[str, str]]:
    if variant_set == "top3":
        return list(TOP3_FALLBACK_HEADS)
    if variant_set == "top5":
        return _head_pairs_from_discharge_heads(_load_ranked_drift_heads()[:5])
    raise ValueError(f"Unsupported variant set: {variant_set}")


def _build_cached_fold_data(
    fold_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[Any, torch.Tensor, torch.Tensor, dict[str, Any] | None, dict[str, Any]]:
    from src.trainers.forecasted_pipeline import (
        _assign_joint_cache_split,
        _init_caches,
        _init_joint_soft_discharge_cache,
    )
    from src.trainers.run_kfold_cv import _build_dataset

    split_payload = _load_json(fold_dir / "joint_forecast_pipeline_splits.json")
    experiment_id = _extract_experiment_id(str(cfg.get("run_name") or fold_dir.parent.parent.name))

    base_dataset = _build_dataset(copy.deepcopy(cfg), str(_PROJECT_ROOT / "src" / "data"))
    input_metadata = _load_json(fold_dir / "forecast_input_metadata.json")
    joint_mode = str(split_payload["joint_mode"]).lower()

    x_cache, los_cache, _ = _init_caches(
        base_dataset,
        "distribution" if joint_mode == "distribution" else "hard",
        discharge_provider=None,
        input_metadata=input_metadata,
    )

    outer_test_idx = split_payload["outer_test_idx"]
    outer_test_payload = _load_outer_test_joint_cache_payload(
        fold_dir=fold_dir,
        experiment_id=experiment_id,
        split_payload=split_payload,
        device=device,
    )

    soft_discharge_cache = None
    if joint_mode == "distribution":
        soft_discharge_cache = _init_joint_soft_discharge_cache(
            base_dataset,
            len(base_dataset),
            outer_test_payload,
            input_metadata,
        )
    _assign_joint_cache_split(
        base_dataset=base_dataset,
        split_name="outer_test",
        cache_payload=outer_test_payload,
        expected_indices=outer_test_idx,
        x_cache=x_cache,
        los_cache=los_cache,
        soft_discharge_cache=soft_discharge_cache,
        joint_mode=joint_mode,
    )
    return base_dataset, x_cache, los_cache, soft_discharge_cache, split_payload


def _joint_payload_to_cache_payload(
    payload: dict[str, Any],
    dataset,
    joint_cfg: dict[str, Any],
    split_name: str,
) -> dict[str, Any]:
    from src.models.discharge_predictor.conditioners import resolve_joint_heads

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
    return {
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
        "caseid": None,
        "metadata": {
            **dataset.schema_metadata,
            "joint_direction": joint_cfg["joint_predictor"]["joint_direction"],
            "condition_mode": joint_cfg["joint_predictor"]["condition_mode"],
            "detach_condition": bool(joint_cfg["joint_predictor"]["detach_condition"]),
            "joint_heads": resolve_joint_heads(
                dataset.target_col_names,
                joint_cfg["joint_predictor"]["joint_heads"],
            ),
            "seed": int(joint_cfg["train"]["seed"]),
            "fold": int(joint_cfg["train"]["fold"]),
            "split": split_name,
            "default_discharge_representation": "hard_argmax",
            "default_los_representation": "distribution",
            "final_los_pred_space": (
                "coarse_class" if dataset.los_target_mode == "coarse" else "raw_los"
            ),
        },
    }


def _export_joint_cache_from_fold_predictor(
    fold_dir: Path,
    split_payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    from src.models.discharge_predictor.joint_consistency_loss import JointConsistencyLoss
    from src.models.discharge_predictor.joint_consistent_predictor import JointConsistentPredictor
    from src.models.discharge_predictor.los_utils import infer_los_target_from_cfg
    from src.trainers.run_joint_consistent_predictor import (
        JointPredictionDataset,
        _evaluate,
        _make_loader,
        _pin_memory_for_device,
    )

    joint_run_dir = fold_dir / "joint_predictor"
    joint_cfg = _load_yaml(joint_run_dir / "config.final.yaml")
    dataset = JointPredictionDataset(
        root=str(_PROJECT_ROOT / "src" / "data"),
        do_preprocess=bool(joint_cfg["train"].get("do_preprocess", False)),
        los_target_mode=infer_los_target_from_cfg(joint_cfg["joint_predictor"]),
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
    checkpoint = torch.load(joint_run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = JointConsistencyLoss(
        lambda_los=float(joint_cfg["joint_predictor"]["lambda_los"]),
        lambda_aux=float(joint_cfg["joint_predictor"]["lambda_aux"]),
        lambda_joint=float(joint_cfg["joint_predictor"]["lambda_joint"]),
        joint_head_names=model.selected_joint_heads,
    )
    batch_size = int(joint_cfg["train"]["batch_size"])
    num_workers = int(joint_cfg["train"].get("num_workers", 0))
    pin_memory = _pin_memory_for_device(device)
    indices = np.asarray(split_payload["outer_test_idx"], dtype=np.int64)
    loader = _make_loader(
        dataset,
        indices,
        batch_size,
        num_workers,
        False,
        pin_memory,
    )
    _metrics, payload = _evaluate(model, loader, criterion, device, dataset)
    return _joint_payload_to_cache_payload(
        payload=payload,
        dataset=dataset,
        joint_cfg=joint_cfg,
        split_name="outer_test",
    )


def _load_outer_test_joint_cache_payload(
    *,
    fold_dir: Path,
    experiment_id: int,
    split_payload: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    joint_run_dir = fold_dir / "joint_predictor"
    if (joint_run_dir / "checkpoints" / "best.pt").exists():
        return _export_joint_cache_from_fold_predictor(fold_dir, split_payload, device)

    cache_paths = _resolve_original_joint_cache_paths(experiment_id)
    return torch.load(cache_paths["outer_test"], map_location="cpu", weights_only=False)


def _clone_soft_discharge_cache(
    soft_discharge_cache: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if soft_discharge_cache is None:
        return None
    cloned: dict[str, Any] = {
        "head_names": list(soft_discharge_cache["head_names"]),
        "soft_head_names": list(soft_discharge_cache["soft_head_names"]),
        "heads": {},
        "metadata": json.loads(json.dumps(soft_discharge_cache["metadata"])),
    }
    for head_name, head_payload in soft_discharge_cache["heads"].items():
        cloned["heads"][head_name] = {
            key: value.clone() if torch.is_tensor(value) else value
            for key, value in head_payload.items()
        }
    return cloned


def _compute_train_mode_values(
    base_dataset,
    indices: list[int] | np.ndarray,
    head_pairs: list[tuple[str, str]],
) -> dict[str, int]:
    col_list, col_dims, _ad_idx, _dis_idx = base_dataset.col_info
    col_index = {str(name): idx for idx, name in enumerate(col_list)}
    mode_values: dict[str, int] = {}
    index_list = [int(idx) for idx in indices]
    for discharge_head, _admission_head in head_pairs:
        discharge_idx = int(col_index[discharge_head])
        discharge_dim = int(col_dims[discharge_idx])
        values = torch.tensor(
            [int(base_dataset[idx][0][discharge_idx]) for idx in index_list],
            dtype=torch.long,
        )
        counts = torch.bincount(values, minlength=discharge_dim)
        mode_values[discharge_head] = int(torch.argmax(counts).item())
    return mode_values


def _apply_fallback_strategy(
    base_dataset,
    x_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None,
    indices: list[int],
    head_pairs: list[tuple[str, str]],
    strategy: str,
    train_mode_values: dict[str, int] | None = None,
) -> None:
    col_list, col_dims, _ad_idx, _dis_idx = base_dataset.col_info
    col_index = {str(name): idx for idx, name in enumerate(col_list)}

    for discharge_head, admission_head in head_pairs:
        if discharge_head not in col_index or admission_head not in col_index:
            raise KeyError(f"Missing fallback mapping columns: {admission_head}, {discharge_head}")
        discharge_idx = int(col_index[discharge_head])
        admission_idx = int(col_index[admission_head])
        discharge_dim = int(col_dims[discharge_idx])
        admission_dim = int(col_dims[admission_idx])
        if discharge_dim != admission_dim:
            raise ValueError(
                f"Cardinality mismatch for {admission_head}->{discharge_head}: "
                f"{admission_dim} != {discharge_dim}"
            )

        if strategy == "admission":
            replacement_values = x_cache[indices, admission_idx].to(dtype=torch.long)
        elif strategy == "train_mode":
            if train_mode_values is None or discharge_head not in train_mode_values:
                raise ValueError(f"Missing train-mode value for {discharge_head}")
            replacement_values = torch.full(
                (len(indices),),
                int(train_mode_values[discharge_head]),
                dtype=torch.long,
            )
        else:
            raise ValueError(f"Unsupported fallback strategy: {strategy}")
        x_cache[indices, discharge_idx] = replacement_values.to(dtype=x_cache.dtype)

        if soft_discharge_cache is None:
            continue
        if discharge_head not in soft_discharge_cache["heads"]:
            continue

        head_payload = soft_discharge_cache["heads"][discharge_head]
        probs = torch.nn.functional.one_hot(replacement_values, num_classes=discharge_dim).to(torch.float32)
        logits = probs.clamp_min(1.0e-12).log()
        head_payload["hard"][indices] = replacement_values
        head_payload["probs"][indices] = probs
        head_payload["logits"][indices] = logits
        head_payload["mask"][indices] = True


def _apply_top3_admission_fallback(
    base_dataset,
    x_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None,
    indices: list[int],
) -> None:
    _apply_fallback_strategy(
        base_dataset,
        x_cache,
        soft_discharge_cache,
        indices,
        list(TOP3_FALLBACK_HEADS),
        strategy="admission",
    )


def _build_test_loader(
    base_dataset,
    x_cache: torch.Tensor,
    los_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None,
    test_idx: list[int],
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    from src.trainers.forecasted_pipeline import ForecastCacheDataset, _make_loader

    cached_dataset = ForecastCacheDataset(
        base_dataset,
        x_cache,
        los_cache,
        soft_discharge_cache,
    )
    return _make_loader(
        cached_dataset,
        test_idx,
        batch_size,
        num_workers,
        shuffle=False,
    )


def _load_model_and_edge(
    fold_dir: Path,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor]:
    from src.models.factory import build_model

    checkpoint = torch.load(fold_dir / "checkpoints" / "best.pt", map_location=device)
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

    edge_index = torch.load(fold_dir / "edge_index.pt", map_location=device)
    if hasattr(model, "precompute_edge_index_2"):
        model.precompute_edge_index_2(edge_index, int(model_cfg["train"]["batch_size"]))
    return model, edge_index


def _evaluate_fold(
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    dataloader: DataLoader,
    cfg: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    from src.trainers.base import evaluate

    criterion: nn.Module
    if bool(cfg["train"].get("binary", True)):
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
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
        "test_loss": float(loss),
        "test_acc": float(acc),
        "test_precision": float(precision),
        "test_recall": float(recall),
        "test_f1": float(f1),
        "test_auc": float(auc),
    }


def _evaluate_variant(
    *,
    variant_name: str,
    strategy: str,
    head_pairs: list[tuple[str, str]],
    base_dataset,
    x_cache: torch.Tensor,
    los_cache: torch.Tensor,
    soft_discharge_cache: dict[str, Any] | None,
    outer_test_idx: list[int],
    train_core_idx: list[int],
    batch_size: int,
    num_workers: int,
    model: torch.nn.Module,
    edge_index: torch.Tensor,
    cfg: dict[str, Any],
    device: torch.device,
    baseline_metrics: dict[str, float],
) -> dict[str, Any]:
    fallback_x_cache = x_cache.clone()
    fallback_soft_cache = _clone_soft_discharge_cache(soft_discharge_cache)
    train_mode_values = None
    if strategy == "train_mode":
        train_mode_values = _compute_train_mode_values(base_dataset, train_core_idx, head_pairs)
    _apply_fallback_strategy(
        base_dataset,
        fallback_x_cache,
        fallback_soft_cache,
        outer_test_idx,
        head_pairs,
        strategy=strategy,
        train_mode_values=train_mode_values,
    )
    fallback_loader = _build_test_loader(
        base_dataset,
        fallback_x_cache,
        los_cache.clone(),
        fallback_soft_cache,
        outer_test_idx,
        batch_size,
        num_workers,
    )
    metrics = _evaluate_fold(model, edge_index, fallback_loader, cfg, device)
    return {
        "variant_name": variant_name,
        "strategy": strategy,
        "head_pairs": [{"discharge_head": d, "admission_head": a} for d, a in head_pairs],
        "train_mode_values": train_mode_values,
        "metrics": metrics,
        "delta_auc": float(metrics["test_auc"] - baseline_metrics["test_auc"]),
        "delta_f1": float(metrics["test_f1"] - baseline_metrics["test_f1"]),
        "passes_auc_0_90": bool(metrics["test_auc"] >= 0.90),
    }


def _collect_run_dirs(run_dirs: list[str], ids: list[int]) -> list[Path]:
    resolved = [Path(p).resolve() for p in run_dirs]
    if ids:
        known = list(RUNS_DIR.glob("20260519-*ctmp_gin_joint_fresh_id*"))
        for exp_id in ids:
            matches = [p.resolve() for p in known if f"id{exp_id}" in p.name]
            if not matches:
                raise FileNotFoundError(f"No downstream run found for id{exp_id}")
            resolved.extend(matches)
    if not resolved:
        resolved = sorted(p.resolve() for p in RUNS_DIR.glob("20260519-*ctmp_gin_joint_fresh_id*"))
    uniq: list[Path] = []
    seen = set()
    for path in resolved:
        if str(path) in seen:
            continue
        seen.add(str(path))
        uniq.append(path)
    return uniq


def _parse_selection(raw_value: str, *, allowed: set[str], all_value: str) -> list[str]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if not parts or all_value in parts:
        return sorted(allowed)
    unknown = [part for part in parts if part not in allowed]
    if unknown:
        raise ValueError(f"Unsupported selection values: {unknown}; allowed={sorted(allowed)}")
    return parts


def _run_one_fold(
    run_dir: Path,
    fold_dir: Path,
    device_override: str | None,
    variant_sets: list[str],
    strategies: list[str],
) -> dict[str, Any]:
    from src.utils.device_set import device_set

    cfg = _load_yaml(fold_dir / "config.final.yaml")
    device = _safe_device(device_set(device_override or cfg.get("device")))
    experiment_id = _extract_experiment_id(str(cfg.get("run_name") or run_dir.name))
    fold_num = int(fold_dir.name.split("_")[-1])

    base_dataset, x_cache, los_cache, soft_discharge_cache, split_payload = _build_cached_fold_data(
        fold_dir,
        cfg,
        device,
    )
    outer_test_idx = [int(idx) for idx in split_payload["outer_test_idx"]]
    train_core_idx = [int(idx) for idx in split_payload["train_core_idx"]]
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    baseline_loader = _build_test_loader(
        base_dataset,
        x_cache.clone(),
        los_cache.clone(),
        _clone_soft_discharge_cache(soft_discharge_cache),
        outer_test_idx,
        batch_size,
        num_workers,
    )
    model, edge_index = _load_model_and_edge(fold_dir, cfg, device)
    baseline_metrics = _evaluate_fold(model, edge_index, baseline_loader, cfg, device)
    variants: dict[str, Any] = {}
    for variant_set in variant_sets:
        head_pairs = _resolve_variant_head_pairs(variant_set)
        for strategy in strategies:
            variant_name = f"{variant_set}_{strategy}"
            variants[variant_name] = _evaluate_variant(
                variant_name=variant_name,
                strategy=strategy,
                head_pairs=head_pairs,
                base_dataset=base_dataset,
                x_cache=x_cache,
                los_cache=los_cache,
                soft_discharge_cache=soft_discharge_cache,
                outer_test_idx=outer_test_idx,
                train_core_idx=train_core_idx,
                batch_size=batch_size,
                num_workers=num_workers,
                model=model,
                edge_index=edge_index,
                cfg=cfg,
                device=device,
                baseline_metrics=baseline_metrics,
            )

    stored_metrics = _load_stored_test_metrics(fold_dir / "metrics.jsonl")
    return {
        "run_dir": str(run_dir),
        "fold_dir": str(fold_dir),
        "id": experiment_id,
        "fold": fold_num,
        "stored_metrics": stored_metrics,
        "baseline_metrics": baseline_metrics,
        "variants": variants,
    }


def run_ablation(
    run_dirs: list[Path],
    device_override: str | None,
    variant_sets: list[str],
    strategies: list[str],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        fold_root = run_dir / "folds"
        if not fold_root.exists():
            continue
        for fold_dir in sorted(p for p in fold_root.iterdir() if p.is_dir() and p.name.startswith("fold_")):
            if not (fold_dir / "checkpoints" / "best.pt").exists():
                continue
            if not (fold_dir / "joint_forecast_pipeline_splits.json").exists():
                continue
            result = _run_one_fold(run_dir, fold_dir, device_override, variant_sets, strategies)
            result_path = fold_dir / FALLBACK_RESULT_FILENAME
            _save_json(result_path, result)
            results.append(result)
    return results


def _write_summary(results: list[dict[str, Any]]) -> tuple[Path, Path]:
    DIAGNOSTICS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d-%H%M%S")
    json_path = DIAGNOSTICS_DIR / f"summary_{ts}.json"
    csv_path = DIAGNOSTICS_DIR / f"summary_{ts}.csv"
    _save_json(json_path, {"results": results})

    fieldnames = [
        "id",
        "run_dir",
        "fold",
        "variant",
        "baseline_test_auc",
        "variant_test_auc",
        "delta_auc",
        "baseline_test_f1",
        "variant_test_f1",
        "delta_f1",
        "passes_auc_0_90",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            for variant_name, variant in row["variants"].items():
                writer.writerow(
                    {
                        "id": row["id"],
                        "run_dir": row["run_dir"],
                        "fold": row["fold"],
                        "variant": variant_name,
                        "baseline_test_auc": row["baseline_metrics"]["test_auc"],
                        "variant_test_auc": variant["metrics"]["test_auc"],
                        "delta_auc": variant["delta_auc"],
                        "baseline_test_f1": row["baseline_metrics"]["test_f1"],
                        "variant_test_f1": variant["metrics"]["test_f1"],
                        "delta_f1": variant["delta_f1"],
                        "passes_auc_0_90": variant["passes_auc_0_90"],
                    }
                )
    return json_path, csv_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval-only fallback ablation variants.")
    parser.add_argument("--run-dir", action="append", default=[], help="Explicit downstream run dir.")
    parser.add_argument("--ids", default="", help="Comma-separated experiment ids, e.g. 9,22,26,38")
    parser.add_argument(
        "--variant-set",
        default="all",
        help="Comma-separated variant sets: top3,top5 or all",
    )
    parser.add_argument(
        "--strategy",
        default="all",
        help="Comma-separated strategies: admission,train_mode or all",
    )
    parser.add_argument("--device", default=None, help="Optional device override.")
    args = parser.parse_args()

    ids = [int(part.strip()) for part in args.ids.split(",") if part.strip()]
    variant_sets = _parse_selection(args.variant_set, allowed={"top3", "top5"}, all_value="all")
    strategies = _parse_selection(args.strategy, allowed={"admission", "train_mode"}, all_value="all")
    run_dirs = _collect_run_dirs(args.run_dir, ids)
    if not run_dirs:
        raise SystemExit("No matching downstream runs found.")

    results = run_ablation(run_dirs, args.device, variant_sets, strategies)
    json_path, csv_path = _write_summary(results)
    print(f"saved_json={json_path}")
    print(f"saved_csv={csv_path}")
    for row in results:
        for variant_name, variant in row["variants"].items():
            print(
                f"id={row['id']} fold={row['fold']} variant={variant_name} "
                f"baseline_auc={row['baseline_metrics']['test_auc']:.4f} "
                f"variant_auc={variant['metrics']['test_auc']:.4f} "
                f"delta_auc={variant['delta_auc']:+.4f} "
                f"passes_auc_0_90={variant['passes_auc_0_90']}"
            )


if __name__ == "__main__":
    main()
