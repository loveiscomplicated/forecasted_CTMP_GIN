from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import numpy as np
import torch
import yaml

from src.models.discharge_predictor.los_utils import (
    get_los_coarse_num_classes,
    infer_los_coarse_breakdown_from_cfg,
    map_los_array_to_coarse_bins,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose joint predictor conditional statistics"
    )
    parser.add_argument("--train-cache-path", type=str, required=True)
    parser.add_argument("--eval-cache-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def _conditional_js_divergence(lhs: np.ndarray, rhs: np.ndarray) -> float:
    eps = 1.0e-12
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    if lhs.sum() <= 0.0 and rhs.sum() <= 0.0:
        return 0.0
    lhs = lhs / max(lhs.sum(), eps)
    rhs = rhs / max(rhs.sum(), eps)
    mid = 0.5 * (lhs + rhs)
    kl_lm = float(
        np.sum(
            lhs * (np.log(np.clip(lhs, eps, None)) - np.log(np.clip(mid, eps, None)))
        )
    )
    kl_rm = float(
        np.sum(
            rhs * (np.log(np.clip(rhs, eps, None)) - np.log(np.clip(mid, eps, None)))
        )
    )
    return 0.5 * (kl_lm + kl_rm)


def _normalize_raw_los(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.size == 0:
        return values
    if np.any((values < 1) | (values > 37)):
        raise ValueError("Expected raw LOS values in 1..37 for raw37 joint statistics.")
    return values - 1


def _load_run_config_near_cache(cache_path: str | None) -> dict[str, Any] | None:
    if not cache_path:
        return None
    config_path = os.path.join(os.path.dirname(cache_path), "config.final.yaml")
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f)
    return loaded if isinstance(loaded, dict) else None


def infer_los_bin_mode(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
) -> str:
    metadata_candidates = [
        eval_cache.get("metadata", {}),
        train_cache.get("metadata", {}),
    ]
    for metadata in metadata_candidates:
        los_num_classes = metadata.get("los_num_classes")
        if los_num_classes == 37:
            return "raw37"
        if los_num_classes == 9:
            return "coarse9"
        if los_num_classes == 6:
            return "coarse6"
        los_target_mode = str(metadata.get("los_target_mode", "")).lower()
        if los_target_mode == "raw37":
            return "raw37"
        if los_target_mode == "coarse":
            return "coarse6"
        final_los_pred_space = str(metadata.get("final_los_pred_space", "")).lower()
        if final_los_pred_space == "raw_los":
            return "raw37"
        if final_los_pred_space == "coarse_class":
            return "coarse6"
        final_los_probs = eval_cache.get("final_los_probs")
        if final_los_probs is None:
            final_los_probs = train_cache.get("final_los_probs")
        if final_los_probs is not None:
            width = int(final_los_probs.shape[1])
            if width == 37:
                return "raw37"
            if width == 9:
                return "coarse9"
            if width == 6:
                return "coarse6"

    config_candidates = [
        _load_run_config_near_cache(eval_cache.get("_path")),
        _load_run_config_near_cache(train_cache.get("_path")),
    ]
    for cfg in config_candidates:
        if not cfg:
            continue
        coarse_source = cfg.get("joint_predictor", cfg)
        try:
            if int(coarse_source.get("num_classes")) == 9:
                return "coarse9"
        except (TypeError, ValueError, AttributeError):
            pass
        if infer_los_coarse_breakdown_from_cfg(coarse_source):
            return "coarse9"
        los_target_mode = str(
            cfg.get("joint_predictor", {}).get(
                "los_target_mode",
                cfg.get("los_target_mode", ""),
            )
        ).lower()
        if los_target_mode == "raw37":
            return "raw37"
        if los_target_mode == "coarse":
            return "coarse6"

    raise ValueError(
        "Could not infer LOS bin mode from cache metadata or adjacent config.final.yaml."
    )


def _row_to_los_bins(
    cache_payload: dict[str, Any],
    *,
    from_targets: bool,
    los_bin_mode: str,
) -> np.ndarray:
    los_bin_mode = str(los_bin_mode).lower()
    if from_targets:
        raw = cache_payload["targets"]["los_raw"].cpu().numpy()
        if los_bin_mode == "raw37":
            return _normalize_raw_los(raw)
        return map_los_array_to_coarse_bins(
            raw,
            breakdown=los_bin_mode == "coarse9",
        )
    metadata = cache_payload["metadata"]
    pred = cache_payload["final_los_pred"].cpu().numpy()
    pred_space = str(metadata.get("final_los_pred_space"))
    if los_bin_mode == "raw37":
        if pred_space != "raw_los":
            raise ValueError(
                f"raw37 joint statistics require eval LOS predictions in raw_los space, got {pred_space!r}."
            )
        return _normalize_raw_los(pred)
    if pred_space == "coarse_class":
        return pred.astype(np.int64)
    final_los_probs = cache_payload.get("final_los_probs")
    if final_los_probs is not None and int(final_los_probs.shape[1]) in {
        get_los_coarse_num_classes(breakdown=False),
        get_los_coarse_num_classes(breakdown=True),
    }:
        return pred.astype(np.int64)
    return map_los_array_to_coarse_bins(
        pred,
        breakdown=los_bin_mode == "coarse9",
    )


def _conditional_table(
    target_values: np.ndarray,
    condition_values: np.ndarray,
    *,
    num_target_classes: int,
    num_condition_classes: int,
) -> np.ndarray:
    table = np.zeros((num_condition_classes, num_target_classes), dtype=np.float64)
    for cond_idx in range(num_condition_classes):
        mask = condition_values == cond_idx
        if not np.any(mask):
            continue
        counts = np.bincount(
            target_values[mask].astype(np.int64), minlength=num_target_classes
        ).astype(np.float64)
        table[cond_idx] = counts / max(counts.sum(), 1.0)
    return table


def _conditional_count_table(
    target_values: np.ndarray,
    condition_values: np.ndarray,
    *,
    num_target_classes: int,
    num_condition_classes: int,
) -> np.ndarray:
    table = np.zeros((num_condition_classes, num_target_classes), dtype=np.int64)
    for cond_idx in range(num_condition_classes):
        mask = condition_values == cond_idx
        if not np.any(mask):
            continue
        counts = np.bincount(
            target_values[mask].astype(np.int64), minlength=num_target_classes
        )
        table[cond_idx] = counts.astype(np.int64)
    return table


def _rare_combo_reference(
    target_values: np.ndarray,
    condition_values: np.ndarray,
    *,
    num_target_classes: int,
    num_condition_classes: int,
    threshold: float,
) -> np.ndarray:
    counts = _conditional_count_table(
        target_values,
        condition_values,
        num_target_classes=num_target_classes,
        num_condition_classes=num_condition_classes,
    )
    total = max(int(len(target_values)), 1)
    return (counts.astype(np.float64) / float(total)) < float(threshold)


def _rare_combo_rate(
    target_values: np.ndarray,
    condition_values: np.ndarray,
    rare_map: np.ndarray,
) -> float:
    hits: list[float] = []
    for target_value, condition_value in zip(
        target_values.tolist(), condition_values.tolist()
    ):
        cond_idx = int(condition_value)
        target_idx = int(target_value)
        if not (0 <= cond_idx < rare_map.shape[0]):
            continue
        if not (0 <= target_idx < rare_map.shape[1]):
            continue
        hits.append(1.0 if bool(rare_map[cond_idx, target_idx]) else 0.0)
    if not hits:
        return 0.0
    return float(np.mean(hits))


def compute_joint_stats(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
    *,
    los_bin_mode: str | None = None,
    rare_threshold: float = 0.0025,
) -> dict[str, Any]:
    head_names = list(eval_cache["final_d_pred"].keys())
    if los_bin_mode is None:
        los_bin_mode = infer_los_bin_mode(train_cache, eval_cache)
    los_bin_mode = str(los_bin_mode).lower()
    if los_bin_mode not in {"coarse6", "coarse9", "raw37"}:
        raise ValueError(f"Unsupported los_bin_mode: {los_bin_mode}")
    eval_los_bin = _row_to_los_bins(
        eval_cache, from_targets=False, los_bin_mode=los_bin_mode
    )
    train_los_bin = _row_to_los_bins(
        train_cache, from_targets=True, los_bin_mode=los_bin_mode
    )
    if los_bin_mode == "raw37":
        num_los_classes = 37
    elif los_bin_mode == "coarse9":
        num_los_classes = 9
    else:
        num_los_classes = 6
    rows: list[dict[str, Any]] = []
    los_given_d_rows: list[dict[str, Any]] = []
    aggregate_d_given_los: list[float] = []
    aggregate_los_given_d: list[float] = []

    for head_name in head_names:
        train_d_target = train_cache["targets"]["d"][head_name].cpu().numpy()
        eval_d_pred = eval_cache["final_d_pred"][head_name].cpu().numpy()
        d_num_classes = int(eval_cache["final_d_probs"][head_name].shape[1])

        pred_d_given_los = _conditional_table(
            eval_d_pred,
            eval_los_bin,
            num_target_classes=d_num_classes,
            num_condition_classes=num_los_classes,
        )
        train_d_given_los = _conditional_table(
            train_d_target,
            train_los_bin,
            num_target_classes=d_num_classes,
            num_condition_classes=num_los_classes,
        )
        d_given_los_js = float(
            np.mean(
                [
                    _conditional_js_divergence(
                        pred_d_given_los[idx], train_d_given_los[idx]
                    )
                    for idx in range(num_los_classes)
                ]
            )
        )

        pred_los_given_d = _conditional_table(
            eval_los_bin,
            eval_d_pred,
            num_target_classes=num_los_classes,
            num_condition_classes=d_num_classes,
        )
        train_los_given_d = _conditional_table(
            train_los_bin,
            train_d_target,
            num_target_classes=num_los_classes,
            num_condition_classes=d_num_classes,
        )
        pred_los_given_d_counts = _conditional_count_table(
            eval_los_bin,
            eval_d_pred,
            num_target_classes=num_los_classes,
            num_condition_classes=d_num_classes,
        )
        train_los_given_d_counts = _conditional_count_table(
            train_los_bin,
            train_d_target,
            num_target_classes=num_los_classes,
            num_condition_classes=d_num_classes,
        )
        los_given_d_js = float(
            np.mean(
                [
                    _conditional_js_divergence(
                        pred_los_given_d[idx], train_los_given_d[idx]
                    )
                    for idx in range(d_num_classes)
                ]
            )
        )
        rare_map = _rare_combo_reference(
            train_d_target,
            train_los_bin,
            num_target_classes=d_num_classes,
            num_condition_classes=num_los_classes,
            threshold=float(rare_threshold),
        )
        rare_combo_rate_predicted = _rare_combo_rate(
            eval_d_pred,
            eval_los_bin,
            rare_map,
        )
        rare_combo_rate_train_reference = _rare_combo_rate(
            train_d_target,
            train_los_bin,
            rare_map,
        )

        aggregate_d_given_los.append(d_given_los_js)
        aggregate_los_given_d.append(los_given_d_js)
        rows.append(
            {
                "head_name": head_name,
                "js_d_given_los": d_given_los_js,
                "js_los_given_d": los_given_d_js,
                "rare_combo_rate_predicted": rare_combo_rate_predicted,
                "rare_combo_rate_train_reference": rare_combo_rate_train_reference,
                "num_d_classes": d_num_classes,
                "eval_unique_pred_classes": int(np.unique(eval_d_pred).size),
            }
        )
        for d_value in range(d_num_classes):
            d_value_js = float(
                _conditional_js_divergence(
                    pred_los_given_d[d_value], train_los_given_d[d_value]
                )
            )
            eval_d_count = int(np.sum(eval_d_pred == d_value))
            train_d_count = int(np.sum(train_d_target == d_value))
            for los_bin in range(num_los_classes):
                los_given_d_rows.append(
                    {
                        "head_name": head_name,
                        "d_value": d_value,
                        "los_bin": los_bin,
                        "train_count": int(train_los_given_d_counts[d_value, los_bin]),
                        "train_prob": float(train_los_given_d[d_value, los_bin]),
                        "eval_count": int(pred_los_given_d_counts[d_value, los_bin]),
                        "eval_prob": float(pred_los_given_d[d_value, los_bin]),
                        "train_d_count": train_d_count,
                        "eval_d_count": eval_d_count,
                        "js_los_given_d_for_d_value": d_value_js,
                        "js_los_given_d_head": los_given_d_js,
                        "los_bin_mode": los_bin_mode,
                    }
                )

    final_los_probs = eval_cache["final_los_probs"].cpu().numpy()
    los_entropy = -np.sum(
        final_los_probs * np.log(np.clip(final_los_probs, 1.0e-12, None)), axis=1
    )
    summary = {
        "train_cache_path": train_cache.get("_path"),
        "eval_cache_path": eval_cache.get("_path"),
        "split": eval_cache.get("split"),
        "los_bin_mode": los_bin_mode,
        "rare_threshold": float(rare_threshold),
        "num_los_classes": num_los_classes,
        "num_rows_eval": int(eval_cache["row_idx"].numel()),
        "num_rows_train": int(train_cache["row_idx"].numel()),
        "mean_js_d_given_los": (
            float(np.mean(aggregate_d_given_los)) if aggregate_d_given_los else 0.0
        ),
        "max_js_d_given_los": (
            float(np.max(aggregate_d_given_los)) if aggregate_d_given_los else 0.0
        ),
        "mean_js_los_given_d": (
            float(np.mean(aggregate_los_given_d)) if aggregate_los_given_d else 0.0
        ),
        "max_js_los_given_d": (
            float(np.max(aggregate_los_given_d)) if aggregate_los_given_d else 0.0
        ),
        "los_pred_entropy_mean": (
            float(np.mean(los_entropy)) if los_entropy.size else 0.0
        ),
        "los_pred_entropy_std": float(np.std(los_entropy)) if los_entropy.size else 0.0,
        "mean_rare_combo_rate_predicted": (
            float(np.mean([row["rare_combo_rate_predicted"] for row in rows]))
            if rows
            else 0.0
        ),
        "max_rare_combo_rate_predicted": (
            float(np.max([row["rare_combo_rate_predicted"] for row in rows]))
            if rows
            else 0.0
        ),
        "per_head": rows,
        "los_given_d_rows": los_given_d_rows,
    }
    return summary


def _write_rows_csv(path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    train_cache = torch.load(args.train_cache_path, map_location="cpu", weights_only=False)
    eval_cache = torch.load(args.eval_cache_path, map_location="cpu", weights_only=False)
    train_cache["_path"] = str(args.train_cache_path)
    eval_cache["_path"] = str(args.eval_cache_path)
    summary = compute_joint_stats(train_cache, eval_cache)
    with open(
        os.path.join(args.output_dir, "joint_stats_summary.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(summary, f, indent=2)
    _write_rows_csv(
        os.path.join(args.output_dir, "joint_stats_per_head.csv"), summary["per_head"]
    )
    _write_rows_csv(
        os.path.join(args.output_dir, "per_head_conditional_los_given_d.csv"),
        summary["los_given_d_rows"],
    )
    downstream_payload = dict(eval_cache)
    downstream_payload["joint_stats"] = {
        "summary_path": os.path.join(args.output_dir, "joint_stats_summary.json"),
        "per_head_csv_path": os.path.join(args.output_dir, "joint_stats_per_head.csv"),
        "los_given_d_csv_path": os.path.join(
            args.output_dir, "per_head_conditional_los_given_d.csv"
        ),
        "mean_js_d_given_los": summary["mean_js_d_given_los"],
        "mean_js_los_given_d": summary["mean_js_los_given_d"],
    }
    torch.save(
        downstream_payload,
        os.path.join(args.output_dir, "downstream_compatible_cache.pt"),
    )


if __name__ == "__main__":
    main()
