from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import torch

from src.diagnostics.diagnose_joint_predictor_joint_stats import compute_joint_stats
from src.models.discharge_predictor.joint_generative_predictor import (
    JointGenerativeLoss,
    JointGenerativePredictor,
    kl_beta_for_epoch,
)
from src.models.discharge_predictor.los_utils import infer_los_target_from_cfg
from src.models.discharge_predictor.los_utils import map_los_array_to_coarse_bins
from src.trainers.forecasted_diagnostics import _rare_combo_map, _rare_rate_for_rows
from src.trainers.run_joint_consistent_predictor import (
    JointPredictionDataset,
    _evaluate_generative_prior,
    _make_loader,
)
from src.utils.device_set import device_set


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose a trained joint generative forecast predictor."
    )
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--root", type=str, default="src/data")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--split-indices-path", type=str, default=None)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--train-cache-path", type=str, default=None)
    parser.add_argument("--eval-cache-path", type=str, default=None)
    parser.add_argument("--rare-threshold", type=float, default=0.0001)
    return parser.parse_args()


def _load_indices(path: str | None, split: str, n: int) -> np.ndarray:
    if path is None:
        return np.arange(n, dtype=np.int64)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    key = f"{split}_idx"
    if key not in payload:
        raise ValueError(f"Split indices file lacks key {key!r}.")
    return np.asarray(payload[key], dtype=np.int64)


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _coarse_los_bins_from_cache(cache: dict[str, Any], *, from_targets: bool) -> np.ndarray:
    if from_targets:
        return map_los_array_to_coarse_bins(cache["targets"]["los_raw"].cpu().numpy()).astype(np.int64)
    pred = cache["final_los_pred"].cpu().numpy().astype(np.int64)
    pred_space = str(cache.get("metadata", {}).get("final_los_pred_space", "")).lower()
    if pred_space == "coarse_class" or int(cache["final_los_probs"].shape[1]) == 6:
        return pred
    return map_los_array_to_coarse_bins(pred).astype(np.int64)


def _rare_combo_summary(
    train_cache: dict[str, Any],
    eval_cache: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    train_los = _coarse_los_bins_from_cache(train_cache, from_targets=True)
    eval_los = _coarse_los_bins_from_cache(eval_cache, from_targets=False)
    rows = []
    for head_name, eval_pred in eval_cache["final_d_pred"].items():
        train_d = train_cache["targets"]["d"][head_name].cpu().numpy().astype(np.int64)
        eval_d = eval_pred.cpu().numpy().astype(np.int64)
        d_dim = int(eval_cache["final_d_probs"][head_name].shape[1])
        rare_map, _probability_map = _rare_combo_map(
            train_d,
            train_los,
            d_dim=d_dim,
            threshold=float(threshold),
        )
        rows.append(
            {
                "head_name": str(head_name),
                "rare_combo_rate_predicted": _rare_rate_for_rows(eval_d, eval_los, rare_map),
            }
        )
    return {
        "rare_threshold": float(threshold),
        "mean_rare_combo_rate_predicted": float(
            np.mean([row["rare_combo_rate_predicted"] for row in rows])
        )
        if rows
        else 0.0,
        "per_head": rows,
    }


def main() -> None:
    args = parse_args()
    device = device_set(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    schema = ckpt.get("schema", {})
    predictor_type = str(cfg.get("joint_predictor", {}).get("predictor_type", "")).lower()
    if predictor_type != "joint_generative":
        raise ValueError(f"Expected joint_generative checkpoint, got {predictor_type!r}.")

    dataset = JointPredictionDataset(
        root=os.path.abspath(args.root),
        do_preprocess=bool(cfg.get("train", {}).get("do_preprocess", False)),
        los_target_mode=str(schema.get("los_target_mode") or infer_los_target_from_cfg(cfg.get("joint_predictor", {}))),
    )
    model = JointGenerativePredictor(
        ad_col_dims=[int(v) for v in schema.get("admission_col_dims", dataset.ad_col_dims)],
        target_col_names=[str(v) for v in schema.get("target_col_names", dataset.target_col_names)],
        target_col_dims=[int(v) for v in schema.get("target_col_dims", dataset.target_col_dims)],
        los_num_classes=int(schema.get("los_num_classes", dataset.los_num_classes)),
        **cfg.get("model", {}).get("params", {}),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    criterion = JointGenerativeLoss(
        lambda_los=float(cfg.get("joint_predictor", {}).get("lambda_los", 1.0)),
        prior_recon_weight=float(cfg.get("joint_predictor", {}).get("prior_recon_weight", 0.5)),
    )
    indices = _load_indices(args.split_indices_path, args.split, len(dataset))
    loader = _make_loader(
        dataset,
        indices,
        int(cfg.get("train", {}).get("batch_size", 1024)),
        int(cfg.get("train", {}).get("num_workers", 0)),
        False,
        device.type == "cuda",
    )
    beta_kl = kl_beta_for_epoch(
        int(ckpt.get("epoch", cfg.get("train", {}).get("epochs", 1))),
        beta_start=float(cfg.get("joint_predictor", {}).get("beta_kl_start", 0.0)),
        beta_max=float(cfg.get("joint_predictor", {}).get("beta_kl_max", 0.001)),
        anneal_epochs=int(cfg.get("joint_predictor", {}).get("kl_anneal_epochs", 10)),
    )
    metrics, _payload = _evaluate_generative_prior(
        model,
        loader,
        criterion,
        device,
        dataset,
        beta_kl=float(beta_kl),
        posterior_diagnostics=True,
    )
    result: dict[str, Any] = {
        "checkpoint_path": args.checkpoint_path,
        "split": args.split,
        "num_rows": int(len(indices)),
        "metrics": metrics,
    }
    if args.train_cache_path and args.eval_cache_path:
        train_cache = torch.load(args.train_cache_path, map_location="cpu")
        eval_cache = torch.load(args.eval_cache_path, map_location="cpu")
        train_cache["_path"] = args.train_cache_path
        eval_cache["_path"] = args.eval_cache_path
        result["joint_stats"] = compute_joint_stats(train_cache, eval_cache)
        result["rare_combos"] = _rare_combo_summary(
            train_cache,
            eval_cache,
            threshold=float(args.rare_threshold),
        )

    if args.output_path:
        with open(args.output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=_json_default)
    print(json.dumps(result, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
