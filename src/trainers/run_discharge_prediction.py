import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from src.data_processing.discharge_prediction_dataset import (
    DischargePredictionDataset,
    split_discharge_dataset,
)
from src.models.discharge_predictor import (
    MultiTaskDischargePredictor,
    MultiTaskCategoricalLoss,
    compute_discharge_metrics,
)
from src.trainers.utils.early_stopper import EarlyStopper
from src.utils.device_set import device_set
from src.utils.experiment import ExperimentLogger, make_run_id, ensure_run_dir
from src.utils.seed_set import set_seed


def _train_one_epoch(
    model: MultiTaskDischargePredictor,
    loader,
    criterion: MultiTaskCategoricalLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    target_col_names: List[str],
) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        y_dict = {name: y[:, i] for i, name in enumerate(target_col_names)}
        loss, _ = criterion(logits, y_dict)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        n_batches += 1

    return total_loss / max(n_batches, 1)


def _evaluate(
    model: MultiTaskDischargePredictor,
    loader,
    criterion: MultiTaskCategoricalLoss,
    device: torch.device,
    target_col_names: List[str],
) -> Tuple[Dict[str, float], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    all_logits: Dict[str, List[np.ndarray]] = {name: [] for name in target_col_names}
    all_targets: Dict[str, List[np.ndarray]] = {name: [] for name in target_col_names}
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc="eval", leave=False):
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


def _save_predictions_csv(
    run_dir: str,
    logits_np: Dict[str, np.ndarray],
    targets_np: Dict[str, np.ndarray],
    target_col_names: List[str],
) -> None:
    rows: Dict[str, np.ndarray] = {}
    for name in target_col_names:
        rows[f"true_{name}"] = targets_np[name].astype(int)
        rows[f"pred_{name}"] = np.argmax(logits_np[name], axis=1).astype(int)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(run_dir, "predictions.csv")
    df.to_csv(csv_path, index=False)
    print(f"Predictions saved: {csv_path}  ({len(df):,} rows)")


def run_discharge_prediction(cfg: dict, root: str) -> dict:
    seed = cfg["train"].get("seed", 42)
    set_seed(seed)

    device = device_set(cfg.get("device"))

    run_id = make_run_id(cfg)
    run_dir = ensure_run_dir("runs", run_id)
    logger = ExperimentLogger(cfg, run_dir)

    dataset = DischargePredictionDataset(
        root=root,
        do_preprocess=cfg["train"].get("do_preprocess", False),
        include_los_in_targets=cfg.get("targets", {}).get("include_los", True),
    )
    schema_metadata = dict(dataset.schema_metadata)
    schema_metadata["target_col_names"] = list(dataset.target_col_names)
    schema_metadata["target_col_dims"] = [int(v) for v in dataset.target_col_dims]

    print(f"Admission variables : {len(dataset.ad_col_names)}")
    print(f"Target variables    : {dataset.target_col_names}")
    print(f"Dataset size        : {len(dataset):,}")

    split_ratio = [
        cfg["train"]["train_ratio"],
        cfg["train"]["val_ratio"],
        cfg["train"]["test_ratio"],
    ]
    train_loader, val_loader, test_loader, _ = split_discharge_dataset(
        dataset=dataset,
        batch_size=cfg["train"]["batch_size"],
        ratio=split_ratio,
        seed=seed,
        num_workers=cfg["train"].get("num_workers", 0),
    )

    model = MultiTaskDischargePredictor(
        ad_col_dims=dataset.ad_col_dims,
        target_col_names=dataset.target_col_names,
        target_col_dims=dataset.target_col_dims,
        **cfg["model"].get("params", {}),
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model)
    print(f"Trainable parameters: {total_params:,}")

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
        optimizer, "max", patience=cfg["train"]["lr_scheduler_patience"]
    )
    early_stopper = EarlyStopper(patience=cfg["train"]["early_stopping_patience"])

    monitor_metric = cfg["train"].get("monitor_metric", "valid_mean_macro_f1")
    target_names = dataset.target_col_names
    epochs = cfg["train"]["epochs"]

    best_val = -float("inf")

    for epoch in tqdm(range(1, epochs + 1)):
        train_loss = _train_one_epoch(model, train_loader, criterion, optimizer, device, target_names)

        val_metrics, _, _ = _evaluate(model, val_loader, criterion, device, target_names)
        val_loss = val_metrics["loss"]
        val_mean_acc = val_metrics["mean_accuracy"]
        val_mean_f1 = val_metrics["mean_macro_f1"]

        scheduler.step(val_mean_f1)
        current_lr = optimizer.param_groups[0]["lr"]

        log_metrics = {
            "lr": float(current_lr),
            "train_loss": float(train_loss),
            "valid_loss": float(val_loss),
            "valid_mean_accuracy": float(val_mean_acc),
            "valid_mean_macro_f1": float(val_mean_f1),
            **{f"valid_{k}": v for k, v in val_metrics.items() if k.startswith(("acc_", "f1_"))},
        }

        logger.log_metrics(epoch, log_metrics)
        logger.maybe_save_checkpoint(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            metrics=log_metrics,
            extra={"schema": schema_metadata},
        )

        cur_obj = float(log_metrics.get(monitor_metric, val_mean_f1))
        if cur_obj > best_val:
            best_val = cur_obj

        print(
            f"\n[Epoch {epoch}/{epochs}] LR={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"mean_acc={val_mean_acc:.4f} | "
            f"mean_f1={val_mean_f1:.4f}"
        )

        if early_stopper(-val_mean_f1):
            print("--- Early Stopping activated ---")
            break

    print("\n--- Training Finished ---")

    best_ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Reloaded best checkpoint (epoch={logger.best_epoch})")

    test_metrics, test_logits_np, test_targets_np = _evaluate(
        model, test_loader, criterion, device, target_names
    )

    print(
        f"\n[Test] mean_acc={test_metrics['mean_accuracy']:.4f} | "
        f"mean_f1={test_metrics['mean_macro_f1']:.4f}"
    )
    for name in target_names:
        print(
            f"  {name}: acc={test_metrics[f'acc_{name}']:.4f}  "
            f"f1={test_metrics[f'f1_{name}']:.4f}"
        )

    logger.log_metrics(epochs, {"split": "test", **{f"test_{k}": v for k, v in test_metrics.items()}})

    _save_predictions_csv(run_dir, test_logits_np, test_targets_np, target_names)

    return {
        "best_valid_metric": float(best_val),
        "test_mean_accuracy": float(test_metrics["mean_accuracy"]),
        "test_mean_macro_f1": float(test_metrics["mean_macro_f1"]),
        "run_dir": run_dir,
    }
