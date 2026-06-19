# permutation_main.py

"""
  ---
  최종 실행 순서

# Step 1: k-fold CV 모델들에서 GatedFusion 가중치 추출
python src/analysis/extract_gated_fusion_kfold.py --run_name "20260302-121934__ctmp_gin__bs=256__lr=2.00e-04__seed=3__cv=5__test=0.15" --all_matching_seeds --device mps
# → src/analysis/gated_fusion_w_los_kfold.csv 생성

# Step 2: LOS 그룹 결정 (k-fold CSV 자동 우선 사용)
python src/analysis/los_group_detection.py
# → src/analysis/los_groups.json + 시각화 생성

# Step 3: PI 실행
python src/explainers/permutation_main.py --run_name "20260302-121934__ctmp_gin__bs=256__lr=2.00e-04__seed=3__cv=5__test=0.15" --fold all --device mps --num_repeats 5 --max_test_samples 30000

python src/explainers/permutation_main.py \
      --run_name "(final)20260413-071956__ctmp_gin__bs=1024__lr=6.10e-04__seed=1__cv=5__test=0.15"
      --fold all --device mps --num_repeats 5

# 모든 seed를 한 번에 추출하려면 Step 1에서 --all_matching_seeds 사용
"""

import json
import os
import sys
import subprocess
from pathlib import Path
import argparse
import yaml
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

project_root = Path(__file__).resolve().parent.parent.parent  # Phase_2_public/
sys.path.insert(0, str(project_root))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.splits import (
    holdout_test_split_stratified,
)
from src.models.factory import build_model
from src.utils.device_set import device_set

from src.explainers.permutation_importance import (
    PermutationImportanceConfig,
    compute_permutation_importance,
)

try:
    from src.explainers.stablity_report import (
        importance_mean_std_table,
    )
except Exception:
    from stablity_report import (
        importance_mean_std_table,
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))
root = os.path.join(cur_dir, "..", "data")
save_base = os.path.join(cur_dir, "results", "permutation")


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="Run directory name under runs/protected/k_fold_CV/, or a run directory path.",
    )
    p.add_argument(
        "--fold",
        type=str,
        default="all",
        help="Fold to evaluate: integer 0-4 or 'all'",
    )
    p.add_argument(
        "--los_groups",
        type=str,
        default=None,
        help="Path to JSON file with LOS group definitions (optional)",
    )
    p.add_argument("--device", type=str, default="mps")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--num_repeats", type=int, default=5, help="Permutation repeats per feature"
    )
    p.add_argument(
        "--max_test_samples",
        type=int,
        default=None,
        help="Cap test set size for PI (e.g. 30000). Stratified random sample. "
        "None = use full test set.",
    )
    p.add_argument(
        "--parallel",
        choices=["none", "auto"],
        default="none",
        help="Use fold-level parallel execution. 'auto' assigns folds to detected CUDA GPUs.",
    )
    p.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel fold workers. Default = number of detected GPUs.",
    )
    p.add_argument(
        "--checkpoint",
        choices=["best", "last"],
        default="best",
        help="Fold checkpoint to load for PI. Default = best.pt.",
    )

    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _parse_fold_arg(fold_str: str, n_folds: int) -> list[int]:
    if fold_str.strip().lower() == "all":
        return list(range(n_folds))
    try:
        fold_id = int(fold_str.strip())
        if not (0 <= fold_id < n_folds):
            raise ValueError(f"fold {fold_id} out of range [0, {n_folds})")
        return [fold_id]
    except ValueError:
        raise ValueError(f"--fold must be an integer or 'all', got: {fold_str!r}")


def _load_fold_model(
    fold_dir: str,
    device: torch.device,
    checkpoint: str = "best",
) -> tuple[nn.Module, dict]:
    """Load a fold checkpoint. cfg is embedded in the checkpoint."""
    ckpt_path = os.path.join(fold_dir, "checkpoints", f"{checkpoint}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]

    # Override device in model params
    cfg["model"]["params"]["device"] = str(device)

    model = build_model(
        model_name=cfg["model"]["name"], **cfg["model"].get("params", {})
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    print(f"  Loaded checkpoint: epoch={ckpt.get('epoch')}")
    if "metrics" in ckpt and ckpt["metrics"]:
        m = ckpt["metrics"]
        if "val_auc" in m:
            print(f"  val_auc={m['val_auc']:.4f}")

    return model, cfg


def _reconstruct_test_idx(dataset: TEDSTensorDataset, cfg: dict) -> np.ndarray:
    """
    Reproduce the same test_idx used during k-fold CV training.
    holdout_test_split_stratified is called once before the fold loop,
    so test_idx is identical across all folds.
    """
    seed = cfg["train"]["seed"]
    test_ratio = cfg["train"]["test_ratio"]
    _, test_idx = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed
    )
    return test_idx


def _subsample_test_idx(
    test_idx: np.ndarray,
    max_samples: int,
    seed: int,
) -> np.ndarray:
    """Randomly subsample test_idx to at most max_samples indices."""
    if len(test_idx) <= max_samples:
        return test_idx
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(test_idx), size=max_samples, replace=False)
    return test_idx[np.sort(chosen)]


def _build_fold_edge_index(
    device: torch.device,
    fold_dir: str,
) -> torch.Tensor:
    """Load the exact edge_index saved by the training run for this fold."""
    edge_index_path = os.path.join(fold_dir, "edge_index.pt")
    if not os.path.exists(edge_index_path):
        raise FileNotFoundError(
            f"Saved edge_index.pt not found: {edge_index_path}. "
            "Permutation importance must use the edge_index saved by training."
        )

    print(f"  Loading saved edge_index from {edge_index_path}")
    return torch.load(edge_index_path, map_location=device).to(device)


def _build_test_loader(
    dataset: TEDSTensorDataset,
    test_idx,
    batch_size: int,
    num_workers: int,
) -> DataLoader:
    # The saved edge_index.pt is already batched for the training batch size.
    # Keep drop_last=True so every forward uses the same graph shape as training.
    indices = test_idx.tolist() if hasattr(test_idx, "tolist") else list(test_idx)
    if len(indices) < batch_size:
        raise ValueError(
            f"Not enough samples for one full PI batch: n={len(indices)}, "
            f"batch_size={batch_size}. Reduce --batch_size only if the saved "
            "edge_index.pt was generated with that same batch size."
        )
    dropped = len(indices) % batch_size
    if dropped:
        print(
            f"  drop_last=True: dropping {dropped} samples "
            f"({len(indices)} -> {len(indices) - dropped})"
        )
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )


def df_to_importance_vector(df, V: int) -> torch.Tensor:
    """
    df: output of compute_permutation_importance
    returns vector of length (V + 1): [var0..var(V-1), LOS]
    """
    vec = torch.full((V + 1,), float("nan"), dtype=torch.float32)

    for _, row in df.iterrows():
        kind = row.get("kind", None)
        if kind == "node":
            j = int(row["index"])
            vec[j] = float(row["importance_mean"])
        elif kind == "los":
            vec[V] = float(row["importance_mean"])

    return vec


def _visible_cuda_gpu_ids() -> list[str]:
    """
    Return CUDA device ids to assign to workers.

    If CUDA_VISIBLE_DEVICES is already set by the container, preserve that
    restriction. Otherwise prefer nvidia-smi, which is reliable in Vast.ai.
    """
    visible_env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_env:
        gpu_ids = [part.strip() for part in visible_env.split(",") if part.strip()]
        if gpu_ids and gpu_ids != ["-1"]:
            return gpu_ids

    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=True,
            capture_output=True,
            text=True,
        )
        count = len([line for line in result.stdout.splitlines() if line.strip()])
        if count > 0:
            return [str(i) for i in range(count)]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if torch.cuda.is_available():
        return [str(i) for i in range(torch.cuda.device_count())]
    return []


def _build_worker_cmd(args: argparse.Namespace, fold_id: int) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run_name",
        args.run_name,
        "--fold",
        str(fold_id),
        "--device",
        "cuda:0",
        "--num_repeats",
        str(args.num_repeats),
        "--parallel",
        "none",
        "--checkpoint",
        args.checkpoint,
    ]
    if args.los_groups:
        cmd.extend(["--los_groups", args.los_groups])
    if args.batch_size is not None:
        cmd.extend(["--batch_size", str(args.batch_size)])
    if args.max_test_samples is not None:
        cmd.extend(["--max_test_samples", str(args.max_test_samples)])
    return cmd


def _run_parallel_folds(
    args: argparse.Namespace,
    folds_to_run: list[int],
    out_dir: Path,
) -> bool:
    gpu_ids = _visible_cuda_gpu_ids()
    gpu_count = len(gpu_ids)
    if gpu_count < 1:
        print("No CUDA GPUs detected; falling back to sequential execution.")
        return False

    worker_count = min(gpu_count, len(folds_to_run))
    if args.max_workers is not None:
        if args.max_workers < 1:
            raise ValueError("--max_workers must be >= 1")
        worker_count = min(worker_count, args.max_workers)

    log_dir = out_dir / "launcher_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Parallel fold execution: detected_gpus={gpu_count}, "
        f"workers={worker_count}, folds={folds_to_run}"
    )
    print(f"Worker logs: {log_dir}")

    active: list[dict] = []
    next_idx = 0
    fail_rc = 0
    fail_fold: int | None = None

    def start_fold(fold_id: int, gpu_id: str) -> dict:
        log_path = log_dir / f"fold_{fold_id}.log"
        log_f = open(log_path, "w")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env.setdefault("OMP_NUM_THREADS", "4")
        env.setdefault("MKL_NUM_THREADS", "4")
        cmd = _build_worker_cmd(args, fold_id)
        print(f"  start fold={fold_id} gpu={gpu_id} log={log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
        )
        return {
            "proc": proc,
            "fold": fold_id,
            "gpu": gpu_id,
            "log": log_path,
            "fh": log_f,
        }

    while next_idx < worker_count:
        active.append(start_fold(folds_to_run[next_idx], gpu_ids[next_idx]))
        next_idx += 1

    while active:
        remaining = []
        for item in active:
            proc = item["proc"]
            rc = proc.poll()
            if rc is None:
                remaining.append(item)
                continue

            item["fh"].close()
            fold_id = item["fold"]
            gpu_id = item["gpu"]
            if rc != 0:
                fail_rc = rc
                fail_fold = fold_id
                print(
                    f"  fold={fold_id} failed on gpu={gpu_id} rc={rc}. "
                    f"See {item['log']}"
                )
            else:
                print(f"  fold={fold_id} completed on gpu={gpu_id}")

            if fail_rc == 0 and next_idx < len(folds_to_run):
                remaining.append(start_fold(folds_to_run[next_idx], gpu_id))
                next_idx += 1

        active = remaining
        if active:
            import time

            time.sleep(5)

    if fail_rc != 0:
        raise RuntimeError(
            f"Parallel permutation worker failed: fold={fail_fold}, rc={fail_rc}. "
            f"See logs under {log_dir}"
        )
    return True


def _prepare_shared_inputs(args: argparse.Namespace):
    # ---- Resolve paths ----
    project_root_dir = Path(cur_dir).resolve().parent.parent
    runs_base = project_root_dir / "runs" / "protected" / "k_fold_CV"
    run_name_path = Path(args.run_name).expanduser()
    if run_name_path.is_absolute() or run_name_path.exists():
        run_dir = run_name_path.resolve()
        output_name = run_dir.name
    else:
        run_dir = runs_base / args.run_name
        output_name = args.run_name

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    # Read fold_0 config for shared settings (seed, test_ratio, n_folds, etc.)
    fold_0_cfg_path = run_dir / "folds" / "fold_0" / "config.final.yaml"
    fold_0_cfg = load_yaml(str(fold_0_cfg_path))
    n_folds = fold_0_cfg["train"].get("n_folds", 5)

    folds_to_run = _parse_fold_arg(args.fold, n_folds)
    print(f"Run: {args.run_name}")
    print(f"Run directory: {run_dir}")
    print(f"Folds to evaluate: {folds_to_run}")

    # ---- Dataset (shared across folds) ----
    model_name = fold_0_cfg["model"]["name"]
    remove_los = model_name not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]
    dataset = TEDSTensorDataset(
        root=root,
        binary=fold_0_cfg["train"].get("binary", True),
        ig_label=fold_0_cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=fold_0_cfg["train"].get("do_preprocess", True),
    )
    col_names = dataset.col_info[0]
    V = len(col_names)
    names_with_los = col_names + ["LOS"]

    # ---- Reconstruct test_idx (same for all folds) ----
    test_idx = _reconstruct_test_idx(dataset, fold_0_cfg)
    print(f"Test set size (full): {len(test_idx)}")
    if args.max_test_samples is not None:
        test_idx = _subsample_test_idx(
            test_idx, args.max_test_samples, seed=fold_0_cfg["train"]["seed"]
        )
        print(f"Test set size (subsampled): {len(test_idx)}")

    # ---- Optional LOS groups ----
    los_groups = None
    if args.los_groups:
        with open(args.los_groups, "r") as f:
            los_groups = json.load(f)
        assert isinstance(los_groups, dict)
        print(f"LOS groups loaded: {list(los_groups.keys())}")

    # ---- Output directory ----
    out_dir = Path(save_base) / output_name
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_dir": run_dir,
        "fold_0_cfg": fold_0_cfg,
        "folds_to_run": folds_to_run,
        "dataset": dataset,
        "col_names": col_names,
        "V": V,
        "names_with_los": names_with_los,
        "test_idx": test_idx,
        "los_groups": los_groups,
        "out_dir": out_dir,
    }


def _run_one_fold(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    dataset: TEDSTensorDataset,
    col_names: list[str],
    test_idx,
    los_groups: dict | None,
    out_dir: Path,
    V: int,
    fold_id: int,
    device: torch.device,
) -> None:
    fold_dir = str(run_dir / "folds" / f"fold_{fold_id}")
    print(f"\n=== Fold {fold_id} ===")

    model, fold_cfg = _load_fold_model(fold_dir, device, checkpoint=args.checkpoint)

    train_batch_size = fold_cfg["train"]["batch_size"]
    if args.batch_size is not None and args.batch_size != train_batch_size:
        raise ValueError(
            f"--batch_size={args.batch_size} does not match the training batch size "
            f"used to build edge_index.pt ({train_batch_size}). Omit --batch_size "
            "or rerun with the same value used during training."
        )
    batch_size = train_batch_size
    num_workers = fold_cfg["train"].get("num_workers", 0)

    # Load fold-specific edge_index saved during training.
    print(f"  Loading edge_index for fold {fold_id}...")
    edge_index = _build_fold_edge_index(
        device=device,
        fold_dir=fold_dir,
    )
    print(
        f"  edge_index shape: {tuple(edge_index.shape)}, max node: {edge_index.max().item()}"
    )

    test_loader = _build_test_loader(dataset, test_idx, batch_size, num_workers)

    perm_cfg = PermutationImportanceConfig(
        num_repeats=args.num_repeats,
        seed=fold_cfg["train"]["seed"],
        variable_names=col_names,
    )

    df = compute_permutation_importance(
        model=model,
        dataloader=test_loader,
        edge_index=edge_index,
        device=device,
        config=perm_cfg,
    )

    out_csv = out_dir / f"fold_{fold_id}_pi.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(f"  baseline_auc = {df['baseline_auc'].iloc[0]:.4f}")
    print(f"  Top 5 features:")
    print(
        df[["feature", "importance_mean", "importance_std"]]
        .head(5)
        .to_string(index=False)
    )

    # ---- LOS group sub-analysis (optional) ----
    if los_groups is not None:
        los_tensor = dataset.LOS

        for gname, los_vals in los_groups.items():
            los_set = set(los_vals)
            group_idx = [
                i for i in test_idx if int(round(float(los_tensor[i]))) in los_set
            ]
            if not group_idx:
                print(
                    f"  Warning: no test samples for {gname} (LOS={los_vals}), skipping"
                )
                continue
            if len(group_idx) < batch_size:
                print(
                    f"  Warning: only {len(group_idx)} test samples for {gname}; "
                    f"need at least batch_size={batch_size}, skipping"
                )
                continue

            print(f"\n  --- {gname} (LOS={los_vals}, n={len(group_idx)}) ---")
            g_loader = _build_test_loader(dataset, group_idx, batch_size, num_workers)

            df_g = compute_permutation_importance(
                model=model,
                dataloader=g_loader,
                edge_index=edge_index,
                device=device,
                config=perm_cfg,
                show_progress=False,
            )

            out_g_csv = out_dir / f"fold_{fold_id}_{gname}_pi.csv"
            df_g.to_csv(out_g_csv, index=False)
            print(f"  Saved: {out_g_csv}")


def _aggregate_results(
    *,
    folds_to_run: list[int],
    out_dir: Path,
    V: int,
    names_with_los: list[str],
    los_groups: dict | None,
) -> None:
    fold_vecs: list[torch.Tensor] = []

    for fold_id in folds_to_run:
        out_csv = out_dir / f"fold_{fold_id}_pi.csv"
        if not out_csv.exists():
            raise FileNotFoundError(f"Expected fold output not found: {out_csv}")
        df = pd.read_csv(out_csv)
        fold_vecs.append(df_to_importance_vector(df, V=V))

    # ---- Cross-fold aggregate: overall ----
    if len(fold_vecs) > 1:
        print("\n=== Cross-fold mean±std (overall) ===")
        df_ms = importance_mean_std_table(
            fold_vecs,
            names_with_los,
            save_path=str(out_dir),
            filename="all_folds_pi.csv",
        )
        print("\nTop 30:")
        print(df_ms.head(30).to_string(index=False))
    elif len(fold_vecs) == 1:
        single_fold_id = folds_to_run[0]
        df_single = pd.read_csv(out_dir / f"fold_{single_fold_id}_pi.csv")
        print(f"\n=== Fold {single_fold_id} Top 30 ===")
        print(df_single.head(30).to_string(index=False))

    # ---- Cross-fold aggregate: per group ----
    if los_groups is not None:
        for gname in los_groups:
            vecs = []
            existing_fold_ids = []
            for fold_id in folds_to_run:
                out_g_csv = out_dir / f"fold_{fold_id}_{gname}_pi.csv"
                if out_g_csv.exists():
                    df_g = pd.read_csv(out_g_csv)
                    vecs.append(df_to_importance_vector(df_g, V=V))
                    existing_fold_ids.append(fold_id)
            if len(vecs) > 1:
                print(f"\n=== Cross-fold mean±std: {gname} ===")
                df_g_ms = importance_mean_std_table(
                    vecs,
                    names_with_los,
                    save_path=str(out_dir),
                    filename=f"{gname}_all_folds_pi.csv",
                )
                print(f"  Top 10:")
                print(df_g_ms.head(10).to_string(index=False))
            elif len(vecs) == 1:
                single_fold_id = existing_fold_ids[0]
                print(f"\n=== {gname} (single fold) Top 10 ===")
                print(
                    pd.read_csv(out_dir / f"fold_{single_fold_id}_{gname}_pi.csv")
                    .head(10)
                    .to_string(index=False)
                )


def main():
    args = parse_args()
    shared = _prepare_shared_inputs(args)

    parallel_done = False
    if args.parallel == "auto" and len(shared["folds_to_run"]) > 1:
        parallel_done = _run_parallel_folds(
            args=args,
            folds_to_run=shared["folds_to_run"],
            out_dir=shared["out_dir"],
        )

    if not parallel_done:
        # ---- Device ----
        device = device_set(args.device)

        # ---- Per-fold PI computation ----
        for fold_id in shared["folds_to_run"]:
            _run_one_fold(
                args=args,
                run_dir=shared["run_dir"],
                dataset=shared["dataset"],
                col_names=shared["col_names"],
                test_idx=shared["test_idx"],
                los_groups=shared["los_groups"],
                out_dir=shared["out_dir"],
                V=shared["V"],
                fold_id=fold_id,
                device=device,
            )

    _aggregate_results(
        folds_to_run=shared["folds_to_run"],
        out_dir=shared["out_dir"],
        V=shared["V"],
        names_with_los=shared["names_with_los"],
        los_groups=shared["los_groups"],
    )


if __name__ == "__main__":
    main()
