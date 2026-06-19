"""
reeval_best_val_auc.py
======================
기존 k-fold CV / ablation run들을 best val_auc epoch 체크포인트로 재평가합니다.

기존 runs는 valid_loss 기준으로 best.pt를 저장했고,
테스트 평가는 마지막 epoch 모델로 수행됐습니다.
이 스크립트는 각 fold의 metrics.jsonl에서 best val_auc epoch를 찾아
해당 epoch_{N}.pt로 테스트셋을 재평가하고, 기존 값과 비교합니다.

실행 결과:
  - run_dir/cv_summary_new.json: best val_auc epoch 기준 재평가 결과
  - 콘솔: fold별 비교 테이블 + (--r_output 시) R 벡터

Usage
-----
# 특정 run 지정
python src/analysis/reeval_best_val_auc.py \\
    --runs runs/protected/k_fold_CV/20260302-143833__ctmp_gin__... \\
           runs/protected/ablation/no_gate_20260328-160934__ctmp_gin__... \\
    --device mps

# 디렉터리 자동 스캔 ((past) 접두사 run은 자동 제외)
python src/analysis/reeval_best_val_auc.py \\
    --scan_dirs runs/protected/k_fold_CV runs/protected/ablation \\
    --device mps

# R 벡터 포맷으로 출력 (run_optuna.r 업데이트용)
python src/analysis/reeval_best_val_auc.py \\
    --scan_dirs runs/protected/k_fold_CV \\
    --device mps --r_output
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data_processing.tensor_dataset import TEDSTensorDataset
from src.data_processing.splits import holdout_test_split_stratified, kfold_stratified
from src.models.factory import build_model, build_edge
from src.trainers.base import evaluate
from src.utils.device_set import device_set
from src.utils.seed_set import set_seed

import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_best_val_auc_epoch(metrics_jsonl: str) -> tuple[int, float]:
    """metrics.jsonl에서 val_auc가 가장 높은 epoch 번호와 val_auc 값을 반환."""
    val_records = []
    with open(metrics_jsonl) as f:
        for line in f:
            r = json.loads(line)
            if "valid_auc" in r:
                val_records.append(r)
    if not val_records:
        raise ValueError(f"No valid_auc records in {metrics_jsonl}")
    best = max(val_records, key=lambda r: r["valid_auc"])
    return best["epoch"], best["valid_auc"]


def load_model_from_epoch_ckpt(fold_dir: str, epoch: int, device: torch.device):
    """
    epoch_{N}.pt 또는 last.pt 체크포인트에서 모델 로드.
    cfg에 저장된 col_info / num_nodes 등을 그대로 사용해 모델을 재구성하므로
    현재 데이터셋과 무관하게 훈련 당시의 아키텍처를 복원합니다.
    epoch=-1이면 last.pt를 로드합니다.
    """
    if epoch == -1:
        ckpt_path = os.path.join(fold_dir, "checkpoints", "last.pt")
    else:
        ckpt_path = os.path.join(fold_dir, "checkpoints", f"epoch_{epoch}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt["cfg"]
    cfg["model"]["params"]["device"] = str(device)
    model = build_model(
        model_name=cfg["model"]["name"], **cfg["model"].get("params", {})
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def get_stored_test_auc(fold_dir: str) -> float | None:
    """metrics.jsonl의 test 레코드에서 기존 test_auc 반환."""
    path = os.path.join(fold_dir, "metrics.jsonl")
    last_test = None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "test":
                last_test = r
    return last_test["test_auc"] if last_test else None


def get_train_df_for_fold(
    dataset: TEDSTensorDataset,
    labels: np.ndarray,
    fold_cfg: dict,
    target_fold_id: int,
) -> object:
    """훈련 시 사용된 fold-specific train_df 재구성."""
    seed = fold_cfg["train"]["seed"]
    test_ratio = fold_cfg["train"]["test_ratio"]
    n_folds = fold_cfg["train"]["n_folds"]

    trainval_idx, _ = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )
    for fold, train_idx, _ in kfold_stratified(
        trainval_idx=trainval_idx, labels=labels, n_folds=n_folds, seed=seed
    ):
        if fold == target_fold_id:
            return dataset.processed_df.iloc[train_idx]
    raise ValueError(f"fold {target_fold_id} not found")


def save_cv_summary_new(run_dir: Path, fold_results: list[dict], fold_0_cfg: dict) -> None:
    """
    재평가 결과를 run_dir/cv_summary_new.json으로 저장.
    기존 cv_summary.json의 메타 정보를 기반으로 하되 test 지표를 교체합니다.
    """
    # 기존 cv_summary.json 로드 (있으면 메타 재사용)
    existing_path = run_dir / "cv_summary.json"
    base_meta: dict = {}
    if existing_path.exists():
        with open(existing_path) as f:
            base_meta = json.load(f)

    summary = {
        "cv_id": base_meta.get("cv_id", run_dir.name),
        "K": fold_0_cfg["train"].get("n_folds", 5),
        "test_ratio": fold_0_cfg["train"].get("test_ratio", 0.15),
        "reeval_note": "Re-evaluated using best val_auc epoch checkpoint (reeval_best_val_auc.py)",
        "fold_results": [],
    }

    for r in fold_results:
        # 기존 fold entry에서 변하지 않는 필드 유지
        orig_fold: dict = {}
        for fr in base_meta.get("fold_results", []):
            if fr.get("fold") == r["fold"]:
                orig_fold = fr
                break

        entry = {
            "fold": r["fold"],
            # best val_auc epoch 정보
            "best_val_auc_epoch": r["best_val_auc_epoch"],
            "best_val_auc": r["best_val_auc"],
            # 재평가 test 지표
            "test_auc": r["new_test_auc"],
            "test_acc": r["new_test_acc"],
            # 비교용 기존 값 (last epoch 기준)
            "stored_test_auc_last_epoch": r["stored_test_auc"],
            "delta_auc": r["delta_auc"],
            # 기존 항목에서 valid 지표 보존
            "best_valid_metric": orig_fold.get("best_valid_metric"),
            "best_valid_metrics": orig_fold.get("best_valid_metrics"),
        }
        summary["fold_results"].append(entry)

    out_path = run_dir / "cv_summary_new.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  → cv_summary_new.json saved: {out_path}")


# ---------------------------------------------------------------------------
# Per-run evaluation
# ---------------------------------------------------------------------------


def reeval_run(
    run_dir: Path,
    device: torch.device,
    batch_size_override: int | None = None,
) -> list[dict]:
    """
    run_dir 내 모든 fold를 best val_auc epoch 체크포인트로 재평가.

    Returns
    -------
    list of dicts (에러가 난 fold는 포함되지 않음):
        fold, best_val_auc_epoch, best_val_auc, stored_test_auc,
        new_test_auc, new_test_acc, delta_auc
    """
    fold_0_cfg = load_yaml(str(run_dir / "folds" / "fold_0" / "config.final.yaml"))
    n_folds = fold_0_cfg["train"].get("n_folds", 5)
    model_name = fold_0_cfg["model"]["name"]
    binary = fold_0_cfg["train"].get("binary", True)
    batch_size = batch_size_override or fold_0_cfg["train"]["batch_size"]

    # Dataset — run_kfold_cv.py와 동일한 remove_los 로직
    data_root = str(_PROJECT_ROOT / "src" / "data")
    remove_los = model_name not in ["gin", "a3tgcn_2_points", "gin_gru_2_points"]
    dataset = TEDSTensorDataset(
        root=data_root,
        binary=binary,
        ig_label=fold_0_cfg["train"].get("ig_label", False),
        remove_los=remove_los,
        do_preprocess=fold_0_cfg["train"].get("do_preprocess", True),
    )

    # num_nodes
    if model_name == "gin":
        num_nodes = len(dataset.col_info[0])
    else:
        num_nodes = len(dataset.col_info[2])

    labels = np.array([dataset[i][1] for i in range(len(dataset))])

    # Test split (seed-deterministic, 모든 fold 공통)
    seed = fold_0_cfg["train"]["seed"]
    test_ratio = fold_0_cfg["train"]["test_ratio"]
    _, test_idx = holdout_test_split_stratified(
        dataset, test_ratio=test_ratio, seed=seed, labels=labels
    )
    test_subset = Subset(dataset, test_idx.tolist())

    criterion = nn.BCEWithLogitsLoss() if binary else nn.CrossEntropyLoss()
    decision_threshold = 0.5

    # mi_cached=False 경고 — cv_mi_dict는 numpy 전역 랜덤 상태에 의존하므로
    # 훈련 시와 다른 edge_index가 생성될 수 있음. sanity check로 검증.
    mi_cached = fold_0_cfg.get("edge", {}).get("mi_cached", True)
    if not mi_cached:
        print(
            "  [WARN] mi_cached=False: cv_mi_dict is non-deterministic. "
            "Each fold will be sanity-checked against stored test_auc before reporting."
        )

    SANITY_THRESHOLD = 0.002  # stored vs recomputed last-epoch AUC 허용 오차

    results = []
    for fold_id in range(n_folds):
        fold_dir = str(run_dir / "folds" / f"fold_{fold_id}")
        metrics_path = os.path.join(fold_dir, "metrics.jsonl")

        try:
            best_ep, best_val_auc = find_best_val_auc_epoch(metrics_path)
            stored_auc = get_stored_test_auc(fold_dir)

            print(
                f"  fold_{fold_id}: best_val_auc_epoch={best_ep} (val_auc={best_val_auc:.5f}), "
                f"stored_test_auc={stored_auc:.5f}"
            )

            # Fold-specific edge_index: load saved if available, else recompute
            _, fold_cfg = load_model_from_epoch_ckpt(fold_dir, -1, device)
            edge_index_path = os.path.join(fold_dir, "edge_index.pt")
            if os.path.exists(edge_index_path):
                print(f"  Loading saved edge_index from {edge_index_path}")
                edge_index = torch.load(edge_index_path, map_location=device)
            else:
                print(f"  edge_index.pt not found, recomputing...")
                train_df = get_train_df_for_fold(dataset, labels, fold_cfg, fold_id)
                edge_cfg = fold_cfg.get("edge", {})
                # seed 고정: cv_mi_dict의 numpy/random 전역 상태 의존성 제거
                set_seed(fold_cfg["train"]["seed"])
                built = build_edge(
                    model_name=model_name,
                    root=data_root,
                    seed=fold_cfg["train"]["seed"],
                    train_df=train_df,
                    num_nodes=num_nodes,
                    batch_size=batch_size,
                    **edge_cfg,
                )
                edge_index = (built[0] if isinstance(built, tuple) else built).to(device)

            # DataLoader: 훈련과 동일하게 drop_last=True
            test_loader = DataLoader(
                test_subset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=fold_cfg["train"].get("num_workers", 0),
                drop_last=True,
            )

            # ── Sanity check: last.pt로 재평가해 stored_test_auc와 비교 ──
            last_model, _ = load_model_from_epoch_ckpt(fold_dir, -1, device)
            _, _, _, _, _, sanity_auc = evaluate(
                last_model, test_loader, criterion, decision_threshold,
                device, binary, edge_index,
            )
            del last_model

            if stored_auc is not None:
                sanity_delta = abs(sanity_auc - stored_auc)
                if sanity_delta > SANITY_THRESHOLD:
                    print(
                        f"  fold_{fold_id}: [SKIP] edge_index mismatch detected — "
                        f"sanity_auc={sanity_auc:.5f} vs stored={stored_auc:.5f} "
                        f"(Δ={sanity_delta:.5f} > {SANITY_THRESHOLD}). "
                        f"edge_index from cv_mi_dict differs from training."
                    )
                    continue
                print(f"           sanity OK: last.pt AUC={sanity_auc:.5f} (Δ={sanity_delta:.5f})")
            # ─────────────────────────────────────────────────────────────

            # best val_auc epoch 체크포인트 로드 & 평가
            model, _ = load_model_from_epoch_ckpt(fold_dir, best_ep, device)
            _, test_acc, _, _, _, test_auc = evaluate(
                model,
                test_loader,
                criterion,
                decision_threshold,
                device,
                binary,
                edge_index,
            )

            delta = test_auc - stored_auc if stored_auc is not None else None
            print(f"           → new_test_auc={test_auc:.5f}  Δ={delta:+.5f}")

            results.append(
                {
                    "fold": fold_id,
                    "best_val_auc_epoch": best_ep,
                    "best_val_auc": best_val_auc,
                    "stored_test_auc": stored_auc,
                    "new_test_auc": test_auc,
                    "new_test_acc": test_acc,
                    "delta_auc": delta,
                }
            )

        except Exception as e:
            print(f"  fold_{fold_id}: [SKIP] {e}")
            continue

    # fold가 하나라도 성공했으면 cv_summary_new.json 저장
    if results:
        save_cv_summary_new(run_dir, results, fold_0_cfg)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description="Re-evaluate runs with best val_auc epoch checkpoint"
    )
    p.add_argument(
        "--runs", nargs="*", default=[], help="재평가할 run 디렉터리 경로 (직접 지정)"
    )
    p.add_argument(
        "--scan_dirs",
        nargs="*",
        default=[],
        help="이 디렉터리들 아래의 모든 run을 자동 스캔 ((past) 접두사 run 제외)",
    )
    p.add_argument("--device", default="cpu", help="cpu / cuda / mps")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument(
        "--r_output",
        action="store_true",
        help="run_optuna.r에 붙여넣기 할 R 벡터 포맷으로 추가 출력",
    )
    return p.parse_args()


def _is_valid_run_dir(path: Path) -> bool:
    """
    유효한 run 디렉터리 조건:
    - folds/fold_0/config.final.yaml 존재
    - 디렉터리 이름이 '(past)' 로 시작하지 않음
    """
    if path.name.startswith("(past)"):
        return False
    return (path / "folds" / "fold_0" / "config.final.yaml").exists()


def collect_run_dirs(args) -> list[Path]:
    dirs: list[Path] = []
    for r in args.runs:
        p = Path(r)
        if not p.is_absolute():
            p = _PROJECT_ROOT / p
        dirs.append(p)
    for sd in args.scan_dirs:
        sp = Path(sd)
        if not sp.is_absolute():
            sp = _PROJECT_ROOT / sp
        for child in sorted(sp.iterdir()):
            if child.is_dir() and _is_valid_run_dir(child):
                dirs.append(child)
            elif child.is_dir() and child.name.startswith("(past)"):
                print(f"  [SKIP] {child.name}  ← (past) run 제외")
    return dirs


def main():
    args = parse_args()
    device = device_set(args.device)

    run_dirs = collect_run_dirs(args)
    if not run_dirs:
        print("재평가할 run이 없습니다. --runs 또는 --scan_dirs를 지정하세요.")
        sys.exit(1)

    print(f"총 {len(run_dirs)}개 run을 재평가합니다.\n")

    all_summary: dict[str, list[dict]] = {}

    for run_dir in run_dirs:
        run_name = run_dir.name
        print(f"\n{'='*70}")
        print(f"Run: {run_name}")
        print(f"{'='*70}")
        results = reeval_run(run_dir, device, args.batch_size)
        if results:
            all_summary[run_name] = results

    # ------------------------------------------------------------------
    # 전체 요약 테이블
    # ------------------------------------------------------------------
    print(f"\n\n{'='*90}")
    print("SUMMARY — stored test_auc (last epoch) vs new test_auc (best val_auc epoch)")
    print(f"{'='*90}")
    print(f"{'Run':60s}  fold  stored_auc  new_auc   delta")
    print("-" * 90)

    for run_name, results in all_summary.items():
        short = run_name[-55:] if len(run_name) > 55 else run_name
        for r in results:
            flag = (
                " !!!"
                if r["delta_auc"] is not None and abs(r["delta_auc"]) > 0.001
                else ""
            )
            print(
                f"{short:60s}  {r['fold']}     {r['stored_test_auc']:.5f}   {r['new_test_auc']:.5f}  "
                f"{r['delta_auc']:+.5f}{flag}"
            )

    # ------------------------------------------------------------------
    # R 벡터 출력 (Fold 4→0 순서, Seed별 그룹)
    # ------------------------------------------------------------------
    if args.r_output:
        print(f"\n\n{'='*70}")
        print("R vector output (Fold 4→0 순서, run_optuna.r 업데이트용)")
        print("※ 같은 모델 여러 seed를 Seed 3→2→1 순으로 나열하세요")
        print(f"{'='*70}")
        for run_name, results in all_summary.items():
            sorted_results = sorted(results, key=lambda r: r["fold"], reverse=True)
            auc_vals = [f"{r['new_test_auc']:.5f}" for r in sorted_results]
            print(f"\n# {run_name}")
            print(f"c({', '.join(auc_vals)})  # Fold 4→0")


if __name__ == "__main__":
    main()
