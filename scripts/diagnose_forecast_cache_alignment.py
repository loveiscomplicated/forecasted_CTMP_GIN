"""
  uv run python scripts/diagnose_forecast_cache_alignment.py \
    --config configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml \
    --mode joint_cache_alignment_audit \
    --fold 0 \
    --seed 1 \
    --device mps \
    --discharge-checkpoint-path /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/diagnostics/forecast_cache_alignment/predicted_d_predicted_los/predictors/selection/discharge/checkpoints/best.pt \
    --los-checkpoint-path /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/diagnostics/forecast_cache_alignment/predicted_d_predicted_los/predictors/selection/los/checkpoints/best.pt

uv run python scripts/diagnose_forecast_cache_alignment.py \
    --config configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml \
    --mode joint_plausibility_audit \
    --fold 0 \
    --seed 1 \
    --device mps \
    --discharge-checkpoint-path /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/diagnostics/forecast_cache_alignment/predicted_d_predicted_los/predictors/selection/discharge/checkpoints/best.pt \
    --los-checkpoint-path /Users/jeong-yunseong/Documents/programming/Phase_2_public/runs/diagnostics/forecast_cache_alignment/predicted_d_predicted_los/predictors/selection/los/checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.trainers.forecasted_diagnostics import run_diagnostic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose Forecasted CTMP-GIN cache alignment."
    )
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "audit_only",
            "oracle_cache",
            "predictor_target_cache",
            "predictor_target_cache_transformed",
            "oracle_cache_coarse_los",
            "oracle_d_predicted_los",
            "oracle_d_predicted_los_hard",
            "oracle_d_predicted_los_distribution",
            "predicted_d_oracle_los",
            "predicted_d_predicted_los",
            "oracle_D_predicted_LOS_hard",
            "oracle_D_predicted_LOS_distribution",
            "predicted_D_oracle_LOS",
            "predicted_D_predicted_LOS",
            "predicted_D_predicted_LOS_oracle_head_ablation",
            "oracle_D_predicted_LOS_predicted_head_ablation",
            "joint_cache_alignment_audit",
            "joint_plausibility_audit",
            "los_distribution_basis_audit",
            "los_hard_runtime_audit",
        ),
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--discharge-checkpoint-path", type=str, default=None)
    parser.add_argument("--los-checkpoint-path", type=str, default=None)
    parser.add_argument("--override-head", type=str, default=None)
    parser.add_argument("--rare-threshold", type=float, default=0.0001)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_diagnostic(
        config_path=args.config,
        mode=args.mode,
        fold=args.fold,
        seed=args.seed,
        device_name=args.device,
        discharge_checkpoint_path=args.discharge_checkpoint_path,
        los_checkpoint_path=args.los_checkpoint_path,
        override_head=args.override_head,
        rare_threshold=args.rare_threshold,
        dry_run=args.dry_run,
    )
    print("[JSON SUMMARY]")
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
