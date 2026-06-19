import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root.parent))

import yaml
import argparse
from src.models.forecast_inputs import ensure_model_forecast_defaults
from src.trainers.run_single_experiment import run_single_experiment
from src.trainers.run_kfold_cv import (
    finalize_kfold_summary,
    prepare_kfold_run,
    run_kfold_experiment,
    run_outcome_aware_single_run,
    run_outcome_aware_stage2_only,
    run_single_fold,
)

cur_dir = os.path.dirname(__file__)
root = os.path.join(cur_dir, "data")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)  # config file location
    # overrides
    # p.add_argument("--model", type=str, default=None) no need, model selection only based on config
    # more detailed adjustment able in config file
    p.add_argument("--is_mi_based_edge", type=int, default=None)
    p.add_argument("--edge_cache_path", type=str, default=None)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--learning_rate", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--decision_threshold", type=float, default=None)
    p.add_argument("--binary", type=int, default=None)
    p.add_argument(
        "--los_emb",
        type=str,
        choices=["embedding", "nn_embedding", "hybrid_ordinal"],
        default=None,
    )
    p.add_argument("--cv", type=lambda x: x.lower() not in ("false", "0", "no"), default=None)
    p.add_argument("--fold", type=int, default=None)
    p.add_argument("--cv_run_dir", type=str, default=None)
    p.add_argument("--resume_fold_from_last", action="store_true")
    p.add_argument("--prepare_cv_only", action="store_true")
    p.add_argument("--finalize_cv", action="store_true")
    p.add_argument("--outcome_aware_single_run", action="store_true")
    p.add_argument("--outcome_aware_stage2_only", action="store_true")
    p.add_argument("--source_run_dir", type=str, default=None)
    p.add_argument("--resume_run_dir", type=str, default=None)
    p.add_argument("--stage2_run_dir", type=str, default=None)
    p.add_argument("--stage2_lambda_aux", type=float, default=None)
    return p.parse_args()


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def override_cfg(cfg: dict, args) -> dict:
    ensure_model_forecast_defaults(cfg)
    if args.device is not None:
        cfg["device"] = args.device
    if args.is_mi_based_edge is not None:
        cfg.setdefault("edge", {})["is_mi_based"] = bool(args.is_mi_based_edge)
    if args.edge_cache_path is not None:
        cfg.setdefault("edge", {})["cache_path"] = str(args.edge_cache_path)
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        cfg.setdefault("train", {})["learning_rate"] = args.learning_rate
    if args.epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.epochs
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.binary is not None:
        cfg.setdefault("train", {})["binary"] = bool(args.binary)
    if args.los_emb is not None:
        cfg.setdefault("model", {}).setdefault("params", {})["los_emb"] = str(args.los_emb)
    if args.decision_threshold is not None:
        cfg.setdefault("train", {})["decision_threshold"] = args.decision_threshold
    if args.cv is not None:
        cfg.setdefault("train", {})["cv"] = args.cv
    if args.stage2_lambda_aux is not None:
        cfg.setdefault("joint_forecast_pipeline", {}).setdefault("stage2", {})[
            "lambda_aux"
        ] = float(args.stage2_lambda_aux)
    return cfg


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    cfg = override_cfg(cfg, args)
    if bool(cfg.get("forecasted_pipeline", {}).get("enabled", False)) and bool(
        cfg.get("joint_forecast_pipeline", {}).get("enabled", False)
    ):
        raise ValueError(
            "forecasted_pipeline.enabled and joint_forecast_pipeline.enabled cannot both be true."
        )

    if args.prepare_cv_only:
        prepared = prepare_kfold_run(cfg, root, cv_run_dir=args.cv_run_dir)
        print(f"CV_RUN_DIR={prepared['cv_dir']}")
    elif args.finalize_cv:
        if args.cv_run_dir is None:
            raise ValueError("--finalize_cv requires --cv_run_dir")
        summary = finalize_kfold_summary(args.cv_run_dir)
        print(
            f"CV_SUMMARY_STATUS={summary['status']} completed_folds={summary['completed_folds']}"
        )
    elif args.outcome_aware_single_run:
        result = run_outcome_aware_single_run(cfg, root, resume_run_dir=args.resume_run_dir)
        print(f"SINGLE_RUN_DIR={result['run_dir']}")
        print(
            "SINGLE_RUN_RESULT "
            f"valid_auc={float(result['stage2_valid_auc']):.6f} "
            f"test_auc={float(result['stage2_test_auc']):.6f}"
        )
    elif args.outcome_aware_stage2_only:
        result = run_outcome_aware_stage2_only(
            cfg,
            root,
            fold=0 if args.fold is None else args.fold,
            source_run_dir=args.source_run_dir,
            stage2_run_dir=args.stage2_run_dir,
        )
        print(f"STAGE2_RUN_DIR={result['run_dir']}")
        print(
            "STAGE2_RESULT "
            f"valid_auc={float(result['stage2_valid_auc']):.6f} "
            f"test_auc={float(result['stage2_test_auc']):.6f}"
        )
    elif args.fold is not None:
        if args.cv_run_dir is None:
            raise ValueError("--fold requires --cv_run_dir")
        run_single_fold(
            cfg,
            root,
            fold=args.fold,
            cv_run_dir=args.cv_run_dir,
            resume_from_last=args.resume_fold_from_last,
        )
    elif bool(cfg.get("train", {}).get("cv", False)):
        run_kfold_experiment(cfg, root)
    else:
        run_single_experiment(cfg, root)


if __name__ == "__main__":
    main()
