import os
import copy
import optuna
import yaml
import random
import torch
import numpy as np
import argparse
from src.models.forecast_inputs import ensure_model_forecast_defaults
from src.trainers.run_single_experiment import run_single_experiment
from src.utils.backup_postgresql import backup_to_sql
from src.utils.send_message import send_discord_message
from scripts.request_mi import request_mi
from typing import Optional


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--init-only", action="store_true", help="Initialize DB and exit")
    p.add_argument("--study-name", type=str, default=None)
    p.add_argument("--n-trials", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    return p.parse_args()


def suggest_ctmp_gin_params(trial, cfg):
    ensure_model_forecast_defaults(cfg)
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical(
        "embedding_dim", [16, 32, 64]
    )
    cfg["model"]["params"]["los_embedding_dim"] = trial.suggest_categorical(
        "los_embedding_dim", [4, 8, 16]
    )

    cfg["model"]["params"]["gin_hidden_channel"] = trial.suggest_categorical(
        "gin_hidden_channel", [16, 32, 64, 96]
    )
    cfg["model"]["params"]["gin_hidden_channel_2"] = trial.suggest_categorical(
        "gin_hidden_channel_2", [16, 32, 64, 96]
    )

    cfg["model"]["params"]["gin_1_layers"] = trial.suggest_int("gin_1_layers", 1, 3)
    cfg["model"]["params"]["gin_2_layers"] = trial.suggest_int("gin_2_layers", 1, 3)

    cfg["model"]["params"]["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical(
        "train_eps", [True, False]
    )
    cfg["model"]["params"]["gate_hidden_ch"] = trial.suggest_categorical(
        "gate_hidden_ch", [None, 64, 128, 256]
    )

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical(
        "threshold", [0.0, 0.005, 0.01, 0.02]
    )
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical(
        "pruning_ratio", [0.0, 0.3, 0.5, 0.7]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    cfg["train"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw"]
    )
    # cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


def suggest_gin_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical(
        "embedding_dim", [16, 32, 64]
    )
    cfg["model"]["params"]["gin_dim"] = trial.suggest_categorical(
        "gin_dim", [16, 32, 64, 96]
    )

    cfg["model"]["params"]["gin_layer_num"] = trial.suggest_int("gin_layer_num", 1, 6)

    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical(
        "train_eps", [True, False]
    )

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical(
        "threshold", [0.0, 0.005, 0.01, 0.02]
    )
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical(
        "pruning_ratio", [0.0, 0.3, 0.5, 0.7]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    cfg["train"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw"]
    )
    # cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


def suggest_a3tgcn_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical(
        "embedding_dim", [16, 32, 64]
    )
    cfg["model"]["params"]["hidden_channel"] = trial.suggest_categorical(
        "hidden_channel", [16, 32, 64, 96]
    )

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical(
        "threshold", [0.0, 0.005, 0.01, 0.02]
    )
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical(
        "pruning_ratio", [0.0, 0.3, 0.5, 0.7]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    cfg["train"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw"]
    )
    # cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


def suggest_a3tgcn_2_points_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical(
        "embedding_dim", [16, 32, 64]
    )
    cfg["model"]["params"]["hidden_channel"] = trial.suggest_categorical(
        "hidden_channel", [16, 32, 64, 96]
    )

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical(
        "threshold", [0.0, 0.005, 0.01, 0.02]
    )
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical(
        "pruning_ratio", [0.0, 0.3, 0.5, 0.7]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    cfg["train"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw"]
    )
    # cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


def suggest_gin_gru_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical(
        "embedding_dim", [16, 32, 64]
    )

    cfg["model"]["params"]["gin_hidden_channel"] = trial.suggest_categorical(
        "gin_hidden_channel", [16, 32, 64, 96]
    )
    cfg["model"]["params"]["gin_layers"] = trial.suggest_int("gin_layers", 1, 6)
    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical(
        "train_eps", [True, False]
    )
    cfg["model"]["params"]["gru_hidden_channel"] = trial.suggest_categorical(
        "gru_hidden_channel", [16, 32, 64, 96]
    )
    cfg["model"]["params"]["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical(
        "threshold", [0.0, 0.005, 0.01, 0.02]
    )
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical(
        "pruning_ratio", [0.0, 0.3, 0.5, 0.7]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    cfg["train"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw"]
    )
    # cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


def suggest_gin_gru_2_points_params(trial, cfg):
    cfg["model"]["params"]["embedding_dim"] = trial.suggest_categorical(
        "embedding_dim", [16, 32, 64]
    )

    cfg["model"]["params"]["gin_hidden_channel"] = trial.suggest_categorical(
        "gin_hidden_channel", [16, 32, 64, 96]
    )
    cfg["model"]["params"]["gin_layers"] = trial.suggest_int("gin_layers", 1, 6)
    cfg["model"]["params"]["train_eps"] = trial.suggest_categorical(
        "train_eps", [True, False]
    )
    cfg["model"]["params"]["gru_hidden_channel"] = trial.suggest_categorical(
        "gru_hidden_channel", [16, 32, 64, 96]
    )
    cfg["model"]["params"]["dropout_p"] = trial.suggest_float("dropout_p", 0.0, 0.5)
    cfg["model"]["params"]["gin_layer_out_dropout_p"] = trial.suggest_float(
        "gin_layer_out_dropout_p", 0.0, 0.5
    )
    cfg["model"]["params"]["gru_layer_out_dropout_p"] = trial.suggest_float(
        "gru_layer_out_dropout_p", 0.0, 0.5
    )

    cfg["edge"]["n_neighbors"] = trial.suggest_categorical("n_neighbors", [1, 3, 5, 7])
    cfg["edge"]["top_k"] = trial.suggest_categorical("top_k", [3, 6, 9, 12])
    cfg["edge"]["threshold"] = trial.suggest_categorical(
        "threshold", [0.0, 0.005, 0.01, 0.02]
    )
    cfg["edge"]["pruning_ratio"] = trial.suggest_categorical(
        "pruning_ratio", [0.0, 0.3, 0.5, 0.7]
    )

    cfg["train"]["batch_size"] = trial.suggest_categorical(
        "batch_size", [32, 64, 128, 256, 512]
    )
    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-4, 3e-3, log=True
    )
    cfg["train"]["weight_decay"] = trial.suggest_float(
        "weight_decay", 1e-6, 5e-4, log=True
    )
    cfg["train"]["optimizer"] = trial.suggest_categorical(
        "optimizer", ["adam", "adamw"]
    )
    # cfg["train"]["lr_scheduler_patience"] = trial.suggest_categorical("lr_scheduler_patience", [2, 5, 8])
    # cfg["train"]["early_stopping_patience"] = trial.suggest_categorical("early_stopping_patience", [8, 12, 16])


def suggest_xgboost_params(trial, cfg):
    cfg["train"]["n_estimators"] = trial.suggest_int(
        "n_estimators", 800, 8000, step=200
    )
    cfg["train"]["max_depth"] = trial.suggest_int("max_depth", 3, 12)
    cfg["train"]["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 20)

    cfg["train"]["learning_rate"] = trial.suggest_float(
        "learning_rate", 1e-3, 0.2, log=True
    )
    cfg["train"]["gamma"] = trial.suggest_float("gamma", 0.0, 5.0)

    cfg["train"]["subsample"] = trial.suggest_float("subsample", 0.6, 1.0)
    cfg["train"]["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.6, 1.0)
    cfg["train"]["colsample_bylevel"] = trial.suggest_float(
        "colsample_bylevel", 0.6, 1.0
    )
    cfg["train"]["colsample_bynode"] = trial.suggest_float("colsample_bynode", 0.6, 1.0)

    cfg["train"]["reg_alpha"] = trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True)
    cfg["train"]["reg_lambda"] = trial.suggest_float("reg_lambda", 1e-8, 50.0, log=True)

    cfg["train"]["max_leaves"] = trial.suggest_int(
        "max_leaves", 0, 256, step=16
    )  # 0이면 비활성
    if cfg["train"]["max_leaves"] == 0:
        cfg["train"].pop("max_leaves", None)


PARAM_SUGGESTORS = {
    "ctmp_gin": suggest_ctmp_gin_params,
    "gin": suggest_gin_params,
    "a3tgcn": suggest_a3tgcn_params,
    "a3tgcn_2_points": suggest_a3tgcn_2_points_params,
    "gin_gru": suggest_gin_gru_params,
    "xgboost": suggest_xgboost_params,
    "gin_gru_2_points": suggest_gin_gru_2_points_params,
}


def load_cfg(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ensure_model_forecast_defaults(cfg)
    return cfg


def objective_factory(
    base_cfg,
    root,
    report_metric="valid_auc",
    objective_seeds=(1,),
    bot_name="optuna_worker",
    epochs: int = 50,
):
    # split_seed: Optuna 전 trial에서 동일한 train/val/test split을 보장하는 고정 seed
    SPLIT_SEED = 42

    def objective(trial: optuna.Trial):
        # trial별 고유 seed: model 초기화, dropout 등에만 적용 (split에는 무관)
        trial_seed = 10000 + trial.number

        cfg = copy.deepcopy(base_cfg)
        model_name = cfg["model"]["name"]

        if model_name in PARAM_SUGGESTORS:
            PARAM_SUGGESTORS[model_name](trial, cfg)
        else:
            raise ValueError(f"No suggestor registered for model: {model_name}")

        scores = []
        for seed in objective_seeds:
            cfg_s = copy.deepcopy(cfg)
            cfg_s["train"]["seed"] = int(seed)
            cfg_s["train"][
                "split_seed"
            ] = SPLIT_SEED  # 모든 trial에서 동일한 split 보장
            cfg_s["train"]["epochs"] = epochs  # --epochs 인자로 config 값 override

            try:
                print(
                    f"[Trial {trial.number}] requesting MI (seed={seed}, n_neighbors={cfg_s['edge']['n_neighbors']})..."
                )
                mi_edge_path = request_mi(
                    mode="single",
                    fold=None,
                    seed=seed,
                    cfg=cfg_s,
                    n_neighbors=cfg_s["edge"]["n_neighbors"],
                    verbose_poll=True,
                )
                out = run_single_experiment(
                    cfg_s,
                    root=root,
                    trial=trial,
                    report_metric=report_metric,
                    mi_cache_path=mi_edge_path,
                    model_seed=trial_seed,  # split과 독립적으로 trial별 model seed 적용
                )

                if model_name == "xgboost":
                    score = float(out["roc_auc"])
                else:
                    score = float(out["best_valid_metric"])

                if (score is None) or (not np.isfinite(score)):
                    raise optuna.TrialPruned()

                scores.append(score)

            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"[Trial {trial.number}] failed:", repr(e))
                try:
                    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
                    send_discord_message(
                        f"[TRIAL FAIL] trial={trial.number} gpu={gpu_id} seed={seed}\n{repr(e)}",
                        bot_name=bot_name,
                    )
                except Exception:
                    pass
                raise optuna.TrialPruned()

        return float(sum(scores) / len(scores))

    return objective


def run_optuna(
    config_path: str,
    root: str,
    n_trials: int = 50,
    epochs: int = 50,
    study_name: Optional[str] = None,
    db="postgresql",
):

    os.makedirs("runs", exist_ok=True)

    config_path = os.path.abspath(config_path)
    root = os.path.abspath(root)

    base_cfg = load_cfg(config_path)

    sampler = optuna.samplers.TPESampler(seed=42, multivariate=True)
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    pruner = optuna.pruners.HyperbandPruner(
        min_resource=5, max_resource=epochs, reduction_factor=3
    )  # aggressive pruning

    model_name = base_cfg["model"]["name"]
    # [수정 부분] storage 분기 처리
    if db == "postgresql":
        storage = "postgresql+psycopg2://optuna:optuna_pw@127.0.0.1:5432/optuna_db"
    else:
        # 현재 경로에 sqlite db 파일을 생성합니다.
        storage = f"sqlite:///runs/{model_name}_optuna.db"

    study_name = study_name or model_name
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
    )

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "all")
    objective = objective_factory(
        base_cfg=base_cfg,
        root=root,
        report_metric="valid_auc",
        objective_seeds=(1,),
        bot_name=f"optuna_{study_name}_gpu{gpu_id}",
        epochs=epochs,
    )

    print(
        f"[Worker GPU={gpu_id}] study={study.study_name}  model={model_name}  n_trials={n_trials}"
    )
    study.optimize(
        objective, n_trials=n_trials, show_progress_bar=True, gc_after_trial=True
    )

    # 결과 CSV 저장
    safe = study.study_name.replace("/", "_")
    csv_dir = f"runs/optuna_logs/{safe}"
    os.makedirs(csv_dir, exist_ok=True)
    study.trials_dataframe().to_csv(f"{csv_dir}/{safe}_optuna_trials.csv", index=False)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(
            "[WARNING] No trials completed — all were pruned or failed. Check request_mi / rclone setup."
        )
        return study

    print("best value:", study.best_value)
    print("best params:", study.best_params)

    return study


if __name__ == "__main__":
    args = parse_args()

    cur_dir = os.path.dirname(__file__)
    root = os.path.join(cur_dir, "..", "data")
    root = os.path.abspath(root)
    config_path = os.path.abspath(args.config)

    # 1. 모델명 미리 파악 (백업 파일명용)
    base_cfg = load_cfg(config_path)
    model_name = base_cfg["model"]["name"]
    db = "postgresql"
    try:
        if args.init_only:
            # init 시에도 동일한 storage 논리 적용
            storage_url = (
                "postgresql+psycopg2://optuna:optuna_pw@127.0.0.1:5432/optuna_db"
                if db == "postgresql"
                else f"sqlite:///runs/{model_name}_optuna.db"
            )

            print(f"[*] Initializing study: {args.study_name or model_name} on {db}")
            optuna.create_study(
                study_name=args.study_name or model_name,
                storage=storage_url,
                direction="maximize",
                load_if_exists=True,
            )
            print("[+] Initialization complete.")
        else:
            run_optuna(
                config_path=config_path,
                root=root,
                n_trials=args.n_trials or 50,
                epochs=args.epochs or 50,
                study_name=args.study_name,
                db=db,  # 파라미터 전달
            )
    finally:
        if not args.init_only:
            print("\n--- Starting Automatic Backup ---")
            backup_to_sql(model_name=model_name)
