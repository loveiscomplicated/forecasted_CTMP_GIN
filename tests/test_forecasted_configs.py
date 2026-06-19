from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml

from src.main import override_cfg
from src.trainers.forecasted_pipeline import _default_joint_stage1_cfg


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: str) -> dict:
    with (REPO_ROOT / path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _main_override_args(**overrides):
    defaults = {
        "device": None,
        "is_mi_based_edge": None,
        "edge_cache_path": None,
        "batch_size": None,
        "learning_rate": None,
        "epochs": None,
        "seed": None,
        "binary": None,
        "los_emb": None,
        "decision_threshold": None,
        "cv": None,
        "stage2_lambda_aux": None,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_stage2_lambda_aux_cli_override_updates_stage2_cfg() -> None:
    cfg = {
        "model": {"name": "ctmp_gin", "params": {}},
        "joint_forecast_pipeline": {"stage2": {"lambda_aux": 0.1}},
    }

    updated = override_cfg(
        cfg,
        _main_override_args(stage2_lambda_aux=0.03),
    )

    assert updated["joint_forecast_pipeline"]["stage2"]["lambda_aux"] == 0.03


def test_gin_forecasted_leakage_free_config_uses_pipeline_and_hard_los() -> None:
    cfg = _load_yaml("configs/gin_forecast_discharge_los_ce_baseline_leakage_free.yaml")

    assert cfg["model"]["name"] == "gin"
    assert cfg["train"]["cv"] is True
    assert cfg["forecasted_pipeline"]["enabled"] is True
    assert cfg["forecasted_pipeline"]["train_prediction_mode"] == "oof"
    assert cfg["forecasted_discharge"]["enabled"] is True
    assert cfg["forecasted_los"]["enabled"] is True
    assert cfg["forecasted_los"]["return_type"] == "hard"
    assert cfg["forecasted_los"]["target_mode"] == "coarse"


def test_a3tgcn_forecasted_leakage_free_config_uses_pipeline_and_hard_los() -> None:
    cfg = _load_yaml("configs/a3tgcn_forecast_discharge_los_ce_baseline_leakage_free.yaml")

    assert cfg["model"]["name"] == "a3tgcn"
    assert cfg["train"]["cv"] is True
    assert cfg["forecasted_pipeline"]["enabled"] is True
    assert cfg["forecasted_pipeline"]["train_prediction_mode"] == "oof"
    assert cfg["forecasted_discharge"]["enabled"] is True
    assert cfg["forecasted_los"]["enabled"] is True
    assert cfg["forecasted_los"]["return_type"] == "hard"
    assert cfg["forecasted_los"]["target_mode"] == "coarse"


def test_ctmp_distribution_config_declares_soft_discharge_defaults() -> None:
    cfg = _load_yaml("configs/ctmp_gin_forecast_discharge_los_ce_distribution_leakage_free.yaml")

    assert cfg["forecasted_discharge"]["mode"] == "hard"
    assert cfg["forecasted_discharge"]["soft_discharge"]["enabled"] is False
    assert cfg["forecasted_discharge"]["soft_discharge"]["heads"] == "all"
    assert cfg["forecasted_discharge"]["soft_discharge"]["temperature"] == 1.0
    assert cfg["model"]["params"]["forecast_input_encoder"] == "distribution"
    assert cfg["model"]["params"]["distribution_encoder_hidden_dim"] == 64


def test_ctmp_soft_discharge_short_configs_are_present() -> None:
    soft_all = _load_yaml("configs/ctmp_gin_forecast_discharge_softd_all_los_ce_distribution_leakage_free.yaml")
    soft_top3 = _load_yaml("configs/ctmp_gin_forecast_discharge_softd_top3_los_ce_distribution_leakage_free.yaml")

    assert soft_all["forecasted_discharge"]["mode"] == "soft"
    assert soft_all["forecasted_discharge"]["soft_discharge"]["enabled"] is True
    assert soft_all["forecasted_discharge"]["soft_discharge"]["heads"] == "all"
    assert soft_all["forecasted_los"]["return_type"] == "distribution"
    assert soft_all["model"]["params"]["forecast_input_encoder"] == "distribution"

    assert soft_top3["forecasted_discharge"]["mode"] == "mixed"
    assert soft_top3["forecasted_discharge"]["soft_discharge"]["enabled"] is True
    assert soft_top3["forecasted_discharge"]["soft_discharge"]["heads"] == [
        "SERVICES_D",
        "FREQ_ATND_SELF_HELP_D",
        "SUB1_D",
    ]
    assert soft_top3["forecasted_los"]["return_type"] == "distribution"
    assert soft_top3["model"]["params"]["forecast_input_encoder"] == "distribution"


def test_ctmp_joint_forecast_distribution_config_matches_reference_joint_stage1() -> None:
    cfg = _load_yaml("configs/ctmp_gin_joint_forecast_distribution_leakage_free.yaml")

    assert cfg["joint_forecast_pipeline"]["enabled"] is True
    assert cfg["joint_forecast_pipeline"]["joint_forecast_input"]["mode"] == "distribution"
    assert cfg["joint_forecast_pipeline"]["stage1"]["train"]["batch_size"] == 1024
    assert cfg["joint_forecast_pipeline"]["stage1"]["train"]["learning_rate"] == 0.001
    assert cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["joint_direction"] == "los_to_d"
    assert cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["detach_condition"] is True
    assert cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["los_target_mode"] == "coarse"
    assert cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["lambda_joint"] == 0.0
    assert cfg["model"]["params"]["forecast_input_encoder"] == "distribution"
    assert cfg["train"]["cv"] is True


def test_default_joint_stage1_cfg_exposes_disabled_struct_loss_block() -> None:
    cfg = _default_joint_stage1_cfg({"train": {"seed": 1, "n_folds": 5}})

    assert cfg["joint_predictor"]["joint_heads"] == "SERVICES_D,SUB1_D,FREQ_ATND_SELF_HELP_D"
    assert cfg["joint_struct_loss"]["enabled"] is False
    assert cfg["joint_struct_loss"]["risk_head_set"] == "new_dvD_top3"
    assert cfg["joint_struct_loss"]["lambda_struct"] == 0.0


def test_structured_loss_example_configs_are_present() -> None:
    top3_001 = _load_yaml("configs/experiments/joint_predictor_struct_dvD_top3_lambda001.yaml")
    top3_003 = _load_yaml("configs/experiments/joint_predictor_struct_dvD_top3_lambda003.yaml")
    top6_001 = _load_yaml("configs/experiments/joint_predictor_struct_dvD_top6_lambda001.yaml")
    robust_001 = _load_yaml("configs/experiments/joint_predictor_struct_robust_top3_lambda001.yaml")
    robust_003 = _load_yaml("configs/experiments/joint_predictor_struct_robust_top3_lambda003.yaml")

    assert top3_001["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["joint_heads"] == "new_dvD_top3"
    assert top3_001["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["enabled"] is True
    assert top3_001["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["lambda_struct"] == 0.01

    assert top3_003["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["lambda_struct"] == 0.03
    assert top6_001["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["joint_heads"] == "new_dvD_top6"
    assert top6_001["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["risk_head_set"] == "new_dvD_top6"
    assert robust_001["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["joint_heads"] == "new_robust_top3"
    assert robust_001["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["risk_head_set"] == "new_robust_top3"
    assert robust_001["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["lambda_struct"] == 0.01
    assert robust_003["joint_forecast_pipeline"]["stage1"]["joint_struct_loss"]["lambda_struct"] == 0.03


def test_ctmp_joint_generative_config_uses_conservative_kl_default() -> None:
    cfg = _load_yaml("configs/ctmp_gin_joint_generative_coarse_distribution.yaml")

    stage1 = cfg["joint_forecast_pipeline"]["stage1"]
    assert stage1["joint_predictor"]["predictor_type"] == "joint_generative"
    assert stage1["joint_predictor"]["prior_recon_weight"] == 0.5
    assert stage1["joint_predictor"]["beta_kl_max"] == 0.001
    assert stage1["model"]["params"]["latent_dim"] == 64
    assert stage1["model"]["params"]["los_context_dim"] == 32
    assert cfg["joint_forecast_pipeline"]["joint_forecast_input"]["mode"] == "distribution"
    assert cfg["model"]["params"]["forecast_input_encoder"] == "distribution"


def test_ctmp_outcome_aware_stage2_config_defaults_to_variant_a() -> None:
    cfg = _load_yaml("configs/ctmp_gin_joint_generative_outcome_aware_coarse_distribution.yaml")

    stage2 = cfg["joint_forecast_pipeline"]["stage2"]
    joint_input = cfg["joint_forecast_pipeline"]["joint_forecast_input"]
    assert stage2["enabled"] is True
    assert stage2["mode"] == "outcome_aware"
    assert "20260528-232514__ctmp_gin_joint_generative_coarse_distribution" in stage2["source_run_dir"]
    assert stage2["freeze_ctmp_gin"] is True
    assert stage2["freeze_ctmp_gin_backbone"] is True
    assert stage2["train_gated_fusion"] is False
    assert stage2["train_classifier"] is False
    assert stage2["train_predictor"] is True
    assert stage2["lambda_aux"] == 0.1
    assert stage2["selection_metric"] == "valid_auc"
    assert stage2["sample_prior_in_train"] is False
    assert "20260528-232514__ctmp_gin_joint_generative_coarse_distribution" in joint_input["source_run_dir"]


def test_ctmp_outcome_aware_single_run_config_trains_from_scratch() -> None:
    cfg = _load_yaml("configs/ctmp_gin_joint_generative_outcome_aware_single_run.yaml")

    stage2 = cfg["joint_forecast_pipeline"]["stage2"]
    joint_input = cfg["joint_forecast_pipeline"]["joint_forecast_input"]
    assert cfg["train"]["cv"] is False
    assert cfg["joint_forecast_pipeline"]["enabled"] is True
    assert cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["predictor_type"] == "joint_generative"
    assert stage2["enabled"] is True
    assert stage2["mode"] == "outcome_aware"
    assert "source_run_dir" not in stage2
    assert joint_input["mode"] == "distribution"
    assert "source_run_dir" not in joint_input


def test_gin_outcome_aware_single_run_config_trains_from_scratch() -> None:
    cfg = _load_yaml("configs/gin_forecast_joint_generative_outcome_aware_single_run.yaml")

    stage2 = cfg["joint_forecast_pipeline"]["stage2"]
    joint_input = cfg["joint_forecast_pipeline"]["joint_forecast_input"]
    assert cfg["model"]["name"] == "gin"
    assert cfg["train"]["cv"] is True
    assert cfg["joint_forecast_pipeline"]["enabled"] is True
    assert cfg["joint_forecast_pipeline"]["stage1"]["joint_predictor"]["predictor_type"] == "joint_generative"
    assert stage2["enabled"] is True
    assert stage2["mode"] == "outcome_aware"
    assert "source_run_dir" not in stage2
    assert joint_input["mode"] == "distribution"
    assert "source_run_dir" not in joint_input
