from __future__ import annotations

import copy
import json

import pytest
import torch

from src.data_processing.canonical_teds import build_canonical_teds_bundle
from src.models.ctmp_gin.model import CTMPGIN
from src.models.discharge_predictor.joint_generative_predictor import (
    JointGenerativePredictor,
)
from src.models.forecasted_ctmp_gin import (
    CANONICAL_JOINT_FORECAST_HEADS,
    OutcomeAwareForecastedCTMPGIN,
    OutcomeAwareForecastedGIN,
    SoftDischargeContract,
    SoftDischargeHeadContract,
    resolve_joint_forecast_contract,
)
from src.models.gin import GIN
from src.trainers.outcome_aware_stage2 import (
    _apply_ctmp_gin_freeze_policy,
    _build_optimizer,
    _compare_joint_cache_to_live_prior,
    _count_trainable_params,
    _joint_cache_path_for_split,
    _resolve_predictor_admission_col_indices,
    _set_stage2_module_modes,
    resolve_stage2_pretrained_paths,
)


def _build_contract() -> SoftDischargeContract:
    return SoftDischargeContract(
        head_names=("SERVICES_D", "SUB1_D"),
        heads=(
            SoftDischargeHeadContract(
                name="SERVICES_D", target_col_idx=2, num_classes=2
            ),
            SoftDischargeHeadContract(
                name="SUB1_D", target_col_idx=3, num_classes=3
            ),
        ),
    )


def _build_predictor() -> JointGenerativePredictor:
    return JointGenerativePredictor(
        ad_col_dims=[3, 4],
        target_col_names=["SERVICES_D", "SUB1_D"],
        target_col_dims=[2, 3],
        los_num_classes=6,
        input_encoding="onehot",
        hidden_dim=8,
        latent_dim=4,
        los_context_dim=5,
        num_layers=1,
        dropout=0.0,
        target_embedding_dim=3,
    )


def _build_ctmp_gin() -> CTMPGIN:
    return CTMPGIN(
        col_info=(
            ["AD0", "AD1", "SERVICES_D", "SUB1_D"],
            [3, 4, 2, 3],
            [0, 1],
            [2, 3],
        ),
        embedding_dim=5,
        gin_hidden_channel=8,
        gin_1_layers=1,
        gin_hidden_channel_2=8,
        gin_2_layers=1,
        num_classes=2,
        dropout_p=0.0,
        los_embedding_dim=4,
        max_los=37,
        readout_mode="last",
        forecast_input_encoder="distribution",
        distribution_encoder_hidden_dim=6,
        distribution_encoder_out_dim=7,
    )


def _build_ctmp_gin_with_extra_discharge_col() -> CTMPGIN:
    return CTMPGIN(
        col_info=(
            ["AD0", "AD1", "SERVICES_D", "SUB1_D", "EXTRA_D"],
            [3, 4, 2, 3, 2],
            [0, 1],
            [2, 3, 4],
        ),
        embedding_dim=5,
        gin_hidden_channel=8,
        gin_1_layers=1,
        gin_hidden_channel_2=8,
        gin_2_layers=1,
        num_classes=2,
        dropout_p=0.0,
        los_embedding_dim=4,
        max_los=37,
        readout_mode="last",
        forecast_input_encoder="distribution",
        distribution_encoder_hidden_dim=6,
        distribution_encoder_out_dim=7,
    )


def _build_gin() -> GIN:
    return GIN(
        embedding_dim=5,
        col_info=(
            ["AD0", "AD1", "SERVICES_D", "SUB1_D"],
            [3, 4, 2, 3],
            [0, 1],
            [2, 3],
        ),
        gin_dim=8,
        gin_layer_num=1,
        num_classes=2,
        train_eps=True,
        forecast_input_encoder="distribution",
        distribution_encoder_hidden_dim=6,
        distribution_encoder_out_dim=7,
    )


def _build_wrapper(*, placeholder: int = 0) -> OutcomeAwareForecastedCTMPGIN:
    return OutcomeAwareForecastedCTMPGIN(
        predictor=_build_predictor(),
        ctmp_gin=_build_ctmp_gin(),
        contract=_build_contract(),
        admission_col_indices=[0, 1],
        discharge_col_indices=[2, 3],
        sample_prior_in_train=False,
        discharge_placeholder_index=placeholder,
    )


def _build_gin_wrapper(*, placeholder: int = 0) -> OutcomeAwareForecastedGIN:
    return OutcomeAwareForecastedGIN(
        predictor=_build_predictor(),
        gin=_build_gin(),
        contract=_build_contract(),
        admission_col_indices=[0, 1],
        discharge_col_indices=[2, 3],
        sample_prior_in_train=False,
        discharge_placeholder_index=placeholder,
    )


def test_outcome_aware_wrapper_forward_produces_reasonb_logits() -> None:
    wrapper = _build_wrapper()
    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 4, 5], [1, 0, 5, 4]], dtype=torch.long)

    out = wrapper(x, edge_index)

    assert out.reasonb_logits.shape == (2, 1)
    assert set(out.forecast_d_probs.keys()) == {"SERVICES_D", "SUB1_D"}
    assert out.forecast_los_probs.shape == (2, 37)


def test_outcome_aware_gin_wrapper_forward_produces_reasonb_logits() -> None:
    wrapper = _build_gin_wrapper()
    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    out = wrapper(x, edge_index)

    assert out.reasonb_logits.shape == (2, 1)
    assert set(out.forecast_d_probs.keys()) == {"SERVICES_D", "SUB1_D"}
    assert out.forecast_los_probs.shape == (2, 37)


def test_predictor_admission_indices_ignore_base_los_column() -> None:
    class BaseDataset:
        col_info = (
            ["AD0", "LOS", "AD1", "SERVICES_D"],
            [3, 37, 4, 2],
            [0, 1, 2],
            [3],
        )
        processed_tensor = torch.empty(2, 4)

    class JointDataset:
        ad_col_names = ["AD0", "AD1"]
        ad_col_dims = [3, 4]

    indices = _resolve_predictor_admission_col_indices(BaseDataset(), JointDataset())

    assert indices == [0, 1]


def test_predictor_admission_indices_reject_cardinality_mismatch() -> None:
    class BaseDataset:
        col_info = (
            ["AD0", "AD1", "SERVICES_D"],
            [3, 5, 2],
            [0, 1],
            [2],
        )
        processed_tensor = torch.empty(2, 4)

    class JointDataset:
        ad_col_names = ["AD0", "AD1"]
        ad_col_dims = [3, 4]

    with pytest.raises(ValueError, match="cardinalities do not match"):
        _resolve_predictor_admission_col_indices(BaseDataset(), JointDataset())


def test_reasonb_loss_gradients_flow_to_predictor_when_ctmp_is_frozen() -> None:
    wrapper = _build_wrapper()
    stage2_cfg = {
        "freeze_ctmp_gin": True,
        "freeze_ctmp_gin_backbone": True,
        "train_gated_fusion": False,
        "train_classifier": False,
        "train_predictor": True,
        "learning_rate": 1.0e-5,
        "weight_decay": 1.0e-5,
    }
    _apply_ctmp_gin_freeze_policy(wrapper, stage2_cfg)
    _set_stage2_module_modes(wrapper, stage2_cfg, is_training=True)
    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)
    y = torch.tensor([1.0, 0.0], dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 4, 5], [1, 0, 5, 4]], dtype=torch.long)

    out = wrapper(x, edge_index)
    assert out.reasonb_logits.requires_grad
    loss = torch.nn.BCEWithLogitsLoss()(out.reasonb_logits.squeeze(1), y)
    loss.backward()

    predictor_grad = sum(
        float(param.grad.abs().sum().item())
        for param in wrapper.predictor.parameters()
        if param.grad is not None
    )
    ctmp_grad = sum(
        float(param.grad.abs().sum().item())
        for param in wrapper.ctmp_gin.parameters()
        if param.grad is not None
    )
    assert predictor_grad > 0.0
    assert ctmp_grad == pytest.approx(0.0)


def test_placeholder_class_does_not_affect_output_when_soft_discharge_overrides_all_d_heads() -> None:
    wrapper_a = _build_wrapper(placeholder=0)
    wrapper_b = _build_wrapper(placeholder=1)
    wrapper_b.load_state_dict(copy.deepcopy(wrapper_a.state_dict()))
    wrapper_a.eval()
    wrapper_b.eval()
    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)
    edge_index = torch.tensor([[0, 1, 4, 5], [1, 0, 5, 4]], dtype=torch.long)

    with torch.no_grad():
        logits_a = wrapper_a(x, edge_index).reasonb_logits
        logits_b = wrapper_b(x, edge_index).reasonb_logits

    assert torch.allclose(logits_a, logits_b, atol=1.0e-6)


def test_stage2_placeholder_preserves_non_forecast_discharge_columns() -> None:
    wrapper = OutcomeAwareForecastedCTMPGIN(
        predictor=_build_predictor(),
        ctmp_gin=_build_ctmp_gin_with_extra_discharge_col(),
        contract=_build_contract(),
        admission_col_indices=[0, 1],
        discharge_col_indices=[2, 3, 4],
        sample_prior_in_train=False,
        discharge_placeholder_index=0,
    )
    x = torch.tensor([[0, 1, 1, 2, 1], [2, 3, 0, 1, 1]], dtype=torch.long)

    x_stage2 = wrapper._build_ctmp_input(x)

    assert torch.equal(x_stage2[:, 2], torch.zeros(2, dtype=torch.long))
    assert torch.equal(x_stage2[:, 3], torch.zeros(2, dtype=torch.long))
    assert torch.equal(x_stage2[:, 4], x[:, 4])


def test_joint_cache_live_prior_comparison_aligns_by_row_idx() -> None:
    cache_payload = {
        "row_idx": torch.tensor([20, 10], dtype=torch.long),
        "final_d_probs": {
            "SERVICES_D": torch.tensor(
                [[0.1, 0.9], [0.8, 0.2]],
                dtype=torch.float32,
            )
        },
        "final_los_probs": torch.tensor(
            [[0.2, 0.8], [0.7, 0.3]],
            dtype=torch.float32,
        ),
    }
    live_payload = {
        "row_idx": torch.tensor([10, 20], dtype=torch.long),
        "final_d_probs": {
            "SERVICES_D": torch.tensor(
                [[0.8, 0.2], [0.1, 0.9]],
                dtype=torch.float32,
            )
        },
        "prior_los_probs": torch.tensor(
            [[0.7, 0.3], [0.2, 0.8]],
            dtype=torch.float32,
        ),
    }

    result = _compare_joint_cache_to_live_prior(
        split_name="gnn_val",
        cache_payload=cache_payload,
        live_payload=live_payload,
        target_names=["SERVICES_D"],
    )

    assert result["d_heads"]["SERVICES_D"]["mean_abs_diff"] == pytest.approx(0.0)
    assert result["los"]["mean_abs_diff"] == pytest.approx(0.0)


def test_missing_optional_joint_cache_path_is_skipped(tmp_path) -> None:
    split_payload = {
        "joint_cache_paths": {
            "gnn_val": str(tmp_path / "missing_gnn_val.pt"),
        }
    }

    assert (
        _joint_cache_path_for_split(
            split_payload,
            fold_dir=str(tmp_path),
            split_name="gnn_val",
        )
        is None
    )


def test_variant_b_freeze_policy_trains_only_fusion_and_classifier() -> None:
    wrapper = _build_wrapper()
    stage2_cfg = {
        "freeze_ctmp_gin": False,
        "freeze_ctmp_gin_backbone": True,
        "train_gated_fusion": True,
        "train_classifier": True,
        "train_predictor": True,
        "learning_rate": 1.0e-5,
        "weight_decay": 1.0e-5,
    }
    _apply_ctmp_gin_freeze_policy(wrapper, stage2_cfg)
    _set_stage2_module_modes(wrapper, stage2_cfg, is_training=True)

    assert any(param.requires_grad for param in wrapper.predictor.parameters())
    assert all(param.requires_grad for param in wrapper.ctmp_gin.classifier_b.parameters())
    assert wrapper.ctmp_gin.classifier_b.training is True
    assert wrapper.ctmp_gin.gated_fusion is not None
    assert all(param.requires_grad for param in wrapper.ctmp_gin.gated_fusion.parameters())
    assert wrapper.ctmp_gin.gated_fusion.training is True
    frozen_modules = [
        wrapper.ctmp_gin.entity_embedding_layer,
        wrapper.ctmp_gin.distribution_encoders,
        wrapper.ctmp_gin.gin_1,
        wrapper.ctmp_gin.gin_2,
        wrapper.ctmp_gin.proj_ad,
        wrapper.ctmp_gin.proj_dis,
        wrapper.ctmp_gin.proj_merged,
    ]
    for module in frozen_modules:
        assert module.training is False
        assert all(not param.requires_grad for param in module.parameters())


def test_stage2_optimizer_uses_only_trainable_parameters() -> None:
    wrapper = _build_wrapper()
    stage2_cfg = {
        "freeze_ctmp_gin": True,
        "freeze_ctmp_gin_backbone": True,
        "train_gated_fusion": False,
        "train_classifier": False,
        "train_predictor": True,
        "learning_rate": 1.0e-5,
        "weight_decay": 1.0e-5,
    }
    _apply_ctmp_gin_freeze_policy(wrapper, stage2_cfg)

    optimizer = _build_optimizer(wrapper, stage2_cfg)
    optimizer_param_ids = {
        id(param) for group in optimizer.param_groups for param in group["params"]
    }
    trainable_param_ids = {
        id(param) for param in wrapper.parameters() if param.requires_grad
    }

    assert optimizer_param_ids == trainable_param_ids
    assert _count_trainable_params(wrapper.ctmp_gin) == 0
    assert _count_trainable_params(wrapper.predictor) > 0


def test_gin_freeze_policy_trains_only_classifier_and_predictor() -> None:
    wrapper = _build_gin_wrapper()
    stage2_cfg = {
        "freeze_ctmp_gin": False,
        "freeze_ctmp_gin_backbone": True,
        "train_gated_fusion": False,
        "train_classifier": True,
        "train_predictor": True,
        "learning_rate": 1.0e-5,
        "weight_decay": 1.0e-5,
    }
    _apply_ctmp_gin_freeze_policy(wrapper, stage2_cfg)
    _set_stage2_module_modes(wrapper, stage2_cfg, is_training=True)

    assert any(param.requires_grad for param in wrapper.predictor.parameters())
    assert all(param.requires_grad for param in wrapper.gin.classifier.parameters())
    assert wrapper.gin.classifier.training is True
    frozen_modules = [
        wrapper.gin.entity_embedding_layer,
        wrapper.gin.distribution_encoders,
        wrapper.gin.los_embedding_layer,
        wrapper.gin.los_distribution_encoder,
        wrapper.gin.gin_layers,
    ]
    for module in frozen_modules:
        assert module.training is False
        assert all(not param.requires_grad for param in module.parameters())


def test_gin_freeze_policy_rejects_train_gated_fusion() -> None:
    wrapper = _build_gin_wrapper()
    stage2_cfg = {
        "freeze_ctmp_gin": False,
        "freeze_ctmp_gin_backbone": True,
        "train_gated_fusion": True,
        "train_classifier": True,
        "train_predictor": True,
        "learning_rate": 1.0e-5,
        "weight_decay": 1.0e-5,
    }

    with pytest.raises(ValueError, match="does not support train_gated_fusion=true"):
        _apply_ctmp_gin_freeze_policy(wrapper, stage2_cfg)


def test_resolve_joint_forecast_contract_matches_current_canonical_12_head_order() -> None:
    bundle = build_canonical_teds_bundle(
        root="src/data",
        binary=True,
        ig_label=False,
        remove_los=True,
        do_preprocess=False,
        admission_only=False,
    )

    contract = resolve_joint_forecast_contract(
        bundle.col_info,
        list(bundle.discharge_target_col_names),
    )

    assert len(contract.head_names) == 12
    assert contract.head_names == CANONICAL_JOINT_FORECAST_HEADS


def test_resolve_stage2_pretrained_paths_prefers_source_run_dir(tmp_path) -> None:
    source_run_dir = tmp_path / "source_run"
    source_fold_dir = source_run_dir / "folds" / "fold_2"
    (source_fold_dir / "joint_predictor" / "checkpoints").mkdir(parents=True)
    (source_fold_dir / "checkpoints").mkdir(parents=True)
    predictor_ckpt = source_fold_dir / "joint_predictor" / "checkpoints" / "best.pt"
    baseline_ckpt = source_fold_dir / "checkpoints" / "best.pt"
    predictor_ckpt.write_bytes(b"")
    baseline_ckpt.write_bytes(b"")
    fold_result = {
        "best_valid_metrics": {"valid_auc": 0.88, "valid_f1": 0.77, "valid_acc": 0.8},
        "test_auc": 0.89,
        "test_f1": 0.78,
        "test_acc": 0.81,
    }
    (source_fold_dir / "fold_result.json").write_text(
        json.dumps(fold_result),
        encoding="utf-8",
    )

    predictor_path, baseline_path, metrics = resolve_stage2_pretrained_paths(
        fold=2,
        stage2_cfg={"source_run_dir": str(source_run_dir)},
        fallback_predictor_checkpoint_path="/tmp/fallback_predictor.pt",
        fallback_baseline_checkpoint_path="/tmp/fallback_baseline.pt",
        fallback_baseline_metrics={"baseline_valid_auc": 0.1, "baseline_test_auc": 0.2},
    )

    assert predictor_path == str(predictor_ckpt)
    assert baseline_path == str(baseline_ckpt)
    assert metrics["baseline_valid_auc"] == pytest.approx(0.88)
    assert metrics["baseline_test_auc"] == pytest.approx(0.89)


def test_resolve_stage2_pretrained_paths_supports_single_run_without_result_json(
    tmp_path,
) -> None:
    source_run_dir = tmp_path / "single_run"
    (source_run_dir / "joint_predictor" / "checkpoints").mkdir(parents=True)
    (source_run_dir / "checkpoints").mkdir(parents=True)
    (source_run_dir / "config.final.yaml").write_text("{}", encoding="utf-8")
    (source_run_dir / "single_run_splits.json").write_text("{}", encoding="utf-8")
    predictor_ckpt = source_run_dir / "joint_predictor" / "checkpoints" / "best.pt"
    baseline_ckpt = source_run_dir / "checkpoints" / "best.pt"
    predictor_ckpt.write_bytes(b"")
    baseline_ckpt.write_bytes(b"")

    predictor_path, baseline_path, metrics = resolve_stage2_pretrained_paths(
        fold=0,
        stage2_cfg={"source_run_dir": str(source_run_dir)},
        fallback_predictor_checkpoint_path="/tmp/fallback_predictor.pt",
        fallback_baseline_checkpoint_path="/tmp/fallback_baseline.pt",
        fallback_baseline_metrics={"baseline_valid_auc": 0.1, "baseline_test_auc": 0.2},
    )

    assert predictor_path == str(predictor_ckpt)
    assert baseline_path == str(baseline_ckpt)
    assert metrics["baseline_valid_auc"] == pytest.approx(0.1)
    assert metrics["baseline_test_auc"] == pytest.approx(0.2)
