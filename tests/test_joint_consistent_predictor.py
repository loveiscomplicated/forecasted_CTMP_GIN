from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn

from src.diagnostics.diagnose_joint_predictor_joint_stats import (
    _conditional_js_divergence,
    compute_joint_stats,
    parse_args as parse_joint_stats_args,
)
from src.models.discharge_predictor import (
    ExpectedCategoricalEmbedding,
    FixedOneHotBatchEncoder,
    JointConsistencyLoss,
    JointConsistentPredictor,
    MultiHeadExpectedEmbedding,
    SoftJointDriftLoss,
)
from src.models.discharge_predictor.joint_consistency_loss import (
    _conditional_distribution_from_targets,
)
from src.trainers.run_joint_consistent_predictor import (
    DEFAULT_JOINT_HEADS,
    _build_d_target_dict,
    _build_config_from_args,
    _generate_joint_drift_reports,
    _normalize_joint_struct_loss_cfg,
    parse_args as parse_joint_args,
)


def test_expected_categorical_embedding_matches_manual_expectation() -> None:
    module = ExpectedCategoricalEmbedding(num_classes=3, embedding_dim=2)
    with torch.no_grad():
        module.embedding.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]]))
    probs = torch.tensor([[0.25, 0.25, 0.50]], dtype=torch.float32)
    output = module(probs)
    expected = torch.tensor([[1.75, 1.0]], dtype=torch.float32)
    assert torch.allclose(output, expected, atol=1.0e-6)


def test_multi_head_expected_embedding_concatenates_selected_heads() -> None:
    module = MultiHeadExpectedEmbedding(
        head_dims={"A": 2, "B": 3},
        selected_heads=["A", "B"],
        embedding_dim=1,
    )
    with torch.no_grad():
        module.embedders["A"].embedding.weight.copy_(torch.tensor([[1.0], [3.0]]))
        module.embedders["B"].embedding.weight.copy_(torch.tensor([[0.0], [2.0], [4.0]]))
    output = module(
        {
            "A": torch.tensor([[0.5, 0.5]], dtype=torch.float32),
            "B": torch.tensor([[0.25, 0.25, 0.5]], dtype=torch.float32),
        }
    )
    assert output.shape == (1, 2)
    assert torch.allclose(output, torch.tensor([[2.0, 2.5]]), atol=1.0e-6)


@pytest.mark.parametrize("input_encoding", ["embedding", "onehot"])
@pytest.mark.parametrize("joint_direction", ["independent", "los_to_d", "d_to_los", "bidirectional"])
@pytest.mark.parametrize("los_num_classes", [6, 37])
def test_joint_consistent_predictor_output_shapes(
    input_encoding: str,
    joint_direction: str,
    los_num_classes: int,
) -> None:
    model = JointConsistentPredictor(
        ad_col_dims=[4, 5, 6],
        target_col_names=["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
        target_col_dims=[3, 4, 2],
        los_num_classes=los_num_classes,
        joint_direction=joint_direction,
        joint_heads=["SERVICES_D", "SUB1_D"],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        input_encoding=input_encoding,
    )
    x = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    y_d = {
        "SERVICES_D": torch.tensor([0, 1], dtype=torch.long),
        "SUB1_D": torch.tensor([1, 2], dtype=torch.long),
        "FREQ_ATND_SELF_HELP_D": torch.tensor([0, 1], dtype=torch.long),
    }
    y_los = torch.tensor([1, 2], dtype=torch.long)
    output = model(x, d_targets=y_d, los_targets=y_los, oracle_ratio=0.25)
    assert set(output.final_d_logits.keys()) == set(y_d.keys())
    assert set(output.base_d_logits.keys()) == set(y_d.keys())
    for head_name, cardinality in {"SERVICES_D": 3, "SUB1_D": 4, "FREQ_ATND_SELF_HELP_D": 2}.items():
        assert output.base_d_logits[head_name].shape == (2, cardinality)
        assert output.final_d_logits[head_name].shape == (2, cardinality)
    assert output.base_los_logits.shape == (2, los_num_classes)
    assert output.final_los_logits.shape == (2, los_num_classes)
    assert output.base_los_probs.shape == (2, los_num_classes)


def test_fixed_onehot_encoder_output_dimension_and_block_sums() -> None:
    encoder = FixedOneHotBatchEncoder(col_dims=[3, 4, 2])
    x = torch.tensor([[0, 1, 1], [2, 3, 0]], dtype=torch.long)

    output = encoder(x)

    assert output.shape == (2, 9)
    block_ranges = [(0, 3), (3, 7), (7, 9)]
    for start, end in block_ranges:
        block_sum = output[:, start:end].sum(dim=1)
        assert torch.allclose(block_sum, torch.ones_like(block_sum), atol=1.0e-6)


def test_onehot_mode_has_no_trainable_admission_embedding_parameters() -> None:
    model = JointConsistentPredictor(
        ad_col_dims=[4, 5, 6],
        target_col_names=["SERVICES_D"],
        target_col_dims=[3],
        los_num_classes=3,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        input_encoding="onehot",
    )

    assert isinstance(model.admission_encoder, FixedOneHotBatchEncoder)
    assert not any(
        isinstance(module, nn.Embedding)
        for module in model.admission_encoder.modules()
    )
    assert sum(param.numel() for param in model.admission_encoder.parameters()) == 0


def test_embedding_mode_backward_compatibility_uses_entity_embedding() -> None:
    model = JointConsistentPredictor(
        ad_col_dims=[4, 5, 6],
        target_col_names=["SERVICES_D"],
        target_col_dims=[3],
        los_num_classes=3,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        input_encoding="embedding",
    )

    assert model.input_encoding == "embedding"
    assert model.embedding_input_dim == 3 * 32
    assert model.actual_encoder_input_dim == model.embedding_input_dim
    assert any(isinstance(module, nn.Embedding) for module in model.admission_encoder.modules())


@pytest.mark.parametrize("input_encoding", ["embedding", "onehot"])
def test_detach_condition_toggles_gradient_flow(input_encoding: str) -> None:
    x = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    y_d = {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)}
    y_los = torch.tensor([1, 0], dtype=torch.long)
    model_detached = JointConsistentPredictor(
        ad_col_dims=[4, 5, 6],
        target_col_names=["SERVICES_D"],
        target_col_dims=[3],
        los_num_classes=3,
        joint_direction="los_to_d",
        detach_condition=True,
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        input_encoding=input_encoding,
    )
    out = model_detached(x, d_targets=y_d, los_targets=y_los)
    loss = out.final_d_logits["SERVICES_D"].sum()
    model_detached.zero_grad()
    loss.backward()
    detached_grad = model_detached.base_los_head.weight.grad
    assert detached_grad is None or torch.allclose(detached_grad, torch.zeros_like(model_detached.base_los_head.weight))

    model_attached = JointConsistentPredictor(
        ad_col_dims=[4, 5, 6],
        target_col_names=["SERVICES_D"],
        target_col_dims=[3],
        los_num_classes=3,
        joint_direction="los_to_d",
        detach_condition=False,
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
        input_encoding=input_encoding,
    )
    out = model_attached(x, d_targets=y_d, los_targets=y_los)
    loss = out.final_d_logits["SERVICES_D"].sum()
    model_attached.zero_grad()
    loss.backward()
    attached_grad = model_attached.base_los_head.weight.grad
    assert attached_grad is not None
    assert float(attached_grad.abs().sum()) > 0.0


def test_invalid_input_encoding_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unsupported input_encoding"):
        JointConsistentPredictor(
            ad_col_dims=[4, 5, 6],
            target_col_names=["SERVICES_D"],
            target_col_dims=[3],
            los_num_classes=3,
            input_encoding="bad_mode",
        )


def test_joint_consistency_loss_lambda_zero_is_noop_for_joint_term() -> None:
    model = JointConsistentPredictor(
        ad_col_dims=[3, 4],
        target_col_names=["SERVICES_D"],
        target_col_dims=[2],
        los_num_classes=3,
        joint_direction="bidirectional",
        joint_heads=["SERVICES_D"],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    x = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    y_d = {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)}
    y_los = torch.tensor([1, 2], dtype=torch.long)
    output = model(x, d_targets=y_d, los_targets=y_los)
    criterion = JointConsistencyLoss(lambda_joint=0.0, joint_head_names=["SERVICES_D"])
    total_loss, metrics = criterion(output, d_targets=y_d, los_targets=y_los)
    assert total_loss.ndim == 0
    assert metrics["loss_joint"] == pytest.approx(0.0)


def test_joint_consistency_loss_lambda_positive_accepts_cpu_joint_targets() -> None:
    model = JointConsistentPredictor(
        ad_col_dims=[3, 4],
        target_col_names=["SERVICES_D"],
        target_col_dims=[2],
        los_num_classes=3,
        joint_direction="bidirectional",
        joint_heads=["SERVICES_D"],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    x = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    y_d_cpu = {"SERVICES_D": torch.tensor([0, 1, 0], dtype=torch.long)}
    y_los_cpu = torch.tensor([1, 2, 0], dtype=torch.long)
    output = model(x, d_targets=y_d_cpu, los_targets=y_los_cpu)
    criterion = JointConsistencyLoss(lambda_joint=0.3, joint_head_names=["SERVICES_D"])

    total_loss, metrics = criterion(
        output,
        d_targets=y_d_cpu,
        los_targets=y_los_cpu,
        d_targets_for_joint=y_d_cpu,
        los_targets_for_joint=y_los_cpu,
    )

    assert total_loss.ndim == 0
    assert metrics["loss_joint"] > 0.0
    assert metrics["joint_consistency_terms"] == pytest.approx(2.0)
    assert "joint_js_d_given_los_SERVICES_D" in metrics
    assert "joint_js_los_given_d_SERVICES_D" in metrics


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS cross-device smoke test requires Apple MPS",
)
def test_joint_consistency_loss_mps_output_accepts_cpu_joint_targets() -> None:
    device = torch.device("mps")
    model = JointConsistentPredictor(
        ad_col_dims=[3, 4],
        target_col_names=["SERVICES_D"],
        target_col_dims=[2],
        los_num_classes=3,
        joint_direction="bidirectional",
        joint_heads=["SERVICES_D"],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    ).to(device)
    x = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long)
    y_d_cpu = {"SERVICES_D": torch.tensor([0, 1, 0], dtype=torch.long)}
    y_los_cpu = torch.tensor([1, 2, 0], dtype=torch.long)
    y_d_device = {name: target.to(device) for name, target in y_d_cpu.items()}
    y_los_device = y_los_cpu.to(device)
    output = model(
        x.to(device),
        d_targets=y_d_device,
        los_targets=y_los_device,
    )
    criterion = JointConsistencyLoss(lambda_joint=0.3, joint_head_names=["SERVICES_D"])

    total_loss, metrics = criterion(
        output,
        d_targets=y_d_device,
        los_targets=y_los_device,
        d_targets_for_joint=y_d_cpu,
        los_targets_for_joint=y_los_cpu,
    )

    assert total_loss.device.type == "mps"
    assert metrics["joint_consistency_terms"] == pytest.approx(2.0)


def test_build_d_target_dict_returns_contiguous_columns() -> None:
    y = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)

    targets = _build_d_target_dict(y, ["A", "B", "C"])

    assert targets["A"].is_contiguous()
    assert targets["B"].is_contiguous()
    assert targets["C"].is_contiguous()
    assert torch.equal(targets["B"], torch.tensor([1, 2], dtype=torch.long))


def test_conditional_distribution_rejects_out_of_range_targets() -> None:
    with pytest.raises(ValueError, match="target labels are outside"):
        _conditional_distribution_from_targets(
            torch.tensor([0, 5], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            num_target_classes=3,
            num_condition_classes=2,
        )


def test_joint_trainer_cli_defaults_match_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.argv", ["run_joint_consistent_predictor.py"])
    args = parse_joint_args()
    assert args.input_encoding == "onehot"
    assert args.joint_direction == "los_to_d"
    assert args.condition_mode == "predicted"
    assert args.detach_condition == "true"
    assert args.los_target_mode == "coarse"
    assert args.lambda_aux == pytest.approx(0.3)
    assert args.lambda_joint == pytest.approx(0.0)
    assert args.joint_heads == DEFAULT_JOINT_HEADS
    cfg = _build_config_from_args(args)
    assert cfg["model"]["params"]["input_encoding"] == "onehot"
    assert cfg["train"]["monitor_metric"] == "valid_balanced_score"
    assert cfg["joint_struct_loss"]["enabled"] is False
    assert cfg["joint_struct_loss"]["risk_head_set"] == "new_dvD_top3"


def test_normalize_joint_struct_loss_cfg_resolves_new_named_set() -> None:
    cfg = {
        "joint_struct_loss": {
            "enabled": True,
            "lambda_struct": 0.01,
            "risk_head_set": "new_dvD_top3",
        }
    }

    normalized, module = _normalize_joint_struct_loss_cfg(
        cfg,
        predictor_type="joint_consistent",
        target_head_names=[
            "SERVICES_D",
            "SUB1_D",
            "FREQ_ATND_SELF_HELP_D",
            "FREQ1_D",
            "FREQ2_D",
            "EMPLOY_D",
            "DETNLF_D",
        ],
    )

    assert module is not None
    assert isinstance(module, SoftJointDriftLoss)
    assert normalized["resolved_risk_heads"] == [
        "FREQ_ATND_SELF_HELP_D",
        "SUB1_D",
        "FREQ2_D",
    ]


def test_normalize_joint_struct_loss_cfg_rejects_joint_generative() -> None:
    cfg = {
        "joint_struct_loss": {
            "enabled": True,
            "lambda_struct": 0.01,
            "risk_head_set": "new_dvD_top3",
        }
    }

    with pytest.raises(ValueError, match="joint_consistent"):
        _normalize_joint_struct_loss_cfg(
            cfg,
            predictor_type="joint_generative",
            target_head_names=[
                "SERVICES_D",
                "SUB1_D",
                "FREQ_ATND_SELF_HELP_D",
                "FREQ1_D",
                "FREQ2_D",
                "EMPLOY_D",
                "DETNLF_D",
            ],
        )


def test_joint_consistency_loss_lambda_struct_zero_matches_disabled_path() -> None:
    model = JointConsistentPredictor(
        ad_col_dims=[3, 4],
        target_col_names=["FREQ_ATND_SELF_HELP_D", "SUB1_D", "FREQ2_D"],
        target_col_dims=[2, 2, 2],
        los_num_classes=2,
        joint_direction="los_to_d",
        joint_heads=["FREQ_ATND_SELF_HELP_D"],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    x = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    y_d = {
        "FREQ_ATND_SELF_HELP_D": torch.tensor([0, 1], dtype=torch.long),
        "SUB1_D": torch.tensor([1, 0], dtype=torch.long),
        "FREQ2_D": torch.tensor([0, 1], dtype=torch.long),
    }
    y_los = torch.tensor([0, 1], dtype=torch.long)
    output = model(x, d_targets=y_d, los_targets=y_los)
    baseline = JointConsistencyLoss(joint_head_names=["FREQ_ATND_SELF_HELP_D"])
    struct_disabled = JointConsistencyLoss(
        joint_head_names=["FREQ_ATND_SELF_HELP_D"],
        lambda_struct=0.0,
        struct_loss_module=SoftJointDriftLoss(
            risk_head_set="new_dvD_top3",
            available_heads=["FREQ_ATND_SELF_HELP_D", "SUB1_D", "FREQ2_D", "FREQ1_D", "EMPLOY_D", "DETNLF_D"],
        ),
    )

    baseline_loss, _ = baseline(output, d_targets=y_d, los_targets=y_los)
    disabled_loss, _ = struct_disabled(output, d_targets=y_d, los_targets=y_los)

    assert torch.allclose(baseline_loss, disabled_loss, atol=1.0e-7)


def test_joint_stats_cli_accepts_cache_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_joint_predictor_joint_stats.py",
            "--train-cache-path",
            "/tmp/train.pt",
            "--eval-cache-path",
            "/tmp/test.pt",
            "--output-dir",
            "/tmp/out",
        ],
    )
    args = parse_joint_stats_args()
    assert args.train_cache_path == "/tmp/train.pt"
    assert args.eval_cache_path == "/tmp/test.pt"


def test_conditional_js_divergence_zero_for_identical_vectors() -> None:
    vec = torch.tensor([0.75, 0.25], dtype=torch.float64).numpy()
    assert _conditional_js_divergence(vec, vec.copy()) == pytest.approx(0.0)


def test_compute_joint_stats_reports_per_head_metrics(tmp_path: Path) -> None:
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.eye(2, dtype=torch.float32).repeat(2, 1)},
        "final_los_pred": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_los_probs": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "los_raw": torch.tensor([1, 1, 2, 2], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "coarse_class"},
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {
            "SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
        },
        "final_los_pred": torch.tensor([0, 1], dtype=torch.long),
        "final_los_probs": torch.tensor(
            [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 1], dtype=torch.long),
            "los_raw": torch.tensor([1, 2], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "coarse_class"},
    }
    train_cache["_path"] = str(tmp_path / "train.pt")
    eval_cache["_path"] = str(tmp_path / "eval.pt")
    summary = compute_joint_stats(train_cache, eval_cache)
    assert summary["split"] == "test"
    assert summary["los_bin_mode"] == "coarse6"
    assert summary["num_los_classes"] == 6
    assert summary["num_rows_eval"] == 2
    assert len(summary["per_head"]) == 1
    assert summary["per_head"][0]["head_name"] == "SERVICES_D"
    assert "rare_combo_rate_predicted" in summary["per_head"][0]
    assert "rare_combo_rate_train_reference" in summary["per_head"][0]
    assert "mean_rare_combo_rate_predicted" in summary
    assert len(summary["los_given_d_rows"]) == 12
    first_row = summary["los_given_d_rows"][0]
    assert first_row["head_name"] == "SERVICES_D"
    assert first_row["los_bin_mode"] == "coarse6"
    assert "js_los_given_d_for_d_value" in first_row


def test_compute_joint_stats_supports_raw37_mode(tmp_path: Path) -> None:
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([1, 1, 4, 4], dtype=torch.long),
        "final_los_probs": torch.eye(37, dtype=torch.float32)[torch.tensor([0, 0, 3, 3], dtype=torch.long)],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 0, 3, 3], dtype=torch.long),
            "los_raw": torch.tensor([1, 1, 4, 4], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "raw_los"},
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {
            "SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
        },
        "final_los_pred": torch.tensor([1, 4], dtype=torch.long),
        "final_los_probs": torch.eye(37, dtype=torch.float32)[torch.tensor([0, 3], dtype=torch.long)],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 3], dtype=torch.long),
            "los_raw": torch.tensor([1, 4], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "raw_los"},
    }
    train_cache["_path"] = str(tmp_path / "train_raw.pt")
    eval_cache["_path"] = str(tmp_path / "eval_raw.pt")

    summary = compute_joint_stats(train_cache, eval_cache, los_bin_mode="raw37")

    assert summary["los_bin_mode"] == "raw37"
    assert summary["num_los_classes"] == 37
    assert summary["mean_js_d_given_los"] == pytest.approx(0.0)
    assert summary["mean_js_los_given_d"] == pytest.approx(0.0)


def test_compute_joint_stats_supports_coarse9_mode(tmp_path: Path) -> None:
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
        "final_d_probs": {
            "SERVICES_D": torch.tensor(
                [[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]],
                dtype=torch.float32,
            )
        },
        "final_los_pred": torch.tensor([0, 5, 6, 8], dtype=torch.long),
        "final_los_probs": torch.eye(9, dtype=torch.float32)[
            torch.tensor([0, 5, 6, 8], dtype=torch.long)
        ],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 5, 6, 8], dtype=torch.long),
            "los_raw": torch.tensor([1, 29, 32, 36], dtype=torch.long),
        },
        "metadata": {
            "los_target_mode": "coarse",
            "los_num_classes": 9,
            "final_los_pred_space": "coarse_class",
        },
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {
            "SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)
        },
        "final_los_pred": torch.tensor([5, 8], dtype=torch.long),
        "final_los_probs": torch.eye(9, dtype=torch.float32)[
            torch.tensor([5, 8], dtype=torch.long)
        ],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([5, 8], dtype=torch.long),
            "los_raw": torch.tensor([29, 36], dtype=torch.long),
        },
        "metadata": {
            "los_target_mode": "coarse",
            "los_num_classes": 9,
            "final_los_pred_space": "coarse_class",
        },
    }
    train_cache["_path"] = str(tmp_path / "train_coarse9.pt")
    eval_cache["_path"] = str(tmp_path / "eval_coarse9.pt")

    summary = compute_joint_stats(train_cache, eval_cache)

    assert summary["los_bin_mode"] == "coarse9"
    assert summary["num_los_classes"] == 9
    assert len(summary["los_given_d_rows"]) == 18
    assert summary["mean_js_d_given_los"] >= 0.0
    assert summary["mean_js_los_given_d"] >= 0.0


def test_compute_joint_stats_infers_raw37_mode_from_metadata(tmp_path: Path) -> None:
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0, 1], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([1, 4], dtype=torch.long),
        "final_los_probs": torch.eye(37, dtype=torch.float32)[torch.tensor([0, 3], dtype=torch.long)],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 3], dtype=torch.long),
            "los_raw": torch.tensor([1, 4], dtype=torch.long),
        },
        "metadata": {"los_target_mode": "raw37", "final_los_pred_space": "raw_los"},
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([1, 4], dtype=torch.long),
        "final_los_probs": torch.eye(37, dtype=torch.float32)[torch.tensor([0, 3], dtype=torch.long)],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 3], dtype=torch.long),
            "los_raw": torch.tensor([1, 4], dtype=torch.long),
        },
        "metadata": {"los_target_mode": "raw37", "final_los_pred_space": "raw_los"},
    }
    train_cache["_path"] = str(tmp_path / "train.pt")
    eval_cache["_path"] = str(tmp_path / "eval.pt")

    summary = compute_joint_stats(train_cache, eval_cache)

    assert summary["los_bin_mode"] == "raw37"


def test_compute_joint_stats_infers_coarse_mode_from_adjacent_final_config(tmp_path: Path) -> None:
    train_path = tmp_path / "train.pt"
    eval_path = tmp_path / "eval.pt"
    config_path = tmp_path / "config.final.yaml"
    config_path.write_text("joint_predictor:\n  los_target_mode: coarse\n", encoding="utf-8")
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0, 1], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([0, 1], dtype=torch.long),
        "final_los_probs": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 1], dtype=torch.long),
            "los_raw": torch.tensor([1, 2], dtype=torch.long),
        },
        "metadata": {},
        "_path": str(train_path),
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([0, 1], dtype=torch.long),
        "final_los_probs": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0, 1], dtype=torch.long)},
            "los_target": torch.tensor([0, 1], dtype=torch.long),
            "los_raw": torch.tensor([1, 2], dtype=torch.long),
        },
        "metadata": {},
        "_path": str(eval_path),
    }

    summary = compute_joint_stats(train_cache, eval_cache)

    assert summary["los_bin_mode"] == "coarse6"


def test_compute_joint_stats_raw37_mode_rejects_coarse_eval_predictions(tmp_path: Path) -> None:
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[1.0, 0.0]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([1], dtype=torch.long),
        "final_los_probs": torch.eye(37, dtype=torch.float32)[torch.tensor([0], dtype=torch.long)],
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0], dtype=torch.long)},
            "los_target": torch.tensor([0], dtype=torch.long),
            "los_raw": torch.tensor([1], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "raw_los"},
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10], dtype=torch.long),
        "final_d_pred": {"SERVICES_D": torch.tensor([0], dtype=torch.long)},
        "final_d_probs": {"SERVICES_D": torch.tensor([[1.0, 0.0]], dtype=torch.float32)},
        "final_los_pred": torch.tensor([0], dtype=torch.long),
        "final_los_probs": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        "targets": {
            "d": {"SERVICES_D": torch.tensor([0], dtype=torch.long)},
            "los_target": torch.tensor([0], dtype=torch.long),
            "los_raw": torch.tensor([1], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "coarse_class"},
    }
    train_cache["_path"] = str(tmp_path / "train_raw.pt")
    eval_cache["_path"] = str(tmp_path / "eval_coarse.pt")

    with pytest.raises(ValueError, match="raw37 joint statistics require eval LOS predictions in raw_los space"):
        compute_joint_stats(train_cache, eval_cache, los_bin_mode="raw37")


def test_generate_joint_drift_reports_writes_summary_and_top3_report(tmp_path: Path) -> None:
    train_cache = {
        "split": "train",
        "row_idx": torch.tensor([0, 1, 2, 3], dtype=torch.long),
        "final_d_pred": {
            "SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "SUB1_D": torch.tensor([0, 1, 0, 1], dtype=torch.long),
        },
        "final_d_probs": {
            "SERVICES_D": torch.tensor([[0.9, 0.1], [0.8, 0.2], [0.1, 0.9], [0.2, 0.8]], dtype=torch.float32),
            "SUB1_D": torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]], dtype=torch.float32),
        },
        "final_los_pred": torch.tensor([0, 0, 1, 1], dtype=torch.long),
        "final_los_probs": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "targets": {
            "d": {
                "SERVICES_D": torch.tensor([0, 0, 1, 1], dtype=torch.long),
                "SUB1_D": torch.tensor([0, 1, 0, 1], dtype=torch.long),
            },
            "los_target": torch.tensor([0, 0, 1, 1], dtype=torch.long),
            "los_raw": torch.tensor([1, 1, 2, 2], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "coarse_class"},
    }
    eval_cache = {
        "split": "test",
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {
            "SERVICES_D": torch.tensor([0, 1], dtype=torch.long),
            "SUB1_D": torch.tensor([1, 1], dtype=torch.long),
        },
        "final_d_probs": {
            "SERVICES_D": torch.tensor([[0.9, 0.1], [0.1, 0.9]], dtype=torch.float32),
            "SUB1_D": torch.tensor([[0.2, 0.8], [0.1, 0.9]], dtype=torch.float32),
        },
        "final_los_pred": torch.tensor([0, 1], dtype=torch.long),
        "final_los_probs": torch.tensor(
            [[0.9, 0.1, 0.0, 0.0, 0.0, 0.0], [0.1, 0.9, 0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "targets": {
            "d": {
                "SERVICES_D": torch.tensor([0, 1], dtype=torch.long),
                "SUB1_D": torch.tensor([1, 1], dtype=torch.long),
            },
            "los_target": torch.tensor([0, 1], dtype=torch.long),
            "los_raw": torch.tensor([1, 2], dtype=torch.long),
        },
        "metadata": {"final_los_pred_space": "coarse_class"},
    }
    cache_dir = tmp_path / "joint_cache"
    cache_dir.mkdir()
    train_path = cache_dir / "train.pt"
    test_path = cache_dir / "test.pt"
    torch.save(train_cache, train_path)
    torch.save(eval_cache, test_path)

    payload = _generate_joint_drift_reports(
        run_dir=str(tmp_path),
        train_cache_path=str(train_path),
        eval_cache_paths={"test": str(test_path)},
    )

    assert "test" in payload
    assert Path(payload["test"]["summary_path"]).exists()
    assert Path(payload["test"]["report_summary_path"]).exists()
    assert payload["test"]["focused_heads"] == ["SERVICES_D", "SUB1_D"]
