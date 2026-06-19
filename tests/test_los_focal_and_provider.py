from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.models.discharge_predictor.los_coarse_model import LOSCoarsePredictor
from src.models.discharge_predictor.los_utils import expand_coarse_distribution_to_raw_los
from src.models.discharge_predictor.loss import multiclass_focal_loss
from src.models.discharge_predictor.ordinal_loss import compute_ce_class_weight
from src.trainers.forecasted_los import ForecastedLOSProvider
from src.trainers.run_los_prediction import (
    _compute_multiclass_nll,
    _compute_top_label_ece,
    _fit_temperature_scaling,
)


class _DummyDataset:
    def __init__(self) -> None:
        self.col_info = (
            ["ad0", "ad1", "dis0"],
            [3, 4, 5],
            [0, 1],
            [2],
        )


def test_multiclass_focal_loss_matches_ce_when_gamma_zero_and_no_alpha() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0], [0.1, -0.3, 1.7]], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.long)

    focal = multiclass_focal_loss(
        logits,
        targets,
        gamma=0.0,
        alpha=None,
        label_smoothing=0.0,
        reduction="mean",
    )
    ce = F.cross_entropy(logits, targets, reduction="mean", label_smoothing=0.0)

    assert torch.allclose(focal, ce, atol=1.0e-7)


def test_compute_ce_class_weight_mean_normalizes_and_clips() -> None:
    labels = torch.tensor([0, 0, 0, 1, 2], dtype=torch.long)
    weights = compute_ce_class_weight(
        labels,
        num_classes=3,
        mode="inverse_sqrt",
        max_weight=1.5,
    )

    assert weights.shape == (3,)
    assert torch.isclose(weights.mean(), torch.tensor(1.0), atol=1.0e-6)
    assert float(weights.max()) <= 1.5 + 1.0e-6


def test_compute_top_label_ece_matches_expected_value() -> None:
    probs = np.array([[0.8, 0.2], [0.6, 0.4]], dtype=np.float32)
    targets = np.array([0, 1], dtype=np.int64)

    ece = _compute_top_label_ece(probs, targets, n_bins=2)

    assert abs(ece - 0.2) < 1.0e-6


def test_fit_temperature_scaling_returns_positive_temperature_and_improves_nll() -> None:
    logits = np.array([[2.0, -2.0], [-2.0, 2.0]], dtype=np.float32)
    targets = np.array([1, 0], dtype=np.int64)

    raw_nll = _compute_multiclass_nll(logits, targets, temperature=1.0)
    temperature = _fit_temperature_scaling(logits, targets)
    calibrated_nll = _compute_multiclass_nll(logits, targets, temperature=temperature)

    assert temperature > 0.0
    assert calibrated_nll <= raw_nll + 1.0e-6


def _build_coarse_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "runs" / "coarse_focal_alpha"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "best.pt"
    calibration_path = run_dir / "calibration.json"

    model = LOSCoarsePredictor(
        ad_col_dims=[3, 4],
        num_classes=6,
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout_p=0.0,
        activation="relu",
        norm_type="layernorm",
    )
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(torch.tensor([2.0, 0.0, -1.0, -2.0, -3.0, -4.0], dtype=torch.float32))

    torch.save(
        {
            "cfg": {
                "loss": {"type": "focal_alpha"},
                "los_target_mode": "coarse",
                "model": {
                    "params": {
                        "embedding_dim": 4,
                        "hidden_dim": 8,
                        "num_layers": 1,
                        "dropout_p": 0.0,
                        "activation": "relu",
                        "norm_type": "layernorm",
                    }
                },
                "num_classes": 6,
            },
            "schema": {
                "admission_col_dims": [3, 4],
                "los_num_classes": 6,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    calibration_path.write_text(
        json.dumps(
            {
                "temperature": {"fitted": 2.0, "source": "validation_nll"},
                "raw": {"valid": {}, "test": {}},
                "calibrated": {"valid": {}, "test": {}},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path, calibration_path


def _build_coarse_breakdown_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    run_dir = tmp_path / "runs" / "coarse_breakdown"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "best.pt"
    calibration_path = run_dir / "calibration.json"

    model = LOSCoarsePredictor(
        ad_col_dims=[3, 4],
        num_classes=9,
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout_p=0.0,
        activation="relu",
        norm_type="layernorm",
    )
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.copy_(
            torch.tensor(
                [2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0],
                dtype=torch.float32,
            )
        )

    torch.save(
        {
            "cfg": {
                "loss": {"type": "focal_alpha"},
                "los_target_mode": "coarse",
                "los_coarse_breakdown": True,
                "model": {
                    "params": {
                        "embedding_dim": 4,
                        "hidden_dim": 8,
                        "num_layers": 1,
                        "dropout_p": 0.0,
                        "activation": "relu",
                        "norm_type": "layernorm",
                    }
                },
                "num_classes": 9,
            },
            "schema": {
                "admission_col_dims": [3, 4],
                "los_num_classes": 9,
                "los_coarse_breakdown": True,
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )

    calibration_path.write_text(
        json.dumps(
            {
                "temperature": {"fitted": 2.0, "source": "validation_nll"},
                "raw": {"valid": {}, "test": {}},
                "calibrated": {"valid": {}, "test": {}},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint_path, calibration_path


def test_forecasted_los_provider_uses_calibrated_temperature_for_coarse_distribution(tmp_path: Path) -> None:
    checkpoint_path, calibration_path = _build_coarse_checkpoint(tmp_path)
    provider = ForecastedLOSProvider(
        {
            "checkpoint_path": str(checkpoint_path),
            "target_mode": "coarse",
            "num_classes": 6,
            "return_type": "distribution",
            "probability_source": "calibrated",
            "calibration_path": str(calibration_path),
            "temperature": None,
        },
        _DummyDataset(),
        torch.device("cpu"),
    )

    x_batch = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
    output = provider(x_batch)
    coarse_probs = F.softmax(
        torch.tensor([[2.0, 0.0, -1.0, -2.0, -3.0, -4.0]], dtype=torch.float32) / 2.0,
        dim=1,
    ).repeat(2, 1)
    expected = expand_coarse_distribution_to_raw_los(coarse_probs)

    assert output.shape == (2, 37)
    assert torch.allclose(output, expected, atol=1.0e-6)


def test_forecasted_los_provider_expands_coarse_breakdown_distribution(tmp_path: Path) -> None:
    checkpoint_path, calibration_path = _build_coarse_breakdown_checkpoint(tmp_path)
    provider = ForecastedLOSProvider(
        {
            "checkpoint_path": str(checkpoint_path),
            "target_mode": "coarse",
            "return_type": "distribution",
            "probability_source": "calibrated",
            "calibration_path": str(calibration_path),
            "temperature": None,
        },
        _DummyDataset(),
        torch.device("cpu"),
    )

    x_batch = torch.tensor([[0, 0, 0], [1, 1, 1]], dtype=torch.long)
    output = provider(x_batch)
    coarse_probs = F.softmax(
        torch.tensor(
            [[2.0, 0.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0]],
            dtype=torch.float32,
        )
        / 2.0,
        dim=1,
    ).repeat(2, 1)
    expected = expand_coarse_distribution_to_raw_los(coarse_probs, breakdown=True)

    assert output.shape == (2, 37)
    assert torch.allclose(output, expected, atol=1.0e-6)


def test_forecasted_los_provider_temperature_override_beats_calibration(tmp_path: Path) -> None:
    checkpoint_path, calibration_path = _build_coarse_checkpoint(tmp_path)
    provider = ForecastedLOSProvider(
        {
            "checkpoint_path": str(checkpoint_path),
            "target_mode": "coarse",
            "num_classes": 6,
            "return_type": "distribution",
            "probability_source": "calibrated",
            "calibration_path": str(calibration_path),
            "temperature": 4.0,
        },
        _DummyDataset(),
        torch.device("cpu"),
    )

    x_batch = torch.tensor([[0, 0, 0]], dtype=torch.long)
    output = provider(x_batch)
    coarse_probs = F.softmax(
        torch.tensor([[2.0, 0.0, -1.0, -2.0, -3.0, -4.0]], dtype=torch.float32) / 4.0,
        dim=1,
    )
    expected = expand_coarse_distribution_to_raw_los(coarse_probs)

    assert torch.allclose(output, expected, atol=1.0e-6)
