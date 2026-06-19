from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from src.models.ctmp_gin.model import CTMPGIN
from src.models.discharge_predictor.model import MultiTaskDischargePredictor
from src.models.gin import GIN
from src.trainers.forecasted_discharge import (
    ForecastedDischargeProvider,
    normalize_forecasted_discharge_cfg,
)
from src.trainers.forecasted_pipeline import ForecastCacheDataset


class _ProviderDataset:
    def __init__(self) -> None:
        self.col_info = (
            ["ad0", "ad1", "SERVICES_D", "SUB1_D"],
            [3, 4, 2, 3],
            [0, 1],
            [2, 3],
        )


class _CacheDataset:
    def __init__(self) -> None:
        self.processed_df = None
        self.col_info = (
            ["ad0", "ad1", "SERVICES_D", "SUB1_D"],
            [3, 4, 2, 3],
            [0, 1],
            [2, 3],
        )
        self.num_classes = 2
        self.samples = [
            (torch.tensor([0, 1, 0, 2]), 1, 4),
            (torch.tensor([1, 2, 1, 1]), 0, 6),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        x, y, los = self.samples[index]
        return x, y, los


def _build_discharge_checkpoint(tmp_path: Path) -> Path:
    checkpoint_path = tmp_path / "discharge_best.pt"
    model = MultiTaskDischargePredictor(
        ad_col_dims=[3, 4],
        target_col_names=["SERVICES_D", "SUB1_D"],
        target_col_dims=[2, 3],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
        model.heads["SERVICES_D"].bias.copy_(torch.tensor([2.0, -1.0], dtype=torch.float32))
        model.heads["SUB1_D"].bias.copy_(torch.tensor([-1.0, 0.5, 1.0], dtype=torch.float32))

    torch.save(
        {
            "cfg": {
                "model": {
                    "params": {
                        "embedding_dim": 4,
                        "hidden_dim": 8,
                        "num_layers": 1,
                        "dropout": 0.0,
                    }
                }
            },
            "schema": {
                "admission_col_names": ["ad0", "ad1"],
                "admission_col_dims": [3, 4],
                "target_col_names": ["SERVICES_D", "SUB1_D"],
                "target_col_dims": [2, 3],
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    return checkpoint_path


def test_forecasted_discharge_provider_returns_soft_payload_shapes(tmp_path: Path) -> None:
    checkpoint_path = _build_discharge_checkpoint(tmp_path)
    provider = ForecastedDischargeProvider(
        {
            "checkpoint_path": str(checkpoint_path),
            "mode": "soft",
            "soft_discharge": {
                "enabled": True,
                "heads": "all",
                "temperature": 1.0,
                "save_logits": True,
                "save_probs": True,
                "use_probs_cache": True,
            },
        },
        _ProviderDataset(),
        torch.device("cpu"),
    )

    x_batch = torch.tensor([[0, 1, 0, 0], [1, 2, 1, 2]], dtype=torch.long)
    x_pred, payload = provider.predict_with_cache_payload(x_batch)

    assert payload is not None
    assert x_pred.shape == x_batch.shape
    assert set(payload["heads"].keys()) == {"SERVICES_D", "SUB1_D"}
    for head_name, expected_classes in {"SERVICES_D": 2, "SUB1_D": 3}.items():
        head_payload = payload["heads"][head_name]
        probs = head_payload["probs"]
        assert probs.ndim == 2
        assert probs.shape == (2, expected_classes)
        assert torch.allclose(probs.sum(dim=1), torch.ones(2), atol=1.0e-6)
        assert head_payload["hard"].shape == (2,)
        assert head_payload["logits"].shape == (2, expected_classes)


def test_forecast_cache_dataset_returns_soft_discharge_sidecar() -> None:
    base = _CacheDataset()
    x_cache = torch.stack([base[i][0] for i in range(len(base))], dim=0)
    los_cache = torch.tensor([base[i][2] for i in range(len(base))], dtype=torch.long)
    soft_cache = {
        "metadata": {"mode": "mixed", "temperature": 1.0},
        "head_names": ["SERVICES_D", "SUB1_D"],
        "soft_head_names": ["SERVICES_D"],
        "heads": {
            "SERVICES_D": {
                "probs": torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32),
                "logits": torch.tensor([[2.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
                "hard": torch.tensor([0, 1], dtype=torch.long),
                "target_col_idx": torch.tensor(2, dtype=torch.long),
                "num_classes": torch.tensor(2, dtype=torch.long),
                "class_to_embedding_idx": torch.tensor([0, 1], dtype=torch.long),
                "mask": torch.tensor([True, True]),
            }
        },
    }

    dataset = ForecastCacheDataset(base, x_cache, los_cache, soft_cache)
    x, y, los, meta = dataset[1]

    assert torch.equal(x, x_cache[1])
    assert int(y) == 0
    assert int(los) == 6
    assert set(meta["soft_discharge"].keys()) == {"SERVICES_D"}
    assert meta["soft_discharge"]["SERVICES_D"]["probs"].shape == (2,)
    assert bool(meta["soft_discharge_mask"]["SERVICES_D"].item()) is True
    assert meta["metadata"]["mode"] == "mixed"


def test_forecasted_discharge_cfg_is_normalized_for_distribution_encoder_path() -> None:
    cfg = {
        "model": {"name": "ctmp_gin", "params": {"forecast_input_encoder": "distribution"}},
        "forecasted_pipeline": {"enabled": True},
    }

    normalized = normalize_forecasted_discharge_cfg(
        cfg,
        {"enabled": True, "mode": "hard", "soft_discharge": {"enabled": False}},
    )

    assert normalized["mode"] == "soft"
    assert normalized["soft_discharge"]["enabled"] is True
    assert normalized["soft_discharge"]["heads"] == "all"
    assert normalized["soft_discharge"]["save_probs"] is True


def test_gin_forecasted_discharge_cfg_is_normalized_for_distribution_encoder_path() -> None:
    cfg = {
        "model": {"name": "gin", "params": {"forecast_input_encoder": "distribution"}},
        "joint_forecast_pipeline": {"enabled": True},
    }

    normalized = normalize_forecasted_discharge_cfg(
        cfg,
        {"enabled": True, "mode": "hard", "soft_discharge": {"enabled": False}},
    )

    assert normalized["mode"] == "soft"
    assert normalized["soft_discharge"]["enabled"] is True
    assert normalized["soft_discharge"]["heads"] == "all"


def test_forecasted_discharge_provider_accepts_named_risk_head_set(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "discharge_top3_best.pt"
    model = MultiTaskDischargePredictor(
        ad_col_dims=[3, 4],
        target_col_names=["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
        target_col_dims=[2, 3, 2],
        embedding_dim=4,
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )
    with torch.no_grad():
        for param in model.parameters():
            param.zero_()
    torch.save(
        {
            "cfg": {
                "model": {
                    "params": {
                        "embedding_dim": 4,
                        "hidden_dim": 8,
                        "num_layers": 1,
                        "dropout": 0.0,
                    }
                }
            },
            "schema": {
                "admission_col_names": ["ad0", "ad1"],
                "admission_col_dims": [3, 4],
                "target_col_names": ["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
                "target_col_dims": [2, 3, 2],
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint_path,
    )
    dataset = type(
        "_ProviderTop3Dataset",
        (),
        {
            "col_info": (
                ["ad0", "ad1", "SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
                [3, 4, 2, 3, 2],
                [0, 1],
                [2, 3, 4],
            )
        },
    )()
    provider = ForecastedDischargeProvider(
        {
            "checkpoint_path": str(checkpoint_path),
            "mode": "soft",
            "soft_discharge": {
                "enabled": True,
                "heads": "old_total_drift_top3",
                "temperature": 1.0,
                "save_logits": True,
                "save_probs": True,
                "use_probs_cache": True,
            },
        },
        dataset,
        torch.device("cpu"),
    )

    assert provider.soft_head_names == ["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"]


def test_gin_distribution_encoder_uses_probs_for_forecasted_discharge_and_one_hot_for_known_vars() -> None:
    model = GIN(
        embedding_dim=5,
        col_info=(
            ["ad0", "ad1", "SERVICES_D", "SUB1_D"],
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

    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)
    probs = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    target_col_idx = torch.tensor([2, 2], dtype=torch.long)
    num_classes = torch.tensor([2, 2], dtype=torch.long)

    encoded = model._embed_x_with_optional_soft_discharge(
        x,
        soft_discharge={
            "SERVICES_D": {
                "probs": probs,
                "target_col_idx": target_col_idx,
                "num_classes": num_classes,
            }
        },
    )

    expected_services = model.distribution_encoders[2](probs)
    expected_sub1 = model.distribution_encoders[3](
        F.one_hot(x[:, 3], num_classes=3).to(dtype=torch.float32)
    )
    expected_ad0 = model.distribution_encoders[0](
        F.one_hot(x[:, 0], num_classes=3).to(dtype=torch.float32)
    )

    assert encoded.shape == (2, 4, 7)
    assert torch.allclose(encoded[:, 2, :], expected_services, atol=1.0e-6)
    assert torch.allclose(encoded[:, 3, :], expected_sub1, atol=1.0e-6)
    assert torch.allclose(encoded[:, 0, :], expected_ad0, atol=1.0e-6)


def test_gin_distribution_encoder_excludes_col_info_los_from_x_features() -> None:
    model = GIN(
        embedding_dim=5,
        col_info=(
            ["ad0", "LOS", "SERVICES_D", "SUB1_D"],
            [3, 38, 2, 3],
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

    x = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    probs = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    encoded = model._embed_x_with_optional_soft_discharge(
        x,
        soft_discharge={
            "SERVICES_D": {
                "probs": probs,
                "target_col_idx": torch.tensor([2, 2], dtype=torch.long),
                "num_classes": torch.tensor([2, 2], dtype=torch.long),
            }
        },
    )

    expected_services = model.distribution_encoders[1](probs)
    expected_sub1 = model.distribution_encoders[2](
        F.one_hot(x[:, 2], num_classes=3).to(dtype=torch.float32)
    )

    assert len(model.distribution_encoders) == 3
    assert encoded.shape == (2, 3, 7)
    assert torch.allclose(encoded[:, 1, :], expected_services, atol=1.0e-6)
    assert torch.allclose(encoded[:, 2, :], expected_sub1, atol=1.0e-6)


def test_gin_distribution_los_encoder_uses_same_path_for_hard_and_soft_inputs() -> None:
    model = GIN(
        embedding_dim=5,
        col_info=(
            ["ad0", "ad1", "SERVICES_D", "SUB1_D"],
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
    los_hard = torch.tensor([1, 12], dtype=torch.long)
    los_soft = F.one_hot(los_hard - 1, num_classes=37).to(dtype=torch.float32)

    hard_emb = model.encode_los(los_hard)
    soft_emb = model.encode_los(los_soft)

    assert torch.allclose(hard_emb, soft_emb, atol=1.0e-6)


def test_ctmp_gin_distribution_encoder_uses_probs_for_forecasted_discharge_and_one_hot_for_known_vars() -> None:
    model = CTMPGIN(
        col_info=(
            ["ad0", "ad1", "SERVICES_D", "SUB1_D"],
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

    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)
    probs = torch.tensor([[0.25, 0.75], [0.6, 0.4]], dtype=torch.float32)
    target_col_idx = torch.tensor([2, 2], dtype=torch.long)
    num_classes = torch.tensor([2, 2], dtype=torch.long)

    encoded = model._embed_x_with_optional_soft_discharge(
        x,
        soft_discharge={
            "SERVICES_D": {
                "probs": probs,
                "target_col_idx": target_col_idx,
                "num_classes": num_classes,
            }
        },
    )

    expected_services = model.distribution_encoders[2](probs)
    expected_sub1 = model.distribution_encoders[3](
        F.one_hot(x[:, 3], num_classes=3).to(dtype=torch.float32)
    )
    expected_ad0 = model.distribution_encoders[0](
        F.one_hot(x[:, 0], num_classes=3).to(dtype=torch.float32)
    )

    assert encoded.shape == (2, 4, 7)
    assert torch.allclose(encoded[:, 2, :], expected_services, atol=1.0e-6)
    assert torch.allclose(encoded[:, 3, :], expected_sub1, atol=1.0e-6)
    assert torch.allclose(encoded[:, 0, :], expected_ad0, atol=1.0e-6)


def test_ctmp_gin_entity_embedding_mode_keeps_oracle_lookup() -> None:
    model = CTMPGIN(
        col_info=(
            ["ad0", "ad1", "SERVICES_D", "SUB1_D"],
            [3, 4, 2, 3],
            [0, 1],
            [2, 3],
        ),
        embedding_dim=4,
        gin_hidden_channel=8,
        gin_1_layers=1,
        gin_hidden_channel_2=8,
        gin_2_layers=1,
        num_classes=2,
        dropout_p=0.0,
        los_embedding_dim=4,
        max_los=37,
        readout_mode="last",
    )
    x = torch.tensor([[0, 1, 1, 2], [2, 3, 0, 1]], dtype=torch.long)

    encoded = model._embed_x_with_optional_soft_discharge(x)
    expected = model.entity_embedding_layer(x)

    assert torch.allclose(encoded, expected, atol=1.0e-6)
