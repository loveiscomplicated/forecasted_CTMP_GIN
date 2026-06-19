from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from src.models.ctmp_gin.los_encoder import (
    ensure_ctmp_gin_los_encoder_defaults,
    resolve_ctmp_gin_input_metadata,
)


class DistributionEncoder(nn.Module):
    def __init__(self, num_classes: int, out_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = int(hidden_dim or out_dim)
        self.num_classes = int(num_classes)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim != 2:
            raise ValueError(
                f"DistributionEncoder expected rank-2 probs, got shape={tuple(probs.shape)}"
            )
        if probs.shape[1] != self.num_classes:
            raise ValueError(
                f"DistributionEncoder width mismatch: expected {self.num_classes}, got {probs.shape[1]}"
            )
        return self.net(probs)


def resolve_constant_long_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        if value.numel() == 0:
            raise ValueError(f"{name} is empty")
        if not torch.equal(value, value[0].expand_as(value)):
            raise ValueError(f"{name} must be constant across the batch")
        return value[:1]
    if value.ndim == 2:
        if value.size(0) == 0:
            raise ValueError(f"{name} is empty")
        ref = value[0:1].expand_as(value)
        if not torch.equal(value, ref):
            raise ValueError(f"{name} must be constant across the batch")
        return value[0]
    raise ValueError(f"Unsupported ndim for {name}: {value.ndim}")


def ensure_model_forecast_defaults(cfg: dict[str, Any]) -> None:
    ensure_ctmp_gin_los_encoder_defaults(cfg)

    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("name") != "gin":
        return

    model_params = model_cfg.setdefault("params", {})
    forecast_enabled = bool(cfg.get("forecasted_pipeline", {}).get("enabled", False))
    joint_forecast_enabled = bool(cfg.get("joint_forecast_pipeline", {}).get("enabled", False))
    if forecast_enabled or joint_forecast_enabled:
        model_params.setdefault("forecast_input_encoder", "distribution")
    else:
        model_params.setdefault("forecast_input_encoder", "entity_embedding")
    model_params.setdefault("distribution_encoder_hidden_dim", 64)
    model_params.setdefault(
        "distribution_encoder_out_dim",
        int(model_params.get("embedding_dim", 16)),
    )


def resolve_model_forecast_input_metadata(cfg: dict[str, Any]) -> dict[str, object]:
    model_name = str(cfg.get("model", {}).get("name", ""))
    if model_name == "ctmp_gin":
        return resolve_ctmp_gin_input_metadata(cfg)
    if model_name != "gin":
        return {}

    model_params = cfg.get("model", {}).get("params", {})
    forecast_enabled = bool(cfg.get("forecasted_pipeline", {}).get("enabled", False))
    joint_forecast_enabled = bool(cfg.get("joint_forecast_pipeline", {}).get("enabled", False))
    forecast_input_encoder = str(
        model_params.get("forecast_input_encoder", "entity_embedding")
    ).lower()
    joint_mode = str(
        cfg.get("joint_forecast_pipeline", {})
        .get("joint_forecast_input", {})
        .get("mode", "distribution")
    ).lower()

    if joint_forecast_enabled and forecast_input_encoder == "distribution":
        d_input_type = "predictor_probs" if joint_mode == "distribution" else "one_hot"
        los_input_type = "predictor_probs" if joint_mode == "distribution" else "one_hot"
        known_variable_input_type = "one_hot"
    elif forecast_enabled and forecast_input_encoder == "distribution":
        d_input_type = (
            "predictor_probs"
            if bool(cfg.get("forecasted_discharge", {}).get("enabled", False))
            else "one_hot"
        )
        los_input_type = (
            "predictor_probs"
            if bool(cfg.get("forecasted_los", {}).get("enabled", False))
            else "one_hot"
        )
        known_variable_input_type = "one_hot"
    else:
        d_input_type = "entity_embedding_lookup"
        los_input_type = "entity_embedding_lookup"
        known_variable_input_type = "entity_embedding"

    return {
        "forecast_input_encoder": forecast_input_encoder,
        "d_input_type": d_input_type,
        "los_input_type": los_input_type,
        "known_variable_input_type": known_variable_input_type,
        "oracle_pipeline_unchanged": True,
        "joint_forecast_enabled": joint_forecast_enabled,
    }
