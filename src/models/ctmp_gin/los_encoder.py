from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
import torch.nn as nn


DEFAULT_TEDS_LOS_REP_DAYS: list[float] = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
    38,
    53,
    75,
    105,
    150,
    273,
    425,
]


def ensure_ctmp_gin_los_encoder_defaults(cfg: dict[str, Any]) -> None:
    model_cfg = cfg.setdefault("model", {})
    if model_cfg.get("name") != "ctmp_gin":
        return
    model_params = model_cfg.setdefault("params", {})
    model_params.setdefault("los_emb", "embedding")
    model_params.setdefault("los_ordinal_basis_dim", 8)
    model_params.setdefault("los_ordinal_dim", 8)
    model_params.setdefault("los_ordinal_hidden_dim", 32)
    forecast_enabled = bool(cfg.get("forecasted_pipeline", {}).get("enabled", False))
    joint_forecast_enabled = bool(cfg.get("joint_forecast_pipeline", {}).get("enabled", False))
    default_encoder = "distribution" if forecast_enabled else "entity_embedding"
    if joint_forecast_enabled:
        default_encoder = "distribution"
    model_params.setdefault("forecast_input_encoder", default_encoder)
    model_params.setdefault("distribution_encoder_hidden_dim", 64)
    model_params.setdefault(
        "distribution_encoder_out_dim",
        int(model_params.get("embedding_dim", 16)),
    )


def resolve_ctmp_gin_input_metadata(cfg: dict[str, Any]) -> dict[str, object]:
    model_cfg = cfg.get("model", {})
    if model_cfg.get("name") != "ctmp_gin":
        return {}

    model_params = model_cfg.get("params", {})
    forecast_enabled = bool(cfg.get("forecasted_pipeline", {}).get("enabled", False))
    joint_forecast_enabled = bool(cfg.get("joint_forecast_pipeline", {}).get("enabled", False))
    forecast_input_encoder = str(model_params.get("forecast_input_encoder", "entity_embedding")).lower()
    joint_mode = str(
        cfg.get("joint_forecast_pipeline", {}).get("joint_forecast_input", {}).get("mode", "distribution")
    ).lower()

    if joint_forecast_enabled and forecast_input_encoder == "distribution":
        d_input_type = "predictor_probs" if joint_mode == "distribution" else "one_hot"
        los_input_type = "predictor_probs" if joint_mode == "distribution" else "one_hot"
        known_variable_input_type = "one_hot"
    elif forecast_enabled and forecast_input_encoder == "distribution":
        d_input_type = "predictor_probs" if bool(cfg.get("forecasted_discharge", {}).get("enabled", False)) else "one_hot"
        los_input_type = "predictor_probs" if bool(cfg.get("forecasted_los", {}).get("enabled", False)) else "one_hot"
        known_variable_input_type = "one_hot"
    elif forecast_enabled and forecast_input_encoder == "expected_embedding_diagnostic":
        d_input_type = "predictor_probs"
        los_input_type = "predictor_probs"
        known_variable_input_type = "entity_embedding"
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


class HybridOrdinalLOSEncoder(nn.Module):
    def __init__(
        self,
        num_los_classes: int,
        out_dim: int,
        basis_dim: int = 8,
        ordinal_dim: int = 8,
        hidden_dim: int = 32,
        dropout: float = 0.0,
        rep_days: Optional[Sequence[float]] = None,
    ) -> None:
        super().__init__()
        if num_los_classes <= 0:
            raise ValueError(f"num_los_classes must be positive, got {num_los_classes}")
        self.num_los_classes = int(num_los_classes)
        self.out_dim = int(out_dim)
        self.basis_dim = int(basis_dim)
        self.ordinal_dim = int(ordinal_dim)
        self.hidden_dim = int(hidden_dim)

        rep_days_tensor = self._resolve_rep_days(rep_days)
        self.register_buffer("rep_days", rep_days_tensor)

        log_duration = torch.log1p(rep_days_tensor)
        self.register_buffer("log_duration", log_duration)
        rank = torch.linspace(0.0, 1.0, steps=self.num_los_classes, dtype=torch.float32)
        self.register_buffer("normalized_rank", rank.unsqueeze(1))

        log_mean = log_duration.mean()
        log_std = log_duration.std(unbiased=False).clamp_min(1.0e-6)
        normalized_log_duration = ((log_duration - log_mean) / log_std).unsqueeze(1)
        self.register_buffer("normalized_log_duration", normalized_log_duration)

        if self.basis_dim > 0:
            centers = torch.linspace(float(log_duration.min()), float(log_duration.max()), steps=self.basis_dim)
            if self.basis_dim > 1:
                basis_width = float(centers[1] - centers[0])
            else:
                basis_width = 1.0
            gamma = 1.0 / max(basis_width * basis_width, 1.0e-6)
            self.register_buffer("rbf_centers", centers)
            self.register_buffer("rbf_gamma", torch.tensor(gamma, dtype=torch.float32))
        else:
            self.register_buffer("rbf_centers", torch.empty(0, dtype=torch.float32))
            self.register_buffer("rbf_gamma", torch.tensor(1.0, dtype=torch.float32))

        self.ordinal_increments = nn.Parameter(
            torch.empty(self.num_los_classes, self.ordinal_dim, dtype=torch.float32)
        )

        input_dim = 1 + 1 + self.basis_dim + self.ordinal_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

        self.reset_parameters()

    def _resolve_rep_days(self, rep_days: Optional[Sequence[float]]) -> torch.Tensor:
        if rep_days is None:
            if self.num_los_classes == len(DEFAULT_TEDS_LOS_REP_DAYS):
                values = DEFAULT_TEDS_LOS_REP_DAYS
            else:
                values = list(range(1, self.num_los_classes + 1))
        else:
            values = list(rep_days)

        if len(values) != self.num_los_classes:
            raise ValueError(
                f"rep_days length mismatch: expected {self.num_los_classes}, got {len(values)}"
            )
        return torch.tensor(values, dtype=torch.float32)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.ordinal_increments, mean=0.0, std=0.02)
        for module in self.proj:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _rbf_features(self) -> torch.Tensor:
        if self.basis_dim == 0:
            return torch.empty(self.num_los_classes, 0, device=self.log_duration.device, dtype=self.log_duration.dtype)
        diff = self.log_duration.unsqueeze(1) - self.rbf_centers.unsqueeze(0)
        return torch.exp(-self.rbf_gamma * diff.pow(2))

    def _all_encoder_inputs(self) -> torch.Tensor:
        cumulative_ordinal_emb = torch.cumsum(self.ordinal_increments, dim=0)
        return torch.cat(
            [
                self.normalized_rank,
                self.normalized_log_duration,
                self._rbf_features(),
                cumulative_ordinal_emb,
            ],
            dim=1,
        )

    def all_embeddings(self) -> torch.Tensor:
        return self.proj(self._all_encoder_inputs())

    def forward(self, los_idx: torch.Tensor) -> torch.Tensor:
        los_idx = los_idx.long()
        if los_idx.numel() == 0:
            return self.all_embeddings().new_empty((*los_idx.shape, self.out_dim))
        idx_min = int(los_idx.min().item())
        idx_max = int(los_idx.max().item())
        if idx_min < 0 or idx_max >= self.num_los_classes:
            raise ValueError(
                f"LOS class index out of range: min={idx_min} max={idx_max} "
                f"valid=[0, {self.num_los_classes - 1}]"
            )
        emb_table = self.all_embeddings()
        return emb_table[los_idx]
