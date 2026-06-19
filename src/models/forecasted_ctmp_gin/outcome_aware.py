from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from src.models.ctmp_gin import CTMPGIN
from src.models.gin import GIN
from src.models.discharge_predictor import expand_coarse_distribution_to_raw_los
from src.models.discharge_predictor.joint_generative_predictor import (
    JointGenerativePredictor,
    JointGenerativePredictorOutput,
)
from src.models.forecasted_ctmp_gin.contract import (
    SoftDischargeContract,
    build_soft_discharge_payload,
)


@dataclass
class OutcomeAwareForecastedCTMPGINOutput:
    reasonb_logits: torch.Tensor
    predictor_output: JointGenerativePredictorOutput
    forecast_d_probs: dict[str, torch.Tensor]
    forecast_los_probs: torch.Tensor
    diagnostics: dict[str, float]


class OutcomeAwareForecastedCTMPGIN(nn.Module):
    def __init__(
        self,
        predictor: JointGenerativePredictor,
        ctmp_gin: CTMPGIN,
        contract: SoftDischargeContract,
        admission_col_indices: list[int],
        discharge_col_indices: list[int],
        *,
        sample_prior_in_train: bool = False,
        discharge_placeholder_index: int = 0,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.ctmp_gin = ctmp_gin
        self.contract = contract
        self.sample_prior_in_train = bool(sample_prior_in_train)
        self.discharge_placeholder_index = int(discharge_placeholder_index)
        self.register_buffer(
            "admission_idx_t", torch.tensor(admission_col_indices, dtype=torch.long)
        )
        self.register_buffer(
            "discharge_idx_t", torch.tensor(discharge_col_indices, dtype=torch.long)
        )
        forecast_discharge_col_indices = [int(head.target_col_idx) for head in contract.heads]
        missing_forecast_cols = sorted(
            set(forecast_discharge_col_indices) - set(int(idx) for idx in discharge_col_indices)
        )
        if missing_forecast_cols:
            raise ValueError(
                "Forecast discharge columns must be included in discharge_col_indices: "
                f"missing={missing_forecast_cols}"
            )
        self.register_buffer(
            "forecast_discharge_idx_t",
            torch.tensor(forecast_discharge_col_indices, dtype=torch.long),
        )

    def _build_ctmp_input(self, x: torch.Tensor) -> torch.Tensor:
        x_stage2 = x.clone()
        x_stage2[:, self.forecast_discharge_idx_t] = int(self.discharge_placeholder_index)
        return x_stage2

    def _diagnostics(
        self,
        predictor_output: JointGenerativePredictorOutput,
    ) -> dict[str, float]:
        d_entropy_values = []
        for probs in predictor_output.prior_d_probs.values():
            entropy = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum(dim=1)
            d_entropy_values.append(entropy)
        if d_entropy_values:
            d_entropy_mean = torch.stack(d_entropy_values, dim=0).mean()
        else:
            d_entropy_mean = predictor_output.prior_los_probs.new_zeros(())
        los_entropy = -(
            predictor_output.prior_los_probs.clamp_min(1.0e-12)
            * predictor_output.prior_los_probs.clamp_min(1.0e-12).log()
        ).sum(dim=1).mean()
        return {
            "d_entropy_mean": float(d_entropy_mean.detach().cpu()),
            "los_entropy_mean": float(los_entropy.detach().cpu()),
            "mu_p_mean": float(predictor_output.mu_p.mean().detach().cpu()),
            "mu_p_std": float(
                predictor_output.mu_p.std(unbiased=False).detach().cpu()
            ),
            "logstd_p_mean": float(
                predictor_output.logstd_p.mean().detach().cpu()
            ),
            "logstd_p_std": float(
                predictor_output.logstd_p.std(unbiased=False).detach().cpu()
            ),
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> OutcomeAwareForecastedCTMPGINOutput:
        ad_x = torch.index_select(x.long(), dim=1, index=self.admission_idx_t)
        predictor_output = self.predictor.forward_prior(
            ad_x,
            sample=bool(self.training and self.sample_prior_in_train),
        )
        x_stage2 = self._build_ctmp_input(x.long())
        los_probs = predictor_output.prior_los_probs
        if los_probs.shape[1] in {6, 9} and int(self.ctmp_gin.max_los) == 37:
            los_probs = expand_coarse_distribution_to_raw_los(los_probs)
        soft_discharge = build_soft_discharge_payload(
            self.contract,
            d_probs=predictor_output.prior_d_probs,
            d_logits=predictor_output.prior_d_logits,
            device=x.device,
        )
        reasonb_logits = self.ctmp_gin(
            x_stage2,
            los_probs,
            edge_index,
            soft_discharge=soft_discharge,
        )
        return OutcomeAwareForecastedCTMPGINOutput(
            reasonb_logits=reasonb_logits,
            predictor_output=predictor_output,
            forecast_d_probs=predictor_output.prior_d_probs,
            forecast_los_probs=los_probs,
            diagnostics=self._diagnostics(predictor_output),
        )


class OutcomeAwareForecastedGIN(nn.Module):
    def __init__(
        self,
        predictor: JointGenerativePredictor,
        gin: GIN,
        contract: SoftDischargeContract,
        admission_col_indices: list[int],
        discharge_col_indices: list[int],
        *,
        sample_prior_in_train: bool = False,
        discharge_placeholder_index: int = 0,
    ) -> None:
        super().__init__()
        self.predictor = predictor
        self.gin = gin
        self.contract = contract
        self.sample_prior_in_train = bool(sample_prior_in_train)
        self.discharge_placeholder_index = int(discharge_placeholder_index)
        self.register_buffer(
            "admission_idx_t", torch.tensor(admission_col_indices, dtype=torch.long)
        )
        self.register_buffer(
            "discharge_idx_t", torch.tensor(discharge_col_indices, dtype=torch.long)
        )
        forecast_discharge_col_indices = [int(head.target_col_idx) for head in contract.heads]
        missing_forecast_cols = sorted(
            set(forecast_discharge_col_indices) - set(int(idx) for idx in discharge_col_indices)
        )
        if missing_forecast_cols:
            raise ValueError(
                "Forecast discharge columns must be included in discharge_col_indices: "
                f"missing={missing_forecast_cols}"
            )
        self.register_buffer(
            "forecast_discharge_idx_t",
            torch.tensor(forecast_discharge_col_indices, dtype=torch.long),
        )

    def _build_gin_input(self, x: torch.Tensor) -> torch.Tensor:
        x_stage2 = x.clone()
        x_stage2[:, self.forecast_discharge_idx_t] = int(self.discharge_placeholder_index)
        return x_stage2

    def _diagnostics(
        self,
        predictor_output: JointGenerativePredictorOutput,
    ) -> dict[str, float]:
        d_entropy_values = []
        for probs in predictor_output.prior_d_probs.values():
            entropy = -(probs.clamp_min(1.0e-12) * probs.clamp_min(1.0e-12).log()).sum(dim=1)
            d_entropy_values.append(entropy)
        if d_entropy_values:
            d_entropy_mean = torch.stack(d_entropy_values, dim=0).mean()
        else:
            d_entropy_mean = predictor_output.prior_los_probs.new_zeros(())
        los_entropy = -(
            predictor_output.prior_los_probs.clamp_min(1.0e-12)
            * predictor_output.prior_los_probs.clamp_min(1.0e-12).log()
        ).sum(dim=1).mean()
        return {
            "d_entropy_mean": float(d_entropy_mean.detach().cpu()),
            "los_entropy_mean": float(los_entropy.detach().cpu()),
            "mu_p_mean": float(predictor_output.mu_p.mean().detach().cpu()),
            "mu_p_std": float(
                predictor_output.mu_p.std(unbiased=False).detach().cpu()
            ),
            "logstd_p_mean": float(
                predictor_output.logstd_p.mean().detach().cpu()
            ),
            "logstd_p_std": float(
                predictor_output.logstd_p.std(unbiased=False).detach().cpu()
            ),
        }

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> OutcomeAwareForecastedCTMPGINOutput:
        ad_x = torch.index_select(x.long(), dim=1, index=self.admission_idx_t)
        predictor_output = self.predictor.forward_prior(
            ad_x,
            sample=bool(self.training and self.sample_prior_in_train),
        )
        x_stage2 = self._build_gin_input(x.long())
        los_probs = predictor_output.prior_los_probs
        if los_probs.shape[1] in {6, 9}:
            los_probs = expand_coarse_distribution_to_raw_los(los_probs)
        soft_discharge = build_soft_discharge_payload(
            self.contract,
            d_probs=predictor_output.prior_d_probs,
            d_logits=predictor_output.prior_d_logits,
            device=x.device,
        )
        reasonb_logits = self.gin(
            x_stage2,
            los_probs,
            edge_index,
            soft_discharge=soft_discharge,
        )
        return OutcomeAwareForecastedCTMPGINOutput(
            reasonb_logits=reasonb_logits,
            predictor_output=predictor_output,
            forecast_d_probs=predictor_output.prior_d_probs,
            forecast_los_probs=los_probs,
            diagnostics=self._diagnostics(predictor_output),
        )
