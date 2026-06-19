from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discharge_predictor.conditioners import (
    MultiHeadExpectedEmbedding,
    ExpectedCategoricalEmbedding,
    mix_condition_distributions,
    one_hot_distribution,
    resolve_joint_heads,
)
from src.models.entity_embedding import EntityEmbeddingBatch3


@dataclass
class JointPredictorOutput:
    base_d_logits: Dict[str, torch.Tensor]
    final_d_logits: Dict[str, torch.Tensor]
    base_d_probs: Dict[str, torch.Tensor]
    final_d_probs: Dict[str, torch.Tensor]
    base_los_logits: torch.Tensor
    final_los_logits: torch.Tensor
    base_los_probs: torch.Tensor
    final_los_probs: torch.Tensor
    shared_hidden: torch.Tensor


class FixedOneHotBatchEncoder(nn.Module):
    """Encode admission categorical columns as a fixed concatenated one-hot vector."""

    def __init__(self, col_dims: Sequence[int]) -> None:
        super().__init__()
        col_dims_t = torch.as_tensor(col_dims, dtype=torch.long)
        self.register_buffer("col_dims", col_dims_t)

    @property
    def output_dim(self) -> int:
        return int(self.col_dims.sum().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        if x.ndim != 2:
            raise ValueError(f"FixedOneHotBatchEncoder expected rank-2 input, got shape={tuple(x.shape)}")
        if x.shape[1] != int(self.col_dims.numel()):
            raise ValueError(
                f"FixedOneHotBatchEncoder column mismatch: expected {int(self.col_dims.numel())}, got {x.shape[1]}"
            )
        parts = [
            F.one_hot(x[:, idx], num_classes=int(cardinality)).to(dtype=torch.float32)
            for idx, cardinality in enumerate(self.col_dims.tolist())
        ]
        return torch.cat(parts, dim=1)


class JointConsistentPredictor(nn.Module):
    """Shared admission-only predictor for discharge targets and LOS."""

    def __init__(
        self,
        ad_col_dims: List[int],
        target_col_names: Sequence[str],
        target_col_dims: Sequence[int],
        *,
        los_num_classes: int,
        joint_direction: str = "los_to_d",
        condition_mode: str = "predicted",
        detach_condition: bool = True,
        joint_heads: str | Sequence[str] | None = None,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        conditioner_embedding_dim: int | None = None,
        input_encoding: str = "onehot",
        **kwargs,
    ) -> None:
        super().__init__()
        self.target_col_names = list(target_col_names)
        self.target_col_dims = [int(dim) for dim in target_col_dims]
        self.ad_col_dims = [int(dim) for dim in ad_col_dims]
        self.target_dim_map = {
            name: int(dim) for name, dim in zip(self.target_col_names, self.target_col_dims)
        }
        self.los_num_classes = int(los_num_classes)
        self.joint_direction = str(joint_direction).lower()
        self.condition_mode = str(condition_mode).lower()
        self.detach_condition = bool(detach_condition)
        self.input_encoding = str(input_encoding).lower()
        if self.joint_direction not in {"independent", "los_to_d", "d_to_los", "bidirectional"}:
            raise ValueError(f"Unsupported joint_direction: {joint_direction}")
        if self.condition_mode not in {"predicted", "oracle", "scheduled"}:
            raise ValueError(f"Unsupported condition_mode: {condition_mode}")
        if self.input_encoding not in {"embedding", "onehot"}:
            raise ValueError(
                f"Unsupported input_encoding: {input_encoding}. Expected one of ['embedding', 'onehot']."
            )

        cond_embedding_dim = int(conditioner_embedding_dim or embedding_dim)
        self.embedding_input_dim = len(self.ad_col_dims) * int(embedding_dim)
        self.onehot_input_dim = int(sum(self.ad_col_dims))
        if self.input_encoding == "embedding":
            self.admission_encoder: nn.Module = EntityEmbeddingBatch3(
                col_dims=self.ad_col_dims,
                embedding_dim=embedding_dim,
            )
            self.embedding: EntityEmbeddingBatch3 | None = self.admission_encoder
            encoder_input_dim = self.embedding_input_dim
        else:
            self.admission_encoder = FixedOneHotBatchEncoder(col_dims=self.ad_col_dims)
            self.embedding = None
            encoder_input_dim = self.onehot_input_dim
        self.actual_encoder_input_dim = int(encoder_input_dim)
        layers: List[nn.Module] = []
        in_dim = encoder_input_dim
        for _ in range(int(num_layers)):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        first_linear = next((module for module in self.encoder if isinstance(module, nn.Linear)), None)
        if first_linear is None:
            raise RuntimeError("JointConsistentPredictor encoder must include at least one nn.Linear layer")

        self.base_d_heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, int(dim))
                for name, dim in zip(self.target_col_names, self.target_col_dims)
            }
        )
        self.base_los_head = nn.Linear(hidden_dim, self.los_num_classes)

        self.los_conditioner = ExpectedCategoricalEmbedding(
            num_classes=self.los_num_classes,
            embedding_dim=cond_embedding_dim,
        )
        self.selected_joint_heads = resolve_joint_heads(self.target_col_names, joint_heads)
        self.discharge_conditioner = MultiHeadExpectedEmbedding(
            head_dims=self.target_dim_map,
            selected_heads=self.selected_joint_heads,
            embedding_dim=cond_embedding_dim,
        )

        d_in_dim = hidden_dim + cond_embedding_dim
        los_in_dim = hidden_dim + self.discharge_conditioner.output_dim
        self.final_d_heads = nn.ModuleDict(
            {
                name: nn.Linear(d_in_dim, int(dim))
                for name, dim in zip(self.target_col_names, self.target_col_dims)
            }
        )
        self.final_los_head = nn.Linear(los_in_dim, self.los_num_classes)

        print("[JointConsistentPredictor admission encoder]")
        print(f"input_encoding={self.input_encoding}")
        print(f"num_admission_columns={len(self.ad_col_dims)}")
        print(f"sum_admission_cardinalities={self.onehot_input_dim}")
        print(f"embedding_input_dim={self.embedding_input_dim}")
        print(f"onehot_input_dim={self.onehot_input_dim}")
        print(f"actual_encoder_input_dim={self.actual_encoder_input_dim}")
        print(f"shared_encoder_first_linear_param_count={int(first_linear.weight.numel() + first_linear.bias.numel())}")

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.admission_encoder(x)
        if self.input_encoding == "embedding":
            z = encoded.reshape(encoded.shape[0], -1)
        else:
            z = encoded
        return self.encoder(z)

    def _base_logits(self, h: torch.Tensor) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        return (
            {name: head(h) for name, head in self.base_d_heads.items()},
            self.base_los_head(h),
        )

    def _resolve_condition_probs(
        self,
        predicted_probs: torch.Tensor,
        *,
        oracle_targets: torch.Tensor | None,
        num_classes: int,
        oracle_ratio: float,
    ) -> torch.Tensor:
        oracle_probs = None
        if oracle_targets is not None:
            oracle_probs = one_hot_distribution(oracle_targets, num_classes=num_classes).to(
                device=predicted_probs.device,
                dtype=predicted_probs.dtype,
            )
        if self.condition_mode == "predicted":
            condition_probs = predicted_probs
        elif oracle_probs is None:
            raise ValueError(
                f"condition_mode={self.condition_mode} requires oracle targets for conditional branches"
            )
        else:
            condition_probs = mix_condition_distributions(
                predicted_probs,
                oracle_probs,
                mode=self.condition_mode,
                oracle_ratio=oracle_ratio,
            )
        if self.detach_condition:
            condition_probs = condition_probs.detach()
        return condition_probs

    def forward(
        self,
        x: torch.Tensor,
        *,
        los_targets: torch.Tensor | None = None,
        d_targets: Dict[str, torch.Tensor] | None = None,
        oracle_ratio: float = 0.0,
    ) -> JointPredictorOutput:
        h = self._encode(x)
        base_d_logits, base_los_logits = self._base_logits(h)
        base_d_probs = {name: F.softmax(logits, dim=1) for name, logits in base_d_logits.items()}
        base_los_probs = F.softmax(base_los_logits, dim=1)

        final_d_logits = dict(base_d_logits)
        final_los_logits = base_los_logits

        if self.joint_direction in {"los_to_d", "bidirectional"}:
            los_condition_probs = self._resolve_condition_probs(
                base_los_probs,
                oracle_targets=los_targets,
                num_classes=self.los_num_classes,
                oracle_ratio=oracle_ratio,
            )
            e_los = self.los_conditioner(los_condition_probs)
            d_input = torch.cat([h, e_los], dim=1)
            final_d_logits = {
                name: head(d_input) for name, head in self.final_d_heads.items()
            }

        if self.joint_direction in {"d_to_los", "bidirectional"}:
            conditioned_probs: Dict[str, torch.Tensor] = {}
            for head_name in self.selected_joint_heads:
                conditioned_probs[head_name] = self._resolve_condition_probs(
                    base_d_probs[head_name],
                    oracle_targets=None if d_targets is None else d_targets[head_name],
                    num_classes=self.target_dim_map[head_name],
                    oracle_ratio=oracle_ratio,
                )
            s_d = self.discharge_conditioner(conditioned_probs)
            los_input = torch.cat([h, s_d], dim=1)
            final_los_logits = self.final_los_head(los_input)

        final_d_probs = {name: F.softmax(logits, dim=1) for name, logits in final_d_logits.items()}
        final_los_probs = F.softmax(final_los_logits, dim=1)
        return JointPredictorOutput(
            base_d_logits=base_d_logits,
            final_d_logits=final_d_logits,
            base_d_probs=base_d_probs,
            final_d_probs=final_d_probs,
            base_los_logits=base_los_logits,
            final_los_logits=final_los_logits,
            base_los_probs=base_los_probs,
            final_los_probs=final_los_probs,
            shared_hidden=h,
        )
