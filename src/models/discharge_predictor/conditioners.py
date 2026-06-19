from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discharge_predictor.risk_heads import resolve_risk_head_selection


def parse_bool_flag(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"Unsupported boolean flag: {value}")


def resolve_joint_heads(
    target_names: Sequence[str],
    joint_heads: str | Sequence[str] | None,
) -> list[str]:
    return resolve_risk_head_selection(
        joint_heads,
        available_heads=target_names,
        mode="legacy_or_named_set",
        allow_all=True,
        field_name="joint_heads",
    )


def one_hot_distribution(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets.long(), num_classes=num_classes).to(dtype=torch.float32)


def mix_condition_distributions(
    predicted_probs: torch.Tensor,
    oracle_probs: torch.Tensor,
    *,
    mode: str,
    oracle_ratio: float = 0.0,
) -> torch.Tensor:
    resolved_mode = str(mode).lower()
    if resolved_mode == "predicted":
        return predicted_probs
    if resolved_mode == "oracle":
        return oracle_probs
    if resolved_mode == "scheduled":
        alpha = float(max(0.0, min(1.0, oracle_ratio)))
        return alpha * oracle_probs + (1.0 - alpha) * predicted_probs
    raise ValueError(f"Unsupported condition_mode: {mode}")


class ExpectedCategoricalEmbedding(nn.Module):
    """Map a categorical distribution to its expected embedding."""

    def __init__(self, num_classes: int, embedding_dim: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(self.num_classes, self.embedding_dim)

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim != 2 or probs.shape[1] != self.num_classes:
            raise ValueError(
                f"Expected probs shape [B, {self.num_classes}], got {tuple(probs.shape)}."
            )
        weight = self.embedding.weight.to(dtype=probs.dtype)
        return probs @ weight


class MultiHeadExpectedEmbedding(nn.Module):
    """Concatenate expected embeddings from selected categorical heads."""

    def __init__(
        self,
        head_dims: Dict[str, int],
        selected_heads: Sequence[str],
        embedding_dim: int,
    ) -> None:
        super().__init__()
        self.selected_heads = list(selected_heads)
        missing = sorted(set(self.selected_heads) - set(head_dims))
        if missing:
            raise ValueError(f"Missing head dimensions for selected heads: {missing}")
        self.embedding_dim = int(embedding_dim)
        self.embedders = nn.ModuleDict(
            {
                head_name: ExpectedCategoricalEmbedding(
                    num_classes=int(head_dims[head_name]),
                    embedding_dim=self.embedding_dim,
                )
                for head_name in self.selected_heads
            }
        )

    @property
    def output_dim(self) -> int:
        return len(self.selected_heads) * self.embedding_dim

    def forward(self, probs_by_head: Dict[str, torch.Tensor]) -> torch.Tensor:
        parts: List[torch.Tensor] = []
        for head_name in self.selected_heads:
            if head_name not in probs_by_head:
                raise ValueError(f"Missing probability tensor for head {head_name}")
            parts.append(self.embedders[head_name](probs_by_head[head_name]))
        if not parts:
            raise ValueError("MultiHeadExpectedEmbedding requires at least one selected head")
        return torch.cat(parts, dim=1)
