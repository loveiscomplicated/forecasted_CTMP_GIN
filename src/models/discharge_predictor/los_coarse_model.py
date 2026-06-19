from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

from src.models.entity_embedding import EntityEmbeddingBatch3


class LOSCoarsePredictor(nn.Module):
    """Predict coarse LOS duration regimes from admission-only variables."""

    def __init__(
        self,
        ad_col_dims: List[int],
        num_classes: int = 6,
        embedding_dim: int = 32,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout_p: float = 0.2,
        activation: str = "gelu",
        norm_type: str = "layernorm",
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)

        self.embedding = EntityEmbeddingBatch3(col_dims=ad_col_dims, embedding_dim=embedding_dim)
        input_dim = len(ad_col_dims) * embedding_dim

        act_cls: type[nn.Module]
        if activation.lower() == "relu":
            act_cls = nn.ReLU
        elif activation.lower() == "gelu":
            act_cls = nn.GELU
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        def make_norm(dim: int) -> nn.Module:
            if norm_type.lower() in {"layernorm", "ln"}:
                return nn.LayerNorm(dim)
            if norm_type.lower() in {"batchnorm", "bn", "batchnorm1d"}:
                return nn.BatchNorm1d(dim)
            raise ValueError(f"Unsupported norm_type: {norm_type}")

        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    make_norm(hidden_dim),
                    act_cls(),
                    nn.Dropout(dropout_p),
                ]
            )
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dim, self.num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return coarse LOS logits."""
        emb = self.embedding(x)
        z = emb.reshape(emb.shape[0], -1)
        h = self.encoder(z)
        return self.head(h)

