from typing import List

import torch
import torch.nn as nn

from src.models.entity_embedding import EntityEmbeddingBatch3


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout_p: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(dim, dim),
            nn.Dropout(dropout_p),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class LOSOrdinalPredictor(nn.Module):
    """Predicts LOS as an ordinal class from admission-only variables.

    Architecture:
        EntityEmbeddingBatch3 → flatten → projection (Linear+LayerNorm+GELU)
        → num_layers × ResidualBlock → Linear head (hidden_dim → los_num_classes - 1)

    The head outputs K-1 raw logits for cumulative ordinal regression.
    Use ordinal_logits_to_class() to decode to a predicted class.
    """

    def __init__(
        self,
        ad_col_dims: List[int],
        los_num_classes: int,
        embedding_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout_p: float = 0.2,
        output_mode: str = "ordinal",
        **kwargs,
    ) -> None:
        super().__init__()

        self.los_num_classes = los_num_classes
        self.output_mode = output_mode

        self.embedding = EntityEmbeddingBatch3(
            col_dims=ad_col_dims, embedding_dim=embedding_dim
        )

        flat_dim = len(ad_col_dims) * embedding_dim
        self.proj = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout_p) for _ in range(num_layers)]
        )

        self.head = nn.Linear(hidden_dim, los_num_classes - 1)
        self.ce_head = nn.Linear(hidden_dim, los_num_classes)

        if self.output_mode not in {"ordinal", "ce", "hybrid"}:
            raise ValueError(f"Unsupported output_mode: {self.output_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_ad_vars] integer tensor of admission variable indices

        Returns:
            ordinal: [batch, los_num_classes - 1] raw ordinal logits
            ce:      [batch, los_num_classes] raw class logits
            hybrid:  dict with both heads
        """
        emb = self.embedding(x)                      # [B, num_vars, emb_dim]
        z = emb.reshape(emb.shape[0], -1)            # [B, flat_dim]
        h = self.proj(z)                             # [B, hidden_dim]
        h = self.blocks(h)                           # [B, hidden_dim]
        if self.output_mode == "ce":
            return self.ce_head(h)                    # [B, K]
        if self.output_mode == "hybrid":
            return {
                "ordinal": self.head(h),              # [B, K-1]
                "ce": self.ce_head(h),                # [B, K]
            }
        return self.head(h)                           # [B, K-1]
