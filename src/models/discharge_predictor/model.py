from typing import Dict, List

import torch
import torch.nn as nn

from src.models.entity_embedding import EntityEmbeddingBatch3


class MultiTaskDischargePredictor(nn.Module):
    """Predicts discharge-side categorical variables and LOS from admission variables.

    Architecture:
        admission embeddings (EntityEmbeddingBatch3)
        → flatten
        → shared BatchNorm+ReLU MLP encoder
        → one linear head per target variable
    """

    def __init__(
        self,
        ad_col_dims: List[int],
        target_col_names: List[str],
        target_col_dims: List[int],
        embedding_dim: int = 16,
        hidden_dim: int = 256,
        num_layers: int = 3,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__()

        self.target_col_names = list(target_col_names)

        self.embedding = EntityEmbeddingBatch3(
            col_dims=ad_col_dims, embedding_dim=embedding_dim
        )

        encoder_input_dim = len(ad_col_dims) * embedding_dim
        layers: List[nn.Module] = []
        in_dim = encoder_input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        self.heads = nn.ModuleDict(
            {
                name: nn.Linear(hidden_dim, cardinality)
                for name, cardinality in zip(target_col_names, target_col_dims)
            }
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [batch, num_ad_vars] integer tensor of admission variable indices

        Returns:
            Dict mapping target variable name to logits [batch, cardinality_k]
        """
        emb = self.embedding(x)  # [batch, num_ad_vars, emb_dim]
        batch_size = emb.shape[0]
        z = emb.reshape(batch_size, -1)  # [batch, num_ad_vars * emb_dim]
        h = self.encoder(z)  # [batch, hidden_dim]
        return {name: head(h) for name, head in self.heads.items()}
