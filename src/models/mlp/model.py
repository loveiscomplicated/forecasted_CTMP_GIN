import torch
import torch.nn as nn

from src.models.entity_embedding import EntityEmbeddingBatch3


class MLP(nn.Module):
    def __init__(
        self,
        embedding_dim,
        col_info,
        hidden_dim,
        num_layers,
        num_classes,
        dropout=0.0,
        use_los=True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.col_dims = list(col_info[1])
        self.use_los = use_los
        self.num_classes = num_classes

        self.entity_embedding_layer = EntityEmbeddingBatch3(
            col_dims=self.col_dims, embedding_dim=embedding_dim
        )

        num_features = len(self.col_dims)
        input_dim = num_features * embedding_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim

        out_dim = 1 if num_classes == 2 else num_classes
        layers.append(nn.Linear(hidden_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x, los, edge_index, **kwargs):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        if self.use_los:
            los = los.unsqueeze(dim=1)
            x = torch.cat((x, los), dim=1)

        x_embedded = self.entity_embedding_layer(x)  # [batch, F, emb_dim]
        batch_size = x_embedded.shape[0]
        x_flat = x_embedded.reshape(batch_size, -1)  # [batch, F * emb_dim]

        return self.mlp(x_flat)
