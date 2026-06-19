import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch.nn import GRU

cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)

from src.models.entity_embedding import EntityEmbeddingBatch3


def append_los_to_vars(x: torch.Tensor, los_batch: torch.Tensor, max_los: float = 37.0):
    """
    Append LOS as an additional feature to every variable/node feature vector.

    Args:
        x: [B, V, F]  (V = num variables, e.g., 72)
        los_batch: [B]
        max_los: used for min-max style scaling (los/max_los)

    Returns:
        [B, V, F+1]
    """
    B, V, _ = x.shape
    los_feat = los_batch.float().unsqueeze(1).unsqueeze(2).expand(B, V, 1)

    # (권장) 스케일 맞추기: 0~1 근처로
    if max_los is not None and max_los > 0:
        los_feat = los_feat / float(max_los)

    return torch.cat([x, los_feat], dim=-1)


def to_two_step_sequence(x: torch.Tensor):
    """
    Convert concatenated [ad; dis] graph embeddings into a 2-step GRU sequence.

    Args:
        x: [B*2, F]  (first B: admission, next B: discharge)

    Returns:
        seq: [2, B, F]  (GRU input with batch_first=False)
    """
    B = x.shape[0] // 2
    ad = x[:B]
    dis = x[B:]
    return torch.stack([ad, dis], dim=0)


def separate_x(x: torch.Tensor, ad_idx_t: torch.Tensor, dis_idx_t: torch.Tensor):
    """
    Split x into admission and discharge variables, then concat along batch dimension.

    Args:
        x: [B, V, F]
        ad_idx_t: [N] indices for admission variables
        dis_idx_t: [N] indices for discharge variables

    Returns:
        [B*2, N, F]
    """
    ad_tensor = torch.index_select(x, dim=1, index=ad_idx_t)   # [B, N, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t) # [B, N, F]
    return torch.cat([ad_tensor, dis_tensor], dim=0)           # [B*2, N, F]


class GinGru_2_Point(nn.Module):
    def __init__(
        self,
        col_info,
        embedding_dim,
        gin_hidden_channel,
        train_eps,
        gin_layers,
        gru_hidden_channel,
        num_classes,
        dropout_p: float = 0.2,
        gin_layer_out_dropout_p: float = 0.2,
        gru_layer_out_dropout_p: float = 0.2,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channel = gin_hidden_channel
        self.dropout_p = float(dropout_p)
        self.gin_layer_out_dropout_p = float(gin_layer_out_dropout_p) 
        self.gru_layer_out_dropout_p = float(gru_layer_out_dropout_p)
        self.max_los = int(kwargs.get("max_los", 37))

        self.gin_layer_out_dropout = nn.Dropout(self.gin_layer_out_dropout_p)
        self.gru_layer_out_dropout = nn.Dropout(self.gru_layer_out_dropout_p)

        # col_info: (col_list, col_dims, ad_col_index, dis_col_index)
        self.col_list, self.col_dims, ad_col_index, dis_col_index = col_info
        self.register_buffer('ad_idx_t', torch.tensor(ad_col_index, dtype=torch.long))
        self.register_buffer('dis_idx_t', torch.tensor(dis_col_index, dtype=torch.long))

        # Entity Embedding
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)

        # GIN MLPs
        gin_nn_input = nn.Sequential(
            nn.Linear(embedding_dim, gin_hidden_channel),
            nn.LayerNorm(gin_hidden_channel),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(gin_hidden_channel, gin_hidden_channel),
        )

        gin_nn = nn.Sequential(
            nn.Linear(gin_hidden_channel, gin_hidden_channel),
            nn.LayerNorm(gin_hidden_channel),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(gin_hidden_channel, gin_hidden_channel),
        )

        # GIN layers
        self.gin_layers = nn.ModuleList()
        self.gin_layers.append(GINConv(nn=gin_nn_input, eps=0, train_eps=train_eps))
        for _ in range(gin_layers - 1):
            self.gin_layers.append(GINConv(nn=gin_nn, eps=0, train_eps=train_eps))

        # GRU
        gru_input_ch = gin_hidden_channel * gin_layers
        self.gru_layer = GRU(input_size=gru_input_ch, hidden_size=gru_hidden_channel)  # batch_first=False

        # Classifier
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_channel, gru_hidden_channel * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(gru_hidden_channel * 2, out_dim),
        )

        # keep refs for reset_parameters
        self.gin_nn_input = gin_nn_input
        self.gin_nn = gin_nn

        self.reset_parameters()

    def forward(self, x_batch: torch.Tensor, LOS_batch: torch.Tensor, template_edge_index, device):
        """
        Args:
            x_batch: [B, V]  categorical variable indices (pre-embedding)
            LOS_batch: [B]
            template_edge_index: edge_index for the *flattened batch graph* (supersized / block-diagonal)
            device: torch device

        Returns:
            logits: [B, 1]
        """

        batch_size = x_batch.shape[0]
        los_idx = LOS_batch.long().unsqueeze(1)
        x_combined = torch.cat([x_batch, los_idx], dim=1) # [B, 73]
        num_nodes = x_combined.shape[1]

        # Embed variables: [B, V, embedding_dim]
        x_embedded = self.entity_embedding_layer(x_combined)

        # Append LOS as node/variable feature: [B, V, embedding_dim + 1] --> updated to making LOS as node.
        # x_embedded = append_los_to_vars(x_embedded, LOS_batch, max_los=self.max_los_norm)

        # Select admission/discharge variable sets and concat along batch: [B*2, N, F]
        x_separated = separate_x(x_embedded, ad_idx_t=self.ad_idx_t, dis_idx_t=self.dis_idx_t) # type: ignore
        # Flatten into [B*2*N, F]
        num_separated_nodes = int(self.ad_idx_t.numel())
        x_after_gin = x_separated.reshape(batch_size * 2 * num_separated_nodes, -1)

        # Apply GIN layers, sum-pool per graph at each layer, then concat pooled outputs
        sum_pooled = []
        for layer in self.gin_layers:
            x_after_gin = layer(x_after_gin, template_edge_index)   # [B*2*N, gin_hidden_channel]
            x_after_gin = self.gin_layer_out_dropout(x_after_gin)

            x_graph = x_after_gin.reshape(batch_size * 2, num_separated_nodes, self.hidden_channel)  # [B*2, N, H]
            x_sum = torch.sum(x_graph, dim=1)  # [B*2, H]
            sum_pooled.append(x_sum)

        gin_result = torch.cat(sum_pooled, dim=1)  # [B*2, H * num_layers]

        # 2-step temporal sequence for GRU: [2, B, H*num_layers]
        gru_input = to_two_step_sequence(gin_result)

        # GRU
        _, gru_h = self.gru_layer(gru_input)  # gru_h: [1, B, gru_hidden_channel]
        gru_h = gru_h.squeeze(0)              # [B, gru_hidden_channel]
        gru_h = self.gru_layer_out_dropout(gru_h)

        # Classifier
        return self.classifier(gru_h)       # [B, 1]

    def reset_parameters(self):
        # GIN MLPs
        for block in [self.gin_nn_input, self.gin_nn]:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)