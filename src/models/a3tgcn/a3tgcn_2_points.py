import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from src.models.entity_embedding import EntityEmbeddingBatch3
from src.models.a3tgcn.attentiontemporalgcn import A3TGCN2


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


def dual_time_stamp(x: torch.Tensor, ad_idx_t: torch.Tensor, dis_idx_t: torch.Tensor):
    """
    Split x into admission and discharge variables, then stack along a new period dimension.

    Args:
        x: [B, V, F]
        ad_idx_t: [N] indices for admission variables
        dis_idx_t: [N] indices for discharge variables

    Returns:
        [B, N, F, 2]
    """
    ad_tensor = torch.index_select(x, dim=1, index=ad_idx_t)   # [B, N, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t) # [B, N, F]
    return torch.stack([ad_tensor, dis_tensor], dim=-1)

class A3TGCN_2_points(nn.Module):
    '''
    tensor 연산 위주로 수행하는 모델
    '''
    def __init__(self, 
                 batch_size, 
                 col_info, 
                 embedding_dim, 
                 hidden_channel, 
                 num_classes,
                 device,
                 cached=True,
                 **kwargs):
        '''
        Args:
            col_info(list): [col_dims, col_list]
                            col_list(list): 데이터에서 나타나는 변수의 순서
                            col_dims(list): 각 변수 별 범주의 개수, 순서는 col_list를 따라야 함
            embedding_dim(int): 엔티티 임베딩 후의 차원
            hidden_channel(int): TGCN의 hidden channel
        '''
        super().__init__()
        self.batch_size = batch_size
        self.hidden_channel = hidden_channel
        self.num_classes = num_classes

        # col_info: (col_list, col_dims, ad_col_index, dis_col_index)
        self.col_list, self.col_dims, self.ad_col_index, self.dis_col_index = col_info

        self.register_buffer('ad_idx_t', torch.tensor(self.ad_col_index, dtype=torch.long))
        self.register_buffer('dis_idx_t', torch.tensor(self.dis_col_index, dtype=torch.long))

        # EntityEmbedding 레이어 정의
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        
        # A3TGCN2 레이어 정의
        # append_los_to_vars adds 1 more feature (LOS) to the embedded features.
        a3tgcn_input_channel = embedding_dim

        self.a3tgcn_layer = A3TGCN2(in_channels=a3tgcn_input_channel,
                        out_channels=hidden_channel,
                        periods=2,
                        batch_size=batch_size,
                        device=device,
                        cached=cached)

        # 분류기 레이어 정의
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel * 2),
            nn.ReLU(),
            nn.Linear(hidden_channel * 2, out_dim),
        )
    
    def forward(self, x_batch: torch.Tensor, LOS_batch: torch.Tensor, template_edge_index: torch.Tensor, device:torch.device):
        '''
        Args:
            template_edge_index(torch.Tensor): edge_index는 동일하므로 template_edge_index로 한꺼번에 전달
        '''
        los_idx = LOS_batch.long().unsqueeze(1)
        x_combined = torch.cat([x_batch, los_idx], dim=1) # [B, 73]
        
        # Embed variables: [B, V, embedding_dim]
        x_embedded = self.entity_embedding_layer(x_combined)

        # Select admission/discharge variable sets and concat along batch: [B*2, N, F]
        x_temporal = dual_time_stamp(x_embedded, ad_idx_t=self.ad_idx_t, dis_idx_t=self.dis_idx_t) # type: ignore

        after_GNN = self.a3tgcn_layer(x_temporal, template_edge_index)  # [B, N, hidden_channel]

        # global mean pooling: [B, N, hidden_channel] -> [B, hidden_channel]
        mean_pooled = torch.mean(after_GNN, dim=1)

        return self.classifier(mean_pooled)
    
