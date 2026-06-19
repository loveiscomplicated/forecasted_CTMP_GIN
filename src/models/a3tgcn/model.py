import torch
import torch.nn as nn
import sys
import os
cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from src.models.entity_embedding import EntityEmbeddingBatch3
from src.models.a3tgcn.attentiontemporalgcn import A3TGCN2

def to_temporal(x_tensor: torch.Tensor, # shape: [batch_size, num_var, feature_dim] (=[32, 72, 25])
                ad_col_index: list, 
                dis_col_index: list,
                LOS: torch.Tensor,
                device,
                max_los=37):
    batch_size, _, num_features = x_tensor.shape
    num_nodes = len(ad_col_index)

    ad_idx_t = torch.tensor(ad_col_index, device=device)
    dis_idx_t = torch.tensor(dis_col_index, device=device)

    ad_tensor = torch.index_select(x_tensor, dim=1, index=ad_idx_t)
    dis_tensor = torch.index_select(x_tensor, dim=1, index=dis_idx_t)

    # Create a temporal mask based on LOS
    los_mask = torch.arange(max_los, device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0) < LOS.unsqueeze(1).unsqueeze(1).unsqueeze(1)
    los_mask = los_mask.expand(batch_size, num_nodes, num_features, max_los)
    
    # Create the temporal tensor
    temporal_tensor = torch.where(los_mask, ad_tensor.unsqueeze(-1), dis_tensor.unsqueeze(-1))

    return temporal_tensor



class A3TGCN_manual(nn.Module):
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

        # EntityEmbedding 레이어 정의
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        
        # A3TGCN2 레이어 정의
        a3tgcn_input_channel = embedding_dim

        self.a3tgcn_layer = A3TGCN2(in_channels=a3tgcn_input_channel,
                        out_channels=hidden_channel,
                        periods=37,
                        batch_size=batch_size,
                        device=device,
                        cached=cached) # 이거 지이이이이이이이인짜 중요함 이걸 해야 성능이 완전 좋아짐

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
        x_embedded = self.entity_embedding_layer(x_batch)
        x = to_temporal(x_embedded, self.ad_col_index, self.dis_col_index, LOS_batch, device)
        after_GNN = self.a3tgcn_layer(x, template_edge_index) # [32, 60, 32]
        
        # global mean pooling [32, 60, 32] -> [32, 32]
        mean_pooled = torch.mean(after_GNN, dim=1)

        return self.classifier(mean_pooled)
    
