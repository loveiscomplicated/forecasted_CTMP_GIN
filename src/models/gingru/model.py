
import os
import sys
import torch
import torch.nn as nn
from torch_geometric.nn import GINConv
from torch.nn import GRU
from torch.nn.utils.rnn import pack_padded_sequence


cur_dir = os.path.dirname(__file__)
parent_dir = os.path.join(cur_dir, '..')
sys.path.append(parent_dir)
from src.models.entity_embedding import EntityEmbeddingBatch3

def get_mask(los_batch: torch.Tensor):
    '''
    주어진 배치별 각 케이스의 시간 길이 (sequence length) 정보를 담고 있는 텐서를 사용하여, 
    해당 길이에 해당하는 부분은 1로, 나머지 부분은 0으로 채워진 마스크 텐서를 생성
    '''
    max_los = 37

    indices = torch.arange(max_los, device=los_batch.device) 
    mask = (indices < los_batch.unsqueeze(1))

    return mask.int().unsqueeze(2)


def to_temporal_gingru(x: torch.Tensor, los_batch: torch.Tensor):
    # process:  [batch, T, gin_h_dim]으로 변환하기 (gin_h_dim = F*num_layers)
    # 1. [batch, 2, gin_h_dim]으로 변환하기 --> .reshape(-1, 2, F)
    # 2. [batch, T, gin_h_dim]으로 변환하기
    # process: PackedSequence로 변환하기

    batch_size = int(x.shape[0] / 2) # [batch(twiced), f]

    ad = x[:batch_size].unsqueeze(1) # 브로드캐스팅을 위해 [B, 1, F]
    dis = x[batch_size:]
    
    # los_batch를 가지고 0,1 마스킹 행렬 생성
    mask = get_mask(los_batch=los_batch) # [B, 37, 1]
    padded = ad * mask # [B, 1, F] * [B, 37, 1] --> [B, 37, F] @@ 브로드캐스팅 !! @@

    batch_indices = torch.arange(batch_size, device=ad.device)
    last_time_indices = los_batch - 1

    padded[batch_indices, last_time_indices, :] = dis

    # -----PackedSequence 만들기-----
    lengths = los_batch.cpu() 

    # 내림차순(descending=True)으로 정렬
    lengths_sorted, sorted_indices = torch.sort(lengths, descending=True)

    padded_sorted = padded[sorted_indices]

    packed_data = pack_padded_sequence(
        padded_sorted,
        lengths_sorted,
        batch_first=True,
        enforce_sorted=True
    )

    return packed_data, sorted_indices # padded_sorted의 디바이스를 따라감


def seperate_x(x: torch.Tensor, ad_idx_t, dis_idx_t, device):
    
    ad_tensor = torch.index_select(x, dim=1, index=ad_idx_t) # [B, 60, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t) # [B, 60, F]

    return torch.concatenate((ad_tensor, dis_tensor), dim=0) # [B*2, 60, F]

class GinGru(nn.Module):
    def __init__(self, 
                 col_info,
                 embedding_dim, 
                 gin_hidden_channel, 
                 train_eps, 
                 gin_layers, 
                 gru_hidden_channel, 
                 num_classes,
                 dropout_p: float = 0.2,          
                 gin_out_dropout_p = None,
                 **kwargs):
        '''
        Args:
            col_info(list): [col_dims, col_list]
                            col_list(list): 데이터에서 나타나는 변수의 순서
                            col_dims(list): 각 변수 별 범주의 개수, 순서는 col_list를 따라야 함
            embedding_dim(int): 엔티티 임베딩 후의 차원
            gin_hidden_channel(int): GIN의 hidden channel, 일반적으로 인풋, 히든, 아웃풋 차원을 동일하게 설정하는 것이 흔하고 효율적임
            train_eps(bool): GIN의 epsilon을 훈련할 것인지, 고정할 것인지

                                Gemini의 당부
                                다만, 모델을 로드하기 위해 새 인스턴스를 만들 때(e.g., model = MyGINModel(...)), 
                                해당 모델이 epsilon 슬롯을 가지고 있도록 반드시 train_eps=True를 동일하게 설정하여 초기화해야 합니다. 
                                그렇지 않으면 저장된 state_dict와 새 모델의 구조가 일치하지 않아 로딩에 실패합니다.

            gin_layers(int): GIN의 레이어 개수
            gru_hidden_channel(int): GRU의 hidden channel, 이는 모델의 기억 용량을 의미함
                                     우리 데이터는 크게 설정할 필요가 없을 것으로 보임 (오직 1번 달라지기 때문)
                                     시간축으로만 보면 그렇지만, 
                                     GIN이 출력하는 특징의 복잡도를 수용할 수 있는 적절한 최소 크기(Medium-sized*로 설정하는 것이 가장 좋다.
                                     gin_hidden_channel과 동일하게 설정하여 시작하는 것이 좋음
        '''
        super().__init__()
        self.num_classes = num_classes
        self.hidden_channel = gin_hidden_channel

        self.dropout_p = float(dropout_p)
        self.gin_out_dropout_p = float(gin_out_dropout_p) if gin_out_dropout_p is not None else float(dropout_p)

        self.dropout_mlp = nn.Dropout(self.dropout_p)
        self.dropout_gin_out = nn.Dropout(self.gin_out_dropout_p)
        self.dropout_after_gru = nn.Dropout(self.dropout_p)

        # col_info: (col_list, col_dims, ad_col_index, dis_col_index)
        self.col_list, self.col_dims, ad_col_index, dis_col_index = col_info
        self.register_buffer("ad_idx_t", torch.tensor(ad_col_index, dtype=torch.long))
        self.register_buffer("dis_idx_t", torch.tensor(dis_col_index, dtype=torch.long))

        # EntityEmbedding 레이어 정의
        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        
        # GIN 레이어 정의
        # MLP 구조는 GIN 논문의 권장 사항(2-Layer MLP, 배치 정규화 )
        gin_nn_input = nn.Sequential(
             nn.Linear(embedding_dim, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),
             nn.Dropout(self.dropout_p),  
             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn = nn.Sequential(
             nn.Linear(gin_hidden_channel, gin_hidden_channel),
             nn.LayerNorm(gin_hidden_channel),
             nn.ReLU(),
             nn.Dropout(self.dropout_p),
             nn.Linear(gin_hidden_channel, gin_hidden_channel) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        self.gin_layers = nn.ModuleList()

        gin_layer1 = GINConv(nn=gin_nn_input, eps=0, train_eps=train_eps)
        self.gin_layers.append(gin_layer1)
        
        for _ in range(gin_layers - 1):
            gin_layer_hidden = GINConv(nn=gin_nn, eps=0, train_eps=train_eps)
            self.gin_layers.append(gin_layer_hidden)
        
        gru_input_ch = gin_hidden_channel * gin_layers
        self.gru_layer = GRU(input_size=gru_input_ch, hidden_size=gru_hidden_channel)

        # 분류기 레이어 정의
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_channel, gru_hidden_channel * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(gru_hidden_channel * 2, out_dim)
        )
        self.gin_nn_input = gin_nn_input
        self.gin_nn = gin_nn
        self.reset_parameters() # He initialization (added 1218)

    def forward(self, x_batch: torch.Tensor, LOS_batch: torch.Tensor, template_edge_index, device):
        '''
        template_edge_index: supersized edge_index
        '''
        # ad_idx_t / dis_idx_t are registered buffers — they move with the model automatically.
        batch_size = x_batch.shape[0]
        num_nodes = len(self.ad_idx_t)

        # x_batch shape: [batch_size, num_var(=72)]
        x_embedded = self.entity_embedding_layer(x_batch) # shape: [batch, num_var, feature_dim]

        # process: [batch * 2, num_nodes, feature_dim]으로 변환하기
        # 위와 같이 되어야 함: 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, ...
        x_seperated = seperate_x(x=x_embedded, # [B*2, 60, 32]
                                 ad_idx_t=self.ad_idx_t, 
                                 dis_idx_t=self.dis_idx_t, 
                                 device=device)

        # GIN에 입력
        x_flatten = x_seperated.reshape(batch_size * 2 * num_nodes, -1)

        x_after_gin = x_flatten
        sum_pooled = []
        for layer in self.gin_layers:
            x_after_gin = layer(x_after_gin, template_edge_index) # [B * 2 * N, F(32)]
            x_after_gin = self.dropout_gin_out(x_after_gin)         # GIN 레이어 출력에도 Dropout
            x_graph = x_after_gin.reshape(batch_size * 2, num_nodes, self.hidden_channel) # [B * 2, N, F]
            x_sum = torch.sum(x_graph, dim=1) # [B * 2, N(60), F(32)] --> [B * 2, F(32)]
            sum_pooled.append(x_sum)

        gin_result = torch.concatenate(sum_pooled, dim=1) # [B*2, F*num_gin_layers]
            
        # [B*2, F*num_gin_layers] --> [B, 37, F*num_gin_layers]
        temporal_embedding, sorted_indices = to_temporal_gingru(x=gin_result, los_batch=LOS_batch)

        # GRU에 입력
        gru_out, gru_h = self.gru_layer(temporal_embedding)

        gru_h = gru_h.squeeze(0)

        # GRU dropout이 없으니 여기서 한 번
        gru_h = self.dropout_after_gru(gru_h)


        # PackedSequence로 만들었기 때문에 (시간 길이 순 정렬) 역정렬해야 함
        inv_indices = torch.argsort(sorted_indices.to(device))
        gru_h = gru_h[inv_indices]             
                      

        # Classifier에 입력
        return self.classifier(gru_h)
    
    def reset_parameters(self):
        # GIN MLP들
        for block in [self.gin_nn_input, self.gin_nn]:
            for m in block.modules():
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # classifier
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
