import torch
import torch.nn as nn
import pandas as pd
from torch_geometric.data import Batch, Data

class EntityEmbedding(torch.nn.Module):
    def __init__(self, col_dims: list, col_list: list):
        '''
        Args:
            cat_dims: 변수별 범주의 개수 리스트
            col_list: 원본 데이터프레임에서 변수들의 순서 왼->오
        '''
        super().__init__()
        self.col_dims = col_dims
        self.col_list = col_list
        # proj_dim must be calculated with the final, corrected col_dims,
        # so we do it here with the provided ones, but it will be re-calculated in the test block
        self.proj_dim = int(max(self.col_dims)**0.5) if self.col_dims else 1

        self.embs = nn.ModuleList([
            nn.Embedding(num_categories, self.proj_dim)
            for num_categories in self.col_dims
        ])

    def forward(self, x_cats):
        '''
        This forward method assumes x_cats is a 2-D tensor of shape [N, F]
        where N is batch_size and F is number of features.
        The current data pipeline produces a different shape, so this is not used.
        '''
        # This logic is for a different data shape and is preserved for potential future use.
        x_cats = x_cats.long()
        outs = []
        for i, emb in enumerate(self.embs):
            out = emb(x_cats[:, i])
            outs.append(out)
        outs_tensor = torch.stack(outs, dim = 1)
        return outs_tensor
    

class EntityEmbeddingBatch(EntityEmbedding):
    def forward(self, batch: Batch):
        '''
        using pyg, deprecated
        '''
        # 1. features 텐서 준비
        features = batch.x.long() # 현재 features.device == cpu (문제의 원인)
        TARGET_DEVICE = self.embs[0].weight.device
        features = features.to(TARGET_DEVICE)
        DEVICE = features.device 
        
        # features의 shape가 [X, 1]이면 [X]로 squeeze (nn.Embedding을 위해)
        if features.dim() > 1 and features.size(1) == 1:
            features = features.squeeze(1) # (N_total * F_count)
        
        N_total_F = features.shape[0]
        F_count = len(self.col_list)
        
        # 2. 텐서 분리 및 인덱스 생성
        # col_indices 생성 시 이미 DEVICE를 사용하고 있으므로, 이제 DEVICE는 MPS/GPU입니다.
        col_indices = torch.arange(F_count, device=DEVICE).repeat(N_total_F // F_count)
        
        # 3. 각 임베딩 레이어를 사용하여 해당하는 인덱스를 한 번에 처리
        all_embedded_features = []
        for i, emb in enumerate(self.embs):
            
            mask = (col_indices == i)
            data_to_embed = features[mask]
            
            embedded_data = emb(data_to_embed) # 이제 CPU 텐서가 아닌 GPU 텐서 입력
            
            all_embedded_features.append((embedded_data, torch.where(mask)[0]))

        # 4. 원래의 순서대로 결과를 재조립
        outs_tensor = torch.zeros(N_total_F, self.proj_dim, device=DEVICE) 
        for embedded_data, indices in all_embedded_features:
            outs_tensor[indices] = embedded_data

        return outs_tensor
    

class EntityEmbeddingBatch2(EntityEmbedding):
    '''
    for Tensor based process
    '''
    def forward(self, batch: torch.Tensor):
        '''
        Args:
            batch (torch.Tensor): shape: [BATCH_SIZE, num_var(=72)]
        '''
        embedded_list = []
        for idx, emb_func in enumerate(self.embs):
            current_input = batch[:, idx] # shape: [BATCH_SIZE]
            embedded_vec = emb_func(current_input)
            embedded_list.append(embedded_vec)
        return torch.stack(embedded_list, dim=1) # shape: [BATCH_SIZE, NUM_VAR, FEATURE_DIM] (=[32, 72, 25])






    """
    EntityEmbeddingBatch3

    이 모듈은 다수의 범주형 변수(예: TEDS-D의 72개 변수)를 하나의 전역 임베딩
    테이블로 처리하기 위한 전용 임베딩 레이어이다. 기존 방식처럼 변수마다
    개별적으로 nn.Embedding 레이어를 생성하고 for-loop로 순차적으로 호출하면
    GPU 커널 호출이 매우 많아져 학습 속도가 저하된다. 본 모듈은 모든 변수의
    범주(category)를 하나의 전역 vocabulary 공간으로 합쳐 단일 Embedding 레이어를
    사용함으로써 연산 효율을 크게 향상시킨다.

    -------------------------------------------------------------------------
    왜 이런 방식이 필요한가?
    -------------------------------------------------------------------------
    - 범주형 변수가 많을수록(예: 72개) 변수별 개별 임베딩 레이어를 호출하는 것은
      비효율적이다. 각 변수마다 embedding lookup을 수행하면 forward마다 72개의
      GPU 커널 호출이 발생하여 병목이 된다.
    - 모든 범주를 하나의 “전역 인덱스(global index)” 공간에 배치하면nn.Embedding을
      단 한 번만 호출하면 되므로 연산량이 대폭 감소한다.
    - 더 빠르고 간결한 구조이며, 파라미터 수는 기존 방식과 동일하다.
      (기존에는 여러 임베딩 테이블을 합쳐놓았던 것을 물리적으로 하나로 묶은 것뿐)

    -------------------------------------------------------------------------
    어떻게 동작하는가?
    -------------------------------------------------------------------------
    1. col_dims: 각 변수(컬럼)의 고유 카테고리 개수 리스트를 입력 받는다.
       예: [3, 6, 7, 4, …]

    2. offsets 계산:
       각 변수가 전역 임베딩 테이블에서 시작하는 위치를 누적합으로 계산한다.
       예: col_dims = [3, 6, 7] → offsets = [0, 3, 9]

    3. batch(로컬 인덱스):
       입력 batch는 (batch_size, num_features) 형태이며, 각 값은 해당 변수 안에서의
       로컬 카테고리 인덱스(0~V_j-1)이다.

    4. 전역 인덱스로 변환:
       glob_batch = batch + offsets
       브로드캐스팅으로 각 변수 위치에 맞는 offset이 자동으로 더해진다.

    5. embedding lookup:
       전역 인덱스 텐서를 단일 Embedding 레이어에 넣어
       (batch_size, num_features, embedding_dim) 형태의 임베딩을 얻는다.

    -------------------------------------------------------------------------
    입력
    -------------------------------------------------------------------------
    batch : torch.Tensor
        shape = [BATCH_SIZE, NUM_VAR]
        각 요소는 "해당 변수에서의 로컬 카테고리 인덱스"를 의미한다.

    -------------------------------------------------------------------------
    출력
    -------------------------------------------------------------------------
    torch.Tensor
        shape = [BATCH_SIZE, NUM_VAR, embedding_dim]
        모든 범주형 변수가 단일 임베딩 테이블에서 변환된 임베딩 벡터.

    -------------------------------------------------------------------------
    주요 장점
    -------------------------------------------------------------------------
    - 기존 방식(변수별 72개의 embedding layer + for-loop)보다 훨씬 빠름.
    - 파라미터 수는 동일하지만 연산량과 GPU kernel 호출 횟수가 압도적으로 감소함.
    - 유지보수 간단: embedding layer 하나만 관리하면 됨.
    - offsets와 col_dims는 register_buffer로 관리되어 장치 이동(to(device))과
      state_dict 저장 시 안정적으로 포함된다.
    """


class EntityEmbeddingBatch3(nn.Module):
    def __init__(self, col_dims, embedding_dim=32):
        super().__init__()
        col_dims = torch.as_tensor(col_dims, dtype=torch.long)

        offsets = torch.zeros_like(col_dims)
        offsets[1:] = torch.cumsum(col_dims[:-1], dim=0) # 누적합 계산

        # 학습 파라미터는 아니지만, 모델과 함께 device를 따라가야 하므로 buffer로 등록
        self.register_buffer("offsets", offsets)
        self.register_buffer("col_dims", col_dims)

        total_dim = int(col_dims.sum().item())  # 전체 카테고리 수
        self.embedding_layer = nn.Embedding(total_dim, embedding_dim)


    def forward(self, batch: torch.Tensor): # batch shape: [BATCH_SIZE, NUM_VAR] (=[32, 72])
        batch = batch.long()
        glob_batch = batch + self.offsets # type: ignore
        return self.embedding_layer(glob_batch) # shape: [BATCH_SIZE, NUM_VAR, FEATURE_DIM]
        


