import torch
import torch_geometric
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from src.models.entity_embedding import EntityEmbeddingBatch3
from src.models.forecast_inputs import DistributionEncoder, resolve_constant_long_tensor

class GIN(nn.Module):
    def __init__(self, 
                 embedding_dim, 
                 col_info, 
                 gin_dim, 
                 gin_layer_num, 
                 num_classes,
                 train_eps=True,
                 use_los=True,
                 **kwargs,
                 ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        original_col_list = list(col_info[0])
        original_col_dims = list(col_info[1])
        self.full_col_list = original_col_list
        self.full_col_dims = original_col_dims
        self.feature_col_indices = [
            int(idx) for idx, name in enumerate(original_col_list) if str(name) != "LOS"
        ]
        self._feature_index_by_original = {
            original_idx: feature_idx
            for feature_idx, original_idx in enumerate(self.feature_col_indices)
        }
        self.col_list = [original_col_list[idx] for idx in self.feature_col_indices]
        self.col_dims = [original_col_dims[idx] for idx in self.feature_col_indices]
        self.discharge_col_index = [
            self._feature_index_by_original[int(idx)]
            for idx, name in enumerate(original_col_list)
            if str(name).endswith("_D") and int(idx) in self._feature_index_by_original
        ]
        self.use_los = use_los
        self.max_los = int(kwargs.get("max_los", 37))
        # self.col_dims.append(self.max_los + 1) # LOS needs to be included in GIN, as it's excluded in col_info. --> remove_LOS parameter makes LOS included in col_info
        self.gin_dim = gin_dim
        self.gin_layer_num = gin_layer_num
        self.train_eps = train_eps
        self.num_classes = num_classes
        self.forecast_input_encoder = str(
            kwargs.get("forecast_input_encoder", "entity_embedding")
        ).lower()
        if self.forecast_input_encoder not in {"entity_embedding", "distribution"}:
            raise ValueError(
                "forecast_input_encoder must be one of ['entity_embedding', 'distribution']"
            )
        self.distribution_encoder_hidden_dim = int(
            kwargs.get("distribution_encoder_hidden_dim", 64)
        )
        self.distribution_encoder_out_dim = int(
            kwargs.get("distribution_encoder_out_dim", embedding_dim)
        )
        self.node_feature_dim = (
            self.distribution_encoder_out_dim
            if self.forecast_input_encoder == "distribution"
            else embedding_dim
        )

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, 
                                                            embedding_dim=embedding_dim)
        self.los_embedding_layer = nn.Embedding(self.max_los + 1, self.node_feature_dim)
        self.distribution_encoders = nn.ModuleList(
            DistributionEncoder(
                num_classes=int(col_dim),
                out_dim=self.distribution_encoder_out_dim,
                hidden_dim=self.distribution_encoder_hidden_dim,
            )
            for col_dim in self.col_dims
        )
        self.los_distribution_encoder = DistributionEncoder(
            num_classes=self.max_los,
            out_dim=self.node_feature_dim,
            hidden_dim=self.distribution_encoder_hidden_dim,
        )
        self._distribution_input_debug_printed = False
        
        gin_nn_input = nn.Sequential(
             nn.Linear(self.node_feature_dim, gin_dim),
             nn.LayerNorm(gin_dim),
             nn.ReLU(),

             nn.Linear(gin_dim, gin_dim) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        gin_nn = nn.Sequential(
             nn.Linear(gin_dim, gin_dim),
             nn.LayerNorm(gin_dim),
             nn.ReLU(),

             nn.Linear(gin_dim, gin_dim) # 논문에서 적용된 배치 정규화 
             # nn.LayerNorm(h_dim),  # 마지막 레이어 이후에는 선택적
        )

        self.gin_layers = nn.ModuleList()

        gin_layer1 = GINConv(nn=gin_nn_input, eps=0, train_eps=self.train_eps)
        self.gin_layers.append(gin_layer1)
        
        for _ in range(self.gin_layer_num - 1):
            gin_layer_hidden = GINConv(nn=gin_nn, eps=0, train_eps=self.train_eps)
            self.gin_layers.append(gin_layer_hidden)

        # 분류기 레이어 정의
        out_dim = 1 if self.num_classes == 2 else self.num_classes
        self.classifier_dim = self.gin_dim * self.gin_layer_num
        self.classifier = nn.Sequential(
            nn.Linear(self.classifier_dim, self.classifier_dim * 2),
            nn.ReLU(),
            nn.Linear(self.classifier_dim * 2, out_dim)
        )

    def _validate_x_feature_width(self, x: torch.Tensor) -> None:
        expected_width = len(self.col_dims)
        if x.shape[1] != expected_width:
            raise ValueError(
                "GIN x feature width mismatch: "
                f"expected {expected_width} non-LOS feature columns, got {x.shape[1]}. "
                "LOS must be passed through the separate los argument."
            )

    def _resolve_soft_discharge_overrides(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None,
    ) -> dict[int, torch.Tensor]:
        if not soft_discharge:
            return {}

        discharge_col_idx = set(self.discharge_col_index)
        overrides: dict[int, torch.Tensor] = {}
        for head_name, payload in soft_discharge.items():
            probs = payload["probs"]
            if probs.ndim != 2:
                raise ValueError(f"{head_name}: probs must be rank-2, got shape={tuple(probs.shape)}")
            if probs.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{head_name}: probs batch mismatch probs_B={probs.shape[0]} x_B={x.shape[0]}"
                )
            target_col_idx = int(
                resolve_constant_long_tensor(
                    payload["target_col_idx"],
                    name=f"{head_name}.target_col_idx",
                )[0].item()
            )
            target_col_idx = self._feature_index_by_original.get(
                target_col_idx, target_col_idx
            )
            if target_col_idx not in discharge_col_idx:
                raise ValueError(
                    f"{head_name}: target_col_idx={target_col_idx} is not a discharge column"
                )
            num_classes = int(
                resolve_constant_long_tensor(
                    payload["num_classes"],
                    name=f"{head_name}.num_classes",
                )[0].item()
            )
            col_dim = int(self.col_dims[target_col_idx])
            if probs.shape[1] != num_classes or num_classes != col_dim:
                raise ValueError(
                    f"{head_name}: probability width/cardinality mismatch "
                    f"probs={probs.shape[1]} num_classes={num_classes} col_dim={col_dim}"
                )
            overrides[target_col_idx] = probs.to(device=x.device, dtype=torch.float32)
        return overrides

    def _encode_x_from_distributions(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        self._validate_x_feature_width(x)
        overrides = self._resolve_soft_discharge_overrides(x, soft_discharge)
        features = []
        for col_idx, encoder in enumerate(self.distribution_encoders):
            probs = overrides.get(col_idx)
            if probs is None:
                probs = F.one_hot(
                    x[:, col_idx].long(),
                    num_classes=int(self.col_dims[col_idx]),
                ).to(dtype=torch.float32)
            features.append(encoder(probs))
        x_encoded = torch.stack(features, dim=1)
        if soft_discharge and not self._distribution_input_debug_printed:
            print("[GIN FORECAST DISTRIBUTION INPUT CHECK]")
            print(f"num_soft_columns={len(overrides)}")
            print(f"x_encoded_shape={tuple(x_encoded.shape)}")
            self._distribution_input_debug_printed = True
        return x_encoded

    def _embed_x_with_optional_soft_discharge(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if self.forecast_input_encoder == "distribution":
            return self._encode_x_from_distributions(x, soft_discharge=soft_discharge)
        self._validate_x_feature_width(x)
        return self.entity_embedding_layer(x)

    def encode_los(self, los: torch.Tensor) -> torch.Tensor:
        if self.forecast_input_encoder == "distribution":
            if los.ndim == 1:
                if los.numel() == 0:
                    return self.los_distribution_encoder.net[0].weight.new_empty(
                        (0, self.node_feature_dim)
                    )
                los_prob = F.one_hot(los.long() - 1, num_classes=self.max_los).to(
                    dtype=torch.float32
                )
            elif los.ndim == 2:
                if los.shape[1] != self.max_los:
                    raise ValueError(
                        f"LOS distribution width mismatch: expected {self.max_los}, got {los.shape[1]}"
                    )
                los_prob = los.to(dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported LOS shape: {tuple(los.shape)}")
            return self.los_distribution_encoder(los_prob)
        if los.ndim != 1:
            raise ValueError("entity_embedding LOS path expects rank-1 LOS indices")
        return self.los_embedding_layer(los.long())

    def forward(self, x, los, edge_index, **kwargs):
        # initial setting
        if x.ndim == 1:
            batch_size = 1
            x = x.unsqueeze(dim=0)
        elif x.ndim == 2:
            batch_size = x.shape[0]
        else:
            raise ValueError("incorrect x dim")
        
        if self.use_los:
            if los.ndim == 1:
                los_feature = los.unsqueeze(dim=1)
            elif los.ndim == 2:
                los_feature = self.encode_los(los).unsqueeze(dim=1)
            else:
                raise ValueError(f"Unsupported LOS input rank: {los.ndim}")
        else:
            los_feature = None

        num_nodes = x.shape[1] + (1 if los_feature is not None else 0)

        # entity embedding
        soft_discharge = kwargs.get("soft_discharge")
        x_embedded = self._embed_x_with_optional_soft_discharge(
            x.long(),
            soft_discharge=soft_discharge,
        )
        if los_feature is not None:
            if los.ndim == 1:
                los_embedded = self.encode_los(los.long()).unsqueeze(dim=1)
            else:
                los_embedded = los_feature
            x_embedded = torch.cat((x_embedded, los_embedded), dim=1)

        # gin layers
        node_embeddings = x_embedded.reshape(batch_size * num_nodes, -1) # [batch * num_var, entity_emb_dim]
        sum_pooled = []
        for layer in self.gin_layers:
            node_embeddings = layer(node_embeddings, edge_index) # [batch * num_var, feature_dim]
            x_temp = node_embeddings.reshape(batch_size, num_nodes, -1) # [batch, num_var, feature_dim]
            x_sum = torch.sum(x_temp, dim=1) # [batch, feature_dim]
            sum_pooled.append(x_sum)
        graph_emb = torch.cat(sum_pooled, dim=1) # [batch, feature_dim * layer_num]

        # classifier
        return self.classifier(graph_emb)
    
