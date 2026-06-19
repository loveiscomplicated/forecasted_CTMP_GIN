# CTMP-GIN
from __future__ import annotations

import os
import sys
from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, GINEConv

cur_dir = os.path.dirname(__file__)
par_dir = os.path.join(cur_dir, '..')
sys.path.append(par_dir)

from src.models.entity_embedding import EntityEmbeddingBatch3
from src.models.ctmp_gin.los_encoder import (
    DEFAULT_TEDS_LOS_REP_DAYS,
    HybridOrdinalLOSEncoder,
)
from src.utils.device_set import device_set


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _make_gin_mlp(in_dim: int, out_dim: int) -> nn.Sequential:
    """Standard 2-layer MLP used inside GINConv / GINEConv."""
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.ReLU(),
        nn.Linear(out_dim, out_dim),
    )


class DistributionEncoder(nn.Module):
    def __init__(self, num_classes: int, out_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = int(hidden_dim or out_dim)
        self.num_classes = int(num_classes)
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            nn.Linear(self.num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.out_dim),
        )

    def forward(self, probs: torch.Tensor) -> torch.Tensor:
        if probs.ndim != 2:
            raise ValueError(f"DistributionEncoder expected rank-2 probs, got shape={tuple(probs.shape)}")
        if probs.shape[1] != self.num_classes:
            raise ValueError(
                f"DistributionEncoder width mismatch: expected {self.num_classes}, got {probs.shape[1]}"
            )
        return self.net(probs)


def seperate_x(x: torch.Tensor, ad_idx_t: torch.Tensor, dis_idx_t: torch.Tensor) -> torch.Tensor:
    """Split node features into admission / discharge halves and stack batch-wise.

    Args:
        x: [B, num_vars, emb_dim]
    Returns:
        [B*2, num_nodes, emb_dim]  — first B slices are admission, next B are discharge
    """
    ad_tensor  = torch.index_select(x, dim=1, index=ad_idx_t)   # [B, N, F]
    dis_tensor = torch.index_select(x, dim=1, index=dis_idx_t)  # [B, N, F]
    return torch.cat([ad_tensor, dis_tensor], dim=0)             # [B*2, N, F]


def _is_los_distribution(los: torch.Tensor, batch_size: int) -> bool:
    return los.ndim == 2 and los.shape[0] == batch_size


def _resolve_constant_long_tensor(value: torch.Tensor, *, name: str) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1)
    if value.ndim == 1:
        if value.numel() == 0:
            raise ValueError(f"{name} is empty")
        if not torch.equal(value, value[0].expand_as(value)):
            raise ValueError(f"{name} must be constant across the batch")
        return value[:1]
    if value.ndim == 2:
        if value.size(0) == 0:
            raise ValueError(f"{name} is empty")
        ref = value[0:1].expand_as(value)
        if not torch.equal(value, ref):
            raise ValueError(f"{name} must be constant across the batch")
        return value[0]
    raise ValueError(f"Unsupported ndim for {name}: {value.ndim}")


# ---------------------------------------------------------------------------
# GatedFusion
# ---------------------------------------------------------------------------

class GatedFusion(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        self.dropout = nn.Dropout(dropout)
        self.out_dim = out_dim

    def forward(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x      = torch.cat([A, B, C], dim=-1)   # [B, 3*d]
        logits = self.score(x)                  # [B, 3]
        w      = F.softmax(logits, dim=-1)       # [B, 3]
        A = self.dropout(A)
        B = self.dropout(B)
        C = self.dropout(C)
        fused = w[:, 0:1] * A + w[:, 1:2] * B + w[:, 2:3] * C
        return fused, w, logits


# ---------------------------------------------------------------------------
# CTMPGIN
# ---------------------------------------------------------------------------

ReadoutMode = Literal["concat", "sum", "last"]


class CTMPGIN(nn.Module):
    """Cross-Temporal Message Passing GIN.

    Args:
        readout_mode: How to aggregate node embeddings across GIN layers.
            - ``"concat"``: Concatenate readouts from all layers (GIN paper).
              Projection input dim = ``gin_hidden_channel * gin_1_layers``.
            - ``"sum"``:    Sum readouts across layers. Projection dim unchanged.
            - ``"last"``:   Use only the final layer readout (pre-fix behaviour).
    """

    def __init__(
        self,
        col_info,
        embedding_dim: int,
        gin_hidden_channel: int,
        gin_1_layers: int,
        gin_hidden_channel_2: int,
        gin_2_layers: int,
        num_classes: int,
        dropout_p: float = 0.2,
        los_embedding_dim: int = 8,
        max_los: int = 37,
        train_eps: bool = True,
        gate_hidden_ch: Optional[int] = None,
        remove_proj_ad_dis: bool = False,
        remove_all_proj: bool = False,
        remove_gated_fusion: bool = False,
        readout_mode: ReadoutMode = "concat",
        **kwargs,
    ):
        super().__init__()
        self.device = device_set(kwargs.get("device", "cpu"))
        self.dropout_p = dropout_p
        self.gin_hidden_channel   = gin_hidden_channel
        self.gin_hidden_channel_2 = gin_hidden_channel_2
        self.gin_1_layers = gin_1_layers
        self.gin_2_layers = gin_2_layers
        self.max_los = int(max_los)
        self.los_embedding_dim = int(los_embedding_dim)
        self.los_emb = str(kwargs.get("los_emb", "embedding")).lower()
        self.forecast_input_encoder = str(kwargs.get("forecast_input_encoder", "entity_embedding")).lower()
        if self.forecast_input_encoder not in {
            "entity_embedding",
            "distribution",
            "expected_embedding_diagnostic",
        }:
            raise ValueError(
                "forecast_input_encoder must be one of "
                "['entity_embedding', 'distribution', 'expected_embedding_diagnostic']"
            )
        self.distribution_encoder_hidden_dim = int(kwargs.get("distribution_encoder_hidden_dim", 64))
        self.distribution_encoder_out_dim = int(kwargs.get("distribution_encoder_out_dim", embedding_dim))
        self.los_ordinal_basis_dim = int(kwargs.get("los_ordinal_basis_dim", 8))
        self.los_ordinal_dim = int(kwargs.get("los_ordinal_dim", 8))
        self.los_ordinal_hidden_dim = int(kwargs.get("los_ordinal_hidden_dim", 32))
        self.node_feature_dim = (
            self.distribution_encoder_out_dim
            if self.forecast_input_encoder == "distribution"
            else embedding_dim
        )

        assert readout_mode in ("concat", "sum", "last"), \
            f"readout_mode must be 'concat', 'sum', or 'last', got {readout_mode!r}"
        self.readout_mode = readout_mode

        self.col_list, self.col_dims, self.ad_col_index, self.dis_col_index = col_info
        self.register_buffer("col_dims_t", torch.tensor(self.col_dims, dtype=torch.long))
        self.register_buffer("ad_idx_t",  torch.tensor(self.ad_col_index,  dtype=torch.long))
        self.register_buffer("dis_idx_t", torch.tensor(self.dis_col_index, dtype=torch.long))

        self.entity_embedding_layer = EntityEmbeddingBatch3(col_dims=self.col_dims, embedding_dim=embedding_dim)
        self.distribution_encoders = nn.ModuleList(
            DistributionEncoder(
                num_classes=int(col_dim),
                out_dim=self.distribution_encoder_out_dim,
                hidden_dim=self.distribution_encoder_hidden_dim,
            )
            for col_dim in self.col_dims_t.tolist()
        )
        self.los_distribution_encoder = DistributionEncoder(
            num_classes=self.max_los,
            out_dim=self.los_embedding_dim,
            hidden_dim=self.distribution_encoder_hidden_dim,
        )
        self.embed_los: EntityEmbeddingBatch3 | None
        self.los_encoder: HybridOrdinalLOSEncoder | None
        if self.los_emb in {"embedding", "nn_embedding"}:
            self.embed_los = EntityEmbeddingBatch3(col_dims=[self.max_los + 1], embedding_dim=self.los_embedding_dim)
            self.los_encoder = None
            self.los_none_embedding = None
        elif self.los_emb == "hybrid_ordinal":
            rep_days = DEFAULT_TEDS_LOS_REP_DAYS if self.max_los == len(DEFAULT_TEDS_LOS_REP_DAYS) else None
            self.embed_los = None
            self.los_encoder = HybridOrdinalLOSEncoder(
                num_los_classes=self.max_los,
                out_dim=self.los_embedding_dim,
                basis_dim=self.los_ordinal_basis_dim,
                ordinal_dim=self.los_ordinal_dim,
                hidden_dim=self.los_ordinal_hidden_dim,
                dropout=dropout_p,
                rep_days=rep_days,
            )
            self.los_none_embedding = nn.Parameter(torch.empty(self.los_embedding_dim, dtype=torch.float32))
        else:
            raise ValueError(
                f"Unsupported los_emb={self.los_emb!r}. Expected one of "
                f"['embedding', 'nn_embedding', 'hybrid_ordinal']."
            )

        # GIN stacks
        self.gin_1 = self._build_gin_stack(
            GINConv, self.node_feature_dim, gin_hidden_channel, gin_1_layers, train_eps,
        )
        self.gin_2 = self._build_gin_stack(
            GINEConv, gin_hidden_channel, gin_hidden_channel_2, gin_2_layers, train_eps,
            edge_dim=los_embedding_dim,
        )

        # Projection input dimensions depend on readout_mode
        readout_dim_1 = gin_hidden_channel   * (gin_1_layers if readout_mode == "concat" else 1)
        readout_dim_2 = gin_hidden_channel_2 * (gin_2_layers if readout_mode == "concat" else 1)

        d = gin_hidden_channel
        self.fuse_dim = d

        # Ablation: remove_proj flags require Identity() to be dimension-safe
        if (remove_all_proj or remove_proj_ad_dis) and readout_mode == "concat":
            assert gin_1_layers == 1 and gin_2_layers == 1, (
                "remove_proj_ad_dis / remove_all_proj requires gin_1_layers=1 and "
                "gin_2_layers=1 when readout_mode='concat' (Identity cannot change dims). "
                "Use readout_mode='sum' or 'last' for ablation with multiple layers."
            )

        if remove_all_proj:
            print("proj_ad_dis_merged removed...")
            self.proj_ad     = nn.Identity()
            self.proj_dis    = nn.Identity()
            self.proj_merged = nn.Identity()
        elif remove_proj_ad_dis:
            print("proj_ad_dis removed...")
            self.proj_ad     = nn.Identity()
            self.proj_dis    = nn.Identity()
            self.proj_merged = nn.Linear(readout_dim_2, d)
        else:
            self.proj_ad     = nn.Linear(readout_dim_1, d)
            self.proj_dis    = nn.Linear(readout_dim_1, d)
            self.proj_merged = nn.Linear(readout_dim_2, d)

        self.remove_gated_fusion = remove_gated_fusion
        self._soft_discharge_debug_printed = False
        self._distribution_input_debug_printed = False
        if not remove_gated_fusion:
            self.gated_fusion = GatedFusion(
                in_dim=3 * self.fuse_dim,
                out_dim=self.fuse_dim,
                hidden_dim=gate_hidden_ch,
                dropout=dropout_p,
            )
        else:
            print("gated_fusion removed...")
            self.gated_fusion = None

        self.classifier_b = nn.Sequential(
            nn.Linear(self.fuse_dim, self.fuse_dim * 2),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(self.fuse_dim * 2, 1),
        )

        self.reset_parameters()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _build_gin_stack(
        self, conv_cls, in_dim: int, hidden_dim: int, num_layers: int, train_eps: bool, **conv_kwargs
    ) -> nn.ModuleList:
        """Build a stack of GINConv / GINEConv layers."""
        layers = nn.ModuleList()
        layers.append(conv_cls(_make_gin_mlp(in_dim, hidden_dim), train_eps=train_eps, **conv_kwargs))
        for _ in range(num_layers - 1):
            layers.append(conv_cls(_make_gin_mlp(hidden_dim, hidden_dim), train_eps=train_eps, **conv_kwargs))
        return layers

    # ------------------------------------------------------------------
    # Core GIN runner with hierarchical readout
    # ------------------------------------------------------------------

    def _run_gin(
        self,
        layers: nn.ModuleList,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        hidden_ch: int,
        batch_factor: int,
        num_nodes: int,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run a GIN stack and return (final node emb, hierarchical graph readout).

        Returns:
            h:         [batch_factor * num_nodes, hidden_ch]  — final node embeddings
            graph_emb: readout shape depends on ``self.readout_mode``:
                       "concat" → [batch_factor, hidden_ch * num_layers]
                       "sum"    → [batch_factor, hidden_ch]
                       "last"   → [batch_factor, hidden_ch]
        """
        h = x_flat
        readouts: list[torch.Tensor] = []
        for layer in layers:
            if edge_attr is None:
                h = layer(x=h, edge_index=edge_index)
            else:
                h = layer(x=h, edge_index=edge_index, edge_attr=edge_attr)
            readouts.append(h.reshape(batch_factor, num_nodes, hidden_ch).mean(dim=1))

        if self.readout_mode == "concat":
            graph_emb = torch.cat(readouts, dim=-1)        # [B_f, F * L]
        elif self.readout_mode == "sum":
            graph_emb = torch.stack(readouts, dim=0).sum(0)  # [B_f, F]
        else:  # "last"
            graph_emb = readouts[-1]                       # [B_f, F]

        return h, graph_emb

    # ------------------------------------------------------------------
    # Shared forward backbone
    # ------------------------------------------------------------------

    def _backbone(
        self,
        x_flat: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int,
        los: Optional[torch.Tensor] = None,
        edge_index_2: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        return_internals: bool = False,
    ):
        # --- GIN_1: intra-graph ---
        x_node, intra_readout = self._run_gin(
            self.gin_1, x_flat, edge_index,
            self.gin_hidden_channel, batch_size * 2, num_nodes,
        )
        ad_readout  = intra_readout[:batch_size]   # [B, readout_dim_1]
        dis_readout = intra_readout[batch_size:]   # [B, readout_dim_1]

        # --- GIN_2: inter-graph ---
        if edge_index_2 is None:
            edge_index_2, edge_attr = self.get_new_edge(edge_index, los, batch_size)
        _, inter_readout = self._run_gin(
            self.gin_2, x_node, edge_index_2,
            self.gin_hidden_channel_2, batch_size * 2, num_nodes,
            edge_attr=edge_attr,
        )
        merged_readout = 0.5 * (inter_readout[:batch_size] + inter_readout[batch_size:])  # [B, readout_dim_2]

        # --- Projection ---
        ad_f     = self.proj_ad(ad_readout)          # [B, d]
        dis_f    = self.proj_dis(dis_readout)        # [B, d]
        merged_f = self.proj_merged(merged_readout)  # [B, d]

        # --- Fusion ---
        if self.remove_gated_fusion:
            fused = (ad_f + dis_f + merged_f) / 3.0
            w = logits_gate = None
        else:
            fused, w, logits_gate = self.gated_fusion(ad_f, dis_f, merged_f)

        logit = self.classifier_b(fused)

        if return_internals:
            return logit, fused, w, ad_f, dis_f, merged_f, logits_gate
        return logit

    # ------------------------------------------------------------------
    # Public forward methods
    # ------------------------------------------------------------------

    def _embed_x_with_optional_soft_discharge(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        if self.forecast_input_encoder == "distribution":
            return self._encode_x_from_distributions(x, soft_discharge=soft_discharge)
        if self.forecast_input_encoder == "expected_embedding_diagnostic":
            return self._embed_x_with_expected_soft_discharge(x, soft_discharge=soft_discharge)
        return self.entity_embedding_layer(x)

    def _embed_x_with_expected_soft_discharge(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        x_embedded = self.entity_embedding_layer(x)
        if not soft_discharge:
            return x_embedded

        embedding_weight = self.entity_embedding_layer.embedding_layer.weight
        offsets = self.entity_embedding_layer.offsets
        col_dims = self.entity_embedding_layer.col_dims
        discharge_col_idx = set(int(idx) for idx in self.dis_idx_t.detach().cpu().tolist())

        for head_name, payload in soft_discharge.items():
            probs = payload["probs"]
            if probs.ndim != 2:
                raise ValueError(f"{head_name}: probs must be rank-2, got shape={tuple(probs.shape)}")
            if probs.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{head_name}: probs batch mismatch probs_B={probs.shape[0]} x_B={x.shape[0]}"
                )
            if probs.device != x.device:
                raise ValueError(
                    f"{head_name}: probs device mismatch probs={probs.device} x={x.device}"
                )
            if not probs.dtype.is_floating_point:
                raise ValueError(f"{head_name}: probs must be floating dtype, got {probs.dtype}")

            probs_sum = probs.sum(dim=-1)
            if not torch.allclose(
                probs_sum,
                torch.ones_like(probs_sum),
                atol=1.0e-3,
                rtol=1.0e-3,
            ):
                raise ValueError(f"{head_name}: probs rows do not sum to 1 within tolerance")

            target_col_idx = int(_resolve_constant_long_tensor(payload["target_col_idx"], name=f"{head_name}.target_col_idx")[0].item())
            if target_col_idx not in discharge_col_idx:
                raise ValueError(f"{head_name}: target_col_idx={target_col_idx} is not a discharge column")

            class_to_embedding_idx = _resolve_constant_long_tensor(
                payload["class_to_embedding_idx"],
                name=f"{head_name}.class_to_embedding_idx",
            ).to(device=x.device, dtype=torch.long)
            num_classes = int(
                _resolve_constant_long_tensor(payload["num_classes"], name=f"{head_name}.num_classes")[0].item()
            )
            if probs.shape[1] != num_classes:
                raise ValueError(
                    f"{head_name}: probs C={probs.shape[1]} != num_classes={num_classes}"
                )
            if class_to_embedding_idx.numel() != num_classes:
                raise ValueError(
                    f"{head_name}: class_to_embedding_idx size={class_to_embedding_idx.numel()} "
                    f"!= num_classes={num_classes}"
                )

            col_dim = int(col_dims[target_col_idx].item())
            if int(class_to_embedding_idx.max().item()) >= col_dim:
                raise ValueError(
                    f"{head_name}: max embedding row index={int(class_to_embedding_idx.max().item())} "
                    f">= column cardinality={col_dim}"
                )

            col_offset = int(offsets[target_col_idx].item())
            col_emb_weight = embedding_weight[col_offset : col_offset + col_dim]
            emb_rows = col_emb_weight[class_to_embedding_idx]
            expected_emb = torch.matmul(probs.to(dtype=emb_rows.dtype), emb_rows)
            hard_emb_shape = x_embedded[:, target_col_idx, :].shape
            if expected_emb.shape != hard_emb_shape:
                raise ValueError(
                    f"{head_name}: expected_emb shape={tuple(expected_emb.shape)} "
                    f"!= hard_emb shape={tuple(hard_emb_shape)}"
                )
            x_embedded[:, target_col_idx, :] = expected_emb

            if not self._soft_discharge_debug_printed:
                print("[SOFT D FORWARD CHECK]")
                print(f"head={head_name}")
                print(f"probs_sum_mean={float(probs_sum.mean().item()):.6f}")
                print(f"probs_sum_min={float(probs_sum.min().item()):.6f}")
                print(f"probs_sum_max={float(probs_sum.max().item()):.6f}")
                print(f"expected_emb_shape={tuple(expected_emb.shape)}")
                print(f"hard_emb_shape={tuple(hard_emb_shape)}")
                print(f"device={expected_emb.device}")
                print(f"dtype={expected_emb.dtype}")
                self._soft_discharge_debug_printed = True

        return x_embedded

    def _resolve_soft_discharge_overrides(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None,
    ) -> dict[int, torch.Tensor]:
        if not soft_discharge:
            return {}

        discharge_col_idx = set(int(idx) for idx in self.dis_idx_t.detach().cpu().tolist())
        overrides: dict[int, torch.Tensor] = {}
        for head_name, payload in soft_discharge.items():
            probs = payload["probs"]
            if probs.ndim != 2:
                raise ValueError(f"{head_name}: probs must be rank-2, got shape={tuple(probs.shape)}")
            if probs.shape[0] != x.shape[0]:
                raise ValueError(
                    f"{head_name}: probs batch mismatch probs_B={probs.shape[0]} x_B={x.shape[0]}"
                )
            if probs.device != x.device:
                raise ValueError(
                    f"{head_name}: probs device mismatch probs={probs.device} x={x.device}"
                )
            if not probs.dtype.is_floating_point:
                raise ValueError(f"{head_name}: probs must be floating dtype, got {probs.dtype}")

            probs_sum = probs.sum(dim=-1)
            if not torch.allclose(
                probs_sum,
                torch.ones_like(probs_sum),
                atol=1.0e-3,
                rtol=1.0e-3,
            ):
                raise ValueError(f"{head_name}: probs rows do not sum to 1 within tolerance")

            target_col_idx = int(
                _resolve_constant_long_tensor(payload["target_col_idx"], name=f"{head_name}.target_col_idx")[0].item()
            )
            if target_col_idx not in discharge_col_idx:
                raise ValueError(f"{head_name}: target_col_idx={target_col_idx} is not a discharge column")

            num_classes = int(
                _resolve_constant_long_tensor(payload["num_classes"], name=f"{head_name}.num_classes")[0].item()
            )
            col_dim = int(self.col_dims_t[target_col_idx].item())
            if probs.shape[1] != num_classes:
                raise ValueError(
                    f"{head_name}: probs C={probs.shape[1]} != num_classes={num_classes}"
                )
            if num_classes != col_dim:
                raise ValueError(
                    f"{head_name}: num_classes={num_classes} != column cardinality={col_dim}"
                )
            overrides[target_col_idx] = probs.to(dtype=torch.float32)
        return overrides

    def _encode_x_from_distributions(
        self,
        x: torch.Tensor,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None = None,
    ) -> torch.Tensor:
        overrides = self._resolve_soft_discharge_overrides(x, soft_discharge)
        features: list[torch.Tensor] = []
        for col_idx, encoder in enumerate(self.distribution_encoders):
            probs = overrides.get(col_idx)
            if probs is None:
                probs = F.one_hot(x[:, col_idx].long(), num_classes=int(self.col_dims_t[col_idx].item())).to(torch.float32)
            feature = encoder(probs)
            features.append(feature)

        x_encoded = torch.stack(features, dim=1)
        if soft_discharge and not self._distribution_input_debug_printed:
            print("[FORECAST DISTRIBUTION INPUT CHECK]")
            print(f"num_soft_columns={len(overrides)}")
            print(f"x_encoded_shape={tuple(x_encoded.shape)}")
            self._distribution_input_debug_printed = True
        return x_encoded

    def _cross_temporal_los_embedding_table(self) -> torch.Tensor:
        if self.los_encoder is None:
            if self.embed_los is None:
                raise RuntimeError("embed_los is unexpectedly None for embedding LOS mode")
            return self.embed_los.embedding_layer.weight[: self.max_los + 1]

        if self.los_none_embedding is None:
            raise RuntimeError("los_none_embedding is unexpectedly None for hybrid ordinal mode")
        los_table = self.los_encoder.all_embeddings()
        if los_table.shape != (self.max_los, self.los_embedding_dim):
            raise ValueError(
                f"Hybrid ordinal LOS table shape mismatch: expected "
                f"({self.max_los}, {self.los_embedding_dim}), got {tuple(los_table.shape)}"
            )
        return torch.cat([self.los_none_embedding.unsqueeze(0), los_table], dim=0)

    def encode_los_indices(self, los_idx: torch.Tensor) -> torch.Tensor:
        los_idx = los_idx.long()
        if los_idx.numel() == 0:
            return self._cross_temporal_los_embedding_table().new_empty((*los_idx.shape, self.los_embedding_dim))
        idx_min = int(los_idx.min().item())
        idx_max = int(los_idx.max().item())
        if idx_min < 1 or idx_max > self.max_los:
            raise ValueError(
                f"LOS edge index out of range: min={idx_min} max={idx_max} valid=[1, {self.max_los}]"
            )
        if self.los_encoder is None:
            if self.embed_los is None:
                raise RuntimeError("embed_los is unexpectedly None for embedding LOS mode")
            return self.embed_los(los_idx)
        return self.los_encoder(los_idx - 1)

    def encode_los_distribution(self, los_prob: torch.Tensor) -> torch.Tensor:
        if los_prob.ndim != 2:
            raise ValueError(f"LOS distribution must have shape [B, K], got {tuple(los_prob.shape)}")
        if los_prob.shape[1] != self.max_los:
            raise ValueError(
                f"LOS distribution width mismatch: expected K={self.max_los}, got {los_prob.shape[1]}"
            )

        if self.forecast_input_encoder == "distribution":
            return self.los_distribution_encoder(los_prob.to(dtype=torch.float32))

        if self.los_encoder is None:
            los_table = self._cross_temporal_los_embedding_table()[1 : 1 + self.max_los]
        else:
            los_table = self.los_encoder.all_embeddings()

        return torch.matmul(los_prob.to(dtype=los_table.dtype), los_table)

    def _encode_los_hard_as_distribution(self, los_idx: torch.Tensor) -> torch.Tensor:
        los_idx = los_idx.long()
        if los_idx.ndim != 1:
            los_idx = los_idx.view(-1)
        if los_idx.numel() == 0:
            return self.los_distribution_encoder.net[0].weight.new_empty((0, self.los_embedding_dim))
        idx_min = int(los_idx.min().item())
        idx_max = int(los_idx.max().item())
        if idx_min < 1 or idx_max > self.max_los:
            raise ValueError(
                f"LOS edge index out of range: min={idx_min} max={idx_max} valid=[1, {self.max_los}]"
            )
        los_prob = F.one_hot(los_idx - 1, num_classes=self.max_los).to(dtype=torch.float32)
        return self.los_distribution_encoder(los_prob)

    def forward(
        self,
        x: torch.Tensor,
        los: torch.Tensor,
        edge_index: torch.Tensor,
        device=None,
        return_internals: bool = False,
        soft_discharge: dict[str, dict[str, torch.Tensor]] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        batch_size = x.shape[0]
        num_nodes  = len(self.ad_idx_t)

        x_embedded = self._embed_x_with_optional_soft_discharge(
            x,
            soft_discharge=soft_discharge,
        )
        x_sep  = seperate_x(x_embedded, self.ad_idx_t, self.dis_idx_t)
        x_flat = x_sep.reshape(batch_size * 2 * num_nodes, -1)

        return self._backbone(
            x_flat, edge_index, batch_size, num_nodes,
            los=los, return_internals=return_internals,
        )

    def forward_from_x_emb_with_edge_attr(
        self,
        x_embedded: torch.Tensor,    # [B, num_vars, emb_dim]
        edge_index: torch.Tensor,    # [2, E_internal]
        edge_index_2: torch.Tensor,  # [2, E_total]
        edge_attr: torch.Tensor,     # [E_total, D_los]
    ) -> torch.Tensor:
        """Forward with pre-computed embeddings and pre-computed edge attributes.
        Used by the Integrated Gradients explainer for gradient computation.
        """
        batch_size = x_embedded.size(0)
        num_nodes  = len(self.ad_idx_t)

        x_sep  = seperate_x(x_embedded, self.ad_idx_t, self.dis_idx_t)
        x_flat = x_sep.reshape(batch_size * 2 * num_nodes, -1)

        return self._backbone(
            x_flat, edge_index, batch_size, num_nodes,
            edge_index_2=edge_index_2, edge_attr=edge_attr,
        )

    # ------------------------------------------------------------------
    # Edge utilities (external contracts — do not change signatures)
    # ------------------------------------------------------------------

    def precompute_edge_index_2(self, edge_index: torch.Tensor, batch_size: int) -> None:
        """Cache edge_index_2 once per trial to avoid repeated CPU tensor creation."""
        num_nodes        = len(self.ad_col_index)
        merged_num_nodes = num_nodes * batch_size
        start_node       = torch.arange(0, merged_num_nodes, device=edge_index.device)
        end_node         = start_node + merged_num_nodes
        cross_edge_index = torch.stack([start_node, end_node], dim=0)
        edge_index_2     = torch.cat([edge_index, cross_edge_index], dim=1)
        self.register_buffer("_cached_edge_index_2", edge_index_2)

    def get_new_edge(
        self, edge_index: torch.Tensor, los: torch.Tensor, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_nodes = len(self.ad_col_index)
        if hasattr(self, "_cached_edge_index_2") and self._cached_edge_index_2 is not None:
            edge_index_2 = self._cached_edge_index_2
        else:
            edge_index_2 = self._build_edge_index_2(edge_index, num_nodes, batch_size)
        edge_attr = self.get_edge_attr(los=los, edge_index=edge_index, batch_size=batch_size, num_nodes=num_nodes)
        return edge_index_2, edge_attr

    def _build_edge_index_2(
        self, edge_index: torch.Tensor, num_nodes: int, batch_size: int
    ) -> torch.Tensor:
        merged_num_nodes = num_nodes * batch_size
        start_node       = torch.arange(0, merged_num_nodes, device=edge_index.device)
        end_node         = start_node + merged_num_nodes
        cross_edge_index = torch.stack([start_node, end_node], dim=0)
        return torch.cat([edge_index, cross_edge_index], dim=1)

    def get_edge_index_2(
        self, edge_index: torch.Tensor, num_nodes: int, batch_size: int
    ) -> torch.Tensor:
        return self._build_edge_index_2(edge_index, num_nodes, batch_size)

    def get_edge_attr(
        self,
        los: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute edge attributes for internal edges (NONE token) and cross edges (LOS token)."""
        device     = edge_index.device
        E_internal = edge_index.size(1)
        los_table = self._cross_temporal_los_embedding_table().to(device=device)

        # Internal edges → NONE token (index 0), no-copy expand
        none_emb           = los_table[0]                                            # (D,)
        edge_attr_internal = none_emb.unsqueeze(0).expand(E_internal, -1)           # (E_internal, D)

        # Cross edges → LOS token or expected LOS embedding from predicted distribution
        if _is_los_distribution(los, batch_size):
            los_dist = los.to(device=device, dtype=none_emb.dtype)
            edge_attr_cross = self.encode_los_distribution(los_dist).to(device=device).repeat_interleave(num_nodes, dim=0)
        else:
            los_idx = los.view(batch_size).to(device).long().repeat_interleave(num_nodes)  # (B*N,)
            if self.forecast_input_encoder == "distribution":
                edge_attr_cross = self._encode_los_hard_as_distribution(los_idx).to(device=device)
            else:
                edge_attr_cross = self.encode_los_indices(los_idx)                              # (B*N, D)

        return torch.cat([edge_attr_internal, edge_attr_cross], dim=0)  # (E_total, D)

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def reset_parameters(self) -> None:
        def _init(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        self.apply(_init)

        if self.los_none_embedding is not None:
            nn.init.normal_(self.los_none_embedding, mean=0.0, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
