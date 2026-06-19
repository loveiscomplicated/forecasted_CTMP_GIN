# models/factory.py
import pandas
import os
from src.data_processing.mi_dict import search_mi_dict, cv_mi_dict
from src.data_processing.edge import (fully_connected_edge_index_batched, 
                                      mi_edge_index_batched, 
                                      mi_edge_index_batched_for_a3tgcn, 
                                      mi_edge_index_batched_for_gin, 
                                      )
from src.models.ctmp_gin import CTMPGIN
from src.models.gin import GIN
from src.models.mlp import MLP
from src.models.gingru import GinGru, GinGru_2_Point
from src.models.a3tgcn import A3TGCN_manual, A3TGCN_2_points

import torch
MODEL_REGISTRY = {
    "ctmp_gin": CTMPGIN,
    "gin": GIN,
    "mlp": MLP,
    "gin_gru": GinGru,
    "gin_gru_2_points": GinGru_2_Point,
    "a3tgcn": A3TGCN_manual,
    "a3tgcn_2_points": A3TGCN_2_points,
}

def build_model(model_name: str, **kwargs):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](**kwargs)

def build_edge(model_name: str, 
               root: str,
               seed: int,
               train_df: pandas.DataFrame,
               num_nodes: int,
               batch_size: int,
               mi_cached: bool = True,
               **kwargs):
    if not kwargs["is_mi_based"]:
        return fully_connected_edge_index_batched(num_nodes=num_nodes, batch_size=batch_size)

    if mi_cached:
        mi_ad_dict, mi_dis_dict, mi_avg_dict, mi_dict = search_mi_dict(root=root,
                                    seed=seed,
                                    train_df=train_df,
                                    n_neighbors=kwargs['n_neighbors'],
                                    cache_path=kwargs.get('cache_path', None),
                                    remove_los=kwargs.get("remove_los", True),
                                    )
    else:
        mi_ad_dict, mi_dis_dict, mi_avg_dict, mi_dict = cv_mi_dict(root=root,
                            seed=seed,
                            train_df=train_df,
                            n_neighbors=kwargs['n_neighbors'],
                            remove_los=kwargs.get("remove_los", True),
                            )

    if model_name in ["a3tgcn", "a3tgcn_2_points"]:
        edge_index = mi_edge_index_batched_for_a3tgcn(batch_size=batch_size, 
                                                        num_nodes=num_nodes, 
                                                        mi_avg_dict=mi_avg_dict, # 여기에 LOS가 들어가냐 마냐는 위의 remove_los에 달려 있음
                                                        top_k=kwargs["top_k"],
                                                        threshold=kwargs["threshold"],
                                                        pruning_ratio=kwargs["pruning_ratio"],
                                                        return_edge_attr=kwargs["return_edge_attr"])
        return edge_index
    
    if model_name in ["gin", "mlp"]:
        edge_index = mi_edge_index_batched_for_gin(batch_size=batch_size,
                                                        num_nodes=num_nodes,
                                                        mi_dict_all_variables=mi_dict,
                                                        top_k=kwargs["top_k"],
                                                        threshold=kwargs["threshold"],
                                                        pruning_ratio=kwargs["pruning_ratio"],
                                                        return_edge_attr=kwargs["return_edge_attr"])
        return edge_index
    
    edge_index = mi_edge_index_batched(batch_size=batch_size, # ctmp_gin, gin_gru, gin_gru_2_points
                                       num_nodes=num_nodes,
                                       mi_ad_dict=mi_ad_dict,
                                       mi_dis_dict=mi_dis_dict,
                                       top_k=kwargs["top_k"],
                                       threshold=kwargs["threshold"],
                                       pruning_ratio=kwargs["pruning_ratio"],
                                       return_edge_attr=kwargs["return_edge_attr"])
    return edge_index

