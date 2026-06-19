import torch
import pickle
from copy import deepcopy
from torch_geometric.utils import to_undirected

def fully_connected_edge_index(num_nodes):
    """
    Create a fully connected edge index for a single graph.

    Args:
        num_nodes (int): Number of nodes (variables).
        return_edge_attr (bool): Whether to include edge_attr.

    Returns:
        torch.Tensor or tuple: Batched edge index, optionally with edge attributes.
    """
    nodes = torch.arange(num_nodes)
    row, col = torch.meshgrid(nodes, nodes, indexing="ij")
    edge_index = torch.stack([row.reshape(-1), col.reshape(-1)], dim=0)
    return edge_index

def fully_connected_edge_index_batched(num_nodes, batch_size):
    """
    Create a batched fully connected edge index by concatenation.

    Args:
        num_nodes (int): Number of nodes per graph.
        batch_size (int): Number of graphs in the batch.
        return_edge_attr (bool): Whether to include edge_attr.

    Returns:
        torch.Tensor: Batched edge index of shape [2, total_edges], optionally with edge attributes.
    """
    # first = fully_connected_edge_index(num_nodes=num_nodes)
    # second = first + num_nodes
    # double = torch.cat([first, second], dim=1) # due to admission and discharge
    single = fully_connected_edge_index(num_nodes=num_nodes)
    edge_list = []

    for g in range(batch_size * 2): 
        offset = num_nodes * g
        edge_i = single + offset
        edge_list.append(edge_i)

    batched_edge_index = torch.cat(edge_list, dim=1)
    return batched_edge_index

def mi_edge_index_single(
    mi_dict, top_k=6, threshold=0.01, pruning_ratio=0.5, return_edge_attr=False
):
    """
    Construct a graph edge index from mutual information (MI).

    Applies:
        1) Undirected conversion
        2) Threshold filtering
        3) Top-k neighbor selection
        4) In-degree pruning

    Args:
        mi_dict (dict): MI values per variable.
        top_k (int): Number of top neighbors per node.
        threshold (float): Minimum MI value to keep an edge.
        pruning_ratio (float): Maximum allowed in-degree ratio.
        return_edge_attr (bool): Whether to return MI as edge weights.

    Returns:
        torch.Tensor or tuple: Edge index, optionally with edge attributes.
    """
    cols = list(mi_dict.keys())
    num_nodes = len(cols)
    col_to_idx = {c: i for i, c in enumerate(cols)}

    # 임시 저장을 위한 리스트 (source, target, weight)
    raw_edges = []

    # 1. 초기 유향 엣지 생성 (Threshold & Top-k 적용)
    for src in cols:
        series = mi_dict[src]

        # 유효 변수 필터링 & 자기 자신 제외
        series = series[series.index.isin(cols)]
        if src in series.index:
            series = series.drop(index=src)

        # [Strategy 2] Threshold 적용 (너무 약한 관계 끊기)
        series = series[series >= threshold]

        # Top-k 선택
        top_neighbors = series.head(top_k)

        src_idx = col_to_idx[src]
        for dst, w in top_neighbors.items():
            dst_idx = col_to_idx[dst]
            raw_edges.append((src_idx, dst_idx, float(w)))

    # 2. [Strategy 3] 구조적 Pruning (Hub 노드 견제)
    # Target 노드별로 엣지를 모아서 In-Degree가 너무 높으면 약한 것부터 잘라냅니다.
    
    # Target별로 그룹화: {dst_idx: [(src, dst, w), ...]}
    edges_by_target = {}
    for edge in raw_edges:
        dst = edge[1]
        if dst not in edges_by_target:
            edges_by_target[dst] = []
        edges_by_target[dst].append(edge)
    
    final_edges = []
    max_in_degree = int(num_nodes * pruning_ratio) # 허용 가능한 최대 In-Degree (예: 60개 중 30개)

    for dst, edges in edges_by_target.items():
        # 만약 특정 노드(예: STFIPS)로 들어오는 엣지가 너무 많다면?
        if len(edges) > max_in_degree:
            # 가중치(MI) 기준 내림차순 정렬 후 상위 N개만 남김
            edges.sort(key=lambda x: x[2], reverse=True)
            kept_edges = edges[:max_in_degree]
            final_edges.extend(kept_edges)
        else:
            final_edges.extend(edges)

    # 텐서 변환 준비
    if not final_edges:
        print("⚠️ 주의: 조건에 맞는 엣지가 하나도 없습니다. Threshold를 낮추세요.")
        return torch.empty((2, 0), dtype=torch.long)

    src_list, dst_list, weight_list = zip(*final_edges)
    
    # Directed Edge Index 생성
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float) if return_edge_attr else None

    # 3. [Strategy 1] 무방향(Undirected) 그래프로 변환
    # A->B가 있으면 B->A도 생성 (정보 흐름 개선)
    # to_undirected는 중복된 엣지는 제거하고, 양방향을 보장해줍니다.
    if return_edge_attr:
        edge_index, edge_attr = to_undirected(edge_index, edge_attr, num_nodes=num_nodes)
        return edge_index, edge_attr
    else:
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        return edge_index

def mi_edge_index_batched(
    batch_size,
    num_nodes,
    mi_ad_dict,
    mi_dis_dict,
    top_k=6,
    threshold=0.01,
    pruning_ratio=0.5,
    return_edge_attr=False,
    edge_attr_single=None,
):
    """
    Build batched MI-based edge indices for admission and discharge graphs.

    Args:
        batch_size (int): Batch size.
        num_nodes (int): Number of nodes per graph.
        mi_ad_dict (dict): Admission-stage MI dictionary.
        mi_dis_dict (dict): Discharge-stage MI dictionary.
        top_k (int): Number of top neighbors per node.
        threshold (float): MI threshold.
        pruning_ratio (float): In-degree pruning ratio.
        return_edge_attr (bool): Whether to return edge weights.

    Returns:
        torch.Tensor or tuple: Batched edge index, optionally with edge attributes.
    """
    if return_edge_attr:
        single_ad, edge_attr_ad= mi_edge_index_single(
            mi_dict=mi_ad_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single_ad = mi_edge_index_single(
            mi_dict=mi_ad_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    
    if return_edge_attr:
        single_dis, edge_attr_dis= mi_edge_index_single(
            mi_dict=mi_dis_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single_dis = mi_edge_index_single(
            mi_dict=mi_dis_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )

    edge_list = []
    attr_list = []

    for g in range(batch_size):
        offset = num_nodes * g
        edge_i = single_ad + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr_ad)
    
    offset_ad = num_nodes * batch_size  # dis graphs start after all ad graphs

    for g in range(batch_size):
        offset = num_nodes * g + offset_ad
        edge_i = single_dis + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr_dis)

    batched_edge_index = torch.cat(edge_list, dim=1)
    if return_edge_attr:
        batched_attr_list = torch.cat(attr_list, dim=0)
        return batched_edge_index, batched_attr_list
    return batched_edge_index
    

def mi_edge_index_batched_for_a3tgcn(
    batch_size,
    num_nodes,
    mi_avg_dict,
    top_k=6,
    threshold=0.01,
    pruning_ratio=0.5,
    return_edge_attr=False,
    edge_attr_single=None,
):
    """
    Build a static batched MI-based edge index for baseline models.

    Uses averaged MI values shared across all graphs in the batch.

    Args:
        batch_size (int): Batch size.
        num_nodes (int): Number of nodes per graph.
        mi_avg_dict (dict): Averaged MI dictionary.
        top_k (int): Number of top neighbors per node.
        threshold (float): MI threshold.
        pruning_ratio (float): In-degree pruning ratio.
        return_edge_attr (bool): Whether to return edge weights.

    Returns:
        torch.Tensor or tuple: Batched edge index, optionally with edge attributes.
    """

    batch_size_d = batch_size

    if return_edge_attr:
        single, edge_attr= mi_edge_index_single(
            mi_dict=mi_avg_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
        return single, edge_attr
    else:
        single = mi_edge_index_single(
            mi_dict=mi_avg_dict, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
        return single

def mi_edge_index_batched_for_gin(batch_size, 
                                  num_nodes, 
                                  mi_dict_all_variables, 
                                  top_k=6, 
                                  threshold=0.01, 
                                  pruning_ratio=0.5, 
                                  return_edge_attr=False, 
                                  edge_attr_single=None):
    batch_size_d = batch_size

    if return_edge_attr:
        single, edge_attr= mi_edge_index_single(
            mi_dict=mi_dict_all_variables, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )
    else:
        single = mi_edge_index_single(
            mi_dict=mi_dict_all_variables, 
            top_k=top_k, 
            threshold=threshold, 
            pruning_ratio=pruning_ratio,
            return_edge_attr=return_edge_attr
        )

    edge_list = []
    attr_list = []

    for g in range(batch_size_d):
        offset = num_nodes * g
        edge_i = single + offset
        edge_list.append(edge_i)

        if return_edge_attr:
            attr_list.append(edge_attr)
    
    batched_edge_index = torch.cat(edge_list, dim=1)

    if return_edge_attr:
        batched_attr_list = torch.cat(attr_list, dim=0)
        return batched_edge_index, batched_attr_list
    
    return batched_edge_index
