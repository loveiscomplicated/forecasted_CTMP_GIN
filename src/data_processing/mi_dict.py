import os
import pickle
from copy import deepcopy
from tqdm import tqdm
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from .data_utils import get_ad_dis_col

def _remove_target(df: pd.DataFrame):
    cols = set(df.columns)
    if 'REASON' in cols:
        cols.discard('REASON')
    if 'REASONb' in cols:
        cols.discard('REASONb')
    return df[list(cols)].copy(deep=True)

def _get_mi_helper(df: pd.DataFrame, seed: int, n_neighbors: int):
    """
    Compute mutual information (MI) between each column and all remaining columns.

    Args:
        df (pd.DataFrame): Training DataFrame.
        seed (int): Random seed for MI estimation.
        n_neighbors (int): Number of neighbors for MI estimation.

    Returns:
        dict: Mapping {target_column: MI Series over remaining columns}.
    """
    mi_dict = {}
    for col in tqdm(df.columns):
        x = df.drop(col, axis=1)
        y = df[col]
        mi = mutual_info_classif(x, y, discrete_features=True, n_neighbors=n_neighbors, random_state=seed)
        mi_series = pd.Series(mi, index=x.columns)
        mi_dict[col] = mi_series
    return mi_dict

def get_mi_dict(train_df: pd.DataFrame, seed: int, mi_dict_path: str | None, n_neighbors=3):
    """
    Compute and save mutual information dictionary for all variables.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        seed (int): Random seed.
        mi_dict_path (str): Path to save the MI dictionary.
        n_neighbors (int): Number of neighbors for MI estimation.

    Returns:
        dict: Mutual information dictionary.
    """
    train_df = _remove_target(train_df)
    mi_dict = _get_mi_helper(train_df, seed, n_neighbors)
    if mi_dict_path is not None:
        with open(mi_dict_path, 'wb') as f:
            pickle.dump(mi_dict, f)
    return mi_dict

def _seperate_ad(mi_dict: dict, ad_col_list):
    """
    Extract admission-stage mutual information values.

    Args:
        mi_dict (dict): Full MI dictionary.
        ad_col_list (list): Admission column list.

    Returns:
        dict: Admission-only MI dictionary.
    """
    mi_ad_dict = {}
    for key, value in mi_dict.items():
        if key not in ad_col_list:
            continue
        cur_col = [c for c in ad_col_list if c != key and c in value.index]
        mi_ad_dict[key] = value[cur_col]
    return mi_ad_dict

def _seperate_dis(mi_dict: dict, dis_col_list):
    """
    Extract discharge-stage mutual information values and align variable names.

    Args:
        mi_dict (dict): Full MI dictionary.
        dis_col_list (list): Discharge column list.

    Returns:
        dict: Discharge-only MI dictionary.
    """
    mi_dis_dict = {}
    for key, value in mi_dict.items():
        if key not in dis_col_list:
            continue
        
        cur_col = [c for c in dis_col_list if c != key and c in value.index]
        new_value = value[cur_col].copy()
        
        # Rename index by removing "_D" suffix to align with admission variables
        new_value.index = [i[:-2] if i.endswith("_D") else i for i in new_value.index]
        
        # If the key itself is a discharge variable, rename it too
        new_key = key[:-2] if key.endswith("_D") else key
        mi_dis_dict[new_key] = new_value
        
    return mi_dis_dict

def _get_avg(mi_ad_dict: dict, mi_dis_dict: dict):
    """
    Compute the average of admission and discharge MI values.

    Args:
        mi_ad_dict (dict): Admission MI dictionary.
        mi_dis_dict (dict): Discharge MI dictionary.

    Returns:
        dict: Averaged MI dictionary.
    """
    mi_avg_dict = {}
    # Only average variables that exist in both dictionaries
    common_vars = set(mi_ad_dict.keys()) & set(mi_dis_dict.keys())
    
    for var in common_vars:
        # Align indices before averaging to handle cases where one side might have missing features
        ad_series = mi_ad_dict[var]
        dis_series = mi_dis_dict[var]
        
        # Intersection of indices to avoid NaNs during addition
        common_idx = ad_series.index.intersection(dis_series.index)
        if not common_idx.empty:
            avg_value = (ad_series[common_idx] + dis_series[common_idx]) / 2
            mi_avg_dict[var] = avg_value
    
    return mi_avg_dict

def seperate_ad_dis(mi_dict: dict, ad_col_list, dis_col_list):
    """
    Split MI dictionary into admission, discharge, and averaged components.

    Args:
        mi_dict (dict): Full MI dictionary.
        ad_col_list (list): Admission column list.
        dis_col_list (list): Discharge column list.

    Returns:
        tuple: (mi_ad_dict, mi_dis_dict, mi_avg_dict)
    """
    mi_ad_dict = _seperate_ad(mi_dict=mi_dict, ad_col_list=ad_col_list)
    mi_dis_dict = _seperate_dis(mi_dict=mi_dict, dis_col_list=dis_col_list)
    mi_avg_dict = _get_avg(mi_ad_dict=mi_ad_dict, mi_dis_dict=mi_dis_dict)
    return mi_ad_dict, mi_dis_dict, mi_avg_dict


def search_mi_dict(root: str, seed: int, train_df: pd.DataFrame, n_neighbors=3, remove_los=True, cache_path: str | None = None):
    """
    Load cached MI results or compute them, then split by admission/discharge.

    Args:
        root (str): Root directory for MI cache.
        seed (int): Random seed.
        train_df (pd.DataFrame): Training DataFrame.
        n_neighbors (int): Number of neighbors for MI estimation. 
        cache_path (None or str): if set to str, it searches the selected path priorly. 
                                  if None, it automatically searches by its seed.

    Returns:
        tuple: (mi_ad_dict, mi_dis_dict, mi_avg_dict)
    """
    print("Buliding Edge Index Based on Mutual Information")
    mi_dict = None

    # 1. Prioritize loading from the user-specified path (cache_path)
    if cache_path is not None:
        try:
            print("Loading cached file...")
            with open(cache_path, 'rb') as f:
                mi_dict = pickle.load(f)
        except Exception as e:
            print(f"FAILED: {e}. Moving to default search...")
            print("Searching cached file by its key...")

    # 2. If loading failed or cache_path was not provided, check the default path (seed-based)
    if mi_dict is None:
        mi_dict_path = os.path.join(root, 'mi', f'mi_dict_seed_{seed}_n_neighbors_{n_neighbors}_remove_los_{remove_los}.pickle')
        
        if os.path.exists(mi_dict_path):
            print("Loading cached file...")
            with open(mi_dict_path, 'rb') as f:
                mi_dict = pickle.load(f)
            
        else:
            print("Calculating MI...")
            mi_dict = get_mi_dict(train_df=train_df, seed=seed, mi_dict_path=mi_dict_path, n_neighbors=n_neighbors)

    # Guard: strip target columns if they were accidentally included in cached dict
    for _label in ["REASON", "REASONb"]:
        if _label in mi_dict:
            del mi_dict[_label]

    ad_col_list, dis_col_list = get_ad_dis_col(df=train_df, remove_los=remove_los)
    mi_ad_dict, mi_dis_dict, mi_avg_dict = seperate_ad_dis(mi_dict=mi_dict, ad_col_list=ad_col_list, dis_col_list=dis_col_list)
    return mi_ad_dict, mi_dis_dict, mi_avg_dict, mi_dict


def cv_mi_dict(root: str, seed: int, train_df: pd.DataFrame, n_neighbors=3, remove_los=True):
    """
    calculate mi_dict every time.
    Args:
        root (str): Root directory for MI cache.
        seed (int): Random seed.
        train_df (pd.DataFrame): Training DataFrame.
        n_neighbors (int): Number of neighbors for MI estimation. # NOTE: cv result of n_neighbors: 3

    Returns:
        tuple: (mi_ad_dict, mi_dis_dict, mi_avg_dict)
    """
    print("Buliding Mutual Information based edge index")
    mi_dict_path = os.path.join(root, 'mi', f'mi_dict_seed_{seed}_n_neighbors_{n_neighbors}_remove_los_{remove_los}.pickle')
    print("Calculating MI...")
    mi_dict = get_mi_dict(train_df=train_df, seed=seed, mi_dict_path=mi_dict_path, n_neighbors=n_neighbors) 

    ad_col_list, dis_col_list = get_ad_dis_col(df=train_df, remove_los=remove_los)
    mi_ad_dict, mi_dis_dict, mi_avg_dict = seperate_ad_dis(mi_dict=mi_dict, ad_col_list=ad_col_list, dis_col_list=dis_col_list)
    return mi_ad_dict, mi_dis_dict, mi_avg_dict, mi_dict
