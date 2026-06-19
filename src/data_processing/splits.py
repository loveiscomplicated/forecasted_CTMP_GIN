# src/data_processing/splits.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

try:
    # sklearn is the most reliable for stratified splitting
    from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
except ImportError as e:
    raise ImportError(
        "This module requires scikit-learn. Install with: pip install scikit-learn"
    ) from e


def _to_numpy_labels(labels) -> np.ndarray:
    """
    Convert various label containers to a 1D numpy array of ints.
    Supports list, numpy array, torch tensor.
    """
    if isinstance(labels, np.ndarray):
        y = labels
    elif torch.is_tensor(labels):
        y = labels.detach().cpu().numpy()
    else:
        y = np.asarray(labels)

    y = y.reshape(-1)
    # handle float labels like 0.0/1.0
    if np.issubdtype(y.dtype, np.floating):
        y = y.astype(np.int64)
    return y


def get_labels_from_dataset(dataset) -> np.ndarray:
    """
    Extract labels by indexing dataset[i][1] for i in range(len(dataset)).
    Matches your current convention.

    Returns:
        y: shape [N] numpy int array
    """
    N = len(dataset)
    y = np.array([dataset[i][1] for i in range(N)])
    return _to_numpy_labels(y)


def holdout_test_split_stratified(
    dataset,
    test_ratio: float = 0.15,
    seed: int = 42,
    labels: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified holdout split:
      - returns (trainval_idx, test_idx) as numpy integer arrays
      - does NOT create DataLoaders (indices only)

    Args:
        dataset: any object with __len__ and __getitem__ returning (x, y) at [1]
        test_ratio: fraction assigned to test
        seed: random seed
        labels: optional precomputed labels array [N]; if None, extracted from dataset

    Returns:
        trainval_idx: indices for train+val pool
        test_idx: indices for test holdout
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError(f"test_ratio must be in (0,1), got {test_ratio}")

    y = get_labels_from_dataset(dataset) if labels is None else _to_numpy_labels(labels)
    N = len(y)
    all_idx = np.arange(N)

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    trainval_pos, test_pos = next(sss.split(all_idx, y))

    trainval_idx = all_idx[trainval_pos].astype(np.int64)
    test_idx = all_idx[test_pos].astype(np.int64)

    return trainval_idx, test_idx


def kfold_stratified(
    trainval_idx: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    seed: int = 42,
) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
    """
    Stratified K-fold on the *trainval pool*:
      - yields (fold, train_idx, val_idx) where indices are in the original dataset index space
      - does NOT create DataLoaders (indices only)

    Args:
        trainval_idx: indices of the train+val pool in original dataset space, shape [M]
        labels: labels for the full dataset, shape [N]
        n_folds: K
        seed: random seed (used when shuffle=True)

    Yields:
        fold: 0..K-1
        train_idx: original-space indices for training in this fold
        val_idx: original-space indices for validation in this fold
    """
    trainval_idx = np.asarray(trainval_idx, dtype=np.int64).reshape(-1)
    y_all = _to_numpy_labels(labels)

    if trainval_idx.size == 0:
        raise ValueError("trainval_idx is empty.")
    if n_folds < 2:
        raise ValueError(f"n_folds must be >=2, got {n_folds}")

    y_trainval = y_all[trainval_idx]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # skf gives positions within trainval_idx
    for fold, (tr_pos, va_pos) in enumerate(skf.split(np.zeros_like(y_trainval), y_trainval)):
        train_idx = trainval_idx[tr_pos]
        val_idx = trainval_idx[va_pos]
        yield fold, train_idx.astype(np.int64), val_idx.astype(np.int64)

def make_loaders(dataset, 
                 train_idx, 
                 val_idx, 
                 test_idx, 
                 batch_size, 
                 num_workers, 
                 drop_last=True):
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)

    print(f"Train Set Size: {len(train_dataset)}")
    print(f"Valid Set Size: {len(val_dataset)}")
    print(f"Test Set Size: {len(test_dataset)}")

    # val/test must use drop_last=True regardless of the parameter:
    # edge_index is built with a fixed batch_size, so a short last-batch
    # would cause an out-of-range node index error at inference time.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True, num_workers=num_workers, drop_last=drop_last)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=False, num_workers=num_workers, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers, drop_last=True)
    
    return train_dataloader, val_dataloader, test_dataloader