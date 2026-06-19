from __future__ import annotations

import numpy as np

from src.trainers.forecasted_pipeline import (
    _inherit_dataset_settings,
    split_outer_train_for_forecasted_pipeline,
)


class _DummyTEDS:
    def __init__(self, n: int = 100) -> None:
        self.labels = np.asarray([i % 2 for i in range(n)], dtype=np.int64)
        self.los = np.asarray([(i % 37) + 1 for i in range(n)], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return None, int(self.labels[index]), int(self.los[index])


def test_forecasted_pipeline_internal_split_is_disjoint_and_sized() -> None:
    dataset = _DummyTEDS(100)
    cfg = {
        "forecasted_pipeline": {
            "split": {
                "train_core_ratio": 0.8,
                "predictor_val_ratio": 0.1,
                "gnn_val_ratio": 0.1,
                "stratify_by": "REASONb",
            }
        }
    }
    outer_train_idx = np.arange(80, dtype=np.int64)

    train_core, predictor_val, gnn_val = split_outer_train_for_forecasted_pipeline(
        dataset, outer_train_idx, cfg, seed=7
    )

    assert len(train_core) == 64
    assert len(predictor_val) == 8
    assert len(gnn_val) == 8
    assert set(train_core).isdisjoint(set(predictor_val))
    assert set(train_core).isdisjoint(set(gnn_val))
    assert set(predictor_val).isdisjoint(set(gnn_val))


def test_forecasted_pipeline_aux_stratify_falls_back_when_too_sparse() -> None:
    dataset = _DummyTEDS(40)
    cfg = {
        "forecasted_pipeline": {
            "split": {
                "train_core_ratio": 0.8,
                "predictor_val_ratio": 0.1,
                "gnn_val_ratio": 0.1,
                "stratify_by": "REASONb",
                "stratify_aux": ["LOS_coarse"],
            }
        }
    }
    outer_train_idx = np.arange(40, dtype=np.int64)

    train_core, predictor_val, gnn_val = split_outer_train_for_forecasted_pipeline(
        dataset, outer_train_idx, cfg, seed=11
    )

    assert len(train_core) + len(predictor_val) + len(gnn_val) == 40
    assert set(train_core) | set(predictor_val) | set(gnn_val) == set(outer_train_idx)


def test_forecasted_pipeline_inherits_parent_preprocess_setting() -> None:
    parent_cfg = {"train": {"do_preprocess": True}}
    child_cfg = {"train": {"do_preprocess": False}}

    _inherit_dataset_settings(parent_cfg, child_cfg)

    assert child_cfg["train"]["do_preprocess"] is True
