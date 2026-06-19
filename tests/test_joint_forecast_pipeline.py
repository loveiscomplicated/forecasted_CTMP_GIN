from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.models.forecast_inputs import resolve_model_forecast_input_metadata
from src.trainers.forecasted_pipeline import (
    _assign_joint_cache_split,
    _cache_positions_for_expected_indices,
    _init_caches,
    _init_joint_soft_discharge_cache,
    _resolve_joint_cache_paths,
)


class _DummyBaseDataset:
    def __init__(self) -> None:
        self._x = [
            torch.tensor([0, 1, 0, 0], dtype=torch.long),
            torch.tensor([1, 0, 1, 1], dtype=torch.long),
            torch.tensor([2, 1, 2, 0], dtype=torch.long),
            torch.tensor([0, 0, 1, 1], dtype=torch.long),
        ]
        self.labels = [0, 1, 0, 1]
        self.los = [1, 2, 3, 4]
        self.processed_df = None
        self.num_classes = 2
        self.col_info = (
            ["AD_A", "AD_B", "SERVICES_D", "SUB1_D"],
            [3, 2, 3, 2],
            [0, 1],
            [2, 3],
        )
        self.raw_row_index = pd.Series([100, 101, 102, 103])

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, index: int):
        return self._x[index], self.labels[index], self.los[index]


def _joint_cache_payload() -> dict[str, object]:
    return {
        "row_idx": torch.tensor([101, 103], dtype=torch.long),
        "final_d_pred": {
            "SERVICES_D": torch.tensor([2, 1], dtype=torch.long),
            "SUB1_D": torch.tensor([1, 0], dtype=torch.long),
        },
        "final_d_probs": {
            "SERVICES_D": torch.tensor(
                [[0.1, 0.2, 0.7], [0.2, 0.6, 0.2]],
                dtype=torch.float32,
            ),
            "SUB1_D": torch.tensor(
                [[0.3, 0.7], [0.9, 0.1]],
                dtype=torch.float32,
            ),
        },
        "final_d_logits": {
            "SERVICES_D": torch.tensor(
                [[1.0, 2.0, 3.0], [1.5, 2.5, 0.5]],
                dtype=torch.float32,
            ),
            "SUB1_D": torch.tensor(
                [[0.1, 0.9], [0.8, 0.2]],
                dtype=torch.float32,
            ),
        },
        "final_los_probs": torch.tensor(
            [[0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.2, 0.0, 0.0, 0.0, 0.8, 0.0]],
            dtype=torch.float32,
        ),
        "final_los_pred": torch.tensor([1, 4], dtype=torch.long),
        "metadata": {
            "target_col_names": ["SERVICES_D", "SUB1_D"],
            "final_los_pred_space": "coarse_class",
        },
    }


def test_joint_forecast_input_metadata_uses_predictor_prob_contract() -> None:
    cfg = {
        "model": {
            "name": "ctmp_gin",
            "params": {"forecast_input_encoder": "distribution"},
        },
        "joint_forecast_pipeline": {
            "enabled": True,
            "joint_forecast_input": {"mode": "distribution"},
        },
    }

    metadata = resolve_model_forecast_input_metadata(cfg)

    assert metadata["d_input_type"] == "predictor_probs"
    assert metadata["los_input_type"] == "predictor_probs"
    assert metadata["joint_forecast_enabled"] is True


def test_joint_cache_positions_follow_dataset_row_index() -> None:
    dataset = _DummyBaseDataset()
    payload = _joint_cache_payload()

    positions = _cache_positions_for_expected_indices(
        payload,
        dataset,
        np.array([1, 3], dtype=np.int64),
        split_name="train",
    )

    assert positions.tolist() == [1, 3]


def test_assign_joint_cache_split_populates_soft_discharge_and_expands_coarse_los() -> None:
    dataset = _DummyBaseDataset()
    payload = _joint_cache_payload()
    x_cache, los_cache, _ = _init_caches(dataset, "distribution")
    soft_cache = _init_joint_soft_discharge_cache(
        dataset,
        len(dataset),
        payload,
        {"forecast_input_encoder": "distribution"},
    )

    _assign_joint_cache_split(
        base_dataset=dataset,
        split_name="train",
        cache_payload=payload,
        expected_indices=np.array([1, 3], dtype=np.int64),
        x_cache=x_cache,
        los_cache=los_cache,
        soft_discharge_cache=soft_cache,
        joint_mode="distribution",
    )

    assert int(x_cache[1, 2].item()) == 2
    assert int(x_cache[3, 2].item()) == 1
    assert int(x_cache[1, 3].item()) == 1
    assert int(x_cache[3, 3].item()) == 0

    assert los_cache.shape == (4, 37)
    assert torch.isclose(los_cache[1].sum(), torch.tensor(1.0))
    assert torch.isclose(los_cache[3].sum(), torch.tensor(1.0))
    assert torch.isclose(los_cache[1, 1:7].sum(), torch.tensor(1.0))
    assert torch.isclose(los_cache[3, 21:28].sum(), torch.tensor(0.8))

    services_head = soft_cache["heads"]["SERVICES_D"]
    sub1_head = soft_cache["heads"]["SUB1_D"]
    assert bool(services_head["mask"][1].item()) is True
    assert bool(services_head["mask"][3].item()) is True
    assert torch.allclose(
        services_head["probs"][1],
        torch.tensor([0.1, 0.2, 0.7], dtype=torch.float32),
    )
    assert torch.allclose(
        sub1_head["probs"][3],
        torch.tensor([0.9, 0.1], dtype=torch.float32),
    )


def test_resolve_joint_cache_paths_prefers_explicit_paths() -> None:
    cfg = {
        "joint_forecast_input": {
            "train_cache_path": "/tmp/train.pt",
            "gnn_val_cache_path": "/tmp/gnn_val.pt",
            "outer_test_cache_path": "/tmp/outer_test.pt",
        }
    }

    resolved = _resolve_joint_cache_paths(cfg, run_dir=None)

    assert resolved == {
        "train": "/tmp/train.pt",
        "gnn_val": "/tmp/gnn_val.pt",
        "outer_test": "/tmp/outer_test.pt",
    }
