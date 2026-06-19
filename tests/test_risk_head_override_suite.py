from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.run_risk_head_override_suite import (
    _SplitForecastCacheDataset,
    _find_run_dir,
    _load_joint_cache_payload,
    _rebuild_cached_predictions_payload_from_joint_cache,
    apply_oracle_head_override,
    parse_args,
)


class _BaseDataset:
    def __init__(self) -> None:
        self.col_info = (
            ["ad0", "SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
            [3, 2, 2, 2],
            [0],
            [1, 2, 3],
        )
        self.raw_row_index = SimpleNamespace(
            to_numpy=lambda dtype=None, copy=True: torch.tensor([10, 11], dtype=torch.int64).numpy()
        )
        self.samples = [
            (torch.tensor([0, 0, 0, 0], dtype=torch.long), 1, 3),
            (torch.tensor([1, 1, 1, 1], dtype=torch.long), 0, 4),
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def test_parse_args_accepts_named_head_sets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_risk_head_override_suite.py",
            "--runs-root",
            "runs",
            "--registry-path",
            "registry.md",
            "--run-ids",
            "9",
            "22",
            "--head-sets",
            "old_total_drift_top3",
            "new_dvD_top3",
            "--override-mode",
            "oracle",
            "--split",
            "valid",
            "--out-dir",
            "reports/risk_head_override/test",
        ],
    )
    args = parse_args()
    assert args.run_ids == [9, 22]
    assert args.head_sets == ["old_total_drift_top3", "new_dvD_top3"]


def test_apply_oracle_override_uses_true_discharge_targets_not_admission_values() -> None:
    base_dataset = _BaseDataset()
    cached_predictions_payload = {
        "x": torch.tensor([[0, 0, 0, 0], [1, 0, 0, 0]], dtype=torch.long),
        "los": torch.tensor([3, 4], dtype=torch.long),
        "indices": torch.tensor([0, 1], dtype=torch.long),
        "soft_discharge": {
            "metadata": {},
            "head_names": ["SERVICES_D"],
            "soft_head_names": ["SERVICES_D"],
            "heads": {
                "SERVICES_D": {
                    "hard": torch.tensor([0, 0], dtype=torch.long),
                    "probs": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
                    "logits": torch.tensor([[0.0, -10.0], [0.0, -10.0]], dtype=torch.float32),
                    "target_col_idx": torch.tensor(1, dtype=torch.long),
                    "num_classes": torch.tensor(2, dtype=torch.long),
                    "class_to_embedding_idx": torch.tensor([0, 1], dtype=torch.long),
                    "mask": torch.tensor([True, True]),
                }
            },
        },
    }
    joint_cache_payload = {
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "targets": {
            "d": {
                "SERVICES_D": torch.tensor([1, 1], dtype=torch.long),
            }
        },
    }

    overridden = apply_oracle_head_override(
        base_dataset=base_dataset,
        cached_predictions_payload=cached_predictions_payload,
        joint_cache_payload=joint_cache_payload,
        override_heads=["SERVICES_D"],
    )

    assert overridden["x"][:, 1].tolist() == [1, 1]
    assert overridden["x"][:, 0].tolist() == [0, 1]
    assert overridden["soft_discharge"]["heads"]["SERVICES_D"]["hard"].tolist() == [1, 1]


def test_apply_oracle_override_only_replaces_selected_heads() -> None:
    base_dataset = _BaseDataset()
    cached_predictions_payload = {
        "x": torch.tensor([[0, 0, 1, 0], [1, 0, 1, 0]], dtype=torch.long),
        "los": torch.tensor([3, 4], dtype=torch.long),
        "indices": torch.tensor([0, 1], dtype=torch.long),
        "soft_discharge": None,
    }
    joint_cache_payload = {
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "targets": {
            "d": {
                "SERVICES_D": torch.tensor([1, 1], dtype=torch.long),
                "SUB1_D": torch.tensor([0, 0], dtype=torch.long),
            }
        },
    }

    overridden = apply_oracle_head_override(
        base_dataset=base_dataset,
        cached_predictions_payload=cached_predictions_payload,
        joint_cache_payload=joint_cache_payload,
        override_heads=["SERVICES_D"],
    )

    assert overridden["x"][:, 1].tolist() == [1, 1]
    assert overridden["x"][:, 2].tolist() == [1, 1]


def test_missing_joint_cache_returns_missing_cache_status(tmp_path) -> None:
    payload, warning = _load_joint_cache_payload(tmp_path / "fold_0", "valid", torch.device("cpu"))
    assert payload is None
    assert warning == "missing_cache"


def test_split_forecast_cache_dataset_accepts_numpy_indices() -> None:
    base_dataset = _BaseDataset()
    dataset = _SplitForecastCacheDataset(
        base_dataset=base_dataset,
        x_cache=torch.tensor([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=torch.long),
        los_cache=torch.tensor([3, 4], dtype=torch.long),
        indices=np.array([0, 1], dtype=np.int64),
        soft_discharge_cache=None,
    )

    x, y, los = dataset[1]

    assert x.tolist() == [1, 1, 1, 1]
    assert int(y) == 0
    assert int(los) == 4


def test_rebuild_cached_predictions_payload_from_existing_joint_cache(tmp_path) -> None:
    fold_dir = tmp_path / "fold_0"
    fold_dir.mkdir()
    (fold_dir / "joint_forecast_pipeline_splits.json").write_text(
        json.dumps(
            {
                "gnn_val_idx": [0, 1],
                "outer_test_idx": [0, 1],
                "joint_mode": "distribution",
                "forecast_input_metadata": {"forecast_input_encoder": "distribution"},
            }
        ),
        encoding="utf-8",
    )
    cfg = {
        "model": {"name": "ctmp_gin", "params": {"forecast_input_encoder": "distribution"}},
        "joint_forecast_pipeline": {
            "enabled": True,
            "joint_forecast_input": {"mode": "distribution"},
        },
    }
    joint_cache_payload = {
        "row_idx": torch.tensor([10, 11], dtype=torch.long),
        "final_d_pred": {
            "SERVICES_D": torch.tensor([1, 0], dtype=torch.long),
            "SUB1_D": torch.tensor([0, 1], dtype=torch.long),
            "FREQ_ATND_SELF_HELP_D": torch.tensor([1, 0], dtype=torch.long),
        },
        "final_d_probs": {
            "SERVICES_D": torch.tensor([[0.1, 0.9], [0.8, 0.2]], dtype=torch.float32),
            "SUB1_D": torch.tensor([[0.7, 0.3], [0.2, 0.8]], dtype=torch.float32),
            "FREQ_ATND_SELF_HELP_D": torch.tensor([[0.4, 0.6], [0.9, 0.1]], dtype=torch.float32),
        },
        "final_d_logits": {
            "SERVICES_D": torch.tensor([[-1.0, 1.0], [1.0, -1.0]], dtype=torch.float32),
            "SUB1_D": torch.tensor([[1.0, -1.0], [-1.0, 1.0]], dtype=torch.float32),
            "FREQ_ATND_SELF_HELP_D": torch.tensor([[-0.2, 0.2], [1.1, -1.1]], dtype=torch.float32),
        },
        "final_los_probs": torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        "final_los_pred": torch.tensor([0, 2], dtype=torch.long),
        "metadata": {
            "target_col_names": ["SERVICES_D", "SUB1_D", "FREQ_ATND_SELF_HELP_D"],
            "final_los_pred_space": "coarse_class",
        },
    }

    payload, warning = _rebuild_cached_predictions_payload_from_joint_cache(
        fold_dir=fold_dir,
        cfg=cfg,
        base_dataset=_BaseDataset(),
        split="valid",
        joint_cache_payload=joint_cache_payload,
    )

    assert payload is not None
    assert "rebuilt cached_predictions/gnn_val_joint.pt" in warning
    assert not (fold_dir / "cached_predictions" / "gnn_val_joint.pt").exists()
    assert payload["indices"].tolist() == [0, 1]
    assert payload["x"][:, 1].tolist() == [1, 0]
    assert payload["x"][:, 2].tolist() == [0, 1]
    assert payload["los"].shape == (2, 37)
    assert torch.isclose(payload["los"][0].sum(), torch.tensor(1.0))
    services_head = payload["soft_discharge"]["heads"]["SERVICES_D"]
    assert services_head["mask"].tolist() == [True, True]
    assert torch.allclose(services_head["probs"][0], torch.tensor([0.1, 0.9]))


def test_find_run_dir_prefers_downstream_ctmp_run(tmp_path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    predictor_dir = runs_root / "20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1"
    predictor_dir.mkdir()
    downstream_dir = runs_root / "20260519-030924__ctmp_gin_joint_fresh_id9__bs=512__lr=6.10e-04__seed=1__cv=5__test=0.2"
    (downstream_dir / "folds").mkdir(parents=True)
    registry_path = tmp_path / "registry.md"
    registry_path.write_text(
        "| ID | Batch | Run ID |\n| ---: | --- | --- |\n| 9 | 2026-05-16 | `20260516-125415__joint_consistent_predictor__bs=1024__lr=1.00e-03__seed=1` |\n",
        encoding="utf-8",
    )

    resolved = _find_run_dir(runs_root, 9, registry_path, {})

    assert resolved == downstream_dir
