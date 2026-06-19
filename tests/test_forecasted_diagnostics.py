from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from src.trainers.forecasted_diagnostics import (
    _apply_head_overrides,
    _conditional_js_divergence,
    _cramers_v,
    _rare_combo_map,
    _rare_rate_for_rows,
    _summarize_joint_alignment,
    audit_los_distribution_basis,
    audit_los_hard_runtime,
    audit_forecast_value_space,
    build_oracle_coarse_los_forecast_cache,
    build_oracle_forecast_cache,
    build_predictor_target_forecast_cache,
    build_predictor_target_transformed_forecast_cache,
)
from src.trainers.forecasted_pipeline import ForecastCacheDataset
from scripts.diagnose_forecast_cache_alignment import parse_args


class _BaseDataset:
    def __init__(self) -> None:
        self.processed_tensor = torch.tensor(
            [
                [0, 10, 20, 0, 1],
                [1, 11, 21, 1, 0],
                [2, 12, 22, 2, 1],
            ],
            dtype=torch.long,
        )
        self.LOS = torch.tensor([1, 2, 3], dtype=torch.long)
        self.col_info = (
            ["A", "B", "B_D", "C_D"],
            [3, 13, 23, 3],
            [0, 1],
            [0, 2, 3],
        )
        self.processed_df = pd.DataFrame({"A": [0, 1, 2]})
        self.num_classes = 2

    def __len__(self) -> int:
        return self.processed_tensor.shape[0]

    def __getitem__(self, index: int):
        return self.processed_tensor[index, :-1], self.processed_tensor[index, -1], self.LOS[index]


class _DischargeDataset:
    def __init__(self, *, mismatch: bool = False, missing_name: bool = False) -> None:
        self.ad_col_names = ["A", "B"]
        self.x = torch.tensor([[0, 10], [1, 11], [2, 12]], dtype=torch.long)
        self.target_col_names = ["MISSING_D" if missing_name else "B_D", "C_D"]
        self.y = torch.tensor([[20, 0], [21, 1], [22, 2]], dtype=torch.long)
        if mismatch:
            self.y[1, 0] = 99


class _LOSDataset:
    def __init__(self, *, mismatch: bool = False) -> None:
        self.ad_col_names = ["A", "B"]
        self.x = torch.tensor([[0, 10], [1, 11], [2, 12]], dtype=torch.long)
        self.y = torch.tensor([1, 2, 3], dtype=torch.long)
        self.los_raw = torch.tensor([1, 2, 3], dtype=torch.long)
        if mismatch:
            self.los_raw[1] = 9


def test_build_oracle_forecast_cache_roundtrips_with_forecast_cache_dataset() -> None:
    base = _BaseDataset()

    x_cache, los_cache = build_oracle_forecast_cache(base, indices=[2, 0])
    cached = ForecastCacheDataset(base, x_cache, los_cache)

    x, y, los = cached[2]
    assert torch.equal(x, base[2][0])
    assert int(y) == int(base[2][1])
    assert int(los) == int(base[2][2])


def test_build_predictor_target_forecast_cache_accepts_matching_value_space() -> None:
    base = _BaseDataset()
    discharge = _DischargeDataset()
    los = _LOSDataset()

    x_cache, los_cache = build_predictor_target_forecast_cache(base, discharge, los)

    assert torch.equal(x_cache, base.processed_tensor[:, :-1])
    assert torch.equal(los_cache, base.LOS)


def test_predictor_target_cache_rejects_discharge_value_space_mismatch() -> None:
    with pytest.raises(RuntimeError, match="B_D"):
        build_predictor_target_forecast_cache(_BaseDataset(), _DischargeDataset(mismatch=True), _LOSDataset())


def test_predictor_target_cache_rejects_missing_discharge_target_name() -> None:
    with pytest.raises(RuntimeError, match="MISSING_D"):
        build_predictor_target_forecast_cache(_BaseDataset(), _DischargeDataset(missing_name=True), _LOSDataset())


def test_audit_rejects_los_value_space_mismatch() -> None:
    summary = audit_forecast_value_space(
        _BaseDataset(),
        _DischargeDataset(),
        _LOSDataset(),
        {"model": {"params": {"max_los": 37}}},
        los_cfg={"los_target_mode": "coarse"},
    )

    assert summary["los_value_match"] is False


def test_audit_summary_reports_matching_value_space() -> None:
    summary = audit_forecast_value_space(
        _BaseDataset(),
        _DischargeDataset(),
        _LOSDataset(),
        {"model": {"params": {"max_los": 37}}},
        los_cfg=None,
    )

    assert summary["admission_x_match_discharge_dataset"] is True
    assert summary["admission_x_match_los_dataset"] is True
    assert summary["all_discharge_value_match"] is True
    assert summary["los_value_match"] is True
    assert summary["cache_roundtrip_match"] is True


def test_predictor_target_transformed_cache_maps_coarse_los_to_raw_representatives() -> None:
    class _CoarseLOSDataset(_LOSDataset):
        def __init__(self) -> None:
            super().__init__()
            self.los_raw = torch.tensor([1, 2, 8], dtype=torch.long)

    base = _BaseDataset()
    base.LOS = torch.tensor([1, 2, 8], dtype=torch.long)
    _, los_cache, payload = build_predictor_target_transformed_forecast_cache(
        base,
        _DischargeDataset(),
        _CoarseLOSDataset(),
        los_cfg={"los_target_mode": "coarse"},
    )

    assert los_cache.tolist() == [1, 4, 11]
    assert payload["coarse_to_raw"] == {0: 1, 1: 4, 2: 11, 3: 18, 4: 25, 5: 33}
    assert payload["uses_los_zero"] is False


def test_oracle_coarse_los_cache_reports_compression_metrics() -> None:
    base = _BaseDataset()
    base.LOS = torch.tensor([1, 7, 37], dtype=torch.long)

    _, los_cache, payload = build_oracle_coarse_los_forecast_cache(base)

    assert los_cache.tolist() == [1, 4, 33]
    assert payload["los_zero_used"] is False
    assert payload["coarse_compression_mae"] > 0.0
    assert payload["coarse_compression_within_2"] < 1.0


def test_distribution_basis_audit_uses_expanded_raw_distribution_not_rows_0_to_5() -> None:
    payload = audit_los_distribution_basis(
        {
            "model": {"params": {"max_los": 37, "los_embedding_dim": 4}},
            "forecasted_los": {"target_mode": "coarse"},
        },
        _BaseDataset(),
        torch.device("cpu"),
    )

    assert payload["basis_valid"] is True
    assert payload["uses_embedding_zero"] is False
    assert payload["uses_rows_0_to_5_directly"] is False
    assert payload["used_embedding_indices"][0] == 1


def test_hard_runtime_audit_mapping_never_uses_zero() -> None:
    payload = audit_los_hard_runtime(
        {"forecasted_los": {"target_mode": "coarse"}},
        _BaseDataset(),
        torch.device("cpu"),
    )

    assert payload["uses_los_zero"] is False
    assert payload["injected_los_unique"] == [1, 4, 11, 18, 25, 33]


def test_joint_alignment_summary_accepts_perfect_match() -> None:
    rows = [
        {
            "split": "train",
            "position_in_loader": 0,
            "gnn_row_idx": 3,
            "d_cache_row_idx": 3,
            "los_cache_row_idx": 3,
            "gnn_caseid": "1003",
            "d_cache_caseid": "1003",
            "los_cache_caseid": "1003",
            "y": 1,
            "match_row_idx": True,
            "match_caseid": True,
            "split_name_gnn": "train",
            "split_name_d_cache": "train",
            "split_name_los_cache": "train",
        },
        {
            "split": "valid",
            "position_in_loader": 0,
            "gnn_row_idx": 4,
            "d_cache_row_idx": 4,
            "los_cache_row_idx": 4,
            "gnn_caseid": "1004",
            "d_cache_caseid": "1004",
            "los_cache_caseid": "1004",
            "y": 0,
            "match_row_idx": True,
            "match_caseid": True,
            "split_name_gnn": "valid",
            "split_name_d_cache": "valid",
            "split_name_los_cache": "valid",
        },
    ]

    summary = _summarize_joint_alignment(
        rows,
        caseid_available=True,
        duplicate_caseid_count=0,
        missing_caseid_count=0,
    )

    assert summary["gnn_row_id_match_d_cache"] is True
    assert summary["gnn_row_id_match_los_cache"] is True
    assert summary["d_cache_match_los_cache"] is True
    assert summary["caseid_available"] is True
    assert summary["caseid_match"] is True
    assert summary["num_mismatches"] == 0


def test_joint_alignment_summary_reports_mismatch_and_fallback_caseids() -> None:
    rows = [
        {
            "split": "test",
            "position_in_loader": 2,
            "gnn_row_idx": 9,
            "d_cache_row_idx": 9,
            "los_cache_row_idx": 10,
            "gnn_caseid": "9",
            "d_cache_caseid": "9",
            "los_cache_caseid": "10",
            "y": 1,
            "match_row_idx": False,
            "match_caseid": False,
            "split_name_gnn": "test",
            "split_name_d_cache": "test",
            "split_name_los_cache": "test",
        }
    ]

    summary = _summarize_joint_alignment(
        rows,
        caseid_available=False,
        duplicate_caseid_count=0,
        missing_caseid_count=0,
    )

    assert summary["caseid_available"] is False
    assert summary["caseid_match"] is False
    assert summary["num_mismatches"] == 1
    assert summary["first_20_mismatches"][0]["los_cache_row_idx"] == 10


def test_cli_accepts_joint_alignment_mode_and_discharge_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_forecast_cache_alignment.py",
            "--config",
            "configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml",
            "--mode",
            "joint_cache_alignment_audit",
            "--discharge-checkpoint-path",
            "/tmp/discharge.ckpt",
            "--los-checkpoint-path",
            "/tmp/los.ckpt",
        ],
    )

    args = parse_args()

    assert args.mode == "joint_cache_alignment_audit"
    assert args.discharge_checkpoint_path == "/tmp/discharge.ckpt"
    assert args.los_checkpoint_path == "/tmp/los.ckpt"


def test_conditional_js_divergence_is_zero_for_identical_tables() -> None:
    table = np.array([[3.0, 1.0], [2.0, 2.0]])

    score = _conditional_js_divergence(table, table.copy())

    assert score == pytest.approx(0.0)


def test_cramers_v_is_zero_for_independent_table() -> None:
    table = np.array([[5.0, 5.0], [5.0, 5.0]])

    score = _cramers_v(table)

    assert score == pytest.approx(0.0)


def test_rare_combo_map_and_rate_use_train_reference_threshold() -> None:
    rare_map, prob_map = _rare_combo_map(
        np.array([0, 0, 1, 1]),
        np.array([0, 0, 1, 1]),
        d_dim=3,
        threshold=0.20,
    )

    rate = _rare_rate_for_rows(
        np.array([0, 2]),
        np.array([0, 1]),
        rare_map,
    )

    assert prob_map[(0, 0)] == pytest.approx(0.5)
    assert rare_map[1, 2]
    assert rate == pytest.approx(0.5)


def test_cli_accepts_joint_plausibility_mode_and_rare_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_forecast_cache_alignment.py",
            "--config",
            "configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml",
            "--mode",
            "joint_plausibility_audit",
            "--discharge-checkpoint-path",
            "/tmp/discharge.ckpt",
            "--los-checkpoint-path",
            "/tmp/los.ckpt",
            "--rare-threshold",
            "0.0025",
        ],
    )

    args = parse_args()

    assert args.mode == "joint_plausibility_audit"
    assert args.rare_threshold == pytest.approx(0.0025)


def test_apply_head_overrides_replaces_only_requested_columns() -> None:
    oracle_x = torch.tensor([[1, 10, 100], [2, 20, 200], [3, 30, 300]], dtype=torch.long)
    predicted_x = torch.tensor([[9, 10, 101], [8, 20, 202], [7, 30, 303]], dtype=torch.long)

    x_cache, payload = _apply_head_overrides(
        oracle_x=oracle_x,
        predicted_x=predicted_x,
        override_x_col_idx_list=[0, 2],
        indices=np.array([0, 2]),
        base_source="predicted",
        override_source="oracle",
    )

    assert x_cache.tolist() == [[1, 10, 100], [8, 20, 202], [3, 30, 300]]
    assert payload["num_override_rows"] == 2
    assert payload["num_override_heads"] == 2
    assert payload["num_changed_rows"] == 2
    assert len(payload["per_head_summary"]) == 2
    assert payload["per_head_summary"][0]["predicted_oracle_match_rate"] == pytest.approx(0.0)


def test_cli_accepts_multi_head_ablation_mode_and_override_head(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "diagnose_forecast_cache_alignment.py",
            "--config",
            "configs/ctmp_gin_forecast_discharge_los_ce_baseline_leakage_free.yaml",
            "--mode",
            "predicted_D_predicted_LOS_oracle_head_ablation",
            "--override-head",
            "SERVICES_D,SUB1_D,FREQ_ATND_SELF_HELP_D",
        ],
    )

    args = parse_args()

    assert args.mode == "predicted_D_predicted_LOS_oracle_head_ablation"
    assert args.override_head == "SERVICES_D,SUB1_D,FREQ_ATND_SELF_HELP_D"
