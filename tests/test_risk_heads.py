from __future__ import annotations

import pytest

from src.models.discharge_predictor.risk_heads import (
    RISK_HEAD_SETS,
    resolve_risk_head_selection,
)


AVAILABLE_HEADS = [
    "SERVICES_D",
    "SUB1_D",
    "FREQ_ATND_SELF_HELP_D",
    "FREQ1_D",
    "FREQ2_D",
    "EMPLOY_D",
    "DETNLF_D",
]


def test_legacy_mode_preserves_explicit_comma_string_behavior() -> None:
    resolved = resolve_risk_head_selection(
        "SERVICES_D,SUB1_D",
        available_heads=AVAILABLE_HEADS,
        mode="legacy_or_named_set",
        field_name="joint_heads",
    )
    assert resolved == ["SERVICES_D", "SUB1_D"]


def test_legacy_mode_expands_named_set_exact_match() -> None:
    resolved = resolve_risk_head_selection(
        "old_total_drift_top3",
        available_heads=AVAILABLE_HEADS,
        mode="legacy_or_named_set",
        field_name="joint_heads",
    )
    assert resolved == list(RISK_HEAD_SETS["old_total_drift_top3"])


def test_strict_mode_requires_known_named_set() -> None:
    with pytest.raises(ValueError, match="Unknown named risk-head set"):
        resolve_risk_head_selection(
            "SERVICES_D,SUB1_D",
            available_heads=AVAILABLE_HEADS,
            mode="strict_named_set",
            field_name="head_set_name",
        )


def test_unknown_final_head_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="Unknown joint_heads"):
        resolve_risk_head_selection(
            ["SERVICES_D", "MISSING_D"],
            available_heads=AVAILABLE_HEADS,
            mode="legacy_or_named_set",
            field_name="joint_heads",
        )


def test_new_dvd_sets_exclude_services_d() -> None:
    for key in ("new_dvD_top3", "new_robust_top3", "new_dvD_top6"):
        assert "SERVICES_D" not in RISK_HEAD_SETS[key]
