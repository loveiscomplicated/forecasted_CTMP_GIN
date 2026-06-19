from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis.joint_drift_decomposition import (
    cramers_v,
    decompose_head_drift,
    js_divergence,
)


def _df(true_d, pred_d, true_los, pred_los) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "true_H": np.asarray(true_d, dtype=int),
            "pred_H": np.asarray(pred_d, dtype=int),
            "true_los_bin": np.asarray(true_los, dtype=int),
            "pred_los_bin": np.asarray(pred_los, dtype=int),
        }
    )


def test_js_divergence_uses_base2_and_is_symmetric() -> None:
    assert js_divergence(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(1.0)
    lhs = js_divergence(np.array([4.0, 1.0]), np.array([1.0, 4.0]))
    rhs = js_divergence(np.array([1.0, 4.0]), np.array([4.0, 1.0]))
    assert lhs == pytest.approx(rhs)


def test_cramers_v_handles_empty_and_degenerate_tables() -> None:
    assert cramers_v(np.zeros((2, 3))) == 0.0
    assert cramers_v(np.array([[1.0, 2.0, 3.0]])) == 0.0


def test_perfect_d_prediction_has_zero_d_attributable_drift() -> None:
    true_los = np.tile(np.arange(3), 40)
    true_d = true_los % 2
    pred_los = (true_los + 1) % 3
    result = decompose_head_drift(_df(true_d, true_d, true_los, pred_los), "H", 3)
    assert result["acc"] == pytest.approx(1.0)
    assert result["dV_D"] == pytest.approx(0.0, abs=1.0e-12)
    assert result["js_D"] == pytest.approx(0.0, abs=1.0e-12)


def test_los_misgrouping_only_changes_los_component_not_d_component() -> None:
    rng = np.random.default_rng(5)
    true_los = np.tile(np.arange(3), 60)
    true_d = true_los.copy()
    pred_los = rng.permutation(true_los)
    result = decompose_head_drift(_df(true_d, true_d, true_los, pred_los), "H", 3)
    assert abs(result["dV_LOS"]) > 0.05
    assert result["dV_D"] == pytest.approx(0.0, abs=1.0e-12)


def test_los_structured_d_error_has_larger_dv_d_than_unstructured_noise() -> None:
    rng = np.random.default_rng(7)
    pred_los = np.tile(np.arange(3), 300)
    true_los = pred_los.copy()
    true_d = rng.integers(0, 2, size=pred_los.size)
    structured_pred_d = (pred_los > 0).astype(int)
    noisy_pred_d = rng.integers(0, 2, size=pred_los.size)

    structured = decompose_head_drift(_df(true_d, structured_pred_d, true_los, pred_los), "H", 3)
    noisy = decompose_head_drift(_df(true_d, noisy_pred_d, true_los, pred_los), "H", 3)

    assert structured["dV_D"] > 0.5
    assert structured["dV_D"] > noisy["dV_D"] + 0.3
    assert structured["js_D"] > noisy["js_D"]


def test_unstructured_d_noise_is_not_equivalent_to_one_minus_accuracy() -> None:
    rng = np.random.default_rng(11)
    pred_los = np.tile(np.arange(4), 250)
    true_los = pred_los.copy()
    true_d = (pred_los % 2).astype(int)
    noisy_pred_d = rng.integers(0, 2, size=pred_los.size)
    result = decompose_head_drift(_df(true_d, noisy_pred_d, true_los, pred_los), "H", 4)

    assert result["acc"] < 0.6
    assert result["dV_D"] < 0.0
    assert result["dV_D"] != pytest.approx(1.0 - result["acc"])
