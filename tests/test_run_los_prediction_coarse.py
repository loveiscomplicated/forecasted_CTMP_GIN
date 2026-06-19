import numpy as np

from src.trainers.run_los_prediction import (
    _build_coarse_label_counts,
    _coarse_metrics,
    _majority_baseline_predictions,
    _stratified_baseline_predictions,
)


def test_majority_baseline_predictions_use_train_mode() -> None:
    train_y = np.array([1, 1, 1, 2, 2, 5], dtype=np.int64)
    preds = _majority_baseline_predictions(train_y, size=4)

    assert preds.tolist() == [1, 1, 1, 1]


def test_stratified_baseline_predictions_are_deterministic() -> None:
    train_y = np.array([0, 0, 1, 1, 1, 5], dtype=np.int64)
    preds_1 = _stratified_baseline_predictions(train_y, size=12, seed=7, num_classes=6)
    preds_2 = _stratified_baseline_predictions(train_y, size=12, seed=7, num_classes=6)

    assert preds_1.tolist() == preds_2.tolist()


def test_build_coarse_label_counts_returns_all_classes() -> None:
    y_true = np.array([0, 0, 2, 5], dtype=np.int64)
    counts = _build_coarse_label_counts(y_true, num_classes=6)

    assert counts == {
        "class_0": 2,
        "class_1": 0,
        "class_2": 1,
        "class_3": 0,
        "class_4": 0,
        "class_5": 1,
    }


def test_coarse_metrics_exposes_long_stay_fields() -> None:
    y_true = np.array([5, 5, 0, 1], dtype=np.int64)
    y_pred = np.array([5, 0, 0, 1], dtype=np.int64)
    metrics = _coarse_metrics(y_true, y_pred, num_classes=6)

    assert "precision_5" in metrics
    assert "recall_5" in metrics
    assert "f1_5" in metrics
    assert metrics["recall_5"] == 0.5
