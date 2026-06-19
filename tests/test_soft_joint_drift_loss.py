from __future__ import annotations

import pytest
import torch

from src.models.discharge_predictor.soft_joint_drift_loss import SoftJointDriftLoss


HEADS = [
    "FREQ_ATND_SELF_HELP_D",
    "SUB1_D",
    "FREQ2_D",
    "EMPLOY_D",
    "FREQ1_D",
    "DETNLF_D",
]
RISK_HEADS = ["FREQ_ATND_SELF_HELP_D", "SUB1_D", "FREQ2_D"]


def _make_logits(pred_classes: list[int], num_classes: int = 2, high: float = 6.0, low: float = -6.0) -> torch.Tensor:
    logits = torch.full((len(pred_classes), num_classes), low, dtype=torch.float32)
    for row, cls in enumerate(pred_classes):
        logits[row, int(cls)] = high
    return logits


def _loss_module(*, stopgrad_los: bool = True, use_ema: bool = False) -> SoftJointDriftLoss:
    return SoftJointDriftLoss(
        risk_head_set="new_dvD_top3",
        available_heads=HEADS,
        stopgrad_los=stopgrad_los,
        use_ema=use_ema,
    )


def _full_head_logits(primary_logits: torch.Tensor, primary_targets: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    d_logits = {}
    d_targets = {}
    for head_name in RISK_HEADS:
        d_logits[head_name] = primary_logits.clone()
        d_targets[head_name] = primary_targets.clone()
    return d_logits, d_targets


def test_zero_case_is_near_zero() -> None:
    los_probs = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    logits = _make_logits([0, 1, 0, 1])
    d_logits, d_targets = _full_head_logits(logits, targets)

    loss, metrics = _loss_module()(los_probs=los_probs, d_logits=d_logits, d_targets=d_targets)

    assert float(loss.item()) < 1.0e-5
    assert metrics["loss_struct"] < 1.0e-5


def test_structured_error_exceeds_zero_case() -> None:
    los_probs = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    logits = _make_logits([1, 1, 1, 1])
    d_logits, d_targets = _full_head_logits(logits, targets)

    loss, _ = _loss_module()(los_probs=los_probs, d_logits=d_logits, d_targets=d_targets)

    assert float(loss.item()) > 0.0


def test_structured_error_exceeds_unstructured_error_with_matched_accuracy_and_marginals() -> None:
    los_probs = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 0, 1, 1, 0, 0, 1, 1], dtype=torch.long)
    structured_preds = [1, 1, 1, 1, 0, 0, 1, 1]
    unstructured_preds = [1, 0, 1, 1, 1, 0, 1, 1]

    structured_logits, d_targets = _full_head_logits(_make_logits(structured_preds), targets)
    unstructured_logits, _ = _full_head_logits(_make_logits(unstructured_preds), targets)

    structured_loss, _ = _loss_module()(los_probs=los_probs, d_logits=structured_logits, d_targets=d_targets)
    unstructured_loss, _ = _loss_module()(los_probs=los_probs, d_logits=unstructured_logits, d_targets=d_targets)

    structured_acc = sum(int(p == t) for p, t in zip(structured_preds, targets.tolist())) / len(structured_preds)
    unstructured_acc = sum(int(p == t) for p, t in zip(unstructured_preds, targets.tolist())) / len(unstructured_preds)
    assert structured_acc == pytest.approx(unstructured_acc)
    assert sum(structured_preds) == sum(unstructured_preds)
    assert float(structured_loss.item()) > float(unstructured_loss.item())


def test_stopgrad_los_blocks_los_gradients() -> None:
    los_probs = torch.tensor(
        [[0.8, 0.2], [0.2, 0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    logits = _make_logits([0, 1]).requires_grad_()
    targets = torch.tensor([0, 1], dtype=torch.long)
    d_logits, d_targets = _full_head_logits(logits, targets)

    loss, _ = _loss_module(stopgrad_los=True)(los_probs=los_probs, d_logits=d_logits, d_targets=d_targets)
    loss.backward()

    assert los_probs.grad is None or torch.allclose(los_probs.grad, torch.zeros_like(los_probs.grad))
    assert logits.grad is not None
    assert float(logits.grad.abs().sum()) > 0.0


def test_non_stopgrad_los_allows_los_gradients() -> None:
    los_probs = torch.tensor(
        [[0.8, 0.2], [0.2, 0.8]],
        dtype=torch.float32,
        requires_grad=True,
    )
    logits = _make_logits([0, 1]).requires_grad_()
    targets = torch.tensor([0, 1], dtype=torch.long)
    d_logits, d_targets = _full_head_logits(logits, targets)

    loss, _ = _loss_module(stopgrad_los=False)(los_probs=los_probs, d_logits=d_logits, d_targets=d_targets)
    loss.backward()

    assert los_probs.grad is not None
    assert float(los_probs.grad.abs().sum()) > 0.0


def test_near_empty_los_bin_is_stable() -> None:
    los_probs = torch.tensor(
        [[1.0, 0.0, 0.0], [1.0, 1.0e-12, 0.0], [1.0, 0.0, 1.0e-12]],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 0], dtype=torch.long)
    logits = _make_logits([0, 1, 1])
    d_logits, d_targets = _full_head_logits(logits, targets)

    loss, _ = _loss_module()(los_probs=los_probs, d_logits=d_logits, d_targets=d_targets)

    assert torch.isfinite(loss)


def test_use_ema_is_reserved_for_future() -> None:
    with pytest.raises(ValueError, match="reserved for a future version"):
        _loss_module(use_ema=True)
