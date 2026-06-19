from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.models.discharge_predictor.joint_generative_predictor import (
    JointGenerativeLoss,
    JointGenerativePredictor,
    JointGenerativePredictorOutput,
    clamp_logstd,
    diagonal_gaussian_kl,
    kl_beta_for_epoch,
)
from src.trainers.run_joint_consistent_predictor import (
    _evaluate_generative_prior,
    _export_cache,
)


def _build_model() -> JointGenerativePredictor:
    return JointGenerativePredictor(
        ad_col_dims=[3, 4],
        target_col_names=["A_D", "B_D"],
        target_col_dims=[2, 3],
        los_num_classes=3,
        input_encoding="onehot",
        hidden_dim=8,
        latent_dim=4,
        los_context_dim=5,
        num_layers=1,
        dropout=0.0,
        target_embedding_dim=3,
    )


def _targets() -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    x = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    d_targets = {
        "A_D": torch.tensor([0, 1], dtype=torch.long),
        "B_D": torch.tensor([1, 2], dtype=torch.long),
    }
    los_targets = torch.tensor([0, 2], dtype=torch.long)
    return x, d_targets, los_targets


def test_joint_generative_training_forward_returns_prior_posterior_and_finite_kl() -> None:
    model = _build_model()
    model.train()
    x, d_targets, los_targets = _targets()

    output = model(x, d_targets=d_targets, los_targets=los_targets)

    assert output.kl is not None
    assert output.kl.shape == (2,)
    assert torch.isfinite(output.kl).all()
    assert output.posterior_d_logits is not None
    assert output.posterior_los_logits is not None
    assert output.prior_los_logits.shape == (2, 3)
    assert output.posterior_los_logits.shape == (2, 3)
    assert output.prior_d_logits["A_D"].shape == (2, 2)
    assert output.prior_d_logits["B_D"].shape == (2, 3)


def test_joint_generative_eval_forward_uses_prior_without_targets() -> None:
    model = _build_model()
    model.eval()
    x, _d_targets, _los_targets = _targets()

    with torch.no_grad():
        output = model(x)

    assert output.kl is None
    assert output.mu_q is None
    assert output.posterior_d_logits is None
    assert output.prior_los_logits.shape == (2, 3)
    assert output.d_logits["A_D"].shape == (2, 2)


def test_logstd_is_clamped_before_kl() -> None:
    raw = torch.tensor([[-100.0, 100.0]], dtype=torch.float32)
    clamped = clamp_logstd(raw)
    assert float(clamped.min()) == pytest.approx(-5.0)
    assert float(clamped.max()) == pytest.approx(2.0)

    mu = torch.zeros((1, 2), dtype=torch.float32)
    kl = diagonal_gaussian_kl(mu, raw, mu, -raw)
    assert torch.isfinite(kl).all()


def test_kl_beta_annealing_starts_conservative_and_reaches_max() -> None:
    assert kl_beta_for_epoch(1, beta_start=0.0, beta_max=0.001, anneal_epochs=10) == pytest.approx(0.0)
    assert kl_beta_for_epoch(6, beta_start=0.0, beta_max=0.001, anneal_epochs=10) == pytest.approx(0.0005)
    assert kl_beta_for_epoch(11, beta_start=0.0, beta_max=0.001, anneal_epochs=10) == pytest.approx(0.001)


def test_prior_recon_weight_changes_total_loss_by_prior_reconstruction() -> None:
    model = _build_model()
    model.train()
    x, d_targets, los_targets = _targets()
    output = model(x, d_targets=d_targets, los_targets=los_targets)

    loss_no_prior, metrics_no_prior = JointGenerativeLoss(
        lambda_los=1.0,
        prior_recon_weight=0.0,
    )(output, d_targets=d_targets, los_targets=los_targets, beta_kl=0.0)
    loss_with_prior, _ = JointGenerativeLoss(
        lambda_los=1.0,
        prior_recon_weight=1.0,
    )(output, d_targets=d_targets, los_targets=los_targets, beta_kl=0.0)

    expected_delta = metrics_no_prior["recon_p_D"] + metrics_no_prior["recon_p_LOS"]
    assert float((loss_with_prior - loss_no_prior).detach()) == pytest.approx(expected_delta)


class _PriorOnlySpy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_kwargs: list[dict[str, object]] = []

    def forward(self, x: torch.Tensor, **kwargs) -> JointGenerativePredictorOutput:
        self.forward_kwargs.append(dict(kwargs))
        if kwargs:
            raise AssertionError("cache/prior evaluation must not pass targets into the model")
        batch = x.shape[0]
        prior_d_logits = {
            "A_D": torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32)[:batch]
        }
        prior_los_logits = torch.tensor(
            [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]],
            dtype=torch.float32,
        )[:batch]
        prior_d_probs = {name: torch.softmax(logits, dim=1) for name, logits in prior_d_logits.items()}
        prior_los_probs = torch.softmax(prior_los_logits, dim=1)
        return JointGenerativePredictorOutput(
            prior_d_logits=prior_d_logits,
            prior_los_logits=prior_los_logits,
            prior_d_probs=prior_d_probs,
            prior_los_probs=prior_los_probs,
            posterior_d_logits=None,
            posterior_los_logits=None,
            posterior_d_probs=None,
            posterior_los_probs=None,
            mu_p=torch.zeros((batch, 2), dtype=torch.float32),
            logstd_p=torch.zeros((batch, 2), dtype=torch.float32),
            mu_q=None,
            logstd_q=None,
            kl=None,
            shared_hidden=torch.zeros((batch, 2), dtype=torch.float32),
        )


def test_cache_style_prior_evaluation_does_not_pass_targets_to_model() -> None:
    model = _PriorOnlySpy()
    criterion = JointGenerativeLoss()
    loader = [
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([10, 11], dtype=torch.long),
        )
    ]
    dataset = SimpleNamespace(target_col_names=["A_D"], los_target_mode="coarse")

    metrics, payload = _evaluate_generative_prior(
        model,  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        criterion,
        torch.device("cpu"),
        dataset,  # type: ignore[arg-type]
        beta_kl=0.0,
        posterior_diagnostics=False,
    )

    assert model.forward_kwargs == [{}]
    assert metrics["discharge_mean_accuracy"] == pytest.approx(1.0)
    assert metrics["los_acc"] == pytest.approx(1.0)
    assert payload["row_idx"].tolist() == [10, 11]


class _PriorBeatsPosterior(torch.nn.Module):
    def forward(self, x: torch.Tensor, **kwargs) -> JointGenerativePredictorOutput:
        batch = x.shape[0]
        prior_d_logits = {
            "A_D": torch.tensor([[10.0, -10.0], [-10.0, 10.0]], dtype=torch.float32)[:batch]
        }
        posterior_d_logits = {
            "A_D": torch.tensor([[-10.0, 10.0], [10.0, -10.0]], dtype=torch.float32)[:batch]
        }
        prior_los_logits = torch.tensor(
            [[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]],
            dtype=torch.float32,
        )[:batch]
        posterior_los_logits = torch.tensor(
            [[-10.0, 10.0, -10.0], [10.0, -10.0, -10.0]],
            dtype=torch.float32,
        )[:batch]
        return JointGenerativePredictorOutput(
            prior_d_logits=prior_d_logits,
            prior_los_logits=prior_los_logits,
            prior_d_probs={name: torch.softmax(logits, dim=1) for name, logits in prior_d_logits.items()},
            prior_los_probs=torch.softmax(prior_los_logits, dim=1),
            posterior_d_logits=posterior_d_logits,
            posterior_los_logits=posterior_los_logits,
            posterior_d_probs={name: torch.softmax(logits, dim=1) for name, logits in posterior_d_logits.items()},
            posterior_los_probs=torch.softmax(posterior_los_logits, dim=1),
            mu_p=torch.zeros((batch, 2), dtype=torch.float32),
            logstd_p=torch.zeros((batch, 2), dtype=torch.float32),
            mu_q=torch.zeros((batch, 2), dtype=torch.float32),
            logstd_q=torch.zeros((batch, 2), dtype=torch.float32),
            kl=torch.zeros((batch,), dtype=torch.float32),
            shared_hidden=torch.zeros((batch, 2), dtype=torch.float32),
        )


def test_validation_metrics_for_selection_use_prior_path_even_with_posterior_diagnostics() -> None:
    loader = [
        (
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([[0], [1]], dtype=torch.long),
            torch.tensor([0, 1], dtype=torch.long),
            torch.tensor([1, 2], dtype=torch.long),
            torch.tensor([10, 11], dtype=torch.long),
        )
    ]
    dataset = SimpleNamespace(target_col_names=["A_D"], los_target_mode="coarse")

    metrics, _payload = _evaluate_generative_prior(
        _PriorBeatsPosterior(),  # type: ignore[arg-type]
        loader,  # type: ignore[arg-type]
        JointGenerativeLoss(),
        torch.device("cpu"),
        dataset,  # type: ignore[arg-type]
        beta_kl=0.0,
        posterior_diagnostics=True,
    )

    assert metrics["discharge_mean_accuracy"] == pytest.approx(1.0)
    assert metrics["los_acc"] == pytest.approx(1.0)


def test_joint_generative_cache_export_preserves_row_count_and_order(tmp_path) -> None:
    payload = {
        "d_logits_np": {
            "A_D": torch.tensor([[3.0, 1.0], [0.5, 2.5]], dtype=torch.float32).numpy()
        },
        "d_targets_np": {
            "A_D": torch.tensor([0, 1], dtype=torch.long).numpy()
        },
        "los_logits": torch.tensor([[3.0, 1.0, 0.0], [0.2, 3.0, 0.1]], dtype=torch.float32),
        "los_probs": torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]], dtype=torch.float32),
        "los_targets": torch.tensor([0, 1], dtype=torch.long),
        "los_raw": torch.tensor([1, 2], dtype=torch.long),
        "row_idx": torch.tensor([42, 7], dtype=torch.long),
    }
    dataset = SimpleNamespace(
        target_col_names=["A_D"],
        schema_metadata={
            "target_col_names": ["A_D"],
            "target_col_dims": [2],
            "los_target_mode": "coarse",
            "los_num_classes": 3,
        },
        los_target_mode="coarse",
    )
    cfg = {
        "joint_predictor": {
            "predictor_type": "joint_generative",
            "prior_recon_weight": 0.5,
            "beta_kl_start": 0.0,
            "beta_kl_max": 0.001,
            "kl_anneal_epochs": 10,
        },
        "model": {"params": {"latent_dim": 4, "z_sampling_at_eval": False, "num_eval_samples": 1}},
        "train": {"seed": 1, "fold": 0},
    }

    path = _export_cache(
        output_dir=str(tmp_path),
        split_name="test",
        payload=payload,
        dataset=dataset,  # type: ignore[arg-type]
        cfg=cfg,
        caseid_lookup=None,
    )

    cache = torch.load(path, map_location="cpu")
    assert cache["row_idx"].tolist() == [42, 7]
    assert cache["final_los_probs"].shape[0] == 2
    assert cache["final_d_pred"]["A_D"].shape[0] == 2
    assert cache["metadata"]["predictor_type"] == "joint_generative"
