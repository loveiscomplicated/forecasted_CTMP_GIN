from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.discharge_predictor.joint_consistent_predictor import (
    FixedOneHotBatchEncoder,
)
from src.models.entity_embedding import EntityEmbeddingBatch3


LOGSTD_MIN = -5.0
LOGSTD_MAX = 2.0


@dataclass
class JointGenerativePredictorOutput:
    prior_d_logits: Dict[str, torch.Tensor]
    prior_los_logits: torch.Tensor
    prior_d_probs: Dict[str, torch.Tensor]
    prior_los_probs: torch.Tensor
    posterior_d_logits: Dict[str, torch.Tensor] | None
    posterior_los_logits: torch.Tensor | None
    posterior_d_probs: Dict[str, torch.Tensor] | None
    posterior_los_probs: torch.Tensor | None
    mu_p: torch.Tensor
    logstd_p: torch.Tensor
    mu_q: torch.Tensor | None
    logstd_q: torch.Tensor | None
    kl: torch.Tensor | None
    shared_hidden: torch.Tensor

    @property
    def d_logits(self) -> Dict[str, torch.Tensor]:
        return self.prior_d_logits

    @property
    def los_logits(self) -> torch.Tensor:
        return self.prior_los_logits


def clamp_logstd(logstd: torch.Tensor) -> torch.Tensor:
    return torch.clamp(logstd, min=LOGSTD_MIN, max=LOGSTD_MAX)


def diagonal_gaussian_kl(
    mu_q: torch.Tensor,
    logstd_q: torch.Tensor,
    mu_p: torch.Tensor,
    logstd_p: torch.Tensor,
) -> torch.Tensor:
    logstd_q = clamp_logstd(logstd_q)
    logstd_p = clamp_logstd(logstd_p)
    var_q = torch.exp(2.0 * logstd_q)
    var_p = torch.exp(2.0 * logstd_p)
    return 0.5 * torch.sum(
        2.0 * (logstd_p - logstd_q)
        + (var_q + (mu_q - mu_p).pow(2)) / var_p.clamp_min(1.0e-12)
        - 1.0,
        dim=1,
    )


def reparameterize(mu: torch.Tensor, logstd: torch.Tensor) -> torch.Tensor:
    logstd = clamp_logstd(logstd)
    eps = torch.randn_like(mu)
    return mu + eps * torch.exp(logstd)


def kl_beta_for_epoch(
    epoch: int,
    *,
    beta_start: float,
    beta_max: float,
    anneal_epochs: int,
) -> float:
    if int(anneal_epochs) <= 0:
        return float(beta_max)
    progress = min(max(float(epoch - 1), 0.0) / float(anneal_epochs), 1.0)
    return float(beta_start) + progress * (float(beta_max) - float(beta_start))


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    *,
    use_batch_norm: bool = True,
) -> nn.Sequential:
    if int(num_layers) < 1:
        raise ValueError("num_layers must be >= 1")
    layers: List[nn.Module] = []
    in_dim = int(input_dim)
    for _ in range(int(num_layers)):
        layers.append(nn.Linear(in_dim, int(hidden_dim)))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(int(hidden_dim)))
        layers.extend([nn.ReLU(), nn.Dropout(float(dropout))])
        in_dim = int(hidden_dim)
    return nn.Sequential(*layers)


class AdmissionGraphEncoder(nn.Module):
    """Encode admission-only categorical inputs into a dense admission state."""

    def __init__(
        self,
        ad_col_dims: Sequence[int],
        *,
        embedding_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.2,
        input_encoding: str = "onehot",
    ) -> None:
        super().__init__()
        self.ad_col_dims = [int(dim) for dim in ad_col_dims]
        self.input_encoding = str(input_encoding).lower()
        if self.input_encoding not in {"onehot", "embedding"}:
            raise ValueError(
                f"Unsupported input_encoding: {input_encoding}. Expected one of ['onehot', 'embedding']."
            )
        if self.input_encoding == "embedding":
            self.admission_encoder: nn.Module = EntityEmbeddingBatch3(
                col_dims=self.ad_col_dims,
                embedding_dim=int(embedding_dim),
            )
            input_dim = len(self.ad_col_dims) * int(embedding_dim)
        else:
            self.admission_encoder = FixedOneHotBatchEncoder(self.ad_col_dims)
            input_dim = int(sum(self.ad_col_dims))
        self.output_dim = int(hidden_dim)
        self.encoder = _build_mlp(
            input_dim=input_dim,
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.admission_encoder(x.long())
        if self.input_encoding == "embedding":
            encoded = encoded.reshape(encoded.shape[0], -1)
        return self.encoder(encoded)


class FuturePosteriorEncoder(nn.Module):
    """Training-only q(z | x_ad, D_true, LOS_true)."""

    def __init__(
        self,
        target_col_names: Sequence[str],
        target_col_dims: Sequence[int],
        *,
        los_num_classes: int,
        hidden_dim: int,
        latent_dim: int,
        target_embedding_dim: int = 32,
        posterior_hidden_dim: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.target_col_names = list(target_col_names)
        self.target_col_dims = [int(dim) for dim in target_col_dims]
        self.los_num_classes = int(los_num_classes)
        self.d_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(int(dim), int(target_embedding_dim))
                for name, dim in zip(self.target_col_names, self.target_col_dims)
            }
        )
        self.los_embedding = nn.Embedding(
            self.los_num_classes,
            int(target_embedding_dim),
        )
        in_dim = int(hidden_dim) + (len(self.target_col_names) + 1) * int(target_embedding_dim)
        mid_dim = int(posterior_hidden_dim or hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(mid_dim, 2 * int(latent_dim)),
        )

    def forward(
        self,
        h_ad: torch.Tensor,
        d_targets: Dict[str, torch.Tensor],
        los_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        parts = [h_ad]
        for name in self.target_col_names:
            if name not in d_targets:
                raise ValueError(f"Missing D target for posterior encoder head {name!r}")
            parts.append(self.d_embeddings[name](d_targets[name].long()))
        parts.append(self.los_embedding(los_targets.long()))
        params = self.net(torch.cat(parts, dim=1))
        mu_q, logstd_q = params.chunk(2, dim=1)
        return mu_q, clamp_logstd(logstd_q)


class FuturePrior(nn.Module):
    """Inference prior p(z | x_ad)."""

    def __init__(
        self,
        *,
        hidden_dim: int,
        latent_dim: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 2 * int(latent_dim)),
        )

    def forward(self, h_ad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        params = self.net(h_ad)
        mu_p, logstd_p = params.chunk(2, dim=1)
        return mu_p, clamp_logstd(logstd_p)


class JointFutureDecoder(nn.Module):
    """Decode LOS first, then D heads conditioned on the same z and LOS distribution."""

    def __init__(
        self,
        target_col_names: Sequence[str],
        target_col_dims: Sequence[int],
        *,
        hidden_dim: int,
        latent_dim: int,
        los_num_classes: int,
        los_context_dim: int = 32,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.target_col_names = list(target_col_names)
        self.target_col_dims = [int(dim) for dim in target_col_dims]
        self.los_head = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(latent_dim), int(hidden_dim)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(los_num_classes)),
        )
        self.los_context = nn.Sequential(
            nn.Linear(int(los_num_classes), int(los_context_dim)),
            nn.ReLU(),
        )
        d_in_dim = int(hidden_dim) + int(latent_dim) + int(los_context_dim)
        self.d_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(d_in_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), int(dim)),
                )
                for name, dim in zip(self.target_col_names, self.target_col_dims)
            }
        )

    def forward(
        self,
        h_ad: torch.Tensor,
        z_future: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        hz = torch.cat([h_ad, z_future], dim=1)
        los_logits = self.los_head(hz)
        los_prob = F.softmax(los_logits, dim=1)
        los_ctx = self.los_context(los_prob)
        d_input = torch.cat([h_ad, z_future, los_ctx], dim=1)
        d_logits = {name: head(d_input) for name, head in self.d_heads.items()}
        return d_logits, los_logits


class JointGenerativePredictor(nn.Module):
    """Conditional future VAE for joint LOS and discharge-side D prediction."""

    def __init__(
        self,
        ad_col_dims: Sequence[int],
        target_col_names: Sequence[str],
        target_col_dims: Sequence[int],
        *,
        los_num_classes: int,
        embedding_dim: int = 32,
        hidden_dim: int = 512,
        latent_dim: int = 64,
        los_context_dim: int = 32,
        num_layers: int = 4,
        dropout: float = 0.2,
        input_encoding: str = "onehot",
        target_embedding_dim: int = 32,
        posterior_hidden_dim: int | None = None,
        z_sampling_at_eval: bool = False,
        num_eval_samples: int = 1,
        **kwargs,
    ) -> None:
        super().__init__()
        self.target_col_names = list(target_col_names)
        self.target_col_dims = [int(dim) for dim in target_col_dims]
        self.los_num_classes = int(los_num_classes)
        self.latent_dim = int(latent_dim)
        self.z_sampling_at_eval = bool(z_sampling_at_eval)
        self.num_eval_samples = max(int(num_eval_samples), 1)
        self.admission_encoder = AdmissionGraphEncoder(
            ad_col_dims,
            embedding_dim=int(embedding_dim),
            hidden_dim=int(hidden_dim),
            num_layers=int(num_layers),
            dropout=float(dropout),
            input_encoding=str(input_encoding),
        )
        self.prior = FuturePrior(
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            dropout=float(dropout),
        )
        self.posterior = FuturePosteriorEncoder(
            self.target_col_names,
            self.target_col_dims,
            los_num_classes=self.los_num_classes,
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            target_embedding_dim=int(target_embedding_dim),
            posterior_hidden_dim=posterior_hidden_dim,
            dropout=float(dropout),
        )
        self.decoder = JointFutureDecoder(
            self.target_col_names,
            self.target_col_dims,
            hidden_dim=int(hidden_dim),
            latent_dim=int(latent_dim),
            los_num_classes=self.los_num_classes,
            los_context_dim=int(los_context_dim),
            dropout=float(dropout),
        )

    def _decode_prior_eval(
        self,
        h_ad: torch.Tensor,
        mu_p: torch.Tensor,
        logstd_p: torch.Tensor,
    ) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
        if not self.z_sampling_at_eval:
            return self.decoder(h_ad, mu_p)

        d_accum: Dict[str, list[torch.Tensor]] = {
            name: [] for name in self.target_col_names
        }
        los_accum: list[torch.Tensor] = []
        for _ in range(self.num_eval_samples):
            z = reparameterize(mu_p, logstd_p)
            d_logits, los_logits = self.decoder(h_ad, z)
            for name, logits in d_logits.items():
                d_accum[name].append(logits)
            los_accum.append(los_logits)
        d_mean = {
            name: torch.stack(chunks, dim=0).mean(dim=0)
            for name, chunks in d_accum.items()
        }
        los_mean = torch.stack(los_accum, dim=0).mean(dim=0)
        return d_mean, los_mean

    def forward_prior(
        self,
        x: torch.Tensor,
        *,
        sample: bool | None = None,
    ) -> JointGenerativePredictorOutput:
        h_ad = self.admission_encoder(x)
        mu_p, logstd_p = self.prior(h_ad)
        if sample is None:
            sample = bool(self.training)

        if sample:
            z_p = reparameterize(mu_p, logstd_p)
            prior_d_logits, prior_los_logits = self.decoder(h_ad, z_p)
        elif self.training:
            prior_d_logits, prior_los_logits = self.decoder(h_ad, mu_p)
        else:
            prior_d_logits, prior_los_logits = self._decode_prior_eval(
                h_ad, mu_p, logstd_p
            )

        prior_d_probs = {
            name: F.softmax(logits, dim=1) for name, logits in prior_d_logits.items()
        }
        prior_los_probs = F.softmax(prior_los_logits, dim=1)
        return JointGenerativePredictorOutput(
            prior_d_logits=prior_d_logits,
            prior_los_logits=prior_los_logits,
            prior_d_probs=prior_d_probs,
            prior_los_probs=prior_los_probs,
            posterior_d_logits=None,
            posterior_los_logits=None,
            posterior_d_probs=None,
            posterior_los_probs=None,
            mu_p=mu_p,
            logstd_p=logstd_p,
            mu_q=None,
            logstd_q=None,
            kl=None,
            shared_hidden=h_ad,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        d_targets: Dict[str, torch.Tensor] | None = None,
        los_targets: torch.Tensor | None = None,
    ) -> JointGenerativePredictorOutput:
        prior_output = self.forward_prior(x)
        h_ad = prior_output.shared_hidden
        mu_p = prior_output.mu_p
        logstd_p = prior_output.logstd_p
        prior_d_logits = prior_output.prior_d_logits
        prior_los_logits = prior_output.prior_los_logits

        posterior_d_logits = None
        posterior_los_logits = None
        posterior_d_probs = None
        posterior_los_probs = None
        mu_q = None
        logstd_q = None
        kl = None

        if d_targets is not None or los_targets is not None:
            if d_targets is None or los_targets is None:
                raise ValueError("Both d_targets and los_targets are required for posterior training.")
            mu_q, logstd_q = self.posterior(h_ad, d_targets, los_targets)
            z_q = reparameterize(mu_q, logstd_q) if self.training else mu_q
            posterior_d_logits, posterior_los_logits = self.decoder(h_ad, z_q)
            posterior_d_probs = {
                name: F.softmax(logits, dim=1)
                for name, logits in posterior_d_logits.items()
            }
            posterior_los_probs = F.softmax(posterior_los_logits, dim=1)
            kl = diagonal_gaussian_kl(mu_q, logstd_q, mu_p, logstd_p)

        return JointGenerativePredictorOutput(
            prior_d_logits=prior_d_logits,
            prior_los_logits=prior_los_logits,
            prior_d_probs=prior_output.prior_d_probs,
            prior_los_probs=prior_output.prior_los_probs,
            posterior_d_logits=posterior_d_logits,
            posterior_los_logits=posterior_los_logits,
            posterior_d_probs=posterior_d_probs,
            posterior_los_probs=posterior_los_probs,
            mu_p=mu_p,
            logstd_p=logstd_p,
            mu_q=mu_q,
            logstd_q=logstd_q,
            kl=kl,
            shared_hidden=h_ad,
        )


class JointGenerativeLoss(nn.Module):
    def __init__(self, *, lambda_los: float = 1.0, prior_recon_weight: float = 0.5) -> None:
        super().__init__()
        self.lambda_los = float(lambda_los)
        self.prior_recon_weight = float(prior_recon_weight)
        self.ce = nn.CrossEntropyLoss()

    def reconstruction_terms(
        self,
        d_logits: Dict[str, torch.Tensor],
        los_logits: torch.Tensor,
        *,
        d_targets: Dict[str, torch.Tensor],
        los_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon_d = los_logits.new_zeros(())
        for name, logits in d_logits.items():
            recon_d = recon_d + self.ce(logits, d_targets[name].long())
        recon_los = self.lambda_los * self.ce(los_logits, los_targets.long())
        return recon_d, recon_los

    def forward(
        self,
        output: JointGenerativePredictorOutput,
        *,
        d_targets: Dict[str, torch.Tensor],
        los_targets: torch.Tensor,
        beta_kl: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if output.posterior_d_logits is None or output.posterior_los_logits is None or output.kl is None:
            raise ValueError("JointGenerativeLoss requires posterior outputs and KL.")
        recon_q_d, recon_q_los = self.reconstruction_terms(
            output.posterior_d_logits,
            output.posterior_los_logits,
            d_targets=d_targets,
            los_targets=los_targets,
        )
        recon_p_d, recon_p_los = self.reconstruction_terms(
            output.prior_d_logits,
            output.prior_los_logits,
            d_targets=d_targets,
            los_targets=los_targets,
        )
        kl_mean = output.kl.mean()
        if not torch.isfinite(kl_mean):
            raise FloatingPointError("Non-finite KL in JointGenerativeLoss")
        total = (
            recon_q_d
            + recon_q_los
            + self.prior_recon_weight * (recon_p_d + recon_p_los)
            + float(beta_kl) * kl_mean
        )
        metrics = {
            "recon_q_D": float(recon_q_d.detach().cpu()),
            "recon_q_LOS": float(recon_q_los.detach().cpu()),
            "recon_p_D": float(recon_p_d.detach().cpu()),
            "recon_p_LOS": float(recon_p_los.detach().cpu()),
            "KL": float(kl_mean.detach().cpu()),
            "beta_kl": float(beta_kl),
            "total_loss": float(total.detach().cpu()),
        }
        return total, metrics
