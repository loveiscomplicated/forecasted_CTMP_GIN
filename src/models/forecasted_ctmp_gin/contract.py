from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


CANONICAL_JOINT_FORECAST_HEADS = (
    "SERVICES_D",
    "EMPLOY_D",
    "LIVARAG_D",
    "ARRESTS_D",
    "DETNLF_D",
    "SUB1_D",
    "SUB2_D",
    "SUB3_D",
    "FREQ1_D",
    "FREQ2_D",
    "FREQ3_D",
    "FREQ_ATND_SELF_HELP_D",
)


@dataclass(frozen=True)
class SoftDischargeHeadContract:
    name: str
    target_col_idx: int
    num_classes: int


@dataclass(frozen=True)
class SoftDischargeContract:
    head_names: tuple[str, ...]
    heads: tuple[SoftDischargeHeadContract, ...]

    @property
    def target_to_col_idx(self) -> dict[str, int]:
        return {head.name: int(head.target_col_idx) for head in self.heads}


def resolve_joint_forecast_contract(col_info, target_names: list[str]) -> SoftDischargeContract:
    col_list, col_dims, _ad_col_index, _dis_col_index = col_info
    col_name_to_idx = {str(name): int(idx) for idx, name in enumerate(col_list)}
    missing = [name for name in target_names if name not in col_name_to_idx]
    if missing:
        raise ValueError(
            "Joint forecast target names must be present in the dataset columns. "
            f"missing={missing}"
        )
    heads = tuple(
        SoftDischargeHeadContract(
            name=str(name),
            target_col_idx=int(col_name_to_idx[str(name)]),
            num_classes=int(col_dims[col_name_to_idx[str(name)]]),
        )
        for name in target_names
    )
    return SoftDischargeContract(
        head_names=tuple(str(name) for name in target_names),
        heads=heads,
    )


def build_soft_discharge_payload(
    contract: SoftDischargeContract,
    *,
    d_probs: dict[str, torch.Tensor],
    d_logits: dict[str, torch.Tensor] | None = None,
    device: torch.device | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    payload: dict[str, dict[str, torch.Tensor]] = {}
    for head in contract.heads:
        probs = d_probs[head.name]
        if probs.ndim != 2:
            raise ValueError(
                f"{head.name}: probs must be rank-2, got shape={tuple(probs.shape)}"
            )
        if probs.shape[1] != int(head.num_classes):
            raise ValueError(
                f"{head.name}: probs width={probs.shape[1]} expected={head.num_classes}"
            )
        logits = None if d_logits is None else d_logits.get(head.name)
        target_device = probs.device if device is None else device
        payload[head.name] = {
            "probs": probs.to(device=target_device),
            "target_col_idx": torch.tensor(
                int(head.target_col_idx), dtype=torch.long, device=target_device
            ),
            "num_classes": torch.tensor(
                int(head.num_classes), dtype=torch.long, device=target_device
            ),
            "class_to_embedding_idx": torch.arange(
                int(head.num_classes), dtype=torch.long, device=target_device
            ),
        }
        if logits is not None:
            payload[head.name]["logits"] = logits.to(device=target_device)
    return payload


def assert_soft_discharge_contract_matches_cached_metadata(
    contract: SoftDischargeContract,
    cached_soft_discharge: dict[str, Any],
) -> None:
    cached_head_names = tuple(str(name) for name in cached_soft_discharge["head_names"])
    cached_soft_head_names = tuple(
        str(name) for name in cached_soft_discharge["soft_head_names"]
    )
    if cached_head_names != contract.head_names or cached_soft_head_names != contract.head_names:
        raise ValueError(
            "Stage2 soft-discharge head order does not match cached joint forecast contract."
        )

    metadata = cached_soft_discharge.get("metadata", {})
    cached_target_to_col_idx = {
        str(name): int(idx)
        for name, idx in dict(metadata.get("target_to_col_idx", {})).items()
    }
    if cached_target_to_col_idx != contract.target_to_col_idx:
        raise ValueError(
            "Stage2 soft-discharge target column mapping does not match cached joint forecast contract."
        )

    for head in contract.heads:
        cached_head = cached_soft_discharge["heads"][head.name]
        cached_target_col_idx = int(cached_head["target_col_idx"].item())
        cached_num_classes = int(cached_head["num_classes"].item())
        if cached_target_col_idx != int(head.target_col_idx):
            raise ValueError(
                f"{head.name}: cached target_col_idx={cached_target_col_idx} expected={head.target_col_idx}"
            )
        if cached_num_classes != int(head.num_classes):
            raise ValueError(
                f"{head.name}: cached num_classes={cached_num_classes} expected={head.num_classes}"
            )
