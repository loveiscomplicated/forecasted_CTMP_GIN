from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

LOS_COARSE_BINS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 7),
    (8, 14),
    (15, 21),
    (22, 28),
    (29, 37),
)
LOS_COARSE_BIN_REPRESENTATIVES: tuple[int, ...] = tuple(
    int(round((lo + hi) / 2)) for lo, hi in LOS_COARSE_BINS
)
LOS_COARSE_BREAKDOWN_BINS: tuple[tuple[int, int], ...] = (
    (1, 1),
    (2, 7),
    (8, 14),
    (15, 21),
    (22, 28),
    (29, 31),
    (32, 33),
    (34, 35),
    (36, 37),
)
LOS_COARSE_BREAKDOWN_BIN_REPRESENTATIVES: tuple[int, ...] = tuple(
    int(round((lo + hi) / 2)) for lo, hi in LOS_COARSE_BREAKDOWN_BINS
)
LOS_COARSE_CLASS_NAMES: tuple[str, ...] = (
    "1_day",
    "2_7_days",
    "8_14_days",
    "15_21_days",
    "22_28_days",
    "29_plus_days",
)
LOS_COARSE_BREAKDOWN_CLASS_NAMES: tuple[str, ...] = (
    "1_day",
    "2_7_days",
    "8_14_days",
    "15_21_days",
    "22_28_days",
    "29_31_days",
    "32_33_days",
    "34_35_days",
    "36_37_days",
)


@dataclass(frozen=True)
class LosBinningMetadata:
    """Metadata describing the coarse LOS binning scheme."""

    target_mode: str = "coarse"
    raw_los_min: int = 1
    raw_los_max: int = 37
    breakdown: bool = False
    bins: tuple[tuple[int, int], ...] = LOS_COARSE_BINS


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def infer_los_coarse_breakdown_from_cfg(cfg: dict) -> bool:
    """Return whether coarse LOS should split the long-stay class into sub-bins."""
    for key in ("los_coarse_breakdown", "coarse_breakdown", "breakdown"):
        if key in cfg:
            return _as_bool(cfg[key])
    for section_key in ("los_binning", "coarse_binning"):
        section = cfg.get(section_key)
        if isinstance(section, dict) and "breakdown" in section:
            return _as_bool(section["breakdown"])
    try:
        return int(cfg.get("num_classes")) == len(LOS_COARSE_BREAKDOWN_BINS)
    except (TypeError, ValueError):
        pass
    return False


def get_los_coarse_bins(*, breakdown: bool = False) -> tuple[tuple[int, int], ...]:
    return LOS_COARSE_BREAKDOWN_BINS if breakdown else LOS_COARSE_BINS


def get_los_coarse_bin_representatives(*, breakdown: bool = False) -> tuple[int, ...]:
    return (
        LOS_COARSE_BREAKDOWN_BIN_REPRESENTATIVES
        if breakdown
        else LOS_COARSE_BIN_REPRESENTATIVES
    )


def get_los_coarse_class_names(*, breakdown: bool = False) -> tuple[str, ...]:
    return LOS_COARSE_BREAKDOWN_CLASS_NAMES if breakdown else LOS_COARSE_CLASS_NAMES


def get_los_coarse_class_labels(*, breakdown: bool = False) -> tuple[str, ...]:
    return tuple(
        str(lo) if lo == hi else f"{lo}-{hi}"
        for lo, hi in get_los_coarse_bins(breakdown=breakdown)
    )


def get_los_coarse_num_classes(*, breakdown: bool = False) -> int:
    return len(get_los_coarse_bins(breakdown=breakdown))


def map_los_to_coarse_bin(los: int, breakdown: bool = False) -> int:
    if breakdown:
        return _los_map_breakdown(los)
    return _los_map(los)


def _los_map(los: int) -> int:
    """Map a LOS value to one of six coarse duration bins.

    Args:
        los: Raw LOS codebook value in 1..37.

    Returns:
        Coarse class index in [0, 5].
    """
    raw_los = int(los)
    if raw_los < 1 or raw_los > 37:
        raise ValueError(f"LOS must be a raw codebook value in 1..37, got {los}.")
    if raw_los == 1:
        return 0
    if 2 <= raw_los <= 7:
        return 1
    if 8 <= raw_los <= 14:
        return 2
    if 15 <= raw_los <= 21:
        return 3
    if 22 <= raw_los <= 28:
        return 4
    if 29 <= raw_los <= 37:
        return 5
    raise ValueError(f"LOS must be in 1..37 after decoding, got {raw_los}.")


def _los_map_breakdown(los: int) -> int:
    """Map a LOS value to one of six coarse duration bins.

    Args:
        los: Raw LOS codebook value in 1..37.

    Returns:
        Coarse class index in [0, 5].
    """
    raw_los = int(los)
    if raw_los < 1 or raw_los > 37:
        raise ValueError(f"LOS must be a raw codebook value in 1..37, got {los}.")
    if raw_los == 1:
        return 0
    if 2 <= raw_los <= 7:
        return 1
    if 8 <= raw_los <= 14:
        return 2
    if 15 <= raw_los <= 21:
        return 3
    if 22 <= raw_los <= 28:
        return 4
    if 29 <= raw_los <= 31:
        return 5
    if 32 <= raw_los <= 33:
        return 6
    if 34 <= raw_los <= 35:
        return 7
    if 36 <= raw_los <= 37:
        return 8
    raise ValueError(f"LOS must be in 1..37 after decoding, got {raw_los}.")


def map_los_array_to_coarse_bins(
    values, *, assume_encoded: bool = False, breakdown: bool = False
) -> np.ndarray | torch.Tensor:
    """Vectorized LOS-to-coarse-bin mapping for NumPy, pandas, or PyTorch inputs."""

    def _map_value(value: int) -> int:
        raw_value = int(value) + 1 if assume_encoded else int(value)
        return map_los_to_coarse_bin(raw_value, breakdown=breakdown)

    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
        mapped = np.vectorize(_map_value, otypes=[np.int64])(arr)
        return torch.as_tensor(mapped, dtype=torch.long, device=values.device)

    if isinstance(values, pd.Series):
        arr = values.to_numpy()
        return pd.Series(
            np.vectorize(_map_value, otypes=[np.int64])(arr), index=values.index
        )

    arr = np.asarray(values)
    return np.vectorize(_map_value, otypes=[np.int64])(arr)


def map_coarse_bin_to_raw_los(
    class_idx: int, *, breakdown: bool | None = None
) -> int:
    """Map a coarse LOS class index to its representative raw LOS token."""
    idx = int(class_idx)
    if breakdown is None:
        breakdown = idx >= len(LOS_COARSE_BIN_REPRESENTATIVES)
    representatives = get_los_coarse_bin_representatives(breakdown=breakdown)
    if idx < 0 or idx >= len(representatives):
        upper = len(representatives) - 1
        raise ValueError(f"Coarse LOS class must be in 0..{upper}, got {class_idx}.")
    return int(representatives[idx])


def map_coarse_array_to_raw_los(
    values, *, breakdown: bool | None = None
) -> np.ndarray | torch.Tensor:
    """Vectorized coarse-class to representative raw LOS mapping."""

    if isinstance(values, torch.Tensor):
        arr = values.detach().cpu().numpy()
        resolved_breakdown = (
            bool(arr.size and np.max(arr) >= len(LOS_COARSE_BIN_REPRESENTATIVES))
            if breakdown is None
            else bool(breakdown)
        )

        def _map_value(value: int) -> int:
            return map_coarse_bin_to_raw_los(int(value), breakdown=resolved_breakdown)

        mapped = np.vectorize(_map_value, otypes=[np.int64])(arr)
        return torch.as_tensor(mapped, dtype=torch.long, device=values.device)

    arr = np.asarray(values)
    resolved_breakdown = (
        bool(arr.size and np.max(arr) >= len(LOS_COARSE_BIN_REPRESENTATIVES))
        if breakdown is None
        else bool(breakdown)
    )

    def _map_value(value: int) -> int:
        return map_coarse_bin_to_raw_los(int(value), breakdown=resolved_breakdown)

    return np.vectorize(_map_value, otypes=[np.int64])(arr)


def expand_coarse_distribution_to_raw_los(
    values, *, breakdown: bool | None = None
) -> np.ndarray | torch.Tensor:
    """Spread coarse-bin probabilities uniformly over raw LOS tokens 1..37."""
    if isinstance(values, torch.Tensor):
        if values.ndim != 2:
            raise ValueError(
                f"Expected coarse distribution shape [B, 6] or [B, 9], got {tuple(values.shape)}."
            )
        if breakdown is None:
            if values.shape[1] == len(LOS_COARSE_BINS):
                breakdown = False
            elif values.shape[1] == len(LOS_COARSE_BREAKDOWN_BINS):
                breakdown = True
            else:
                raise ValueError(
                    f"Expected coarse distribution shape [B, 6] or [B, 9], got {tuple(values.shape)}."
                )
        bins = get_los_coarse_bins(breakdown=bool(breakdown))
        num_classes = len(bins)
        if values.shape[1] != num_classes:
            raise ValueError(
                f"Expected coarse distribution shape [B, {num_classes}], got {tuple(values.shape)}."
            )
        expanded = torch.zeros(
            (values.shape[0], 37), dtype=values.dtype, device=values.device
        )
        for class_idx, (lo, hi) in enumerate(bins):
            width = hi - lo + 1
            expanded[:, lo - 1 : hi] = values[:, class_idx : class_idx + 1] / float(
                width
            )
        return expanded

    arr = np.asarray(values)
    if arr.ndim != 2:
        raise ValueError(
            f"Expected coarse distribution shape [B, 6] or [B, 9], got {arr.shape}."
        )
    if breakdown is None:
        if arr.shape[1] == len(LOS_COARSE_BINS):
            breakdown = False
        elif arr.shape[1] == len(LOS_COARSE_BREAKDOWN_BINS):
            breakdown = True
        else:
            raise ValueError(
                f"Expected coarse distribution shape [B, 6] or [B, 9], got {arr.shape}."
            )
    bins = get_los_coarse_bins(breakdown=bool(breakdown))
    num_classes = len(bins)
    if arr.shape[1] != num_classes:
        raise ValueError(
            f"Expected coarse distribution shape [B, {num_classes}], got {arr.shape}."
        )
    expanded = np.zeros((arr.shape[0], 37), dtype=arr.dtype)
    for class_idx, (lo, hi) in enumerate(bins):
        width = hi - lo + 1
        expanded[:, lo - 1 : hi] = arr[:, class_idx : class_idx + 1] / float(width)
    return expanded


def infer_los_target_from_cfg(cfg: dict) -> str:
    """Return the configured LOS target mode, defaulting to fine."""
    return str(cfg.get("los_target_mode", cfg.get("target_mode", "fine"))).lower()


def los_binning_metadata_dict(*, breakdown: bool = False) -> dict[str, object]:
    """Serialize the coarse LOS binning scheme for experiment artifacts."""
    bins = get_los_coarse_bins(breakdown=breakdown)
    representatives = get_los_coarse_bin_representatives(breakdown=breakdown)
    class_names = get_los_coarse_class_names(breakdown=breakdown)
    return {
        "target_mode": "coarse",
        "raw_los_min": 1,
        "raw_los_max": 37,
        "breakdown": bool(breakdown),
        "los_coarse_breakdown": bool(breakdown),
        "num_classes": len(bins),
        "los_bins": [list(pair) for pair in bins],
        "coarse_raw_los_representatives": list(representatives),
        "coarse_class_names": list(class_names),
    }
