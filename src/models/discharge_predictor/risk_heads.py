from __future__ import annotations

from collections.abc import Sequence


RISK_HEAD_SETS: dict[str, tuple[str, ...]] = {
    "old_total_drift_top3": (
        "SERVICES_D",
        "SUB1_D",
        "FREQ_ATND_SELF_HELP_D",
    ),
    "new_dvD_top3": (
        "FREQ_ATND_SELF_HELP_D",
        "SUB1_D",
        "FREQ2_D",
    ),
    "new_robust_top3": (
        "FREQ_ATND_SELF_HELP_D",
        "SUB1_D",
        "FREQ1_D",
    ),
    "new_dvD_top6": (
        "FREQ_ATND_SELF_HELP_D",
        "SUB1_D",
        "FREQ2_D",
        "EMPLOY_D",
        "FREQ1_D",
        "DETNLF_D",
    ),
}

LEGACY_TOP3_HEADS = RISK_HEAD_SETS["old_total_drift_top3"]


def available_risk_head_sets() -> tuple[str, ...]:
    return tuple(RISK_HEAD_SETS.keys())


def get_named_risk_head_set(name: str) -> list[str]:
    key = str(name).strip()
    if key not in RISK_HEAD_SETS:
        available = ", ".join(sorted(RISK_HEAD_SETS))
        raise ValueError(
            f"Unknown named risk-head set: {name!r}. Available sets: {available}"
        )
    return list(RISK_HEAD_SETS[key])


def resolve_risk_head_selection(
    value: str | Sequence[str] | None,
    *,
    available_heads: Sequence[str],
    mode: str,
    allow_all: bool = False,
    field_name: str = "risk heads",
) -> list[str]:
    available = [str(head) for head in available_heads]
    available_set = set(available)
    resolved_mode = str(mode).strip().lower()
    if resolved_mode not in {"legacy_or_named_set", "strict_named_set"}:
        raise ValueError(f"Unsupported resolver mode: {mode}")

    if resolved_mode == "strict_named_set":
        if not isinstance(value, str):
            raise ValueError(
                f"{field_name} must be a named risk-head set string in strict mode."
            )
        resolved = get_named_risk_head_set(value)
        _validate_resolved_heads(resolved, available, available_set, field_name)
        return resolved

    if value is None:
        resolved = list(available)
        _validate_resolved_heads(resolved, available, available_set, field_name)
        return resolved

    if isinstance(value, str):
        text = value.strip()
        if allow_all and text.lower() == "all":
            return list(available)
        if text in RISK_HEAD_SETS:
            resolved = get_named_risk_head_set(text)
        elif text.lower() in {"", "all"}:
            resolved = list(available)
        else:
            resolved = [item.strip() for item in text.split(",") if item.strip()]
            if not resolved:
                resolved = list(available)
    else:
        resolved = [str(item).strip() for item in value if str(item).strip()]

    _validate_resolved_heads(resolved, available, available_set, field_name)
    return resolved


def _validate_resolved_heads(
    resolved: Sequence[str],
    available: Sequence[str],
    available_set: set[str],
    field_name: str,
) -> None:
    missing = sorted(set(resolved) - available_set)
    if missing:
        available_text = ", ".join(available)
        raise ValueError(
            f"Unknown {field_name}: {missing}. Available heads: {available_text}"
        )
