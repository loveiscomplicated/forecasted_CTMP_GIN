from __future__ import annotations

from src.trainers.run_los_prediction import run_los_prediction


def run_los_ordinal_prediction(cfg: dict, root: str) -> dict:
    """Backward-compatible wrapper for the legacy LOS trainer entrypoint."""
    return run_los_prediction(cfg, root)

