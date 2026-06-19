"""

uv run python scripts/analyze_joint_plausibility_report.py \
--diagnostic-dir runs/diagnostics/forecast_cache_alignment/distribution_diagnosis \
--heads SERVICES_D,SUB1_D,FREQ_ATND_SELF_HELP_D \
--discharge-confidence-min 0.9 \
--los-confidence-min 0.5 \
--limit 200
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.forecast_joint_plausibility_report import (
    generate_joint_plausibility_report,
)  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interpretation reports from a joint plausibility audit directory."
    )
    parser.add_argument("--diagnostic-dir", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--split",
        type=str,
        default="overall",
        choices=("train", "valid", "test", "overall"),
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="SERVICES_D,SUB1_D,FREQ_ATND_SELF_HELP_D",
        help="Comma-separated target heads for focused analysis.",
    )
    parser.add_argument("--discharge-confidence-min", type=float, default=0.9)
    parser.add_argument("--los-confidence-min", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads = [part.strip() for part in args.heads.split(",") if part.strip()]
    payload = generate_joint_plausibility_report(
        args.diagnostic_dir,
        top_k=args.top_k,
        split=args.split,
        heads=heads,
        discharge_confidence_min=args.discharge_confidence_min,
        los_confidence_min=args.los_confidence_min,
        limit=args.limit,
        script_path=Path(__file__).resolve(),
    )
    print("[JOINT PLAUSIBILITY REPORT]")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
