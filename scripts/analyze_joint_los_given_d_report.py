from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.joint_los_given_d_report import (  # noqa: E402
    generate_joint_los_given_d_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate interpretation reports from a joint_stats_test directory."
    )
    parser.add_argument("--diagnostic-dir", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--heads",
        type=str,
        default="",
        help="Comma-separated head names for focused analysis. Defaults to the top-ranked heads.",
    )
    parser.add_argument("--limit", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    heads = [part.strip() for part in args.heads.split(",") if part.strip()] or None
    payload = generate_joint_los_given_d_report(
        args.diagnostic_dir,
        top_k=args.top_k,
        heads=heads,
        limit=args.limit,
        script_path=Path(__file__).resolve(),
    )
    print("[JOINT LOS GIVEN D REPORT]")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
