"""CLI entry point for offline Markov regime analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Optional

import yaml

# Ensure project root is on PYTHONPATH when running as a standalone script
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.markov_regime import RegimeConfig, run_and_save_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline Markov regime analysis using historical NT8 data.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/train_config_full.yaml"),
        help="Path to training config to infer instrument/timeframes.",
    )
    parser.add_argument(
        "--instrument",
        type=str,
        default=None,
        help="Override instrument symbol (default inferred from config).",
    )
    parser.add_argument(
        "--timeframes",
        type=int,
        nargs="+",
        default=None,
        help="Override timeframes in minutes (default inferred from config).",
    )
    parser.add_argument(
        "--regimes",
        type=int,
        default=3,
        help="Number of regimes to cluster (default: 3).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/markov_regime_report.json"),
        help="Path to write JSON report.",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Optional start date (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Optional end date (YYYY-MM-DD).",
    )
    return parser.parse_args()


def infer_from_training_config(config_path: Path) -> tuple[Optional[str], Optional[Iterable[int]]]:
    if not config_path.exists():
        return None, None

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    instrument = data.get("environment", {}).get("instrument")
    timeframes = data.get("environment", {}).get("timeframes")

    return instrument, timeframes


def main() -> None:
    args = parse_args()

    inferred_instrument, inferred_timeframes = infer_from_training_config(args.config)

    instrument = args.instrument or inferred_instrument or "ES"
    timeframes = args.timeframes or inferred_timeframes or [1, 5, 15]

    regime_config = RegimeConfig(
        instrument=instrument,
        timeframes=timeframes,
        start_date=args.start_date,
        end_date=args.end_date,
        num_regimes=args.regimes,
    )

    result = run_and_save_report(args.output, regime_config)

    print(f"✅ Markov regime analysis complete. Report saved to {args.output}")
    print("\nTransition Matrix:")
    print(result.transition_matrix)
    print("\nRegime Summary:")
    print(result.regime_summary)
    print("\nStationary Distribution:")
    print(result.stationary_distribution)


if __name__ == "__main__":
    main()


