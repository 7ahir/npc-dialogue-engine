#!/usr/bin/env python3
"""Generate synthetic NPC training data.

Convenience wrapper around src/training/data_generation.py.

Usage:
    python scripts/generate_training_data.py
    python scripts/generate_training_data.py --output data/processed/train.jsonl --examples 4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.data_generation import generate_training_data
from src.utils.logging_config import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic NPC training data")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/train.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=4,
        help="Examples per character×scenario pair",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    setup_logging()
    total = generate_training_data(args.output, args.examples, args.seed)
    print(f"\nGenerated {total} training examples → {args.output}")


if __name__ == "__main__":
    main()
