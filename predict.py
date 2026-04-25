"""Standalone prediction script for Fitaro.

Usage (hardcoded example):
    python predict.py

Usage (command-line arguments):
    python predict.py \
        --height 1.75 --weight 82 --age 30 \
        --chest 42 --length 29 --sleeve 25 --shoulder 18.5 \
        --fit Regular

Requires that main.py has already been run to produce the saved model and
preprocessor in outputs/models/.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.predictor import FitaroPredictor, InputValidationError


# ---------------------------------------------------------------------------
# Default example — matches a typical user who would run this script directly.
# ---------------------------------------------------------------------------

DEFAULT_INPUT = {
    "Height_m": 1.72,
    "Weight_kg": 75,
    "Age": 28,
    "Chest_in": 40,
    "Length_in": 28,
    "Sleeve_in": 24,
    "ShoulderWidth_in": 17.5,
    "FitPreference": "Regular",
}


def parse_args() -> argparse.Namespace:
    """Parse optional command-line measurement overrides."""
    parser = argparse.ArgumentParser(
        description="Fitaro — Garment Size Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--height", type=float, help="Height in metres (e.g. 1.75)")
    parser.add_argument("--weight", type=float, help="Weight in kg (e.g. 82)")
    parser.add_argument("--age", type=int, help="Age in years (e.g. 30)")
    parser.add_argument("--chest", type=float, help="Chest circumference in inches")
    parser.add_argument("--length", type=float, help="Body length in inches")
    parser.add_argument("--sleeve", type=float, help="Sleeve length in inches")
    parser.add_argument("--shoulder", type=float, help="Shoulder width in inches")
    parser.add_argument(
        "--fit",
        choices=["Regular", "Slimfit", "Oversize"],
        help="Fit preference",
    )
    return parser.parse_args()


def build_input_from_args(args: argparse.Namespace) -> dict:
    """Merge command-line arguments over the default example input."""
    raw = DEFAULT_INPUT.copy()
    if args.height is not None:
        raw["Height_m"] = args.height
    if args.weight is not None:
        raw["Weight_kg"] = args.weight
    if args.age is not None:
        raw["Age"] = args.age
    if args.chest is not None:
        raw["Chest_in"] = args.chest
    if args.length is not None:
        raw["Length_in"] = args.length
    if args.sleeve is not None:
        raw["Sleeve_in"] = args.sleeve
    if args.shoulder is not None:
        raw["ShoulderWidth_in"] = args.shoulder
    if args.fit is not None:
        raw["FitPreference"] = args.fit
    return raw


def main() -> None:
    """Entry point for the standalone prediction script."""
    args = parse_args()
    raw_input = build_input_from_args(args)

    print("\nFITARO — Size Recommendation")
    print("=" * 50)
    print("Input measurements:")
    for k, v in raw_input.items():
        print(f"  {k}: {v}")
    print()

    try:
        predictor = FitaroPredictor()
        result = predictor.predict(raw_input)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}")
        print("Please run 'python main.py' first to train and save the model.")
        sys.exit(1)
    except InputValidationError as exc:
        print(f"VALIDATION ERROR: {exc}")
        sys.exit(1)

    print(result["justification"])
    print()
    print("Full probability breakdown:")
    for size, prob in result["probabilities"].items():
        bar = "#" * int(prob * 30)
        print(f"  {size:4s}  {bar:<30s} {prob * 100:.1f}%")


if __name__ == "__main__":
    main()
