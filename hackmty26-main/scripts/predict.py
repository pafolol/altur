"""Offline single-WAV inference. Debug output contains timing features, not audio."""

import argparse
import json
from pathlib import Path
from behavior.inference import BehaviorDetector


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    print(json.dumps(BehaviorDetector(model_path=args.model).predict(args.wav.read_bytes(), args.debug), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
