"""Reproduce the staged local pipeline, with a measured gate before full VAD."""

import argparse
import json
import os
import subprocess
import sys
from behavior.config import ROOT


def run(module, *args):
    print(f"Running {module} {' '.join(args)}", flush=True)
    subprocess.run([sys.executable, "-m", module, *args], cwd=ROOT, check=True,
                   env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download-vad", action="store_true", help="Explicit setup download of public Silero weights if needed")
    parser.add_argument("--max-vad-minutes", type=float, default=20.)
    args = parser.parse_args()
    run("pytest", "-q", "tests/test_turn_logic.py", "tests/test_events.py", "tests/test_features.py")
    run("scripts.extract_features", "--source", "organizer")
    run("scripts.train_behavior", "--source", "organizer")
    run("scripts.inspect_data")
    run("scripts.analyze_organizer")
    if args.download_vad:
        run("scripts.download_vad")
    run("scripts.compare_turns", "--smoke", "3")
    smoke = json.loads((ROOT / "reports/metrics/vad_smoke.json").read_text())
    if smoke["estimated_full_dataset_minutes"] > args.max_vad_minutes:
        raise SystemExit("Full VAD exceeds the local compute budget. Review smoke results and request approval before any remote compute/upload.")
    run("scripts.compare_turns", "--tune", "--extract-all")
    run("scripts.extract_features", "--source", "vad")
    run("scripts.train_behavior", "--source", "vad")
    run("scripts.finalize_model")  # preserve original supported/count candidate and comparison
    run("scripts.run_ablations")
    run("scripts.finalize_model", "--exclude-counts")
    run("scripts.run_ablations")
    run("scripts.evaluate_behavior")
    run("scripts.evaluate_prefixes")
    run("scripts.robustness")
    run("scripts.benchmark_latency", "--calls", "20")
    run("scripts.smoke_service")
    run("scripts.make_figures")
    run("scripts.write_report")
    run("pytest", "-q")
    run("scripts.verify_deliverables")


if __name__ == "__main__":
    main()
