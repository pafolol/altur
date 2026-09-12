"""Evaluate a frozen deployed JSON model; never fits/tunes anything."""

import argparse
import json
import pandas as pd
from behavior.config import ROOT
from behavior.dataset import load_feature_table
from behavior.inference import score_features
from behavior.metrics import evaluate
from .common import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "artifacts/models/behavior.json"))
    args = parser.parse_args()
    artifact = json.loads(open(args.model).read())
    df = load_feature_table("vad")
    comparison = pd.read_csv(ROOT / "reports/private/vad_comparison.csv")
    df = df.merge(comparison[["anon_id", "stability"]], on="anon_id", validate="one_to_one")
    scores = [score_features(artifact, row, row["stability"]) for row in df.to_dict("records")]
    p = [r["synthetic_probability"] for r in scores]
    results = {s: evaluate(df.loc[df.split == s, "y"], [v for v, split in zip(p, df.split) if split == s], artifact["threshold"])
               for s in ("train", "val")}
    write_json(ROOT / "reports/metrics/frozen_evaluation.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
