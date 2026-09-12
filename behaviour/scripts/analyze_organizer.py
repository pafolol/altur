"""Initial organizer ablations/error review before investing in audio extraction."""

import json
import numpy as np
import pandas as pd
from behavior.config import ROOT
from behavior.metrics import bootstrap_auc_difference
from behavior.turns import load_turns, normalize_turns
from .common import write_json


def main():
    model_dir = ROOT / "artifacts" / "models" / "organizer"
    pred = pd.read_csv(model_dir / "predictions.csv")
    pred = pred[pred.split == "val"]
    features = pd.read_csv(ROOT / "artifacts" / "features" / "organizer.csv")
    val = pred.merge(features.drop(columns=["split", "y"]), on="anon_id", validate="one_to_one")
    summary = {"consistency_minus_reaction": bootstrap_auc_difference(val.y, val.E_consistency, val.D_reaction),
               "consistency_minus_shortcut": bootstrap_auc_difference(val.y, val.E_consistency, val.B_shortcut)}
    private = ROOT / "reports" / "private"
    private.mkdir(parents=True, exist_ok=True)
    errors = val[(val.E_consistency >= .5) != (val.y == 1)].copy()
    errors["type"] = np.where(errors.y == 0, "false_positive", "false_negative")
    errors.to_csv(private / "organizer_errors.csv", index=False)
    timelines = []
    for i, row in enumerate(errors.itertuples()):
        turns = normalize_turns(load_turns(ROOT / "turns" / f"{row.anon_id}.json"), row.duration_s)
        timelines.append({"example": f"ORG-ERROR-{i + 1}", "type": row.type,
                          "turns": [{"channel": t.channel, "start": t.start, "end": t.end} for t in turns]})
    write_json(private / "organizer_error_timelines.json", timelines)
    summary["error_counts"] = errors.type.value_counts().to_dict()
    summary["error_event_medians"] = errors[["interruption_count", "barge_in_count", "response_latency_count", "duration_s"]].median().to_dict()
    write_json(ROOT / "reports" / "metrics" / "organizer_analysis.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
