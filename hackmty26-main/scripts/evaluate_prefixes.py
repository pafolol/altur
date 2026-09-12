"""Actual cropped WAV -> same frozen detector, threshold, calibration, quality logic."""

import json
from time import perf_counter
import numpy as np
import pandas as pd
from behavior.audio import load_wav, encode_wav
from behavior.config import ROOT
from behavior.dataset import locate_audio, read_manifest
from behavior.inference import BehaviorDetector
from behavior.metrics import evaluate
from .common import write_json


def main():
    detector = BehaviorDetector()
    val = read_manifest().query("split == 'val'")
    directory = locate_audio()
    records = []
    feature_rows = []
    start = perf_counter()
    for i, row in enumerate(val.itertuples()):
        raw = (directory / f"{row.anon_id}.wav").read_bytes()
        audio = load_wav(raw)
        for prefix in (15, 30, 60, "full"):
            # Re-run VAD on the actual prefix; no future audio in the last partial frame.
            payload = raw if prefix == "full" else encode_wav(audio.samples[:prefix * 8000])
            result = detector.predict(payload, debug=True)
            records.append({"anon_id": row.anon_id, "y": row.y, "prefix": str(prefix),
                            **{k: result[k] for k in ("synthetic_probability", "quality_score", "behavior_confidence", "event_count")},
                            **{f"{k}_ms": v for k, v in result["diagnostics"]["timing_ms"].items()},
                            "insufficient_evidence": result["diagnostics"]["insufficient_evidence"]})
            feature_rows.append({"anon_id": row.anon_id, "y": row.y, "prefix": str(prefix), **result["features"]})
        if (i + 1) % 20 == 0:
            print(f"Prefix evaluation: {i + 1}/{len(val)} calls; {perf_counter() - start:.1f}s", flush=True)
    df = pd.DataFrame(records)
    df.to_csv(ROOT / "reports/private/prefix_predictions.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(ROOT / "reports/private/prefix_features.csv", index=False)
    summary = {}
    for prefix in ("15", "30", "60", "full"):
        subset = df[df.prefix == prefix]
        summary[prefix] = {"metrics": evaluate(subset.y, subset.synthetic_probability),
                           "mean_event_count": subset.event_count.mean(), "mean_quality": subset.quality_score.mean(),
                           "mean_behavior_confidence": subset.behavior_confidence.mean(),
                           "insufficient_evidence_calls": int(subset.insufficient_evidence.sum()),
                           "latency_mean_ms": subset.total_ms.mean(),
                           "latency_p50_ms": subset.total_ms.median(),
                           "latency_p95_ms": float(np.quantile(subset.total_ms, .95)),
                           "mean_component_ms": {key: subset[key].mean() for key in ("decode_ms", "vad_ms", "features_ms", "classifier_ms")}}
    write_json(ROOT / "reports/metrics/prefixes.json", summary)
    print(json.dumps({k: {"auc": v["metrics"]["roc_auc"], "ba": v["metrics"]["balanced_accuracy"],
                          "cm": v["metrics"]["confusion_matrix"], "events": v["mean_event_count"],
                          "quality": v["mean_quality"], "p50_ms": v["latency_p50_ms"]} for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
