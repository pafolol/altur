"""Read-only source-data audit; writes only derived LOCAL reports and aggregate results."""

from collections import Counter
import hashlib
import json
import numpy as np
import pandas as pd
from behavior.audio import load_wav
from behavior.config import ROOT
from behavior.dataset import locate_audio, read_manifest
from behavior.turns import load_turns
from .common import write_json


def main():
    manifest = read_manifest()
    audio_dir = locate_audio()
    records, hashes = [], []
    endpoint_excess = []
    for row in manifest.itertuples():
        payload = (audio_dir / f"{row.anon_id}.wav").read_bytes()
        hashes.append(hashlib.sha256(payload).hexdigest())
        audio = load_wav(payload)
        x = audio.samples.astype(np.float64)
        r = {"anon_id": row.anon_id, "split": row.split, "y": row.y,
             "manifest_duration_s": row.duration_s, "wav_duration_s": audio.duration,
             "duration_discrepancy_s": audio.duration - row.duration_s,
             "sample_rate": 8000, "channels": 2, "bits_per_sample": 16,
             "file_bytes": len(payload), "channel_correlation": np.corrcoef(x.T)[0, 1]}
        for c, label in enumerate(("caller", "agent")):
            r[f"{label}_rms"] = float(np.sqrt(np.mean(x[:, c] ** 2)))
            r[f"{label}_clipped_fraction"] = float(np.mean(np.abs(x[:, c]) >= .999))
            r[f"{label}_digital_zero_fraction"] = float(np.mean(x[:, c] == 0))
        turns = load_turns(ROOT / "turns" / f"{row.anon_id}.json")
        endpoint_excess.extend(max(0., t.end - audio.duration) for t in turns)
        records.append(r)
    audit = pd.DataFrame(records)
    private = ROOT / "reports" / "private"
    private.mkdir(parents=True, exist_ok=True)
    audit.to_csv(private / "audio_audit.csv", index=False)
    feature_table = pd.read_csv(ROOT / "artifacts" / "features" / "organizer.csv")
    combined = audit.merge(feature_table, on=["anon_id", "split", "y"], validate="one_to_one")
    summaries = []
    for split in ("train", "val"):
        data = combined[combined.split == split]
        for col in data.select_dtypes("number").columns:
            if col == "y" or col.endswith("_ms"):
                continue
            summaries.append({"split": split, "feature": col,
                              "human_median": data.loc[data.y == 0, col].median(),
                              "synthetic_median": data.loc[data.y == 1, col].median(),
                              "spearman_label": data[[col, "y"]].corr(method="spearman").iloc[0, 1],
                              "missing_fraction": data[col].isna().mean(), "unique_values": data[col].nunique()})
    pd.DataFrame(summaries).to_csv(ROOT / "reports" / "metrics" / "shortcut_audit.csv", index=False)
    summary = {"split_counts": [{"split": s, "human": int(((manifest.split == s) & (manifest.y == 0)).sum()),
                                "synthetic": int(((manifest.split == s) & (manifest.y == 1)).sum())} for s in ("train", "val")],
               "audio_hours": audit.wav_duration_s.sum() / 3600,
               "duration_s": audit.wav_duration_s.describe().to_dict(),
               "duration_discrepancy_s": audit.duration_discrepancy_s.describe().to_dict(),
               "turn_endpoints_exceeding_wav": sum(e > .001 for e in endpoint_excess),
               "max_endpoint_excess_s": max(endpoint_excess),
               "exact_duplicate_audio_files": sum(n - 1 for n in Counter(hashes).values() if n > 1),
               "speaker_group_ids_available": False, "format": "all stereo PCM16 8kHz"}
    write_json(ROOT / "reports" / "metrics" / "dataset_audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
