"""Extract local feature CSVs. Audio/turn data and per-call features remain local."""

import argparse
import hashlib
import json
from time import perf_counter
import pandas as pd
from behavior.config import ROOT, feature_config
from behavior.dataset import locate_audio, read_manifest, wav_duration
from behavior.features import extract_features, evidence_quality
from behavior.turns import load_turns
from .common import write_json


def build_table(source="organizer", audio_dir=None, prefixes=(None,)):
    audio_dir = locate_audio(audio_dir)
    manifest = read_manifest()
    digest = hashlib.sha256((ROOT / "manifest.csv").read_bytes())
    tables = {prefix: [] for prefix in prefixes}
    vad_meta = {}
    if source == "vad":
        from behavior.vad import MODEL_SHA256
        vad_meta = {"vad_config": json.loads((ROOT / "artifacts/vad/config.json").read_text()),
                    "vad_model_sha256": MODEL_SHA256}
    for row in manifest.itertuples():
        path = audio_dir / f"{row.anon_id}.wav"
        duration = wav_duration(path)
        if source == "organizer":
            turn_path = ROOT / "turns" / f"{row.anon_id}.json"
            turns = load_turns(turn_path)
            stability = 1.
        else:
            turn_path = ROOT / "artifacts" / "vad_turns" / f"{row.anon_id}.json"
            payload = json.loads(turn_path.read_text())
            if payload["vad_config"] != vad_meta["vad_config"] or payload["model_sha256"] != vad_meta["vad_model_sha256"]:
                raise ValueError("VAD turn cache mismatch; rerun compare_turns --extract-all")
            turns = load_turns(turn_path)
            stability = payload["stability"]
            if payload["wav_size"] != path.stat().st_size or payload["wav_mtime_ns"] != path.stat().st_mtime_ns:
                raise ValueError("Stale VAD cache: WAV changed; rerun compare_turns")
        digest.update(turn_path.read_bytes())
        for prefix in prefixes:
            start = perf_counter()
            d = min(duration, prefix) if prefix else duration
            f, _ = extract_features(turns, d)
            q, count = evidence_quality(f, stability)
            tables[prefix].append({"anon_id": row.anon_id, "split": row.split, "y": row.y,
                                   **f, "quality": q, "event_count": count,
                                   "feature_ms": (perf_counter() - start) * 1000})
    output = {}
    for prefix, rows in tables.items():
        name = source if prefix is None else f"{source}_{prefix}s"
        path = ROOT / "artifacts" / "features" / f"{name}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(path, index=False)
        write_json(path.with_suffix(".meta.json"), {"source": source, "prefix": prefix,
                   "input_digest": digest.hexdigest(), "feature_config": feature_config(), "rows": len(df),
                   "table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **vad_meta})
        output[name] = {"rows": len(df), "features": len(f), "mean_event_count": df.event_count.mean(),
                        "mean_feature_ms": df.feature_ms.mean()}
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["organizer", "vad"], default="organizer")
    parser.add_argument("--audio-dir")
    args = parser.parse_args()
    print(json.dumps(build_table(args.source, args.audio_dir), indent=2))


if __name__ == "__main__":
    main()
