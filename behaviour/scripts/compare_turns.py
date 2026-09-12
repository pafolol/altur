"""Label-blind VAD smoke, selection and organizer-reference comparison."""

import argparse
from dataclasses import asdict
import json
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from behavior.audio import load_wav
from behavior.config import ROOT, SEED, VADConfig
from behavior.dataset import read_manifest, locate_audio
from behavior.features import extract_features
from behavior.turns import load_turns, normalize_turns
from behavior.vad import SileroVAD, MODEL_SHA256, probabilities_to_turns, segmentation_stability, speech_iou
from .common import write_json

CONFIG_PATH = ROOT / "artifacts" / "vad" / "config.json"
CANDIDATES = [VADConfig(threshold=.35, min_speech_s=.10, min_silence_s=.10, pad_s=0.),
              VADConfig(), VADConfig(threshold=.65, min_speech_s=.24, min_silence_s=.24)]


def get_probabilities(vad, path, force=False):
    cache = ROOT / "artifacts" / "vad_probabilities" / (path.stem + ".npz")
    size, mtime = path.stat().st_size, path.stat().st_mtime_ns
    if cache.exists() and not force:
        with np.load(cache, allow_pickle=False) as data:
            if int(data["size"]) == size and int(data["mtime"]) == mtime and str(data["model_sha256"]) == MODEL_SHA256:
                return data["probabilities"], float(data["duration"]), float(data["vad_seconds"])
    start = perf_counter()
    audio = load_wav(path.read_bytes())
    probabilities = vad.probabilities(audio)
    seconds = perf_counter() - start
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, probabilities=probabilities, duration=audio.duration,
                        vad_seconds=seconds, size=size, mtime=mtime, model_sha256=MODEL_SHA256)
    return probabilities, audio.duration, seconds


def compare(reference, generated):
    result = {}
    for ch in (0, 1):
        a, b = ([t for t in ts if t.channel == ch] for ts in (reference, generated))
        iou = np.zeros((len(a), len(b)))
        for i, x in enumerate(a):
            for j, y in enumerate(b):
                overlap = max(0., min(x.end, y.end) - max(x.start, y.start))
                iou[i, j] = overlap / (x.duration + y.duration - overlap)
        pairs = list(zip(*linear_sum_assignment(-iou))) if iou.size else []
        pairs = [(i, j) for i, j in pairs if iou[i, j] >= .1]
        result.update({f"ch{ch}_speech_iou": speech_iou(reference, generated, ch),
                       f"ch{ch}_reference_segments": len(a), f"ch{ch}_generated_segments": len(b),
                       f"ch{ch}_matched_segments": len(pairs),
                       f"ch{ch}_onset_mae_s": float(np.mean([abs(a[i].start - b[j].start) for i, j in pairs])) if pairs else np.nan,
                       f"ch{ch}_offset_mae_s": float(np.mean([abs(a[i].end - b[j].end) for i, j in pairs])) if pairs else np.nan,
                       f"ch{ch}_matched_segment_iou": float(np.mean([iou[i, j] for i, j in pairs])) if pairs else np.nan})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--extract-all", action="store_true")
    args = parser.parse_args()
    manifest = read_manifest()
    # Discard labels before ALL segmentation comparisons and subset decisions.
    data = manifest[["anon_id", "split"]].copy()
    audio_dir = locate_audio()
    start = perf_counter()
    vad = SileroVAD()
    load_s = perf_counter() - start
    if args.smoke:
        sample = data[data.split == "train"].sample(args.smoke, random_state=SEED)
        times = [get_probabilities(vad, audio_dir / f"{r.anon_id}.wav", force=True)[1:] for r in sample.itertuples()]
        duration, seconds = map(sum, zip(*times))
        summary = {"calls": len(times), "audio_s": duration, "processing_s": seconds,
                   "realtime_factor": seconds / duration, "model_load_s": load_s,
                   "estimated_full_dataset_minutes": seconds / duration * 14.51 * 60,
                   "cpu_threads": 1, "channels_batched": 2}
        write_json(ROOT / "reports" / "metrics" / "vad_smoke.json", summary)
        print(json.dumps(summary, indent=2))
    if args.tune:
        subset = data[data.split == "train"].sample(24, random_state=SEED)
        results = []
        for candidate in CANDIDATES:
            scores = []
            for row in subset.itertuples():
                p, duration, _ = get_probabilities(vad, audio_dir / f"{row.anon_id}.wav")
                ref = normalize_turns(load_turns(ROOT / "turns" / f"{row.anon_id}.json"), duration)
                turns = probabilities_to_turns(p, duration, candidate)
                scores.append(np.mean([speech_iou(ref, turns, ch) for ch in (0, 1)]))
            results.append({"config": asdict(candidate), "mean_channel_iou": float(np.mean(scores))})
        winner = max(results, key=lambda r: r["mean_channel_iou"])
        write_json(CONFIG_PATH, winner["config"])
        write_json(ROOT / "reports" / "metrics" / "vad_selection.json", {
            "selection": "max mean channel IoU on 24 train calls; no class labels used",
            "sample_seed": SEED, "candidates": results, "chosen": winner["config"]})
        print(json.dumps({"vad_selection": results, "chosen": winner["config"]}, indent=2))
    if args.extract_all:
        config = VADConfig(**json.loads(CONFIG_PATH.read_text()))
        records = []
        start = perf_counter()
        for i, row in enumerate(data.itertuples()):
            path = audio_dir / f"{row.anon_id}.wav"
            p, duration, seconds = get_probabilities(vad, path)
            turns = probabilities_to_turns(p, duration, config)
            ref = normalize_turns(load_turns(ROOT / "turns" / f"{row.anon_id}.json"), duration)
            stability = segmentation_stability(p, duration, config)
            write_json(ROOT / "artifacts" / "vad_turns" / f"{row.anon_id}.json", {
                "turns": [{"channel": t.channel, "start": t.start, "end": t.end} for t in turns],
                "duration": duration, "stability": stability, "vad_config": asdict(config),
                "model_sha256": MODEL_SHA256, "wav_size": path.stat().st_size, "wav_mtime_ns": path.stat().st_mtime_ns})
            a, _ = extract_features(ref, duration)
            b, _ = extract_features(turns, duration)
            r = {"anon_id": row.anon_id, "split": row.split, "vad_s": seconds, "duration_s": duration,
                 "stability": stability, **compare(ref, turns)}
            for key in ("interruption_count", "barge_in_count", "response_latency_count"):
                r[f"{key}_delta"] = b[key] - a[key]
                r[f"{key}_absolute_delta"] = abs(b[key] - a[key])
            records.append(r)
            if (i + 1) % 50 == 0:
                print(f"Local VAD: {i + 1}/{len(data)} calls processed; elapsed {perf_counter() - start:.1f}s", flush=True)
        df = pd.DataFrame(records)
        private = ROOT / "reports" / "private"
        private.mkdir(parents=True, exist_ok=True)
        df.to_csv(private / "vad_comparison.csv", index=False)
        summary = df.groupby("split").mean(numeric_only=True).to_dict("index")
        summary["processing"] = {"summed_uncached_vad_s": df.vad_s.sum(), "this_run_s": perf_counter() - start,
                                 "realtime_factor": df.vad_s.sum() / df.duration_s.sum(), "calls": len(df)}
        write_json(ROOT / "reports" / "metrics" / "vad_comparison.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
