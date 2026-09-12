"""Fixed robustness probes, no augmentation training or validation retuning."""

from dataclasses import replace
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfilt
from behavior.audio import load_wav, encode_wav
from behavior.config import ROOT, SEED, VADConfig
from behavior.dataset import read_manifest, locate_audio
from behavior.features import extract_features
from behavior.inference import BehaviorDetector, score_features
from behavior.metrics import evaluate
from behavior.turns import load_turns, Turn, normalize_turns
from behavior.vad import probabilities_to_turns, segmentation_stability
from .common import write_json


def main():
    detector = BehaviorDetector()
    artifact = detector.artifact
    config = VADConfig(**artifact["metadata"]["vad_config"])
    val = read_manifest().query("split == 'val'")
    original = pd.read_csv(ROOT / "reports/private/final_predictions.csv").query("split == 'val'").set_index("anon_id")
    results = []
    rng = np.random.default_rng(SEED)
    for row in val.itertuples():
        with np.load(ROOT / "artifacts/vad_probabilities" / f"{row.anon_id}.npz", allow_pickle=False) as cache:
            p, duration = cache["probabilities"], float(cache["duration"])
        for delta in (-.1, .1):
            cfg = replace(config, threshold=config.threshold + delta)
            turns = probabilities_to_turns(p, duration, cfg)
            f, _ = extract_features(turns, duration)
            score = score_features(artifact, f, segmentation_stability(p, duration, cfg))
            results.append({"anon_id": row.anon_id, "y": row.y, "variant": f"threshold_{cfg.threshold:.2f}",
                            "p": score["synthetic_probability"], "original_p": original.loc[row.anon_id, "probability"]})
        turns = load_turns(ROOT / "artifacts/vad_turns" / f"{row.anon_id}.json")
        jittered = []
        for t in turns:
            a = np.clip(t.start + rng.uniform(-.032, .032), 0, duration - .001)
            b = np.clip(t.end + rng.uniform(-.032, .032), a + .001, duration)
            jittered.append(Turn(float(a), float(b), t.channel))
        f, _ = extract_features(normalize_turns(jittered, duration), duration)
        score = score_features(artifact, f)
        results.append({"anon_id": row.anon_id, "y": row.y, "variant": "timing_jitter_32ms",
                        "p": score["synthetic_probability"], "original_p": original.loc[row.anon_id, "probability"]})
    # Predefined label-balanced subset; no error/prediction-driven selection.
    subset = val.groupby("y", group_keys=False).sample(n=12, random_state=SEED)
    directory = locate_audio()
    sos = butter(4, [300, 3400], btype="bandpass", fs=8000, output="sos")
    for i, row in enumerate(subset.itertuples()):
        audio = load_wav((directory / f"{row.anon_id}.wav").read_bytes())
        x = audio.samples
        noise = sosfilt(sos, rng.normal(size=x.shape), axis=0)
        noise *= np.sqrt(np.mean(x ** 2, axis=0)) / (np.sqrt(np.mean(noise ** 2, axis=0)) * 10 ** (25 / 20) + 1e-12)
        for variant, altered in (("gain_minus3db", x * 10 ** (-3 / 20)),
                                 ("gain_plus3db", x * 10 ** (3 / 20)),
                                 ("telephony_noise_25db", x + noise)):
            score = detector.predict(encode_wav(altered))
            results.append({"anon_id": row.anon_id, "y": row.y, "variant": variant,
                            "p": score["synthetic_probability"], "original_p": original.loc[row.anon_id, "probability"]})
        if (i + 1) % 8 == 0:
            print(f"Local robustness audio probes: {i + 1}/24 calls", flush=True)
    df = pd.DataFrame(results)
    df.to_csv(ROOT / "reports/private/robustness_predictions.csv", index=False)
    summary = {}
    for name, group in df.groupby("variant"):
        summary[name] = {"metrics": evaluate(group.y, group.p), "original_same_subset": evaluate(group.y, group.original_p),
                         "mean_absolute_probability_change": (group.p - group.original_p).abs().mean(),
                         "max_absolute_probability_change": (group.p - group.original_p).abs().max(),
                         "decision_flips": int(((group.p >= .5) != (group.original_p >= .5)).sum())}
    write_json(ROOT / "reports/metrics/robustness.json", summary)
    print(json.dumps({k: {"n": v["metrics"]["n"], "auc": v["metrics"]["roc_auc"], "ba": v["metrics"]["balanced_accuracy"],
                          "delta_p": v["mean_absolute_probability_change"], "flips": v["decision_flips"]}
                      for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
