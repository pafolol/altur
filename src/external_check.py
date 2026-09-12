"""
EXTERNAL CHECK - does the detector recognise synthetic speech it has never seen, from OUTSIDE the dataset?

The validation split came through the same recording pipeline as the training split, so it cannot tell "detects
synthetic speech" from "detects the pipeline". This script builds a small out-of-distribution test set:

  SYNTHETIC (engines the dataset never used)
    * Microsoft Edge neural TTS (edge-tts): es-MX, es-CO, es-AR, es-US voices, modern neural vocoder
    * Google TTS (gTTS), Spanish
    * Windows SAPI desktop voices (Microsoft Sabina es-MX, Helena es-ES): old concatenative/formant engines
  HUMAN (people the dataset never recorded)
    * FLEURS es_419: read Latin-American Spanish sentences, many speakers, 16 kHz
    * ASVspoof 2019 bonafide (English studio speech) from the previous project, if present on disk

Every clip - human and synthetic alike - goes through the SAME simulated telephone path before scoring:
  resample to 8 kHz -> (optionally) 300-3400 Hz band-pass + G.711 mu-law -> 1.5 s of faint line noise before/after
  -> stereo WAV with a silent agent channel -> the production detector (predict.AcousticDetector).
Two channel variants are reported: "8k" (resample only) and "pstn" (band-pass + mu-law).

Nothing here is used for training or selection; it is a read-out of how the deployed model behaves off-pipeline.

Run:  python src/external_check.py [--n-human 40] [--n-tts 8]
Outputs: outputs/external/{clips/*.wav, results.json, scores.csv}, outputs/figures/external_check.png
"""
import argparse
import asyncio
import io
import json
import os
import subprocess
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import butter, sosfiltfilt

import _bootstrap  # noqa: F401
import audio
import config
from metrics import evaluate_scores, log_experiment, save_json

OUT = config.OUTPUTS_DIR / "external"
CLIPS = OUT / "clips"
SR = config.SOURCE_SAMPLE_RATE

SENTENCES = [
    "Buenas tardes, hablo para consultar el saldo de mi cuenta de ahorros, mi número de cliente es cuatro siete dos nueve.",
    "Sí, claro, mi nombre completo es Mariana López Hernández y nací el doce de marzo de mil novecientos noventa.",
    "No tengo ese producto, nunca he contratado un seguro de auto con ustedes.",
    "Quisiera reportar un cargo que no reconozco en mi tarjeta de crédito, fue el martes pasado.",
    "Perdón, no le escuché bien, ¿me puede repetir la pregunta por favor?",
    "Mi código postal es cero seis seis cero cero y vivo en la colonia Roma Norte.",
    "Sí, confirmo, el monto del último pago fue de dos mil trescientos pesos.",
    "Entiendo, entonces necesito acudir a la sucursal con una identificación oficial.",
    "Disculpe, creo que hubo una confusión, yo no solicité ningún préstamo personal.",
    "Muchas gracias por su ayuda, que tenga buen día.",
    "Me gustaría actualizar mi número de teléfono, el nuevo es cincuenta y cinco doce treinta y cuatro.",
    "¿Podría decirme cuál es el horario de atención de la sucursal del centro?",
]
EDGE_VOICES = ["es-MX-DaliaNeural", "es-MX-JorgeNeural", "es-CO-SalomeNeural", "es-CO-GonzaloNeural",
               "es-AR-ElenaNeural", "es-AR-TomasNeural", "es-US-PalomaNeural", "es-US-AlonsoNeural"]
SAPI_VOICES = ["Microsoft Sabina Desktop", "Microsoft Helena Desktop"]


# ----------------------------------------------------------------------------- sources
def load_any(path_or_bytes):
    """Decode mp3/wav/flac with soundfile (mp3 via libsndfile >= 1.1) or ffmpeg/pyav fallback -> mono float32 + sr."""
    try:
        w, sr = sf.read(path_or_bytes if not isinstance(path_or_bytes, (bytes, bytearray)) else io.BytesIO(path_or_bytes),
                        dtype="float32", always_2d=True)
        return w.mean(axis=1), int(sr)
    except Exception:
        import av
        src = io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes)
        chunks = []
        with av.open(src) as c:
            s = c.streams.audio[0]
            rs = av.AudioResampler(format="flt", layout="mono", rate=s.rate)
            for f in c.decode(s):
                chunks += [x.to_ndarray().reshape(-1) for x in rs.resample(f)]
            chunks += [x.to_ndarray().reshape(-1) for x in rs.resample(None)]
        return np.concatenate(chunks).astype(np.float32), int(s.rate)


def gen_edge(n_per_voice):
    import edge_tts
    out = []

    async def one(voice, text, path):
        await edge_tts.Communicate(text, voice).save(str(path))

    for v in EDGE_VOICES:
        for i in range(n_per_voice):
            text = SENTENCES[(i * len(EDGE_VOICES) + EDGE_VOICES.index(v)) % len(SENTENCES)]
            p = CLIPS / f"tts_edge_{v}_{i}.mp3"
            if not p.exists():
                asyncio.run(one(v, text, p))
            w, sr = load_any(p)
            out.append(("synthetic", f"edge-tts {v}", w, sr))
    return out


def gen_gtts(n):
    from gtts import gTTS
    out = []
    for i in range(n):
        p = CLIPS / f"tts_gtts_{i}.mp3"
        if not p.exists():
            gTTS(SENTENCES[i % len(SENTENCES)], lang="es", tld="com.mx").save(str(p))
        w, sr = load_any(p)
        out.append(("synthetic", "gTTS es-MX", w, sr))
    return out


def gen_sapi(n_per_voice):
    out = []
    for v in SAPI_VOICES:
        for i in range(n_per_voice):
            p = CLIPS / f"tts_sapi_{v.split()[1]}_{i}.wav"
            if not p.exists():
                text = SENTENCES[(i * 2 + SAPI_VOICES.index(v)) % len(SENTENCES)].replace('"', "")
                ps = (f"Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                      f"$s.SelectVoice('{v}'); $s.SetOutputToWaveFile('{p}'); $s.Speak('{text}'); $s.Dispose()")
                subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, capture_output=True)
            w, sr = load_any(p)
            out.append(("synthetic", f"Windows SAPI {v.split()[1]}", w, sr))
    return out


def fetch_fleurs(n):
    os.environ.setdefault("HF_HOME", str(config.PROJECT_ROOT / ".cache" / "huggingface"))
    from datasets import Audio, load_dataset
    out = []
    cache = CLIPS / "fleurs"
    cache.mkdir(exist_ok=True)
    existing = sorted(cache.glob("*.wav"))
    if len(existing) >= n:
        for p in existing[:n]:
            w, sr = load_any(p)
            out.append(("human", "FLEURS es_419", w, sr))
        return out
    ds = load_dataset("google/fleurs", "es_419", split="validation", streaming=True).cast_column("audio", Audio(decode=False))
    seen_speakers = set()
    for ex in ds:
        a = ex["audio"]
        raw = a["bytes"] if a.get("bytes") else open(a["path"], "rb").read()
        w, sr = load_any(raw)
        if len(w) / sr < 4:
            continue
        p = cache / f"fleurs_{ex['id']}.wav"
        sf.write(p, w, sr)
        out.append(("human", "FLEURS es_419", w, sr))
        if len(out) >= n:
            break
    return out


def fetch_asvspoof_bonafide(n):
    root = Path(r"D:\asvspoof-learning\data\LA")
    proto = root / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.dev.trl.txt"
    if not proto.exists():
        return []
    rows = [l.split() for l in proto.read_text().splitlines() if l.endswith("bonafide")]
    rng = np.random.default_rng(config.SEED)
    out = []
    for r in rng.permutation(len(rows))[:n]:
        p = root / "ASVspoof2019_LA_dev" / "flac" / f"{rows[r][1]}.flac"
        if p.exists():
            w, sr = load_any(p)
            out.append(("human", "ASVspoof bonafide (English)", w, sr))
    return out


# ----------------------------------------------------------------------------- telephone simulation
def to_call(wave, sr, variant, seed):
    """mono any-rate -> stereo 8 kHz call-like array: caller = clip through the channel, agent = near-silence."""
    w = audio.resample(wave, sr, SR) if sr != SR else wave.astype(np.float32)
    w = w / (np.abs(w).max() + 1e-9) * 0.5
    if variant == "pstn":
        sos = butter(6, [300 / (SR / 2), 3400 / (SR / 2)], btype="band", output="sos")
        w = sosfiltfilt(sos, w).astype(np.float32)
        import stress_test
        w = stress_test.mulaw(w)
    rng = np.random.default_rng(seed)
    pad = (0.0005 * rng.standard_normal(int(1.5 * SR))).astype(np.float32)
    caller = np.concatenate([pad, w + 0.0005 * rng.standard_normal(len(w)).astype(np.float32), pad])
    agent = (0.0003 * rng.standard_normal(len(caller))).astype(np.float32)
    return np.stack([caller, agent], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-human", type=int, default=40)
    parser.add_argument("--n-tts", type=int, default=4, help="clips per TTS voice")
    args = parser.parse_args()
    CLIPS.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from predict import AcousticDetector
    det = AcousticDetector(verbose=True)

    print("collecting clips ...")
    items = []
    items += gen_edge(args.n_tts)
    items += gen_gtts(args.n_tts * 2)
    try:
        items += gen_sapi(args.n_tts)
    except Exception as exc:
        print(f"  SAPI skipped: {exc}")
    items += fetch_fleurs(args.n_human)
    items += fetch_asvspoof_bonafide(args.n_human // 2)
    print(f"  {len(items)} clips: " + ", ".join(f"{s}={sum(1 for i in items if i[1] == s)}" for s in dict.fromkeys(i[1] for i in items)))

    rows = []
    for k, (label, source, w, sr) in enumerate(items):
        for variant in ("8k", "pstn"):
            stereo = to_call(w, sr, variant, seed=k)
            r = det.predict_stereo(stereo, SR)
            rows.append({"source": source, "label": label, "y": int(label == "synthetic"), "variant": variant, "clip": k,
                         "seconds": round(len(w) / sr, 1), "n_chunks": r["n_chunks"], "score": r["score"],
                         "p_synthetic": r["synthetic_probability"], "pred": int(r["synthetic_probability"] >= 0.5)})
            if variant == "8k" and k < 400:
                sf.write(CLIPS / f"call_{k:03d}_{label}_{source.split()[0]}.wav", stereo, SR, subtype="PCM_16")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "scores.csv", index=False)

    res = {"n_clips": len(items), "per_source": {}, "per_variant": {}}
    print("\nRESULTS (deployed detector, threshold p = 0.5)")
    for variant in ("8k", "pstn"):
        d = df[df["variant"] == variant]
        m = evaluate_scores(d["y"], d["score"])
        res["per_variant"][variant] = {k: m[k] for k in ("accuracy", "balanced_accuracy", "auc", "eer", "far", "frr", "n_calls")}
        print(f"\n  channel variant '{variant}': accuracy {m['accuracy']:.3f}  balanced {m['balanced_accuracy']:.3f}  AUC {m['auc']:.3f}  "
              f"EER {m['eer'] * 100:.1f}%  (synthetic missed {m['far']:.2f}, humans flagged {m['frr']:.2f})")
        for src, g in d.groupby("source", sort=False):
            correct = float((g["pred"] == g["y"]).mean())
            res["per_source"].setdefault(src, {})[variant] = {"n": int(len(g)), "correct": correct,
                                                             "median_p_synthetic": float(g["p_synthetic"].median()),
                                                             "median_score": float(g["score"].median())}
            print(f"    {src:<30} {g['label'].iloc[0]:<9} n={len(g):3d}  correct {correct:.2f}  median p(synthetic) {g['p_synthetic'].median():.3f}  median score {g['score'].median():+.2f}")
    res["runtime_s"] = time.time() - t0
    save_json(res, OUT / "results.json")

    # figure: score distributions per source, both variants
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, variant in zip(axes, ("8k", "pstn")):
        d = df[df["variant"] == variant]
        sources = list(dict.fromkeys(d["source"]))
        for i, src in enumerate(sources):
            g = d[d["source"] == src]
            color = "#d62728" if g["label"].iloc[0] == "synthetic" else "#1f77b4"
            ax.scatter(g["score"], np.full(len(g), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(g)), s=14, color=color, alpha=0.7)
        ax.axvline(0, color="black", ls="--", lw=0.8)
        ax.set_yticks(range(len(sources)), sources, fontsize=8)
        ax.set_xlabel("detector score (log-odds of synthetic; > 0 = called synthetic)")
        ax.set_title(f"channel '{variant}': " + ("resample to 8 kHz only" if variant == "8k" else "300-3400 Hz band-pass + mu-law"), fontsize=9.5)
        ax.grid(alpha=0.3)
    fig.suptitle("Out-of-dataset check: new TTS engines (red) and new human speakers (blue) through a simulated phone line")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "external_check.png", dpi=140)
    plt.close(fig)
    log_experiment(title="external out-of-dataset check", model=f"{det.backbone} layer {det.layer} + mlp (deployed)", dataset_fraction="n/a",
                   segmentation="as pipeline", selected_layer=det.layer, classifier="deployed", hyperparameters="edge-tts/gTTS/SAPI vs FLEURS/ASVspoof bonafide",
                   device=str(det.device), runtime_s=round(res["runtime_s"]),
                   results={v: {k: round(x, 3) for k, x in res["per_variant"][v].items() if k != "n_calls"} for v in res["per_variant"]})
    print(f"\nSaved {OUT / 'results.json'}, scores.csv, clips/ and figures/external_check.png ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
