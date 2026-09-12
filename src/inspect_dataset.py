"""
STAGE 1 - DATASET INSPECTION. Run this before training anything.

Answers, with numbers from the actual files:
  * how many calls per split and class, how balanced
  * audio format of EVERY file (sample rate, channels, bit depth, duration) + full decode check
  * per-call statistics of the caller channel (RMS, peak, clipping, silence, speech time, bandwidth)
  * conversation structure from the official turn files (turn counts, first caller start, latency, overlap)
  * how well our own energy VAD agrees with the official turns (we cannot use the turn files at inference)
  * how many 4 s chunks the segmentation produces (embedding budget)
  * leakage / shortcut checks: duplicate files, id ordering vs label, and the single-feature AUC of every
    trivial statistic (a feature with AUC far from 0.5 is a shortcut a model could learn instead of the voice)

Run:  python src/inspect_dataset.py
Outputs: outputs/dataset_inspection/{summary.json, call_stats.csv, ltas.npy, unreadable_files.txt}
         outputs/figures/{class_distribution, durations, turn_structure, levels, vad_agreement, chunks}.png
"""
import hashlib
from pathlib import Path
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.stats import mannwhitneyu
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset

OUT = config.OUTPUTS_DIR / "dataset_inspection"
FIG = config.FIGURES_DIR
N_FFT, HOP = 512, 80  # 64 ms window, 10 ms hop at 8 kHz -> 257 frequency bins, 15.6 Hz apart
COLORS = {"human": "#1f77b4", "synthetic": "#d62728"}

# the trivial statistics whose class separation we measure (name -> description)
SHORTCUT_FEATURES = {
    "duration_s": "total call duration (manifest)",
    "caller_speech_s": "caller speech time (official turns)",
    "caller_speech_fraction": "caller speech / call duration",
    "n_caller_turns": "number of caller turns",
    "n_agent_turns": "number of agent turns",
    "mean_caller_turn_s": "mean caller turn length",
    "first_caller_start_s": "time until the caller first speaks (leading silence)",
    "trailing_silence_s": "silence after the caller's last turn",
    "agent_speech_s": "agent speech time",
    "median_response_latency_s": "median caller response latency after an agent turn",
    "overlap_s": "seconds of caller/agent overlap",
    "caller_rms_db": "caller channel RMS level (whole channel)",
    "caller_speech_rms_db": "caller RMS level inside speech",
    "caller_peak": "caller peak amplitude",
    "caller_clipping_fraction": "fraction of clipped caller samples",
    "caller_rolloff95_hz": "frequency below which 95% of caller speech energy lies",
    "caller_centroid_hz": "spectral centroid of caller speech",
    "caller_hf_ratio_db": "energy 3.4-4 kHz vs 0.3-3.4 kHz (telephone band edge)",
    "caller_noise_floor_db": "caller channel noise floor (10th percentile frame level)",
    "caller_snr_db": "caller speech level minus noise floor",
    "agent_rms_db": "agent channel RMS level",
    "vad_n_regions": "number of caller speech regions found by our VAD",
    "n_chunks": "number of 4 s chunks our segmentation produces",
}


def analyze_call(row):
    info = sf.info(row["path"])
    stereo, sr = audio.read_wav(row["path"])
    total_s = len(stereo) / sr
    caller = audio.get_channel(stereo, config.CALLER_CHANNEL)
    agent = audio.get_channel(stereo, config.AGENT_CHANNEL)
    r = {"anon_id": row["anon_id"], "label": row["label"], "split": row["split"], "y": row["y"],
         "manifest_duration_s": row["duration_s"], "duration_s": total_s, "sample_rate": info.samplerate,
         "channels": info.channels, "subtype": info.subtype, "format": info.format,
         "file_bytes": (config.AUDIO_DIR / f"{row['anon_id']}.wav").stat().st_size,
         "md5": hashlib.md5(open(row["path"], "rb").read()).hexdigest()}
    r.update({f"caller_{k}": v for k, v in audio.level_stats(caller).items()})
    r.update({f"agent_{k}": v for k, v in audio.level_stats(agent).items()})
    r["channel_correlation"] = float(np.corrcoef(caller, agent)[0, 1]) if caller.std() > 0 and agent.std() > 0 else 0.0

    turns = dataset.load_turns(row["turns_path"])
    r.update(dataset.turn_stats(turns, total_s))
    r["trailing_silence_s"] = total_s - r["last_caller_end_s"]

    regions = audio.detect_speech(caller, sr, other=agent)
    r.update({f"vad_{k}": v for k, v in audio.silence_stats(regions, total_s).items()})
    official = dataset.turns_of_channel(turns, config.CALLER_CHANNEL)
    m_off, m_vad = audio.regions_to_mask(official, total_s), audio.regions_to_mask(regions, total_s)
    n = min(len(m_off), len(m_vad))
    m_off, m_vad = m_off[:n], m_vad[:n]
    tp = np.sum(m_off & m_vad)
    r["vad_precision"] = tp / max(1, m_vad.sum())
    r["vad_recall"] = tp / max(1, m_off.sum())
    r["vad_f1"] = 2 * tp / max(1, m_vad.sum() + m_off.sum())
    chunks = audio.chunk_regions(regions)
    r["n_chunks"] = len(chunks)
    r["chunk_seconds"] = sum(e - s for s, e in chunks)

    # long-term average spectrum of caller SPEECH (our VAD regions) at the native 8 kHz
    import librosa
    S = np.abs(librosa.stft(caller, n_fft=N_FFT, hop_length=HOP)) ** 2
    frame_t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    speech_frames = np.zeros(S.shape[1], dtype=bool)
    for s, e in regions:
        speech_frames[(frame_t >= s) & (frame_t < e)] = True
    ltas = S[:, speech_frames].mean(axis=1) if speech_frames.any() else S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    cum = np.cumsum(ltas) / max(ltas.sum(), 1e-12)
    r["caller_rolloff95_hz"] = float(freqs[np.searchsorted(cum, 0.95)])
    r["caller_centroid_hz"] = float((freqs * ltas).sum() / max(ltas.sum(), 1e-12))
    band = lambda lo, hi: ltas[(freqs >= lo) & (freqs < hi)].sum()
    r["caller_hf_ratio_db"] = float(10 * np.log10(band(3400, 4000) / max(band(300, 3400), 1e-12) + 1e-12))
    speech_samples = np.concatenate([audio.slice_chunk(caller, sr, s, e) for s, e in regions]) if regions else caller
    r["caller_speech_rms_db"] = audio.rms_db(speech_samples)
    _, fdb = audio.frame_db(caller, sr)
    r["caller_noise_floor_db"] = float(np.percentile(fdb, 10))
    r["caller_snr_db"] = r["caller_speech_rms_db"] - r["caller_noise_floor_db"]
    return r, 10 * np.log10(ltas + 1e-12)


def single_feature_auc(values, y):
    """AUC of ranking by one feature (0.5 = no information; far from 0.5 = shortcut)."""
    from sklearn.metrics import roc_auc_score
    ok = np.isfinite(values)
    if ok.sum() < 4 or len(np.unique(y[ok])) < 2:
        return np.nan
    return float(roc_auc_score(y[ok], values[ok]))


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(2)
    for i, (label, color) in enumerate(COLORS.items()):
        counts = [int(((df["split"] == s) & (df["label"] == label)).sum()) for s in config.SPLITS]
        bars = ax.bar(x + (i - 0.5) * 0.36, counts, 0.36, label=label, color=color)
        for b, c, s in zip(bars, counts, config.SPLITS):
            total = int((df["split"] == s).sum())
            ax.text(b.get_x() + b.get_width() / 2, c, f"{c}\n({c / total:.0%})", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"{s}\n({int((df['split'] == s).sum())} calls)" for s in config.SPLITS])
    ax.set_ylabel("number of calls")
    ax.set_ylim(0, df.groupby(["split", "label"]).size().max() * 1.3)
    ax.set_title("Calls per split and class (official manifest)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "class_distribution.png", dpi=140)
    plt.close(fig)


def hist_by_class(ax, df, col, title, bins=25, unit=""):
    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return
    edges = np.linspace(vals.min(), vals.max(), bins)
    for label, color in COLORS.items():
        v = df.loc[df["label"] == label, col].dropna()
        ax.hist(v, bins=edges, alpha=0.55, color=color, label=f"{label} (median {v.median():.2f}{unit})")
    ax.set_title(title, fontsize=9.5)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)


def plot_durations(df):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, split in zip(axes, config.SPLITS):
        hist_by_class(ax, df[df["split"] == split], "duration_s", f"call duration - {split}", unit=" s")
        ax.set_xlabel("seconds")
    fig.suptitle("Call duration per class (a shortcut check: the classes should overlap)")
    fig.tight_layout()
    fig.savefig(FIG / "durations.png", dpi=140)
    plt.close(fig)


def plot_turn_structure(df):
    panels = [("n_caller_turns", "caller turns per call", ""), ("first_caller_start_s", "time until caller speaks", " s"),
              ("caller_speech_s", "caller speech time", " s"), ("median_response_latency_s", "median response latency", " s"),
              ("mean_caller_turn_s", "mean caller turn length", " s"), ("overlap_s", "caller/agent overlap", " s")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (col, title, unit) in zip(axes.ravel(), panels):
        hist_by_class(ax, df[df["split"] == "train"], col, f"{title} (train)", unit=unit)
    fig.suptitle("Conversation structure from the official turn files: behavioural cues that are NOT acoustic")
    fig.tight_layout()
    fig.savefig(FIG / "turn_structure.png", dpi=140)
    plt.close(fig)


def plot_levels(df):
    panels = [("caller_speech_rms_db", "caller loudness inside speech", " dB"), ("caller_peak", "caller peak amplitude", ""),
              ("caller_rolloff95_hz", "95% spectral roll-off of caller speech", " Hz"),
              ("caller_hf_ratio_db", "3.4-4 kHz vs 0.3-3.4 kHz energy", " dB"),
              ("caller_centroid_hz", "spectral centroid of caller speech", " Hz"), ("agent_rms_db", "agent channel RMS", " dB")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4))
    for ax, (col, title, unit) in zip(axes.ravel(), panels):
        hist_by_class(ax, df[df["split"] == "train"], col, f"{title} (train)", unit=unit)
    fig.suptitle("Signal-level statistics of the caller channel: loudness, clipping, bandwidth")
    fig.tight_layout()
    fig.savefig(FIG / "levels.png", dpi=140)
    plt.close(fig)


def plot_vad_and_chunks(df):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].hist(df["vad_f1"], bins=30, color="#2ca02c")
    axes[0].set_title(f"our VAD vs official caller turns: frame F1 (median {df['vad_f1'].median():.3f})", fontsize=9)
    axes[0].set_xlabel("F1 per call")
    axes[1].scatter(df["n_caller_turns"], df["vad_n_regions"], s=10, alpha=0.6,
                    c=[COLORS[l] for l in df["label"]])
    lim = max(df["n_caller_turns"].max(), df["vad_n_regions"].max())
    axes[1].plot([0, lim], [0, lim], "k--", lw=0.8)
    axes[1].set_xlabel("official caller turns")
    axes[1].set_ylabel("VAD speech regions")
    axes[1].set_title("regions per call: official vs ours", fontsize=9)
    hist_by_class(axes[2], df, "n_chunks", f"4 s chunks per call (total {int(df['n_chunks'].sum()):,})")
    axes[2].set_xlabel("chunks")
    fig.tight_layout()
    fig.savefig(FIG / "vad_agreement.png", dpi=140)
    plt.close(fig)


def plot_average_spectra(ltas, df):
    import librosa
    freqs = librosa.fft_frequencies(sr=config.SOURCE_SAMPLE_RATE, n_fft=N_FFT)
    fig, ax = plt.subplots(figsize=(9, 4))
    for split, ls in (("train", "-"), ("val", "--")):
        for label, color in COLORS.items():
            sel = ((df["split"] == split) & (df["label"] == label)).to_numpy()
            mean = ltas[sel].mean(axis=0)
            ax.plot(freqs, mean - mean.max(), ls, color=color, lw=1.8 if split == "train" else 1.0,
                    label=f"{label} - {split} ({sel.sum()} calls)")
    ax.axvline(3400, color="gray", ls=":", lw=1)
    ax.text(3420, -5, "PSTN band edge\n~3.4 kHz", fontsize=7, color="gray")
    ax.set_xlabel("frequency (Hz)   [8 kHz audio: nothing exists above 4 kHz]")
    ax.set_ylabel("average power (dB, peak-normalised)")
    ax.set_title("Long-term average spectrum of caller speech, per class and split")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "average_spectrum.png", dpi=140)
    plt.close(fig)


def id_leakage(df):
    """Do the anonymous ids carry label information? (hex value vs label, and label runs in id order)."""
    hexval = np.array([int(i.split("_")[1], 16) for i in df["anon_id"]], dtype=float)
    y = df["y"].to_numpy()
    auc = single_feature_auc(hexval, y)
    # runs test on the id-sorted label sequence: expected runs for a random sequence
    order = np.argsort(hexval)
    seq = y[order]
    runs = 1 + int(np.sum(seq[1:] != seq[:-1]))
    n0, n1 = int((seq == 0).sum()), int((seq == 1).sum())
    expected = 1 + 2 * n0 * n1 / (n0 + n1)
    return {"id_hex_auc": auc, "label_runs_in_id_order": runs, "expected_runs_if_random": expected,
            "id_pattern": "call_ + 12 hex chars" if all(len(i) == 17 for i in df["anon_id"]) else "mixed"}


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(config.MANIFEST)
    manifest["y"] = manifest["label"].map(config.LABEL2ID)
    manifest["path"] = [str(config.AUDIO_DIR / f"{i}.wav") for i in manifest["anon_id"]]
    manifest["turns_path"] = [str(config.TURNS_DIR / f"{i}.json") for i in manifest["anon_id"]]
    summary = {"manifest_rows": len(manifest), "columns": list(manifest.columns[:4])}

    print("MANIFEST")
    counts = manifest.groupby(["split", "label"]).size().unstack(fill_value=0)
    print(counts)
    summary["counts"] = {s: {l: int(counts.loc[s, l]) for l in counts.columns} for s in counts.index}
    for s in config.SPLITS:
        n = int((manifest["split"] == s).sum())
        summary["counts"][s]["total"] = n
        summary["counts"][s]["synthetic_fraction"] = float((manifest.loc[manifest["split"] == s, "y"]).mean())
    summary["split_overlap_ids"] = int(len(set(manifest.loc[manifest["split"] == "train", "anon_id"]) &
                                           set(manifest.loc[manifest["split"] == "val", "anon_id"])))
    summary["missing_audio"] = [i for i, p in zip(manifest["anon_id"], manifest["path"]) if not Path(p).exists()]
    summary["missing_turns"] = [i for i, p in zip(manifest["anon_id"], manifest["turns_path"]) if not Path(p).exists()]
    print(f"missing audio: {len(summary['missing_audio'])}, missing turns: {len(summary['missing_turns'])}")

    print("\nDecoding and analysing every call ...")
    rows, ltas, bad = [], [], []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), unit="call"):
        try:
            r, spec = analyze_call(row)
            rows.append(r)
            ltas.append(spec)
        except Exception as exc:
            bad.append(row["anon_id"])
            print(f"[unreadable] {row['anon_id']}: {exc}")
    df = pd.DataFrame(rows)
    ltas = np.stack(ltas)
    df.to_csv(OUT / "call_stats.csv", index=False)
    np.save(OUT / "ltas.npy", ltas)
    if bad:
        (OUT / "unreadable_files.txt").write_text("\n".join(bad))
    summary["unreadable_files"] = bad

    # audio format ---------------------------------------------------------------------------------
    fmt = df.groupby(["sample_rate", "channels", "subtype", "format"]).size()
    summary["formats"] = {" | ".join(map(str, k)): int(v) for k, v in fmt.items()}
    summary["formats_by_class"] = {l: {" | ".join(map(str, k)): int(v) for k, v in
                                       df[df["label"] == l].groupby(["sample_rate", "channels", "subtype"]).size().items()}
                                   for l in config.LABELS}
    diff = (df["duration_s"] - df["manifest_duration_s"]).abs()
    summary["manifest_duration_max_abs_error_s"] = float(diff.max())
    summary["duplicate_md5"] = int(df["md5"].duplicated().sum())
    print(f"\nformats: {summary['formats']}")
    print(f"manifest duration vs header: max |error| = {diff.max():.2f} s; duplicate files (md5): {summary['duplicate_md5']}")

    # near-duplicate check on the spectrum fingerprint across splits ---------------------------------
    z = (ltas - ltas.mean(axis=1, keepdims=True)) / (ltas.std(axis=1, keepdims=True) + 1e-9)
    corr = z @ z.T / z.shape[1]
    np.fill_diagonal(corr, -1)
    tr, va = (df["split"] == "train").to_numpy(), (df["split"] == "val").to_numpy()
    cross = corr[np.ix_(tr, va)]
    summary["ltas_fingerprint"] = {"max_cross_split_correlation": float(cross.max()),
                                   "median_cross_split_correlation": float(np.median(cross)),
                                   "pairs_above_0.995": int((cross > 0.995).sum())}

    # per-class statistics + single-feature AUC ------------------------------------------------------
    print("\nSHORTCUT CHECK: single-feature AUC (0.5 = useless, 1.0 or 0.0 = perfect shortcut)")
    feats = {}
    for col, desc in SHORTCUT_FEATURES.items():
        row = {"description": desc}
        for split in config.SPLITS:
            d = df[df["split"] == split]
            row[f"{split}_median_human"] = float(d.loc[d["label"] == "human", col].median())
            row[f"{split}_median_synthetic"] = float(d.loc[d["label"] == "synthetic", col].median())
            row[f"{split}_auc"] = single_feature_auc(d[col].to_numpy(dtype=float), d["y"].to_numpy())
        d = df[df["split"] == "train"]
        a, b = d.loc[d["label"] == "human", col].dropna(), d.loc[d["label"] == "synthetic", col].dropna()
        row["train_mannwhitney_p"] = float(mannwhitneyu(a, b).pvalue) if len(a) > 2 and len(b) > 2 else np.nan
        feats[col] = row
        print(f"  {col:<30} train AUC {row['train_auc']:.3f}  val AUC {row['val_auc']:.3f}   "
              f"median human {row['train_median_human']:9.3f}  synthetic {row['train_median_synthetic']:9.3f}   "
              f"p={row['train_mannwhitney_p']:.1e}")
    summary["shortcut_features"] = feats
    pd.DataFrame(feats).T.to_csv(OUT / "shortcut_features.csv")

    # VAD agreement + chunk budget -------------------------------------------------------------------
    summary["vad_vs_official"] = {k: float(df[k].median()) for k in ("vad_precision", "vad_recall", "vad_f1")}
    summary["vad_vs_official"]["mean_f1"] = float(df["vad_f1"].mean())
    summary["vad_vs_official"]["calls_with_f1_below_0.8"] = int((df["vad_f1"] < 0.8).sum())
    summary["chunks"] = {s: {"n_chunks": int(df.loc[df["split"] == s, "n_chunks"].sum()),
                             "chunk_hours": float(df.loc[df["split"] == s, "chunk_seconds"].sum() / 3600),
                             "chunks_per_call_median": float(df.loc[df["split"] == s, "n_chunks"].median()),
                             "calls_with_zero_chunks": int((df.loc[df["split"] == s, "n_chunks"] == 0).sum())}
                         for s in config.SPLITS}
    summary["caller_speech_hours_official"] = float(df["caller_speech_s"].sum() / 3600)
    summary["total_audio_hours"] = float(df["duration_s"].sum() / 3600)
    summary["id_leakage"] = id_leakage(df)
    print(f"\nVAD vs official turns: {summary['vad_vs_official']}")
    print(f"chunks: {summary['chunks']}")
    print(f"id leakage: {summary['id_leakage']}")

    # figures ---------------------------------------------------------------------------------------
    plot_class_distribution(df)
    plot_durations(df)
    plot_turn_structure(df)
    plot_levels(df)
    plot_vad_and_chunks(df)
    plot_average_spectra(ltas, df)
    summary["runtime_s"] = time.time() - t0
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\nSaved {OUT / 'summary.json'}, call_stats.csv and figures in {FIG}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
