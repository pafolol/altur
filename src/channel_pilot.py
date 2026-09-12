"""
THE PILOT - ten calls, before spending anything on the other three hundred.

The question this script exists to answer is not "did the audio change" (of course it did) but the only one
that matters: DOES THE TELEPHONE CHANNEL ACTUALLY BREAK THE DEPLOYED DETECTOR? We already know from manual
testing that a real phone call hurts it. If the transformation built in phone_channel.py does not reproduce
that effect on these ten calls, the transformation is wrong and generating a whole dataset from it would be
three hundred useless copies. So this script ends in a verdict: GO or STOP.

It also separates two failure modes that look identical from the outside and need completely different fixes:

  segmentation failure   the voice-activity detector finds no caller speech in the channel audio, so the
                         model never gets to vote and the endpoint falls back to 0.5
  representation shift   the model gets its chunks and still gets the answer wrong

Run:  python src/channel_pilot.py [--n 10] [--family seen]
Outputs: channel_dataset/pilot/pilot_table.csv, channel_dataset/figures/pilot_*.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

import _bootstrap  # noqa: F401
import audio
import config
import phone_channel as pc
from build_channel_dataset import MANIFESTS, ROOT
from metrics import log_experiment, save_json

FIGURES = ROOT / "figures"
PILOT = ROOT / "pilot"
SR = pc.SR


def spectrum_db(wave, nperseg=512):
    """Welch power spectral density in dB - the honest way to compare two spectra of unequal noise."""
    f, p = welch(wave, SR, nperseg=min(nperseg, max(64, len(wave))))
    return f, 10 * np.log10(p + 1e-14)


def log_spectral_distance(a, b):
    """Root-mean-square difference between two log spectra, in dB. One number for 'how different do they look'."""
    _, pa = spectrum_db(a)
    _, pb = spectrum_db(b)
    return float(np.sqrt(np.mean((pa - pb) ** 2)))


def band_edge(wave, drop_db=20.0):
    """Highest frequency still within drop_db of the spectral peak: where the channel cuts the audio off."""
    f, p = spectrum_db(wave)
    peak = p.max()
    above = np.flatnonzero(p >= peak - drop_db)
    return float(f[above[-1]]) if len(above) else float("nan")


def score(det, caller, agent, vad):
    """Score one caller channel with a given voice-activity detector, keeping the agent channel untouched."""
    n = min(len(caller), len(agent))
    stereo = np.stack([caller[:n], agent[:n]], axis=1).astype(np.float32)
    det.vad = vad
    return det.predict_stereo(stereo, SR)


def plot_pair(orig, chan, row, path):
    """Waveform, spectrum and spectrogram of one call, before and after the line."""
    fig, axes = plt.subplots(3, 2, figsize=(13, 9))
    t = np.arange(len(orig)) / SR
    for col, (w, name) in enumerate(((orig, "original Altur caller channel"), (chan, f"after the line ({row['codecs']})"))):
        axes[0, col].plot(t[:len(w)], w, lw=0.3, color="#1f77b4" if col == 0 else "#d62728")
        axes[0, col].set_title(name, fontsize=10)
        axes[0, col].set_ylim(-1.05, 1.05)
        axes[0, col].set_xlabel("seconds")
        f, p = spectrum_db(w)
        axes[1, col].semilogx(f[1:], p[1:], color="#1f77b4" if col == 0 else "#d62728")
        axes[1, col].set_xlim(50, SR / 2)
        axes[1, col].set_ylim(-140, -10)
        axes[1, col].set_xlabel("Hz")
        axes[1, col].set_ylabel("dB")
        axes[1, col].grid(alpha=0.3, which="both")
        axes[2, col].specgram(w, NFFT=256, Fs=SR, noverlap=192, cmap="magma", vmin=-140, vmax=-30)
        axes[2, col].set_ylim(0, SR / 2)
        axes[2, col].set_xlabel("seconds")
        axes[2, col].set_ylabel("Hz")
    axes[1, 0].set_title("power spectrum", fontsize=9)
    axes[1, 1].set_title("power spectrum", fontsize=9)
    axes[2, 0].set_title("spectrogram", fontsize=9)
    axes[2, 1].set_title("spectrogram", fontsize=9)
    fig.suptitle(f"{row['call_id']}  ({row['label']})   synthetic probability {row['p_original']:.3f} "
                 f"-> {row['p_channel']:.3f}   AGC {row['agc']}, loss {row['realised_loss']:.1%}, "
                 f"delay {row['estimated_delay_ms']:+.1f} ms", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_summary(table, responses, path):
    """What the channel did, across all pilot calls: measured frequency response, levels, prediction shift."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    freqs, curves = responses
    for c in curves:
        axes[0].semilogx(freqs[1:], c[1:], color="#888", lw=0.7, alpha=0.6)
    axes[0].semilogx(freqs[1:], np.mean(curves, axis=0)[1:], color="#d62728", lw=2.2, label="mean over the pilot")
    axes[0].axhline(0, color="black", lw=0.8, ls="--")
    axes[0].set_xlim(50, SR / 2)
    axes[0].set_ylim(-45, 20)
    axes[0].set_xlabel("Hz")
    axes[0].set_ylabel("dB, channel minus original")
    axes[0].set_title("measured frequency response of the line", fontsize=10)
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)

    x = np.arange(len(table))
    axes[1].bar(x - 0.2, table["rms_original_db"], width=0.4, label="original", color="#1f77b4")
    axes[1].bar(x + 0.2, table["rms_channel_db"], width=0.4, label="after the line", color="#d62728")
    axes[1].set_xticks(x, [f"{r.label[0].upper()}{i}" for i, r in enumerate(table.itertuples())], fontsize=7)
    axes[1].set_ylabel("caller RMS (dB)")
    axes[1].set_title("level: the 6 dB loudness gap the model could lean on", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    colours = ["#d62728" if lab == "synthetic" else "#1f77b4" for lab in table["label"]]
    axes[2].scatter(table["p_original"], table["p_channel"], c=colours, s=70, zorder=3)
    axes[2].plot([0, 1], [0, 1], "k--", lw=0.8, label="no change")
    axes[2].axhline(0.5, color="gray", lw=0.8, ls=":")
    axes[2].axvline(0.5, color="gray", lw=0.8, ls=":")
    axes[2].set_xlabel("synthetic probability, original")
    axes[2].set_ylabel("synthetic probability, after the line")
    axes[2].set_title("what the deployed detector does\n(blue = human, red = synthetic)", fontsize=10)
    axes[2].set_xlim(-0.03, 1.03)
    axes[2].set_ylim(-0.03, 1.03)
    axes[2].grid(alpha=0.3)
    axes[2].legend(fontsize=8)

    fig.suptitle("Pilot: what the telephone channel changes", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="pilot size (half human, half synthetic)")
    parser.add_argument("--family", default="seen", choices=["seen", "unseen"])
    parser.add_argument("--split", default="train", choices=config.SPLITS)
    args = parser.parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    PILOT.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(MANIFESTS / "all.csv")
    man = man[(man["status"] == "ok") & (man["split"] == args.split) & (man["family"] == args.family)]
    if man.empty:
        raise SystemExit(f"no {args.split}/{args.family} samples yet - run build_channel_dataset.py first")
    # distinct CALLS, not distinct samples: with two variants per call, .head() would otherwise hand back the
    # same five voices twice and the pilot would cover half the ground it claims to.
    man = man.drop_duplicates(subset="call_id")
    pilot = pd.concat([g.head(args.n // 2) for _, g in man.groupby("label")]).sort_values("sample_id")
    print(f"[pilot] {len(pilot)} samples from {args.split}/{args.family}\n")

    from predict import AcousticDetector
    det = AcousticDetector(verbose=True)
    native_vad = config.VAD_METHOD

    rows, curves, freqs = [], [], None
    for r in pilot.itertuples():
        stereo, sr = audio.read_wav(r.original_path)
        orig = audio.get_channel(stereo, config.CALLER_CHANNEL)
        agent = audio.get_channel(stereo, config.AGENT_CHANNEL)
        chan, _ = audio.read_wav(r.channel_path)
        chan = chan[:, 0]

        base = score(det, orig, agent, native_vad)
        got = score(det, chan, agent, native_vad)
        got_robust = score(det, chan, agent, "silero")
        base_robust = score(det, orig, agent, "silero")

        f, po = spectrum_db(orig)
        _, pch = spectrum_db(chan)
        freqs = f
        curves.append(pch - po)

        lag, corr = pc.estimate_delay(orig, chan)
        rows.append({
            "call_id": r.call_id, "label": r.label, "codecs": r.codecs, "agc": r.agc,
            "p_original": base["synthetic_probability"], "p_channel": got["synthetic_probability"],
            "delta_p": got["synthetic_probability"] - base["synthetic_probability"],
            "correct_original": (base["synthetic_probability"] >= 0.5) == (r.label == "synthetic"),
            "correct_channel": (got["synthetic_probability"] >= 0.5) == (r.label == "synthetic"),
            "p_channel_silero": got_robust["synthetic_probability"],
            "correct_channel_silero": (got_robust["synthetic_probability"] >= 0.5) == (r.label == "synthetic"),
            "chunks_original": base["n_chunks"], "chunks_channel": got["n_chunks"],
            "chunks_original_silero": base_robust["n_chunks"], "chunks_channel_silero": got_robust["n_chunks"],
            "rms_original_db": audio.rms_db(orig), "rms_channel_db": audio.rms_db(chan),
            "spectral_distance_db": log_spectral_distance(orig, chan),
            "band_edge_original_hz": band_edge(orig), "band_edge_channel_hz": band_edge(chan),
            "estimated_delay_ms": 1000.0 * lag / SR, "alignment_correlation": corr,
            "realised_loss": r.realised_loss,
        })
        print(f"  {r.call_id[-8:]}  {r.label:<9} {str(r.codecs):<22} "
              f"p {base['synthetic_probability']:.3f} -> {got['synthetic_probability']:.3f}   "
              f"chunks {base['n_chunks']:>2} -> {got['n_chunks']:>2} (silero {got_robust['n_chunks']:>2})   "
              f"LSD {rows[-1]['spectral_distance_db']:.1f} dB   delay {rows[-1]['estimated_delay_ms']:+.1f} ms")

    table = pd.DataFrame(rows)
    table.to_csv(PILOT / "pilot_table.csv", index=False)

    for label in ("human", "synthetic"):
        r = table[table["label"] == label].iloc[0]
        src = pilot[pilot["call_id"] == r["call_id"]].iloc[0]
        stereo, _ = audio.read_wav(src["original_path"])
        chan, _ = audio.read_wav(src["channel_path"])
        plot_pair(audio.get_channel(stereo, 0), chan[:, 0], r, FIGURES / f"pilot_{label}.png")
    plot_summary(table, (freqs, np.array(curves)), FIGURES / "pilot_summary.png")

    # ------------------------------------------------------------------ the verdict
    acc_o = table["correct_original"].mean()
    acc_c = table["correct_channel"].mean()
    acc_s = table["correct_channel_silero"].mean()
    mean_shift = table["delta_p"].abs().mean()
    flips = int((table["correct_original"] & ~table["correct_channel"]).sum())
    vad_deaths = int((table["chunks_channel"] == 0).sum())
    summary = {
        "n": len(table), "split": args.split, "family": args.family,
        "accuracy_original": acc_o, "accuracy_channel": acc_c, "accuracy_channel_silero": acc_s,
        "mean_abs_delta_p": mean_shift, "verdict_flips": flips, "segmentation_failures": vad_deaths,
        "mean_spectral_distance_db": float(table["spectral_distance_db"].mean()),
        "mean_level_change_db": float((table["rms_channel_db"] - table["rms_original_db"]).mean()),
        "median_alignment_correlation": float(table["alignment_correlation"].median()),
        "median_delay_ms": float(table["estimated_delay_ms"].median()),
    }
    print("\n" + "=" * 78)
    print(f"  deployed detector accuracy:  original {acc_o:.0%}  ->  through the line {acc_c:.0%}")
    print(f"  same audio, robust segmentation (silero):            {acc_s:.0%}")
    print(f"  verdicts flipped: {flips}/{len(table)}   segmentation failures (0 chunks): {vad_deaths}/{len(table)}")
    print(f"  mean |change in synthetic probability|: {mean_shift:.3f}")
    print(f"  mean spectral distance: {summary['mean_spectral_distance_db']:.1f} dB   "
          f"mean level change: {summary['mean_level_change_db']:+.1f} dB")
    print("=" * 78)
    if mean_shift < 0.05 and flips == 0 and vad_deaths == 0:
        summary["decision"] = "STOP"
        print("  STOP. The channel barely moves the detector, so it is not reproducing the failure we saw\n"
              "  on a real call. Do not generate the dataset until this is understood.")
    else:
        summary["decision"] = "GO"
        print("  GO. The channel reproduces the production failure: the detector loses calls it gets right\n"
              "  on the clean audio. Generating the full dataset is justified.")
    print("=" * 78)
    save_json(summary, PILOT / "pilot_summary.json")
    log_experiment(title=f"channel pilot - {args.split}/{args.family}", model="deployed specialist (Model A)",
                   n_samples=len(table), accuracy_original=round(acc_o, 4), accuracy_channel=round(acc_c, 4),
                   accuracy_channel_silero=round(acc_s, 4), verdict_flips=flips, segmentation_failures=vad_deaths,
                   mean_abs_delta_p=round(mean_shift, 4), decision=summary["decision"],
                   notes="10-call gate before generating the full channel dataset")
    print(f"\ntable   -> {PILOT / 'pilot_table.csv'}")
    print(f"figures -> {FIGURES}")


if __name__ == "__main__":
    main()
