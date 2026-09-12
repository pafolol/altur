"""
Figures for the telephone-robustness report that are not produced by a training or evaluation script.

  channel_dynamic_range.png   the finding that explains the failure: the Altur caller channel is DIGITALLY
                              SILENT between turns, a telephone line never is, and the energy voice-activity
                              detector depends on the difference
  channel_dataset.png         what the dataset factory produced: codecs drawn, impairments drawn, per-split counts
  channel_architecture.png    the pipeline, end to end

Run:  python src/channel_figures.py [--n 40]
Outputs: outputs/figures/channel_*.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from build_channel_dataset import MANIFESTS
from metrics import save_json

FIG = config.FIGURES_DIR


def fig_dynamic_range(n_calls=40):
    """
    Per call: the frame-level noise floor (10th percentile), the speech level (95th), the dynamic range between
    them, and how many speech regions each voice-activity detector finds - before and after the telephone line.
    """
    man = pd.read_csv(MANIFESTS / "all.csv")
    man = man[(man["status"] == "ok") & (man["split"] == "val") & (man["family"] == "seen")]
    man = man.drop_duplicates("call_id").head(n_calls)
    rows = []
    for r in man.itertuples():
        orig, sr = sf.read(r.original_path, dtype="float32", always_2d=True)
        orig = orig[:, config.CALLER_CHANNEL]
        chan, _ = sf.read(r.channel_path, dtype="float32", always_2d=True)
        chan = chan[:, 0]
        for name, w in (("original", orig), ("channel", chan)):
            _, db = audio.frame_db(w, sr)
            rows.append({"call_id": r.call_id, "kind": name, "label": r.label,
                         "floor_db": float(np.percentile(db, 10)), "median_db": float(np.percentile(db, 50)),
                         "speech_db": float(np.percentile(db, 95)),
                         "range_db": float(np.percentile(db, 95) - np.percentile(db, 10)),
                         "n_energy": len(audio.detect_speech_energy(w, sr)),
                         "n_silero": len(audio.detect_speech_silero(w, sr))})
    df = pd.DataFrame(rows)
    df.to_csv(config.OUTPUTS_DIR / "robust" / "dynamic_range.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    o, c = df[df["kind"] == "original"], df[df["kind"] == "channel"]

    bins = np.linspace(-90, -5, 40)
    axes[0].hist(o["floor_db"], bins=bins, alpha=0.75, color="#1f77b4", label=f"original, median {o['floor_db'].median():.0f} dB")
    axes[0].hist(c["floor_db"], bins=bins, alpha=0.75, color="#d62728", label=f"after the line, median {c['floor_db'].median():.0f} dB")
    axes[0].set_xlabel("noise floor of the caller channel (dB, 10th percentile of 20 ms frames)")
    axes[0].set_ylabel("calls")
    axes[0].set_title("the silence is not silent any more", fontsize=10)
    axes[0].legend(fontsize=7.5)
    axes[0].grid(alpha=0.3)

    axes[1].scatter(o["range_db"], c["range_db"], s=34, c=["#d62728" if l == "synthetic" else "#1f77b4" for l in o["label"]])
    lim = [0, max(o["range_db"].max(), c["range_db"].max()) + 4]
    axes[1].plot(lim, lim, "k--", lw=0.8, label="no change")
    axes[1].axhline(config.VAD_THRESHOLD_DB, color="#e08a00", lw=1.4,
                    label=f"energy VAD needs {config.VAD_THRESHOLD_DB:.0f} dB to fire")
    axes[1].scatter([], [], s=34, c="#1f77b4", label="human caller")
    axes[1].scatter([], [], s=34, c="#d62728", label="synthetic caller")
    axes[1].set_xlim(lim)
    axes[1].set_ylim(0, lim[1])
    axes[1].set_xlabel("dynamic range, original (dB)")
    axes[1].set_ylabel("dynamic range, after the line (dB)")
    axes[1].set_title("dynamic range collapses below what the VAD needs", fontsize=10)
    axes[1].legend(fontsize=7.5)
    axes[1].grid(alpha=0.3)

    width = 0.35
    x = np.arange(2)
    energy = [o["n_energy"].mean(), c["n_energy"].mean()]
    silero = [o["n_silero"].mean(), c["n_silero"].mean()]
    axes[2].bar(x - width / 2, energy, width, color="#e08a00", label="energy VAD (deployed)")
    axes[2].bar(x + width / 2, silero, width, color="#2ca02c", label="Silero VAD")
    dead = int((c["n_energy"] == 0).sum())
    for xi, v in zip(x - width / 2, energy):
        axes[2].text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    for xi, v in zip(x + width / 2, silero):
        axes[2].text(xi, v + 0.2, f"{v:.1f}", ha="center", fontsize=8)
    axes[2].set_xticks(x, ["original", "after the line"])
    axes[2].set_ylabel("speech regions found per call (mean)")
    axes[2].set_title(f"the energy VAD finds nothing at all in {dead}/{len(c)} channel calls", fontsize=10)
    axes[2].legend(fontsize=7.5)
    axes[2].grid(alpha=0.3, axis="y")

    fig.suptitle("Why the deployed detector goes blind on a telephone line", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "channel_dynamic_range.png", dpi=140)
    plt.close(fig)
    summary = {"n_calls": int(len(o)),
               "floor_db_original_median": float(o["floor_db"].median()),
               "floor_db_channel_median": float(c["floor_db"].median()),
               "range_db_original_median": float(o["range_db"].median()),
               "range_db_channel_median": float(c["range_db"].median()),
               "energy_vad_dead_calls": dead, "energy_vad_dead_fraction": dead / len(c),
               "silero_dead_calls": int((c["n_silero"] == 0).sum()),
               "mean_regions_energy_original": float(o["n_energy"].mean()),
               "mean_regions_energy_channel": float(c["n_energy"].mean()),
               "mean_regions_silero_original": float(o["n_silero"].mean()),
               "mean_regions_silero_channel": float(c["n_silero"].mean())}
    save_json(summary, config.OUTPUTS_DIR / "robust" / "dynamic_range.json")
    return summary


def fig_dataset():
    """What the factory actually drew: codecs, impairments, and the per-split counts."""
    man = pd.read_csv(MANIFESTS / "all.csv")
    man = man[man["status"] == "ok"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    counts = man.groupby(["split", "family", "label"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", stacked=True, ax=axes[0, 0], color=["#1f77b4", "#d62728"], width=0.7)
    axes[0, 0].set_title("samples generated (the split is never crossed)", fontsize=10)
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("samples")
    axes[0, 0].tick_params(axis="x", rotation=20, labelsize=7.5)
    axes[0, 0].legend(fontsize=7.5)

    codecs = man["codecs"].str.split("+").explode()
    order = codecs.value_counts()
    colours = ["#2ca02c" if c in __import__("phone_channel").SEEN_CODECS else "#d62728" for c in order.index]
    axes[0, 1].barh(order.index[::-1], order.to_numpy()[::-1], color=colours[::-1])
    axes[0, 1].set_title("codec hops drawn\n(green = trainable, red = evaluation only)", fontsize=10)
    axes[0, 1].tick_params(axis="y", labelsize=7.5)
    axes[0, 1].grid(alpha=0.3, axis="x")

    axes[0, 2].hist(man["gain_db"], bins=40, color="#1f77b4")
    axes[0, 2].axvline(0, color="black", ls="--", lw=0.8)
    axes[0, 2].set_xlabel("level change through the line (dB)")
    axes[0, 2].set_ylabel("samples")
    axes[0, 2].set_title(f"AGC moved the level by {man['gain_db'].abs().median():.1f} dB (median)", fontsize=10)
    axes[0, 2].grid(alpha=0.3)

    axes[1, 0].hist(man["realised_loss"] * 100, bins=40, color="#d62728")
    axes[1, 0].set_xlabel("packet loss actually realised (%)")
    axes[1, 0].set_ylabel("samples")
    axes[1, 0].set_title(f"{(man['realised_loss'] > 0).mean():.0%} of samples lost packets", fontsize=10)
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].hist(man["estimated_delay_ms"], bins=50, color="#7f7f7f")
    axes[1, 1].set_xlabel("codec delay measured by cross-correlation (ms)")
    axes[1, 1].set_ylabel("samples")
    axes[1, 1].set_title(f"delay removed before saving: median {man['estimated_delay_ms'].median():.1f} ms", fontsize=10)
    axes[1, 1].grid(alpha=0.3)

    axes[1, 2].scatter(man["alignment_correlation"], man["residual_delay_samples"],
                       s=8, alpha=0.4, c=["#d62728" if a else "#1f77b4" for a in man["agc"]])
    axes[1, 2].set_xlabel("envelope correlation with the original (AGC lowers it by design)")
    axes[1, 2].set_ylabel("residual delay after alignment (samples)")
    axes[1, 2].set_title("every sample is aligned to within a few samples", fontsize=10)
    axes[1, 2].legend(handles=[mpatches.Patch(color="#d62728", label="AGC on"),
                               mpatches.Patch(color="#1f77b4", label="AGC off")], fontsize=7.5)
    axes[1, 2].grid(alpha=0.3)

    fig.suptitle("The telephone-channel dataset factory: 706 samples from 353 calls", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG / "channel_dataset.png", dpi=140)
    plt.close(fig)


def fig_architecture():
    """The pipeline as one picture: where the data comes from, what may be trained on, what may not."""
    fig, ax = plt.subplots(figsize=(13.5, 7.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge, fontsize=8.2, weight="normal", textcolor="black"):
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.7", linewidth=1.3,
                                             facecolor=face, edgecolor=edge))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
                fontweight=weight, color=textcolor)

    def arrow(x1, y1, x2, y2, style="-|>", colour="#444444", ls="-"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=colour, lw=1.3, linestyle=ls))

    # Row 1: the dataset. Row 2: the official split. Row 3: the four domains. Then the model column on the
    # left and the evaluation column on the right, kept far enough apart that no arrow crosses a label.
    box(33, 89, 34, 7, "OFFICIAL ALTUR DATASET\n353 stereo calls, 8 kHz", "#eaf1fb", "#173a5e", 9, "bold")
    arrow(45, 89, 28, 83)
    arrow(55, 89, 72, 83)
    box(6, 75, 38, 7.5, "TRAIN  282 calls\nch0 caller, full timeline", "#ffffff", "#555555")
    box(56, 75, 38, 7.5, "VAL  71 calls\nnever touched by a gradient", "#ffffff", "#555555")

    arrow(16, 75, 12, 68)
    arrow(34, 75, 36, 68)
    arrow(66, 75, 62, 68)
    arrow(84, 75, 86, 68)

    box(2, 58, 20, 9, "original\n282 calls", "#eaf7ea", "#2ca02c")
    box(25, 58, 24, 9, "TELEPHONE CHANNEL\nG.711 / G.726, x2 draws\n564 samples", "#eaf7ea", "#2ca02c", 7.8)
    box(53, 58, 18, 9, "original\n71 calls", "#eaf1fb", "#1f77b4")
    box(75, 58, 23, 9, "CHANNEL\nseen 71  +  unseen 71", "#fdecec", "#d62728")

    # the model column (left), with the arrows on x = 12 and x = 36 so nothing crosses the right-hand column
    arrow(12, 58, 24, 49)
    arrow(36, 58, 32, 49)
    box(12, 40, 36, 9, "DOMAIN-BALANCED SAMPLING\nevery (domain, class) drawn equally", "#fff4e0", "#e08a00")
    arrow(30, 40, 30, 34)
    box(12, 24, 36, 9, "Wav2Vec2 Spanish, FROZEN\nhidden layer, mean over time", "#eaf1fb", "#173a5e", 8.6, "bold")
    arrow(30, 24, 30, 18)
    box(12, 8, 36, 9, "ROBUST ACOUSTIC V2\nMLP 768-256-2", "#173a5e", "#173a5e", 9, "bold", "white")

    # the evaluation column (right), arrows hugging the outer edges so the warning text sits in clear space
    arrow(56, 58, 56, 18, colour="#1f77b4", ls="--")
    arrow(96, 58, 96, 18, colour="#d62728", ls="--")
    ax.text(76, 38, "EVALUATION ONLY", ha="center", fontsize=9.5, color="#d62728", fontweight="bold")
    ax.text(76, 30, "nothing here reaches a gradient,\na scaler, a threshold or a calibrator",
            ha="center", fontsize=8, color="#d62728")
    box(53, 8, 45, 9, "Altur VAL   |   channel seen   |   channel UNSEEN\n(GSM, AMR, iLBC, Speex, Opus, G.722)",
        "#ffffff", "#d62728", 7.6)

    ax.text(25, 54.6, "real codecs through ffmpeg  →  20 ms RTP framing  →  bursty loss + concealment  →  clock drift\n"
                      "→  DTX comfort noise  →  line noise  →  AGC   (the same transform on both sides)",
            ha="center", va="top", fontsize=7.4, style="italic", color="#555555")
    ax.text(50, 2.5, "Model A, the specialist, was trained on the leftmost green box only. Model B adds the channel "
                     "box beside it. Nothing in the right-hand column is ever trained on.",
            ha="center", fontsize=8, color="#555555")
    fig.tight_layout()
    fig.savefig(FIG / "channel_architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40, help="calls used for the dynamic-range figure")
    args = parser.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    (config.OUTPUTS_DIR / "robust").mkdir(parents=True, exist_ok=True)
    print("[figures] dynamic range / VAD collapse ...")
    s = fig_dynamic_range(args.n)
    print(f"  noise floor {s['floor_db_original_median']:.0f} dB -> {s['floor_db_channel_median']:.0f} dB, "
          f"dynamic range {s['range_db_original_median']:.0f} dB -> {s['range_db_channel_median']:.0f} dB, "
          f"energy VAD dead on {s['energy_vad_dead_calls']}/{s['n_calls']} channel calls "
          f"(silero: {s['silero_dead_calls']})")
    print("[figures] dataset composition ...")
    fig_dataset()
    print("[figures] architecture ...")
    fig_architecture()
    print(f"-> {FIG}")


if __name__ == "__main__":
    main()
