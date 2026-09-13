"""
Figures generated for PROJECT_TECHNICAL_REPORT that the project did not already produce.

Everything here is drawn from measured artifacts already on disk - no synthetic example data:
  fig_shortcuts.png      outputs/dataset_inspection/call_stats.csv   (353 real calls)
  fig_v1_v2_domains.png  outputs/robust/evaluation_matrix.csv        (the real cross-domain eval)
  fig_latency.png        outputs/fusion/latency_benchmark.json       (the real 71-call prefix walk)

Palette: categorical slots 1 and 2 of the validated reference palette (blue / orange). Two series only,
both above the 3:1 contrast floor on a light surface, legend plus direct labels so identity is never
carried by colour alone.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report_assets" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1 and 2
INK, DIM, RULE = "#15191f", "#5a6472", "#d4dae2"

plt.rcParams.update({
    "figure.dpi": 190, "savefig.dpi": 190, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "axes.edgecolor": RULE, "axes.labelcolor": DIM, "text.color": INK,
    "xtick.color": DIM, "ytick.color": DIM, "xtick.labelsize": 7.8, "ytick.labelsize": 7.8,
    "axes.grid": True, "grid.color": "#eef1f5", "grid.linewidth": .7,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 8,
})


def strip(ax):
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ===================================================================== 1. the shortcuts
def fig_shortcuts():
    """
    The six call-level statistics that separate the two classes WITHOUT hearing a voice.

    Each panel is a real histogram of all 353 calls. The AUC printed in each panel is the single-feature
    train-split separation recorded by the project's own dataset inspection, so the panel and the number
    come from the same measurement.
    """
    df = pd.read_csv(ROOT / "outputs/dataset_inspection/call_stats.csv")
    summary = json.loads((ROOT / "outputs/dataset_inspection/summary.json").read_text())
    aucs = {}
    for block in ("shortcuts", "shortcut_auc", "shortcut_features"):
        node = summary.get(block)
        if isinstance(node, dict):
            for k, v in node.items():
                if isinstance(v, dict) and "auc" in v:
                    aucs[k] = v["auc"]
                elif isinstance(v, (int, float)):
                    aucs[k] = v

    panels = [
        ("first_caller_start_s", "Leading silence before the caller speaks", "seconds", (0, 30)),
        ("caller_rms_db", "Caller channel loudness", "dBFS", None),
        ("median_response_latency_s", "Median response latency", "seconds", (0, 12)),
        ("mean_caller_turn_s", "Mean caller turn length", "seconds", (0, 8)),
        ("n_caller_turns", "Number of caller turns", "turns", None),
        ("overlap_s", "Talk-over with the agent", "seconds", (0, 30)),
    ]
    panels = [p for p in panels if p[0] in df.columns]

    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.0))
    for ax, (col, title, unit, clip) in zip(axes.ravel(), panels):
        h = df.loc[df.label == "human", col].dropna()
        s = df.loc[df.label == "synthetic", col].dropna()
        lo, hi = (clip if clip else (min(h.min(), s.min()), max(h.max(), s.max())))
        bins = np.linspace(lo, hi, 34)
        ax.hist(np.clip(h, lo, hi), bins=bins, color=BLUE, alpha=.72, label="Human", linewidth=0)
        ax.hist(np.clip(s, lo, hi), bins=bins, color=ORANGE, alpha=.62, label="Synthetic", linewidth=0)
        ax.axvline(h.median(), color=BLUE, lw=1.5, ls="--")
        ax.axvline(s.median(), color=ORANGE, lw=1.5, ls="--")
        ax.set_title(title, color=INK, fontweight="600", pad=5)
        ax.set_xlabel(unit)
        ax.set_ylabel("calls")
        strip(ax)
        auc = aucs.get(col)
        note = f"medians {h.median():.2f} vs {s.median():.2f}"
        if isinstance(auc, (int, float)):
            note += f"\nsingle-feature AUC {auc:.3f}"
        ax.text(.97, .95, note, transform=ax.transAxes, ha="right", va="top",
                fontsize=6.8, color=DIM, linespacing=1.35)
    axes[0, 0].legend(loc="upper right", bbox_to_anchor=(1.0, .72))
    fig.suptitle("Six statistics that split the classes without hearing a single voice",
                 fontsize=11, fontweight="650", color=INK, x=.5, y=1.005)
    fig.tight_layout()
    fig.savefig(OUT / "fig_shortcuts.png")
    plt.close(fig)
    print("fig_shortcuts.png", [p[0] for p in panels])


# ===================================================================== 2. V1 vs V2 by domain
def fig_v1_v2():
    """The headline robustness result: the same six domains, the specialist against Robust V2."""
    df = pd.read_csv(ROOT / "outputs/robust/evaluation_matrix.csv")
    order = [("altur_val", "Altur validation\n(clean, original)"),
             ("channel_val_seen", "Through a line,\nSEEN codecs"),
             ("channel_val_unseen", "Through a line,\nUNSEEN codecs"),
             ("stress", "11 channel\nperturbations"),
             ("external_ood_channel", "Off-pipeline TTS\ndown a line"),
             ("external_ood", "Off-pipeline TTS\nclean (not telephone)")]
    key = df.set_index(["domain", "model"])["accuracy"]
    doms = [(lab, key.get((d, "specialist")), key.get((d, "robust_v2")))
            for d, lab in order if (d, "specialist") in key.index]

    fig, ax = plt.subplots(figsize=(10.2, 4.0))
    y = np.arange(len(doms))
    hgt = .36
    v1 = [d[1] * 100 for d in doms]
    v2 = [d[2] * 100 for d in doms]
    ax.barh(y + hgt / 2, v1, hgt, color=BLUE, label="Specialist V1", zorder=3)
    ax.barh(y - hgt / 2, v2, hgt, color=ORANGE, label="Robust V2  (deployed)", zorder=3)
    for yi, v in zip(y + hgt / 2, v1):
        ax.text(v + 1.1, yi, f"{v:.1f}", va="center", fontsize=7.6, color=BLUE, fontweight="650")
    for yi, v in zip(y - hgt / 2, v2):
        ax.text(v + 1.1, yi, f"{v:.1f}", va="center", fontsize=7.6, color=ORANGE, fontweight="650")
    ax.set_yticks(y, [d[0] for d in doms], fontsize=7.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 112)
    ax.set_xticks(range(0, 101, 20), [f"{v}%" for v in range(0, 101, 20)])
    ax.set_xlabel("Call-level accuracy at the deployed threshold")
    ax.axvline(50, color=RULE, lw=1, ls=":", zorder=1)
    ax.text(50, len(doms) - .35, "chance", ha="center", va="top", fontsize=6.8, color=DIM)
    ax.legend(loc="lower right", ncol=2, bbox_to_anchor=(1.0, -.30))
    ax.grid(axis="y", visible=False)
    strip(ax)
    ax.set_title("Robust V2 wins every telephone domain and loses the one that is not a telephone",
                 fontsize=11, fontweight="650", color=INK, loc="left", pad=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_v1_v2_domains.png")
    plt.close(fig)
    print("fig_v1_v2_domains.png", [(d[0].replace(chr(10), ' '), d[1], d[2]) for d in doms])


# ===================================================================== 3. detection latency
def fig_latency():
    """How much CALLER SPEECH the acoustic layer needed before it committed, over the 71 held-out calls."""
    bench = json.loads((ROOT / "outputs/fusion/latency_benchmark.json").read_text())
    block = bench.get("specialist", bench[list(bench)[0]])
    notable = block.get("notable", {})
    rows = (notable.get("most_gradual", []) or []) + (notable.get("early_disagreed_with_final", []) or [])
    speech = [r["confident_after_caller_speech_sec"] for r in rows
              if isinstance(r.get("confident_after_caller_speech_sec"), (int, float))]
    summary = block["all"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.2, 3.3),
                                   gridspec_kw={"width_ratios": [1.15, 1]})

    # left: the four headline medians, as a labelled bar - a stat row, not a fake distribution
    labels = ["Caller speech\nheard", "Elapsed into\nthe call", "Lead time before\nthe call ended"]
    vals = [summary["median_caller_speech_sec"], summary["median_call_time_sec"],
            summary["median_lead_time_sec"]]
    x = np.arange(len(vals))
    ax1.bar(x, vals, .5, color=[ORANGE, BLUE, BLUE], zorder=3)
    for xi, v in zip(x, vals):
        ax1.text(xi, v + max(vals) * .03, f"{v:.1f} s", ha="center", fontsize=8.6,
                 fontweight="680", color=INK)
    ax1.set_yscale("log")
    ax1.set_ylim(1, max(vals) * 3)
    ax1.set_xticks(x, labels, fontsize=7.6)
    ax1.set_ylabel("seconds (log scale)")
    ax1.grid(axis="x", visible=False)
    strip(ax1)
    ax1.set_title("Median moment of a confident verdict", fontsize=9.6, fontweight="650",
                  color=INK, loc="left")

    # right: the two rates that say whether the early verdict can be trusted
    rates = [("Calls that ever\nbecame confident", summary["n_detected"] / summary["n"] * 100),
             ("Confident before\nhalf the call", summary["detected_before_halfway_pct"]),
             ("Early verdict matched\nthe final one", summary["early_agrees_with_final_pct"])]
    y = np.arange(len(rates))
    ax2.barh(y, [r[1] for r in rates], .5, color=BLUE, zorder=3)
    for yi, (_, v) in zip(y, rates):
        ax2.text(v - 2, yi, f"{v:.0f}%", va="center", ha="right", fontsize=8.6,
                 fontweight="680", color="white")
    ax2.set_yticks(y, [r[0] for r in rates], fontsize=7.6)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 105)
    ax2.set_xticks(range(0, 101, 25), [f"{v}%" for v in range(0, 101, 25)])
    ax2.grid(axis="y", visible=False)
    strip(ax2)
    ax2.set_title(f"Across all {summary['n']} held-out calls", fontsize=9.6, fontweight="650",
                  color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_latency.png")
    plt.close(fig)
    print("fig_latency.png", summary, f"n_gradual_rows={len(speech)}")


if __name__ == "__main__":
    fig_shortcuts()
    fig_v1_v2()
    fig_latency()
