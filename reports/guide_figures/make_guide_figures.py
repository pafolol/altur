"""Figures for reports/ISISI_PRESENTER_GUIDE.md that do not exist elsewhere in the repo.

Every number is copied from README.md (Results), semantic/AUDIT.md, behaviour/reports/metrics/latency.json
and reports/best_model.json; the two timelines are rebuilt from the Scribe word timestamps in
semantic/cache/asr_scribe_v2/. Needs matplotlib only:

    python -m venv .venv && .venv/bin/pip install matplotlib && .venv/bin/python reports/guide_figures/make_guide_figures.py

Then regenerate the PDF from the markdown (see the guide header).
"""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "guide_figures"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
                     "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})

def clean(ax, x=True, y=False):
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(length=0)
    if x: ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    if y: ax.yaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)

def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.15); plt.close(fig); print("wrote", name)

# 1. fusion systems -------------------------------------------------------------
systems = ["Acoustic V1 alone", "Behaviour alone", "Semantic alone", "Fused: 50/50 + verifier", "Flat 3-way vote, no gate"]
acc = [100.0, 97.2, 91.5, 100.0, 100.0]; brier = [0.000, 0.041, 0.070, 0.010, 0.009]
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.0), gridspec_kw={"wspace": 0.55})
y = list(range(len(systems)))[::-1]
cols = [BLUE, BLUE, BLUE, INK, BLUE]
a.barh(y, acc, color=cols, height=0.5); clean(a); a.set_yticks(y, systems); a.set_xlim(85, 103)
a.set_title("Accuracy on the 71 held-out calls"); a.set_xlabel("% of calls right")
for yi, v in zip(y, acc): a.text(v + 0.3, yi, f"{v:.1f}%", va="center", color=INK, fontsize=8.5)
b.barh(y, brier, color=cols, height=0.5); clean(b); b.set_yticks(y, [""] * 5); b.set_xlim(0, 0.09)
b.set_title("Brier score (lower is better)"); b.set_xlabel("mean squared error of the probability")
for yi, v in zip(y, brier): b.text(v + 0.002, yi, f"{v:.3f}", va="center", color=INK, fontsize=8.5)
save(fig, "fusion_systems.png")

# 2. the five escalations ------------------------------------------------------
calls = [("0847d7417bb1", "synthetic", 1.000, 0.435, 0.952, 0.748),
         ("569ffb0869eb", "human", 0.000, 0.964, 0.046, 0.425),
         ("678ee1dd2242", "human", 0.001, 0.476, 0.178, 0.231),
         ("95ccefc6e4ce", "human", 0.013, 0.428, 0.280, 0.228),
         ("b658b1155216", "human", 0.000, 0.434, 0.390, 0.240)]
fig, ax = plt.subplots(figsize=(9, 3.4))
y = list(range(len(calls)))[::-1]
ax.axvspan(0.2, 0.8, color=GRID, alpha=0.45, lw=0)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3)))
for yi, (cid, truth, ac, be, se, fu) in zip(y, calls):
    ax.plot([min(ac, be, se, fu), max(ac, be, se, fu)], [yi, yi], color=GRID, lw=1.2, zorder=1)
    ax.scatter([ac], [yi], s=70, color=BLUE, zorder=3, edgecolor=SURF, lw=1.5)
    ax.scatter([be], [yi], s=70, color=ORANGE, zorder=3, edgecolor=SURF, lw=1.5)
    ax.scatter([se], [yi], s=70, color=AQUA, zorder=3, edgecolor=SURF, lw=1.5)
    ax.scatter([fu], [yi], s=95, marker="D", color=INK, zorder=4, edgecolor=SURF, lw=1.5)
    ax.text(fu, yi - 0.36, f"fused {fu:.3f}", ha="center", fontsize=8, color=INK)
ax.set_yticks(y, [f"{c[0]}  ·  {c[1]}" for c in calls]); ax.set_xlim(-0.03, 1.03); ax.set_ylim(-0.7, 4.5); clean(ax)
ax.set_xlabel("probability of synthetic (0.5 = decision threshold; shaded = where the primaries' average would be under the 80% gate)")
ax.set_title("The five held-out calls the primaries could not settle, and what every layer said")
ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, label="acoustic"), Line2D([], [], marker="o", ls="", color=ORANGE, label="behaviour"),
                   Line2D([], [], marker="o", ls="", color=AQUA, label="semantic (verifier)"), Line2D([], [], marker="D", ls="", color=INK, label="fused verdict")],
          loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=4, frameon=False, fontsize=8.5)
save(fig, "escalations.png")

# 3. how the verdict adds up ----------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 3.2))
rows = [("call_0847d7417bb1 (synthetic)\nstage 1, primaries only", [(0.5, 1.000, BLUE, "acoustic"), (0.5, 0.435, ORANGE, "behaviour")]),
        ("call_0847d7417bb1 (synthetic)\nstage 2, verifier added", [(0.5/1.15, 1.000, BLUE, "acoustic"), (0.5/1.15, 0.435, ORANGE, "behaviour"), (0.15/1.15, 0.952, AQUA, "semantic")]),
        ("call_569ffb0869eb (human)\nstage 1, primaries only", [(0.5, 0.000, BLUE, "acoustic"), (0.5, 0.964, ORANGE, "behaviour")]),
        ("call_569ffb0869eb (human)\nstage 2, verifier added", [(0.5/1.15, 0.000, BLUE, "acoustic"), (0.5/1.15, 0.964, ORANGE, "behaviour"), (0.15/1.15, 0.046, AQUA, "semantic")])]
y = [3.2, 2.2, 0.9, -0.1]
ends = {}
for yi, (label, segs) in zip(y, rows):
    left = 0
    for share, p, col, name in segs:
        w = share * p
        ax.barh(yi, w, left=left, color=col, height=0.55, edgecolor=SURF, lw=1.5)
        if w > 0.06: ax.text(left + w / 2, yi, f"{share:.2f}×{p:.3f}", ha="center", va="center", color="white", fontsize=7.5)
        left += w
    ends[yi] = left
verdicts = ["= 0.718 → 72% confident, under the gate", "= 0.748 → synthetic, 75% confident", "= 0.482 → 52% confident, under the gate", "= 0.425 → human, 57% confident"]
for yi, v in zip(y, verdicts): ax.text(ends[yi] + 0.012, yi, v, va="center", fontsize=8.5, color=INK, bbox=dict(facecolor=SURF, edgecolor="none", pad=1.5))
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3))); ax.text(0.5, 3.75, "0.5 threshold", ha="center", fontsize=8, color=INK2)
ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 1.0); ax.set_ylim(-0.6, 3.9); clean(ax)
ax.set_xlabel("each layer's share of the vote × its probability; the segments add up to the fused probability of synthetic")
ax.set_title("How the two demo verdicts add up (weights 0.50 / 0.50, then 0.50 / 0.50 / 0.15 renormalised)")
save(fig, "verdict_adds_up.png")

# 4. latency budget + gate split -----------------------------------------------
fig, (a, b, c) = plt.subplots(1, 3, figsize=(9.5, 2.9), gridspec_kw={"width_ratios": [1.3, 1.3, 0.9], "wspace": 0.6})
a.barh([1], [1.0], color=BLUE, height=0.5); a.barh([0], [1.0], color=BLUE, height=0.5); a.barh([0], [2.9], left=1.0, color=AQUA, height=0.5, edgecolor=SURF, lw=1.5)
a.text(1.05, 1, "≈ 1.0 s", va="center", fontsize=9, color=INK); a.text(3.95, 0, "≈ 3.9 s", va="center", fontsize=9, color=INK)
a.text(0.5, 0.42, "primaries", ha="center", va="bottom", color=INK2, fontsize=7.5); a.text(2.45, 0.42, "+ semantic verifier", ha="center", va="bottom", color=INK2, fontsize=7.5)
a.set_yticks([1, 0], ["settled by the primaries\n(66 of 71 calls)", "escalated to the verifier\n(5 of 71 calls)"]); a.set_xlim(0, 5); a.set_ylim(-0.5, 1.6); clean(a)
a.set_xlabel("seconds from request to verdict, live /detect"); a.set_title("End-to-end time per call")
layers = ["acoustic, GPU", "acoustic, CPU", "behaviour, CPU (p50)", "semantic service (p50)"]; lat = [0.12, 0.85, 0.86, 2.1]
cols = [BLUE, BLUE, ORANGE, AQUA]; y = [3, 2, 1, 0]
b.barh(y, lat, color=cols, height=0.5); clean(b); b.set_yticks(y, layers); b.set_xlim(0, 2.9)
for yi, v in zip(y, lat): b.text(v + 0.05, yi, f"{v:.2f} s", va="center", fontsize=8.5, color=INK)
b.set_xlabel("seconds per call, one layer alone"); b.set_title("Each layer on its own")
c.barh([0], [66], color=BLUE, height=0.5); c.barh([0], [5], left=66, color=AQUA, height=0.5, edgecolor=SURF, lw=1.5)
c.text(33, 0, "66 settled", ha="center", va="center", color="white", fontsize=8); c.text(66 + 2.5, 0.42, "5 asked the\nverifier", ha="center", fontsize=8, color=INK)
c.set_yticks([]); c.set_xlim(0, 71); c.set_ylim(-0.6, 0.9); clean(c); c.set_xlabel("71 held-out calls"); c.set_title("Where the paid layer is spent")
save(fig, "latency_budget.png")

# 5. semantic ladder -----------------------------------------------------------
ladder = [("dummy: majority class", 0.500), ("shortcut: call duration", 0.517), ("shortcut: seconds of caller speech", 0.451),
          ("shortcut: turn statistics (3)", 0.734), ("F4 text features, clean (8)", 0.912), ("Gemini rubric, 7 dimensions", 0.832),
          ("F4 clean + 2 rubric dims  ← served", 0.921), ("F4 full + 7 dims (leaks turn length)", 0.955)]
fig, ax = plt.subplots(figsize=(8.5, 3.4))
y = list(range(len(ladder)))[::-1]
cols = [MUTED, MUTED, MUTED, MUTED, AQUA, AQUA, INK, MUTED]
ax.barh(y, [v for _, v in ladder], color=cols, height=0.55); clean(ax)
ax.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3))); ax.text(0.505, -0.75, "chance", ha="left", fontsize=8, color=INK2)
ax.set_yticks(y, [n for n, _ in ladder]); ax.set_xlim(0.4, 1.0); ax.set_ylim(-1.0, len(ladder) - 0.4)
for yi, (_, v) in zip(y, ladder): ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8.5, color=INK)
ax.set_xlabel("validation ROC AUC, 71 calls (semantic/AUDIT.md, section 1)"); ax.set_title("The semantic module's own ladder: from no model to the served one")
save(fig, "semantic_ladder.png")

# 6. semantic prefix + TTS probe ------------------------------------------------
fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.1), gridspec_kw={"wspace": 0.45})
px = [30, 60, 90, 120]; pv = [0.937, 0.915, 0.883, 0.912]
a.plot(px, pv, color=AQUA, lw=2, marker="o", ms=7, mec=SURF, mew=1.5); clean(a, x=False, y=True)
a.set_xticks(px, ["first 30 s", "first 60 s", "first 90 s", "whole call"]); a.set_ylim(0.85, 0.96)
for x, v in zip(px, pv): a.text(x, v + (0.004 if v != min(pv) else -0.011), f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
a.set_ylabel("validation AUC, text features only"); a.set_title("The first 30 s already carry the signal")
voices = ["Paulina", "Eddy", "Flo", "Grandma", "Grandpa"]; bot = [0.96, 0.92, 0.94, 0.95, 0.95]; hum = [0.57, 0.59, 0.79, 0.28, 0.66]
y = list(range(5))[::-1]
for yi, bv, hv in zip(y, bot, hum):
    b.plot([hv, bv], [yi, yi], color=GRID, lw=1.5, zorder=1)
    b.scatter([hv], [yi], s=70, color=BLUE, zorder=3, edgecolor=SURF, lw=1.5); b.scatter([bv], [yi], s=70, color=ORANGE, zorder=3, edgecolor=SURF, lw=1.5)
b.axvline(0.5, color=INK2, lw=1, ls=(0, (3, 3))); b.set_yticks(y, voices); b.set_xlim(0, 1.05); clean(b)
b.set_xlabel("P(synthetic) returned by the live service"); b.set_title("Same Apple TTS voice, two scripts")
b.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, label="bot-like script (mean 0.94)"), Line2D([], [], marker="o", ls="", color=BLUE, label="human-like script (mean 0.58)")],
         loc="lower center", bbox_to_anchor=(0.5, -0.5), ncol=1, frameon=False, fontsize=8.5)
save(fig, "semantic_prefix_tts.png")

# 7. demo call timelines --------------------------------------------------------
def segments(call, ch, gap=0.5):
    words = [w for w in json.load(open(ROOT / "semantic/cache/asr_scribe_v2" / f"{call}_ch{ch}.json"))["words"] if w.get("type") == "word"]
    segs = []
    for w in sorted(words, key=lambda w: w["start"]):
        if segs and w["start"] - segs[-1][1] < gap: segs[-1][1] = max(segs[-1][1], w["end"])
        else: segs.append([w["start"], w["end"]])
    return segs
fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.0), sharex=True, gridspec_kw={"hspace": 0.55})
for ax, (call, label, col, notes) in zip(axes, [
        ("call_0847d7417bb1", "call_0847d7417bb1 · synthetic caller", ORANGE, [(67.3, 76.4, "agent cuts in at 67 s; the caller keeps talking for 9 s more")]),
        ("call_569ffb0869eb", "call_569ffb0869eb · human caller (Lucía)", BLUE, [(26.3, 28.0, "answers over the agent; both stop, both restart"), (79.9, 79.9, "agent cuts in; she had already stopped")])]):
    for s, e in segments(call, 1): ax.barh(1, e - s, left=s, height=0.5, color=MUTED)
    for s, e in segments(call, 0): ax.barh(0, e - s, left=s, height=0.5, color=col)
    for s, e, text in notes:
        ax.axvspan(s, max(e, s + 0.3), color=INK, alpha=0.12, lw=0)
        ax.annotate(text, xy=((s + e) / 2, 0.3), xytext=((s + e) / 2 + (8 if s < 100 else -8), 1.75), fontsize=8, color=INK, ha="left" if s < 100 else "right",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))
    ax.set_yticks([1, 0], ["agent (ch 1)", "caller (ch 0)"]); ax.set_ylim(-0.5, 2.2); clean(ax); ax.set_title(label, loc="left")
axes[1].set_xlim(0, 125); axes[1].set_xlabel("seconds into the call · bars = speech, from the Scribe word timestamps in semantic/cache")
save(fig, "demo_timelines.png")
