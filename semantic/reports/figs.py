"""Figures for the study guide. Reads caches only. python reports/figs.py -> reports/figures/*.png"""
import csv, json, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import features, rubric
from model import make, F4, DIMS
from vad import read_wav, vad, SR
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve

ROOT = pathlib.Path(__file__).resolve().parents[1]; OUT = ROOT / "reports/figures"
HUM, SYN, GRAY, SURF, INK, INK2 = "#2a78d6", "#eb6834", "#8a8983", "#fcfcfb", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.family": "Helvetica", "font.size": 9, "axes.edgecolor": "#c9c8c2", "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 10,
                     "grid.color": "#e6e5df", "grid.linewidth": 0.5, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "legend.frameon": False, "savefig.dpi": 200, "savefig.bbox": "tight"})

rows = list(csv.DictReader(open(ROOT / "data/hackmty26/manifest.csv")))
tab = []
for r in rows:
    cid = r["anon_id"]
    words = json.load(open(ROOT / f"cache/asr/{cid}_ch0.json"))["words"]
    x, _ = read_wav(ROOT / f"data/hackmty26/audio/{cid}.wav"); segs = vad(x[:, 0], SR)
    sc = json.load(open(rubric.CACHE / f"{cid}_{rubric.MODEL}_{rubric.PROMPT_ID}.json"))["scores"]
    lens = np.array([e - s for s, e in segs])
    rec = {"y": r["label"] == "synthetic", "val": r["split"] == "val", "duration_s": float(r["duration_s"]),
           "n_turns": float(len(segs)), "mean_turn_len": float(lens.mean()), "longest_turn_share": float(lens.max() / lens.sum())}
    rec.update({d: sc[d]["score"] for d in rubric.DIMS})
    for T in (30, 60, 90, None):
        f = features.f4([w for w in words if T is None or w["start"] < T], [s for s in segs if T is None or s[0] < T])
        rec.update({(k if T is None else f"{k}@{T}"): v for k, v in f.items()})
    tab.append(rec)
y = np.array([t["y"] for t in tab]); va = np.array([t["val"] for t in tab]); tr = ~va
col = lambda k: np.array([t[k] for t in tab], dtype=float)
def fit(cols):
    X = np.column_stack([col(c) for c in cols]); m = make().fit(X[tr], y[tr]); return m.predict_proba(X[va])[:, 1]
PROXY = ["words_per_turn", "n_words"]; TURN = ["n_turns", "mean_turn_len", "longest_turn_share"]

# 1. per-feature val AUC
names = features.NAMES + rubric.DIMS
aucs = [roc_auc_score(y[va], col(n)[va]) for n in names]
order = np.argsort([max(a, 1 - a) for a in aucs])
fig, ax = plt.subplots(figsize=(7, 4.6))
for i, j in enumerate(order):
    a = aucs[j]; up = a >= 0.5; served = names[j] in F4 + DIMS
    ax.barh(i, max(a, 1 - a), color=(SYN if up else HUM) if served else GRAY, height=0.62)
    ax.text(max(a, 1 - a) + 0.005, i, f"{max(a, 1 - a):.2f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(names))); ax.set_yticklabels([names[j] for j in order], fontsize=8)
ax.axvline(0.5, color=GRAY, lw=0.8, ls="--"); ax.set_xlim(0.45, 1.0); ax.grid(axis="x")
ax.set_xlabel("AUC en val (71 llamadas), tomando la dirección que separa"); ax.set_title("Poder individual de cada feature")
ax.plot([], [], "s", color=SYN, label="sube con sintético (servida)"); ax.plot([], [], "s", color=HUM, label="sube con humano (servida)"); ax.plot([], [], "s", color=GRAY, label="no se sirve")
ax.legend(loc="lower right", fontsize=8); fig.savefig(OUT / "feature_auc.png"); plt.close(fig)

# 2. distributions by class
fig, axs = plt.subplots(2, 2, figsize=(7, 4.6))
for ax, (k, lab) in zip(axs.flat, [("logprob_mean", "confianza media del ASR (log-prob por palabra)"), ("disfluency_rate", "muletillas por palabra"),
                                   ("logprob_std", "desviación de la confianza del ASR"), ("sobrecompletitud", "sobrecompletitud (Gemini, 0-1)")]):
    v = col(k); bins = np.linspace(v.min(), v.max(), 22)
    ax.hist(v[~y], bins=bins, color=HUM, alpha=0.75, label=f"humano (n={(~y).sum()})")
    ax.hist(v[y], bins=bins, color=SYN, alpha=0.75, label=f"sintético (n={y.sum()})")
    ax.set_title(lab, fontsize=9); ax.set_ylabel("llamadas"); ax.grid(axis="y")
axs[0, 0].legend(fontsize=8); fig.suptitle("Las 353 llamadas, por clase", fontweight="bold"); fig.tight_layout(); fig.savefig(OUT / "distributions.png"); plt.close(fig)

# 3. ladder
F4C = [n for n in features.NAMES if n not in PROXY]
ladder = [("dummy", 0.5, GRAY), ("atajo: duración", roc_auc_score(y[va], fit(["duration_s"])), GRAY),
          ("atajo: estad. de turnos", roc_auc_score(y[va], fit(TURN)), GRAY),
          ("F5 rúbrica (2 dims)", roc_auc_score(y[va], fit(DIMS)), SYN),
          ("F4 texto limpio (8)", roc_auc_score(y[va], fit(F4C)), SYN),
          ("F4 limpio + F5  (servido)", roc_auc_score(y[va], fit(F4C + DIMS)), SYN),
          ("F4 con proxies + F5 (descartado)", roc_auc_score(y[va], fit(features.NAMES + rubric.DIMS)), HUM)]
fig, ax = plt.subplots(figsize=(7, 3.4))
for i, (n, a, c) in enumerate(ladder):
    ax.barh(i, a, color=c, height=0.62); ax.text(a + 0.005, i, f"{a:.3f}", va="center", fontsize=8, color=INK2)
ax.set_yticks(range(len(ladder))); ax.set_yticklabels([l[0] for l in ladder], fontsize=8); ax.set_xlim(0.45, 1.0); ax.grid(axis="x")
ax.set_xlabel("AUC en val"); ax.set_title("Escalera de modelos: de nada a lo que se sirve"); fig.savefig(OUT / "ladder.png"); plt.close(fig)

# 4. ROC
fig, ax = plt.subplots(figsize=(4.2, 4.2))
for n, cols, c in [("F4 limpio + F5 (servido)", F4C + DIMS, SYN), ("F4 limpio", F4C, HUM), ("F5 rúbrica", DIMS, GRAY)]:
    p = fit(cols); fpr, tpr, _ = roc_curve(y[va], p); ax.plot(fpr, tpr, color=c, lw=2, label=f"{n}  AUC {roc_auc_score(y[va], p):.3f}")
ax.plot([0, 1], [0, 1], color="#c9c8c2", lw=0.8, ls="--"); ax.set_xlabel("tasa de falsos positivos (humano marcado como bot)")
ax.set_ylabel("tasa de verdaderos positivos (bot detectado)"); ax.set_title("Curvas ROC en val"); ax.legend(fontsize=8, loc="lower right"); ax.grid()
fig.savefig(OUT / "roc.png"); plt.close(fig)

# 5. calibration
p = fit(F4C + DIMS)
fig, axs = plt.subplots(1, 2, figsize=(7, 3.2))
frac, mean = calibration_curve(y[va], p, n_bins=5, strategy="quantile")
axs[0].plot([0, 1], [0, 1], color="#c9c8c2", ls="--", lw=0.8); axs[0].plot(mean, frac, "o-", color=SYN, lw=2, ms=7)
axs[0].set_xlabel("score devuelto"); axs[0].set_ylabel("fracción real de sintéticos"); axs[0].set_title("Calibración (val, 5 bins)"); axs[0].grid()
axs[1].hist(p[~y[va]], bins=np.linspace(0, 1, 11), color=HUM, alpha=0.75, label="humano"); axs[1].hist(p[y[va]], bins=np.linspace(0, 1, 11), color=SYN, alpha=0.75, label="sintético")
axs[1].set_xlabel("score devuelto"); axs[1].set_ylabel("llamadas"); axs[1].set_title("Distribución del score por clase"); axs[1].legend(fontsize=8); axs[1].grid(axis="y")
fig.tight_layout(); fig.savefig(OUT / "calibration.png"); plt.close(fig)

# 6. prefix
xs, ys = [30, 60, 90, 148], []
for T in (30, 60, 90, None):
    ys.append(roc_auc_score(y[va], fit([f"{k}@{T}" if T else k for k in F4C])))
fig, ax = plt.subplots(figsize=(5.5, 3))
ax.plot(xs, ys, "o-", color=SYN, lw=2, ms=7); [ax.text(x_, y_ + 0.004, f"{y_:.3f}", ha="center", fontsize=8, color=INK2) for x_, y_ in zip(xs, ys)]
ax.set_xticks(xs); ax.set_xticklabels(["30 s", "60 s", "90 s", "completa\n(media 148 s)"]); ax.set_ylim(0.85, 0.96); ax.grid(axis="y")
ax.set_xlabel("cuánto de la llamada se usa"); ax.set_ylabel("AUC en val (F4 limpio)"); ax.set_title("Los primeros 30 s ya discriminan"); fig.savefig(OUT / "prefix.png"); plt.close(fig)

# 7. latency budget (measured numbers, see FINDINGS.md)
fig, ax = plt.subplots(figsize=(7, 2.6))
stages = [("decode + VAD\n+ modelo", 0.04, GRAY), ("Scribe, ambos canales\n(paralelo, hedged)", 1.24, HUM), ("Gemini flash-lite\n(rúbrica, solo scores)", 0.69, SYN)]
left = 0
for n, d, c in stages:
    ax.barh(0, d, left=left, color=c, height=0.5, edgecolor=SURF, linewidth=2); ax.text(left + d / 2, 0.42, f"{n}\n{d:.2f} s", ha="center", va="bottom", fontsize=8, color=INK2); left += d
ax.axvline(2.8, color=INK, lw=1, ls="--"); ax.text(2.82, -0.05, "plazo de red 2.8 s", fontsize=8, color=INK, va="center")
ax.axvline(2.08, color=SYN, lw=1, ls=":"); ax.text(2.1, -0.32, "p50 real 2.08 s", fontsize=8, color=SYN)
ax.set_xlim(0, 3.2); ax.set_ylim(-0.5, 1.1); ax.set_yticks([]); ax.set_xlabel("segundos"); ax.set_title("Presupuesto de latencia (medias medidas)"); ax.spines["left"].set_visible(False)
fig.savefig(OUT / "latency.png"); plt.close(fig)

# 8. timeline of one call: raw vs compacted
x, _ = read_wav(ROOT / "data/hackmty26/audio/call_acda78859d0d.wav"); T0, T1 = 0, 60
fig, axs = plt.subplots(2, 1, figsize=(7, 2.8), sharex=False)
for ch, c, n in ((1, GRAY, "agente (canal 1)"), (0, SYN, "cliente (canal 0)")):
    for s, e in vad(x[:, ch], SR):
        if s < T1: axs[0].barh(ch, min(e, T1) - s, left=s, color=c, height=0.6)
axs[0].set_yticks([0, 1]); axs[0].set_yticklabels(["cliente", "agente"]); axs[0].set_xlim(T0, T1); axs[0].set_title("Una llamada real: segmentos de voz que encuentra el VAD (primeros 60 s)"); axs[0].set_xlabel("segundos en la llamada")
segs = [s for s in vad(x[:, 0], SR) if s[0] < T1]; t = 0
for s, e in segs:
    axs[1].barh(0, e - s, left=t, color=SYN, height=0.6); t += e - s + 0.5
axs[1].set_yticks([0]); axs[1].set_yticklabels(["cliente\ncompactado"]); axs[1].set_xlim(T0, T1); axs[1].set_xlabel("segundos de audio que se mandan al ASR (0.5 s de silencio entre segmentos)")
axs[1].set_title(f"Lo mismo compactado: {t:.0f} s en vez de 60 s, y sin silencios donde el ASR alucina", fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "compaction.png"); plt.close(fig)
print("figures:", sorted(p.name for p in OUT.glob("*.png")))
