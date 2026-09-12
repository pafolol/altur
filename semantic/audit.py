"""Methodology audit of the semantic module (same ladder the behaviour prompt demands):
dummy -> shortcut baselines (duration, turn stats) -> F4 without turn/length proxies -> F4 -> F4+F5,
per-family ablation, train/val gap, calibration (Brier), confusion matrix, call-prefix experiment,
redundancy with behaviour-module turn statistics, and val error listing. Reads only caches; no API calls.
"""
import csv, json, pathlib, sys
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, balanced_accuracy_score, precision_score, recall_score,
                             f1_score, brier_score_loss, confusion_matrix)
import asr, features, rubric
from model import make, F4
from vad import read_wav, vad, SR

ROOT = pathlib.Path(__file__).parent
rows = list(csv.DictReader(open(ROOT / "data/hackmty26/manifest.csv")))

# ---- build the table once: F4 (full call and prefixes), rubric, shortcut stats
PREFIXES = [30, 60, 90, None]
tab = []
for r in rows:
    cid = r["anon_id"]
    words = json.load(open(asr.CACHE / f"{cid}_ch0.json"))["words"]
    x, _ = read_wav(ROOT / "data/hackmty26/audio" / f"{cid}.wav")
    segs = vad(x[:, 0], SR)
    g = rubric.cache_path(cid)
    sc = json.load(open(g))["scores"]
    lens = np.array([e - s for s, e in segs])
    rec = {"id": cid, "y": r["label"] == "synthetic", "val": r["split"] == "val",
           "duration_s": float(r["duration_s"]), "caller_voice_s": float(lens.sum()), "n_turns": float(len(segs)),
           "mean_turn_len": float(lens.mean()), "longest_turn_share": float(lens.max() / lens.sum()),
           "rubric": {d: sc[d]["score"] for d in rubric.DIMS}}
    for T in PREFIXES:
        w = [w_ for w_ in words if T is None or w_["start"] < T]
        s = [sg for sg in segs if T is None or sg[0] < T]
        rec[f"f4@{T or 'full'}"] = features.f4(w, s) if w else {k: np.nan for k in features.NAMES}
    tab.append(rec)
y = np.array([t["y"] for t in tab]); va = np.array([t["val"] for t in tab]); tr = ~va

def X_of(cols, prefix="full"):
    out = []
    for t in tab:
        v = []
        for c in cols:
            if c in features.NAMES: v.append(t[f"f4@{prefix}"][c])
            elif c in rubric.DIMS: v.append(t["rubric"][c])
            else: v.append(t[c])
        out.append(v)
    X = np.array(out, dtype=float)
    return np.nan_to_num(X, nan=np.nanmean(X[tr], axis=0)) if np.isnan(X).any() else X   # ponytail: mean-impute from train

def evaluate(name, cols, prefix="full", model=None):
    X = X_of(cols, prefix)
    m = (model or make()).fit(X[tr], y[tr])
    p_tr, p = m.predict_proba(X[tr])[:, 1], m.predict_proba(X[va])[:, 1]
    yhat = p >= 0.5
    tn, fp, fn, tp = confusion_matrix(y[va], yhat).ravel()
    res = dict(name=name, n_feat=len(cols), auc_tr=roc_auc_score(y[tr], p_tr), auc=roc_auc_score(y[va], p),
               acc=accuracy_score(y[va], yhat), bacc=balanced_accuracy_score(y[va], yhat),
               prec=precision_score(y[va], yhat, zero_division=0), rec=recall_score(y[va], yhat), f1=f1_score(y[va], yhat),
               brier=brier_score_loss(y[va], p), fpr=fp / (fp + tn), fnr=fn / (fn + tp), cm=(tn, fp, fn, tp))
    print(f"  {name:34s} k={len(cols):2d} AUC tr/val={res['auc_tr']:.3f}/{res['auc']:.3f} acc={res['acc']:.3f} bacc={res['bacc']:.3f} "
          f"P={res['prec']:.2f} R={res['rec']:.2f} F1={res['f1']:.2f} Brier={res['brier']:.3f} FPR={res['fpr']:.2f} FNR={res['fnr']:.2f} cm(tn,fp,fn,tp)={res['cm']}")
    return res, p

LOGPROB = ["logprob_mean", "logprob_std", "logprob_p10"]
STYLE = ["disfluency_rate", "mexicanism_rate", "false_start_rate", "ttr", "mean_sentence_len"]
PROXY = ["words_per_turn", "n_words"]                 # use the turn count / call length: behaviour-module territory
F4_CLEAN = LOGPROB + STYLE
TURN = ["n_turns", "mean_turn_len", "longest_turn_share"]
F5_KEEP = ["sobrecompletitud", "registro_formal"]     # the two rubric dims with val AUC > 0.6

print(f"val prior: {y[va].mean():.3f} synthetic  (train {y[tr].mean():.3f})\n")
print("== ladder (positive class = synthetic; val = 71 speaker-disjoint calls) ==")
d = DummyClassifier(strategy="most_frequent").fit(np.zeros((tr.sum(), 1)), y[tr])
print(f"  {'dummy (most frequent)':34s} acc={accuracy_score(y[va], d.predict(np.zeros((va.sum(), 1)))):.3f} bacc=0.500 AUC=0.500")
R = {}
for name, cols in [("shortcut: duration", ["duration_s"]), ("shortcut: caller voice s", ["caller_voice_s"]),
                   ("shortcut: turn stats (behaviour)", TURN), ("shortcut: duration+voice+turns", ["duration_s", "caller_voice_s"] + TURN),
                   ("F4 clean (no turn/length proxy)", F4_CLEAN), ("F4 full (as shipped)", F4),
                   ("F5 all 7 dims", rubric.DIMS), ("F5 2 useful dims", F5_KEEP),
                   ("F4 clean + F5 (2 dims)", F4_CLEAN + F5_KEEP), ("F4 clean + F5 (7 dims)", F4_CLEAN + rubric.DIMS),
                   ("F4 full + F5 (7 dims, shipped)", F4 + rubric.DIMS)]:
    R[name] = evaluate(name, cols)

print("\n== ablation: remove one family from F4 clean + F5 (2 dims) ==")
full = F4_CLEAN + F5_KEEP
for fam, cols in [("logprob", LOGPROB), ("style/lexical", STYLE), ("rubric", F5_KEEP)]:
    evaluate(f"  - {fam}", [c for c in full if c not in cols])
print("\n== single families ==")
for fam, cols in [("logprob only", LOGPROB), ("style/lexical only", STYLE), ("proxies only (words_per_turn, n_words)", PROXY)]:
    evaluate(fam, cols)

print("\n== redundancy with the behaviour module: |Pearson r| between our features and turn stats (all 353 calls) ==")
Xa = X_of(F4 + F5_KEEP + TURN)
names = F4 + F5_KEEP
for i, n in enumerate(names):
    rs = [abs(np.corrcoef(Xa[:, i], Xa[:, len(names) + j])[0, 1]) for j in range(len(TURN))]
    flag = "  <-- shares >0.5 with a turn statistic" if max(rs) > 0.5 else ""
    print(f"  {n:20s} " + " ".join(f"{t}={r_:.2f}" for t, r_ in zip(TURN, rs)) + flag)

print("\n== call-prefix experiment (F4 clean + F5 uses the full-call rubric; F4 clean alone is prefix-only) ==")
for T in PREFIXES:
    evaluate(f"F4 clean @ first {T or 'full'} s", F4_CLEAN, prefix=T or "full")

print("\n== val errors of F4 clean + F5 (2 dims) at threshold 0.5 ==")
_, p = evaluate("F4 clean + F5 (2 dims)", full)
ids = [t["id"] for t in tab]; vi = np.flatnonzero(va)
for j in vi[np.argsort(-np.abs(p - y[va]))][:8]:
    if (p[list(vi).index(j)] >= 0.5) != y[j]:
        t = tab[j]; f = t["f4@full"]
        print(f"  {t['id']} label={'synthetic' if t['y'] else 'human':9s} p={p[list(vi).index(j)]:.2f} words={f['n_words']:.0f} "
              f"logprob_mean={f['logprob_mean']:.3f} std={f['logprob_std']:.3f} disfl={f['disfluency_rate']:.3f} "
              f"sobrecomp={t['rubric']['sobrecompletitud']:.1f} formal={t['rubric']['registro_formal']:.1f} n_turns={t['n_turns']:.0f}")
