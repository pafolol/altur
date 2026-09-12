"""Fase 4: text features without an LLM, from the cached caller transcript (Scribe words with logprob).
No turn statistics (§1.1): words_per_turn is the only turn-normalized value and is kept separable.

f4(words, turns) -> dict of floats;  NAMES = feature order
"""
import re
import numpy as np

DISFL = {"este", "eh", "em", "mmm", "mm", "ah", "ehm", "pues", "bueno", "o sea", "osea", "digo", "ay", "ándale", "andale"}
MEX = {"mande", "ándale", "andale", "órale", "orale", "ahorita", "fíjese", "fijese", "checar", "sale", "chido", "híjole",
       "hijole", "nomás", "nomas", "ni modo", "güey", "wey", "mero", "chance", "padre", "oiga", "oye"}
EXTRA = ["logprob_min", "logprob_low_frac", "logprob_first30", "guion_rate"]   # Scribe confidence distribution + one script tic (FINDINGS)
AGENT = ["overlap_agent"]   # needs the agent transcript: served only on the f4+f5 path (model.py puts it with the rubric dims)
NAMES = ["logprob_mean", "logprob_std", "logprob_p10", "disfluency_rate", "mexicanism_rate", "false_start_rate",
         "ttr", "mean_sentence_len", "words_per_turn", "n_words"] + EXTRA


def _toks(words):
    return [t for t in (re.sub(r"[^\wáéíóúñü]", "", w["text"].lower()) for w in words) if t]


def f4(words, turns):
    toks = _toks(words)
    n = max(len(toks), 1)
    text = " ".join(toks)
    lp = np.array([w["logprob"] for w in words]) if words else np.zeros(1)   # KeyError on purpose: an ASR without logprob would silently zero the strongest features
    raw = " ".join(w["text"] for w in words)
    sentences = [s for s in re.split(r"[.!?¿¡]+", raw) if s.strip()]
    return {
        "logprob_mean": float(lp.mean()), "logprob_std": float(lp.std()), "logprob_p10": float(np.percentile(lp, 10)),
        "disfluency_rate": sum(t in DISFL for t in toks) / n + text.count("o sea") / n,
        "mexicanism_rate": sum(t in MEX for t in toks) / n + sum(text.count(p) for p in ("ni modo",)) / n,
        "false_start_rate": sum("--" in w["text"] or w["text"].endswith("-") for w in words) / n,   # Scribe marks cut-offs "el--"
        "ttr": len(set(toks)) / n,
        "mean_sentence_len": n / max(len(sentences), 1),
        "words_per_turn": n / max(len(turns), 1),
        "n_words": float(n),
        "logprob_min": float(lp.min()),
        "logprob_low_frac": float((lp < -0.05).mean()),                       # share of words the ASR was unsure about
        "logprob_first30": float(np.mean([w["logprob"] for w in words if w["start"] < 30] or [lp.mean()])),   # AUDIT §3: the opening separates best
        "guion_rate": sum(t in ("guion", "guión") for t in toks) / n,   # the bot reads the hyphen of its reference number aloud (72/203 bots vs 14/150 humans)
    }


def f_agent(words, agent_words):
    """Grounding: share of the caller's vocabulary the agent also said. Humans echo the agent to confirm; bots follow their script."""
    t0, t1 = set(_toks(words)), set(_toks(agent_words))
    return {"overlap_agent": len(t0 & t1) / max(len(t0), 1)}


if __name__ == "__main__":
    # Fase 4 acceptance: per-feature AUC on val, and a logistic regression on F4 alone (train -> val).
    import csv, json, pathlib
    from vad import read_wav, vad
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    root = pathlib.Path(__file__).parent
    rows = list(csv.DictReader(open(root / "data/hackmty26/manifest.csv")))
    X, y, split = [], [], []
    for r in rows:
        p = root / "cache/asr" / f"{r['anon_id']}_ch0.json"
        if not p.exists():
            continue
        w = json.load(open(p))["words"]
        turns = vad(read_wav(root / "data/hackmty26/audio" / f"{r['anon_id']}.wav")[0][:, 0])   # same source as inference
        X.append([f4(w, turns)[k] for k in NAMES]); y.append(r["label"] == "synthetic"); split.append(r["split"])
    X, y, split = np.array(X), np.array(y), np.array(split)
    tr, va = split == "train", split == "val"
    print(f"calls with transcript: {len(y)}  (train {tr.sum()}, val {va.sum()})")
    for i, k in enumerate(NAMES):
        a = roc_auc_score(y[va], X[va, i]) if va.sum() else float("nan")
        print(f"  {k:18s} val AUC={max(a, 1 - a):.3f} ({'↑synth' if a >= 0.5 else '↓synth'})  human={X[~y, i].mean():.3f} synth={X[y, i].mean():.3f}")
    if va.sum() and tr.sum():
        m = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)).fit(X[tr], y[tr])
        print(f"F4 logistic regression: val AUC={roc_auc_score(y[va], m.predict_proba(X[va])[:, 1]):.3f}")
