"""
Train-fit validation scores for the fusion layer.  (Added by the fusion integration; nothing in the
module is modified — this only imports it.)

    .venv/Scripts/python holdout_val.py        ->  scores_val_trainfit.csv   (anon_id,split,score)

WHY THIS EXISTS.  `scores_semantico.csv` is produced by `cross_val_predict(..., cv=5)` over all 353 calls.
Every row is out-of-fold for itself, but a random 5-fold ignores the official split, which is
SPEAKER-DISJOINT — model.py's own comment says so: "the manifest has no speaker ids, so repeated speakers
can leak across folds; slightly optimistic".  Scoring the 71 official validation calls that way reads
about AUC 0.97, while the same configuration trained on the train split and evaluated on val reads 0.921
in AUDIT.md.  The fusion console must show the second number, so this writes it.

It is the same estimator and the same feature vector as model.py — `model.make()`, `model.vector()`,
`model.CONFIG` — fitted on the train split alone and applied to val.  It reads only the caches and the
audio: no ElevenLabs, no Gemini, no network.  It writes ONE new file and overwrites neither
`model.pkl` nor `scores_semantico.csv`.
"""
import csv
import json
import pathlib

import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score

import asr
import features
import model
import rubric
from vad import read_wav, vad

ROOT = pathlib.Path(__file__).parent
OUT = ROOT / "scores_val_trainfit.csv"


def build():
    rows = list(csv.DictReader(open(ROOT / "data/hackmty26/manifest.csv")))
    ids, X, y, split = [], [], [], []
    for i, r in enumerate(rows, 1):
        cid = r["anon_id"]
        caller = asr.CACHE / f"{cid}_ch0.json"
        agent = asr.CACHE / f"{cid}_ch1.json"
        if not caller.exists():
            continue
        words = json.load(open(caller))["words"]
        agent_words = json.load(open(agent))["words"] if agent.exists() else []
        turns = vad(read_wav(ROOT / "data/hackmty26/audio" / f"{cid}.wav")[0][:, 0])
        g = rubric.cache_path(cid)
        scores = json.load(open(g))["scores"] if g.exists() else None
        ids.append(cid)
        X.append(model.vector(features.f4(words, turns), scores, features.f_agent(words, agent_words)))
        y.append(r["label"] == "synthetic")
        split.append(r["split"])
        if i % 50 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
    return np.array(ids), np.array(X), np.array(y), np.array(split)


def main():
    ids, X, y, split = build()
    tr, va = split == "train", split == "val"
    cols = {"F4": slice(0, len(model.F4)), "F5": slice(len(model.F4), None), "F4+F5": slice(0, None)}[model.CONFIG]
    fitted = model.make().fit(X[tr][:, cols], y[tr])
    p = fitted.predict_proba(X[va][:, cols])[:, 1]
    print(f"\nconfig={model.CONFIG}  train={tr.sum()} val={va.sum()}")
    print(f"  val AUC      {roc_auc_score(y[va], p):.3f}   (AUDIT.md reports 0.921 for this configuration)")
    print(f"  val accuracy {accuracy_score(y[va], p >= 0.5):.3f}")
    print(f"  val Brier    {brier_score_loss(y[va], p):.3f}")
    with open(OUT, "w", newline="") as f:
        f.write("anon_id,split,score\n")
        f.writelines(f"{c},val,{s:.4f}\n" for c, s in zip(ids[va], p))
    print(f"\nwrote {OUT.name} ({va.sum()} rows, trained on the {tr.sum()} train calls only)")


if __name__ == "__main__":
    main()
