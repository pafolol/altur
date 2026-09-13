"""Fase 6: output model + calibration.  Vector = F4 (features.py) + F5 rubric scores.
Logistic regression, class_weight balanced, Platt-calibrated by CV on train. Evaluated on val.

python model.py cv         -> repeated 5-fold x10 AUC on all calls (the rule for comparing experiments; saves nothing)
python model.py            -> table (AUC / accuracy / calibration) for F4, F5, F4+F5; saves model.pkl and
                              scores_semantico.csv (out-of-fold scores for every row); model.pkl is fit on all calls
"""
import csv, json, os, pathlib, pickle, platform, sys
import numpy as np
import sklearn
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
try:
    from . import asr, features, rubric
    from .vad import read_wav, vad
except ImportError:  # standalone scripts run from semantic/
    import asr, features, rubric
    from vad import read_wav, vad

ROOT = pathlib.Path(__file__).parent
PROXY = ("words_per_turn", "n_words")     # turn-count / call-length proxies: behaviour-module territory (PLAN §1.1), see AUDIT.md
EXTRA = os.environ.get("SEMANTIC_EXTRA", "1") == "1"     # include features.EXTRA (set 0 to compare against the 8-feature model)
F4 = [n for n in features.NAMES if n not in PROXY and (EXTRA or n not in features.EXTRA)]
DIMS = ["sobrecompletitud", "registro_formal"]   # the only rubric dims with val AUC > 0.6 (AUDIT.md); the prompt still asks all 7
AGENT = features.AGENT   # agent-transcript features; they live in the F5 block because both need the agent channel
CONFIG = os.environ.get("SEMANTIC_CONFIG", "F4+F5")   # served config, chosen by decision (AUDIT.md), not by argmax on 71 val calls


def vector(f4_dict, scores, agent=None):
    """scores: rubric dict or None; agent: features.f_agent dict or None. The F4 model only reads the first len(F4) columns,
    so the server can call this with (f4, None) on the degraded path; the F4+F5 model always gets both."""
    return [f4_dict[k] for k in F4] + [agent[k] if agent else 0.0 for k in AGENT] + [scores[d]["score"] if scores else 0.5 for d in DIMS]


def provenance():
    return {"rubric_model": rubric.MODEL, "prompt_id": rubric.PROMPT_ID, "scribe_model": asr.MODEL, "chunk_s": asr.CHUNK_S, "f4": F4, "agent": AGENT, "dims": DIMS,
            "sklearn": sklearn.__version__, "python": platform.python_version()}


def make():
    return make_pipeline(StandardScaler(), CalibratedClassifierCV(
        LogisticRegression(class_weight="balanced", max_iter=1000), method="sigmoid", cv=5))


if __name__ == "__main__":
    rows = list(csv.DictReader(open(ROOT / "data/hackmty26/manifest.csv")))
    ids, X, y, split = [], [], [], []
    for r in rows:
        cid = r["anon_id"]
        p = asr.CACHE / f"{cid}_ch0.json"
        g = rubric.cache_path(cid)
        if not p.exists():
            continue
        words = json.load(open(p))["words"]; agent_words = json.load(open(asr.CACHE / f"{cid}_ch1.json"))["words"]
        turns = vad(read_wav(ROOT / "data/hackmty26/audio" / f"{cid}.wav")[0][:, 0])
        scores = json.load(open(g))["scores"] if g.exists() else None
        ids.append(cid); X.append(vector(features.f4(words, turns), scores, features.f_agent(words, agent_words))); y.append(r["label"] == "synthetic"); split.append(r["split"])
    X, y, split = np.array(X), np.array(y), np.array(split)
    tr, va = split == "train", split == "val"
    nf4 = len(F4)
    print(f"calls={len(y)} train={tr.sum()} val={va.sum()}  (rubric={rubric.MODEL}/{rubric.PROMPT_ID}, asr={asr.MODEL}, served config={CONFIG})")
    configs = {"F4": slice(0, nf4), "F5": slice(nf4, None), "F4+F5": slice(0, None)}
    if sys.argv[1:] == ["cv"]:
        # measurement rule for experiments: repeated stratified 5-fold on all calls. 71 val calls give AUC +-0.04 of
        # noise; 10 repeats x 5 folds over 353 calls tightens that to ~+-0.01 on the mean. Nothing is saved.
        from sklearn.model_selection import RepeatedStratifiedKFold
        rs = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
        for name, cols in configs.items():
            aucs, accs = [], []
            for a, b in rs.split(X, y):
                p = make().fit(X[a][:, cols], y[a]).predict_proba(X[b][:, cols])[:, 1]
                aucs.append(roc_auc_score(y[b], p)); accs.append(accuracy_score(y[b], p >= 0.5))
            print(f"  {name:6s} cv AUC={np.mean(aucs):.3f}+-{np.std(aucs):.3f} acc={np.mean(accs):.3f}+-{np.std(accs):.3f}  (50 folds)")
        sys.exit(0)
    fitted = {}
    for name, cols in configs.items():
        m = make().fit(X[tr][:, cols], y[tr]); fitted[name] = (cols, m)
        p = m.predict_proba(X[va][:, cols])[:, 1]
        auc, acc = roc_auc_score(y[va], p), accuracy_score(y[va], p >= 0.5)
        frac, mean = calibration_curve(y[va], p, n_bins=5, strategy="quantile")
        ece = float(np.mean(np.abs(frac - mean)))
        print(f"  {name:6s} val AUC={auc:.3f} acc={acc:.3f}  calibration bins (pred -> real): " +
              " ".join(f"{m_:.2f}->{f_:.2f}" for m_, f_ in zip(mean, frac)) + f"  |mean gap|={ece:.3f}")
    # served model: every config refit on all 353 calls (the train->val table above and `model.py cv` are the measurements;
    # the val split is too small to be worth withholding from the final fit). scores_semantico.csv = out-of-fold for every
    # row (random 5-fold: the manifest has no speaker ids, so repeated speakers can leak across folds; slightly optimistic).
    final = {name: (cols, make().fit(X[:, cols], y)) for name, cols in configs.items()}
    pickle.dump({"config": CONFIG, "models": final, **provenance()}, open(ROOT / "model.pkl", "wb"))
    cols = configs[CONFIG]
    oof = cross_val_predict(make(), X[:, cols], y, cv=5, method="predict_proba")[:, 1]
    with open(ROOT / "scores_semantico.csv", "w") as f:
        f.write("anon_id,split,score\n"); f.writelines(f"{c},{sp},{s:.4f}\n" for c, sp, s in zip(ids, split, oof))
    print(f"saved model.pkl ({CONFIG}, fit on all {len(y)}) and scores_semantico.csv ({len(ids)} rows OOF, AUC={roc_auc_score(y, oof):.3f})")
