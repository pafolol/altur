"""
BATCH EVALUATION of the fusion on the held-out calls.

Scores every call of a split through every detection layer ONCE, then re-decides that same set of
scores under many weightings - so the weight sweep costs no model time at all.

    python src/evaluate_fusion.py                     # the 71 held-out val calls, every registered layer
    python src/evaluate_fusion.py --acoustic-model robust_v2
    python src/evaluate_fusion.py --split train --refresh

Writes
    outputs/fusion/validation_layer_scores.json   the per-layer cache the server reads (so the console
                                                  comes up already populated - no 71-call wait in a demo)
    outputs/fusion/<split>_layer_scores.csv       one row per call: every layer's score + the fused verdict
    outputs/fusion/<split>_weight_sweep.json      accuracy / AUC / Brier as the acoustic weight moves 0 -> 1

The validation split was never trained on by either layer, but it HAS been looked at during the
development of both, so treat a sweep maximum as a description of this split and not as a tuned setting
you can promise on the hidden test set. The shipped weights (acoustic 0.50 / behaviour 0.35 / semantic
0.15) are reported first, an even split second.

A layer whose service is not answering still gets its column - every row an abstention - and combine()
hands its share to the others, so the table shows exactly what the deployed fusion would decide today.
"""
import argparse
import json
import time

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import config
import dataset
import fusion

OUT_DIR = config.OUTPUTS_DIR / "fusion"
CACHE_FILE = OUT_DIR / "validation_layer_scores.json"


def load_cache():
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text())
        except Exception:
            pass
    return {}


def score_split(det, df, acoustic_model, refresh=False, cache=None):
    """Run every layer over every call, reusing the on-disk cache unless --refresh."""
    cache = load_cache() if cache is None else cache
    rows = []
    for i, r in enumerate(df.itertuples(), 1):
        key = f"{acoustic_model}|{'+'.join(det.keys())}:{r.anon_id}"
        if refresh or key not in cache:
            t0 = time.perf_counter()
            results = det.score_layers(open(r.path, "rb").read(), anon_id=r.anon_id)
            cache[key] = {"layers": [x.to_dict() for x in results],
                          "scored_ms": round((time.perf_counter() - t0) * 1000),
                          "acoustic_model": acoustic_model}
            state = f"{cache[key]['scored_ms']:5.0f} ms"
        else:
            state = "   cached"
        entry = cache[key]
        line = " ".join(f"{x['key'][:4]}={'abst' if x['abstained'] else format(x['probability'], '.3f')}"
                        for x in entry["layers"])
        print(f"[{i:3d}/{len(df)}] {r.anon_id}  {r.label:9s} {state}  {line}", flush=True)
        rows.append({"anon_id": r.anon_id, "label": r.label, "y": r.y, "duration_s": r.duration_s,
                     "layers": entry["layers"], "scored_ms": entry["scored_ms"]})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, indent=1))
    return rows


def decide(rows, cfg, weights):
    """Re-decide the whole split from cached scores. Pure arithmetic - this is what a slider drag costs."""
    return [fusion.combine(r["layers"], cfg, weights) for r in rows]


def metrics(rows, verdicts):
    y = np.array([r["y"] for r in rows])
    p = np.array([v["synthetic_probability"] for v in verdicts])
    pred = np.array([v["is_synthetic"] for v in verdicts], dtype=int)
    out = {"n": len(y), "accuracy": float((pred == y).mean()),
           "brier": float(np.mean((p - y) ** 2)),
           "n_abstained_calls": int(sum(1 for v in verdicts if not v["decisive"]))}
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    out |= {"confusion_matrix": [[tn, fp], [fn, tp]],
            "fpr": fp / (fp + tn) if fp + tn else None, "fnr": fn / (fn + tp) if fn + tp else None}
    if len(np.unique(y)) == 2:
        from sklearn.metrics import roc_auc_score
        out["roc_auc"] = float(roc_auc_score(y, p))
    return out


def layer_only(rows, key):
    """
    What one layer would decide on its own, for the per-system column of the report.

    Two different numbers, because they answer two different questions:
      accuracy           over the calls the layer was WILLING to judge. An abstention is no answer, not a
                         wrong one - counting it wrong would make an absent service look like a bad model.
      accuracy_forced    over every call, with an abstention falling back to a non-flag at 0.5. This is
                         what the layer alone would actually deliver in production, abstentions included.
    """
    from sklearn.metrics import roc_auc_score
    y = np.array([r["y"] for r in rows])
    p, answered = [], []
    for r in rows:
        x = next((l for l in r["layers"] if l["key"] == key), None)
        blank = x is None or x["abstained"]
        p.append(0.5 if blank else x["probability"])
        answered.append(not blank)
    p, answered = np.array(p), np.array(answered)
    pred = (p >= config.DECISION_THRESHOLD).astype(int)
    out = {"n": len(y), "n_answered": int(answered.sum()), "n_abstained_calls": int((~answered).sum()),
           "accuracy_forced": float((pred == y).mean()), "brier": float(np.mean((p - y) ** 2))}
    if answered.any():
        out["accuracy"] = float((pred[answered] == y[answered]).mean())
        out["brier_answered"] = float(np.mean((p[answered] - y[answered]) ** 2))
        if len(np.unique(y[answered])) == 2:
            out["roc_auc"] = float(roc_auc_score(y[answered], p[answered]))
    else:
        out["accuracy"] = out["roc_auc"] = None
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="val", choices=config.SPLITS)
    parser.add_argument("--acoustic-model", default="wav2vec2_spanish",
                        help="which acoustic model sits in the fusion (wav2vec2_spanish = V1, default)")
    parser.add_argument("--semantic-url", default=None, help="the semantic service's /detect endpoint")
    parser.add_argument("--refresh", action="store_true", help="re-run the layers instead of using the cache")
    parser.add_argument("--mode", default="weighted_mean", choices=fusion.COMBINE_MODES)
    parser.add_argument("--on-abstain", default="renormalise", choices=fusion.ABSTAIN_POLICIES)
    parser.add_argument("--steps", type=int, default=21, help="points in the weight sweep")
    args = parser.parse_args()

    # include_unavailable: the same registry the server serves, so a layer whose service is down still
    # gets its column (every row an abstention) instead of vanishing from the table.
    det = fusion.FusionDetector(layers=fusion.build_layers(args.acoustic_model, include_unavailable=True,
                                                          semantic_url=args.semantic_url))
    if not det.layers:
        raise SystemExit("no detection layer is available: check models/ and behaviour/artifacts/")
    keys = det.keys()
    print(f"[fusion] layers: {', '.join(f'{l.display} [{l.key}]' for l in det.layers)}")
    df = dataset.load_split(args.split)
    print(f"[data]   {args.split}: {len(df)} calls "
          f"({int((df.y == 0).sum())} human / {int((df.y == 1).sum())} synthetic)\n")

    rows = score_split(det, df, args.acoustic_model, args.refresh)
    base = fusion.FusionConfig.from_dict({"mode": args.mode, "on_abstain": args.on_abstain},
                                         det.default_config())
    shipped = base.weight_map()                       # acoustic 0.50 / behaviour 0.35 / semantic 0.15
    equal = {k: 1.0 / len(keys) for k in keys}
    verdicts = decide(rows, base, shipped)

    # -------------------------------------------------------------- per-call table
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = []
    for r, v in zip(rows, verdicts):
        row = {"anon_id": r["anon_id"], "truth": r["label"], "duration_s": r["duration_s"]}
        for l in r["layers"]:
            row[f"{l['key']}_p"] = None if l["abstained"] else round(l["probability"], 6)
            row[f"{l['key']}_quality"] = round(l["quality"], 4)
            row[f"{l['key']}_abstained"] = l["abstained"]
        row |= {"fused_p": v["synthetic_probability"], "verdict": "synthetic" if v["is_synthetic"] else "human",
                "confidence": v["confidence"],
                "correct": bool(v["is_synthetic"] == (r["label"] == "synthetic")),
                "scored_ms": r["scored_ms"]}
        table.append(row)
    csv_path = OUT_DIR / f"{args.split}_layer_scores.csv"
    pd.DataFrame(table).to_csv(csv_path, index=False)

    # -------------------------------------------------------------- the weight sweep (two layers only)
    sweep = []
    if len(keys) == 2:
        a, b = keys
        for w in np.linspace(0, 1, args.steps):
            weights = {a: float(w), b: float(1 - w)}
            m = metrics(rows, decide(rows, base, weights))
            sweep.append({"weights": weights, **m})
    sweep_path = OUT_DIR / f"{args.split}_weight_sweep.json"
    summary = {"split": args.split, "acoustic_model": args.acoustic_model, "mode": args.mode,
               "on_abstain": args.on_abstain, "layers": det.describe(),
               "per_layer_alone": {k: layer_only(rows, k) for k in keys},
               "shipped_weights": {"weights": shipped, **metrics(rows, verdicts)},
               "equal_weights": {"weights": equal, **metrics(rows, decide(rows, base, equal))},
               "sweep": sweep,
               "note": "the val split is held out of training but was seen during development of both "
                       "layers; a sweep maximum describes this split, it is not a tuned setting"}
    sweep_path.write_text(json.dumps(summary, indent=2))

    # -------------------------------------------------------------- report
    print(f"\n{'=' * 78}\n{args.split} split, {len(rows)} calls - {args.acoustic_model} + behaviour, mode={args.mode}\n{'=' * 78}")
    fmt = lambda v: f"{v:7.3f}" if v is not None else "      -"   # ASCII: the Windows console is cp1252
    print(f"{'system':36s} {'acc':>7s} {'AUC':>7s} {'Brier':>7s} {'abstain':>8s}   (acc over the calls it answered)")
    for k in keys:
        m = summary["per_layer_alone"][k]
        display = next(l.display for l in det.layers if l.key == k)
        print(f"{display:36s} {fmt(m['accuracy'])} {fmt(m.get('roc_auc'))} "
              f"{fmt(m.get('brier_answered'))} {m['n_abstained_calls']:8d}"
              + ("   never answered" if not m["n_answered"] else ""))
    for label, m in (("FUSED  " + " / ".join(f"{shipped[k]:.0%}" for k in keys), summary["shipped_weights"]),
                     ("FUSED  " + " / ".join(f"{equal[k]:.0%}" for k in keys), summary["equal_weights"])):
        print(f"{label:36s} {m['accuracy']:7.3f} {m.get('roc_auc', float('nan')):7.3f} {m['brier']:7.3f} "
              f"{m['n_abstained_calls']:8d}    confusion {m['confusion_matrix']}")
    if sweep:
        best = max(sweep, key=lambda s: (s["accuracy"], -s["brier"]))
        print(f"\nweight sweep ({keys[0]} weight -> accuracy):")
        for s in sweep:
            w = s["weights"][keys[0]]
            bar = "#" * int(round(s["accuracy"] * 50))
            print(f"  {keys[0][:9]:>9s}={w:4.2f}  {s['accuracy']:.3f}  {bar}")
        print(f"\nbest on this split: {best['weights']} -> accuracy {best['accuracy']:.3f}, "
              f"AUC {best.get('roc_auc', float('nan')):.3f}  (descriptive only, see note)")
    print(f"\nwrote {csv_path}\n      {sweep_path}\n      {CACHE_FILE}  (the server reads this cache)")


if __name__ == "__main__":
    main()
