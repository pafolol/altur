"""Fixed, small experiment ladder using the official split. No hidden-test access."""

import argparse
from time import perf_counter
import joblib
import pandas as pd
from behavior.config import ROOT
from behavior.dataset import load_feature_table
from behavior.features import FEATURE_SETS
from behavior.metrics import evaluate
from behavior.model import make_model
from .common import provenance, write_json

LADDER = [
    ("A_dummy", "dummy", "dummy", .1, "Majority/prior without interaction evidence"),
    ("B_shortcut", "shortcut", "logistic", .1, "Can global duration and silence explain labels?"),
    ("B_shortcut_trees", "shortcut", "trees", .1, "Nonlinear global shortcut control"),
    ("C_temporal", "temporal", "logistic", .1, "Normal temporal dynamics are predictive"),
    ("D_reaction", "reaction", "logistic", .1, "Agent-conditioned reactions add to normal dynamics"),
    ("E_consistency", "consistency", "logistic", .1, "Repeated-event variability adds information"),
    ("E_less_regularized", "consistency", "logistic", 1., "More linear capacity is beneficial"),
    ("E_trees", "consistency", "trees", .1, "Small nonlinear model improves generalization"),
    ("E_no_counts", "no_counts", "logistic", .1, "Event counts may act as duration proxies"),
    ("E_no_agent", "no_agent", "logistic", .1, "Agent duration may encode dataset flow"),
    ("R_reaction_only", "reaction_only", "logistic", .1, "Reaction signal survives without normal dynamics"),
    ("G_with_global", "with_global", "logistic", .1, "Global shortcuts improve the full model"),
]


def run_ladder(source):
    df = load_feature_table(source)
    train, val = df[df.split == "train"], df[df.split == "val"]
    if not len(train) or not len(val):
        raise ValueError("Both official splits required")
    records = []
    detail = {}
    model_dir = ROOT / "artifacts" / "models" / source
    model_dir.mkdir(parents=True, exist_ok=True)
    predictions = df[["anon_id", "split", "y"]].copy()
    for exp, family, kind, c, hypothesis in LADDER:
        names = FEATURE_SETS[family]
        model = make_model(kind, c)
        start = perf_counter()
        model.fit(train[names], train.y)
        seconds = perf_counter() - start
        start = perf_counter()
        p = model.predict_proba(df[names])[:, 1]
        ms = (perf_counter() - start) * 1000 / len(df)
        predictions[exp] = p
        tr = evaluate(train.y, p[df.split == "train"])
        va = evaluate(val.y, p[df.split == "val"])
        clf = model.named_steps["classifier"]
        capacity = clf.coef_.size + clf.intercept_.size if hasattr(clf, "coef_") else (
            sum(t.tree_.node_count for t in clf.estimators_) if hasattr(clf, "estimators_") else 1)
        result = {"source": source, "experiment": exp, "hypothesis": hypothesis, "feature_set": family,
                  "model": kind, "c": c, "feature_count": len(names), "capacity": capacity,
                  "capacity_unit": "coefficients" if kind == "logistic" else "tree_nodes" if kind == "trees" else "prior",
                  "training_s": seconds, "inference_ms_per_call_batched": ms,
                  **{f"train_{k}": tr[k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "brier")},
                  **{f"val_{k}": va[k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "f1", "fpr", "fnr", "eer", "brier")},
                  "conclusion": "Comparison recorded; see report for scientific interpretation",
                  "decision": "Retain reproducible baseline for comparison"}
        records.append(result)
        detail[exp] = {"features": names, "hyperparameters": clf.get_params(), "preprocessing": "train-only median+missing indicator, z-score, clip +/-6",
                       "train": tr, "val": va}
        joblib.dump({"model": model, "feature_names": names, "source": source}, model_dir / f"{exp}.joblib")
        print(f"{source}/{exp}: train AUC={tr['roc_auc']:.3f}, val AUC={va['roc_auc']:.3f}, "
              f"BA={va['balanced_accuracy']:.3f}, CM={va['confusion_matrix']}, Brier={va['brier']:.3f}")
        # Save after EACH experiment, so interrupted runs leave a usable baseline/log.
        pd.DataFrame(records).to_csv(model_dir / "experiments.csv", index=False)
        write_json(ROOT / "reports" / "metrics" / f"{source}_models.json", detail)
    predictions.to_csv(model_dir / "predictions.csv", index=False)
    write_json(model_dir / "provenance.json", provenance())
    report_log = ROOT / "reports" / "experiments.csv"
    previous = pd.read_csv(report_log) if report_log.exists() else pd.DataFrame()
    if len(previous):
        previous = previous[previous.source != source]
    pd.concat([previous, pd.DataFrame(records)], ignore_index=True).to_csv(report_log, index=False)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=["organizer", "vad"], default="organizer")
    args = parser.parse_args()
    run_ladder(args.source)


if __name__ == "__main__":
    main()
