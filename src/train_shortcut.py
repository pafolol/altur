"""
SHORTCUT CHECK - how far does a model get WITHOUT listening to the voice?

Trains small classifiers on trivial per-call statistics produced by inspect_dataset.py:
  * "metadata" set: duration, caller/agent speech time, turn counts, leading/trailing silence
  * "level" set:    caller loudness, peak, clipping, bandwidth (3.4-4 kHz ratio, roll-off, centroid)
  * "behavioural" set: response latency, overlap, turn lengths (conversation dynamics, NOT acoustic)
  * "all": everything above

This is NOT a production model. It answers: does the dataset contain shortcuts that a model could learn
instead of synthetic-voice artefacts? If it scores far above chance, part of any model's score may come
from those cues, and cues tied to the recording pipeline (e.g. how long the AI caller takes to start
talking, or the gain of its audio) may not survive a different pipeline in the hidden test set.

Run:  python src/train_shortcut.py
Outputs: outputs/shortcut/{results.json, feature_weights.png}, reports/shortcut_check.csv
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _bootstrap  # noqa: F401
import config
from metrics import evaluate_scores, format_metrics, log_experiment, save_json

OUT = config.OUTPUTS_DIR / "shortcut"
FEATURE_SETS = {
    "duration_only": ["duration_s"],
    "duration+silence": ["duration_s", "first_caller_start_s", "trailing_silence_s"],
    "metadata": ["duration_s", "caller_speech_s", "caller_speech_fraction", "n_caller_turns", "n_agent_turns",
                 "agent_speech_s", "first_caller_start_s", "trailing_silence_s", "n_chunks"],
    "loudness": ["caller_rms_db", "caller_speech_rms_db", "caller_peak", "caller_clipping_fraction",
                 "caller_noise_floor_db", "caller_snr_db", "agent_rms_db"],
    "bandwidth": ["caller_rolloff95_hz", "caller_centroid_hz", "caller_hf_ratio_db"],
    "behavioural": ["median_response_latency_s", "std_response_latency_s", "overlap_s", "mean_caller_turn_s",
                    "max_caller_turn_s", "n_caller_turns"],
}
FEATURE_SETS["all_trivial"] = sorted(set(sum(FEATURE_SETS.values(), [])))


def fit_eval(train, val, cols, model="logreg"):
    Xtr, Xva = train[cols].to_numpy(float), val[cols].to_numpy(float)
    med = np.nanmedian(Xtr, axis=0)
    Xtr, Xva = np.where(np.isnan(Xtr), med, Xtr), np.where(np.isnan(Xva), med, Xva)
    if model == "logreg":
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000))
        clf.fit(Xtr, train["y"])
        s_tr, s_va = clf.decision_function(Xtr), clf.decision_function(Xva)
        weights = dict(zip(cols, clf[-1].coef_[0].round(3)))
    else:
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=2, learning_rate=0.05, random_state=config.SEED)
        clf.fit(Xtr, train["y"])
        p_tr, p_va = np.clip(clf.predict_proba(Xtr)[:, 1], 1e-6, 1 - 1e-6), np.clip(clf.predict_proba(Xva)[:, 1], 1e-6, 1 - 1e-6)
        s_tr, s_va = np.log(p_tr / (1 - p_tr)), np.log(p_va / (1 - p_va))
        weights = dict(zip(cols, clf.feature_importances_.round(3)))
    return {"train": evaluate_scores(train["y"], s_tr), "val": evaluate_scores(val["y"], s_va), "weights": weights}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats = pd.read_csv(config.OUTPUTS_DIR / "dataset_inspection" / "call_stats.csv")
    train, val = stats[stats["split"] == "train"], stats[stats["split"] == "val"]
    results, rows = {}, []
    print("SHORTCUT MODELS (validation metrics; primary = accuracy at p=0.5)")
    for name, cols in FEATURE_SETS.items():
        cols = [c for c in cols if c in stats.columns]
        for model in ("logreg", "gbm"):
            r = fit_eval(train, val, cols, model)
            results[f"{name}/{model}"] = r
            v = r["val"]
            rows.append({"feature_set": name, "model": model, "n_features": len(cols), "val_accuracy": v["accuracy"],
                         "val_balanced_accuracy": v["balanced_accuracy"], "val_auc": v["auc"], "val_eer": v["eer"],
                         "val_f1": v["f1"], "train_auc": r["train"]["auc"]})
            print(f"  {name:<18} {model:<6} ({len(cols):2d} feats): {format_metrics(v)}")
    table = pd.DataFrame(rows)
    table.to_csv(config.REPORTS_DIR / "shortcut_check.csv", index=False)
    save_json(results, OUT / "results.json")

    # figure: logistic-regression weights of the "all_trivial" model + AUC per feature set
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    w = results["all_trivial/logreg"]["weights"]
    names, vals = zip(*sorted(w.items(), key=lambda kv: abs(kv[1])))
    axes[0].barh(names, vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals])
    axes[0].axvline(0, color="black", lw=0.8)
    axes[0].set_xlabel("logistic-regression weight (standardised feature)   red: pushes to SYNTHETIC, blue: to HUMAN")
    axes[0].set_title("Shortcut model on ALL trivial statistics: which cues does it use?", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=7)
    t = table[table["model"] == "logreg"]
    axes[1].barh(t["feature_set"], t["val_auc"], color="#9467bd")
    for i, (a, acc) in enumerate(zip(t["val_auc"], t["val_accuracy"])):
        axes[1].text(a + 0.005, i, f"AUC {a:.2f} / acc {acc:.2f}", va="center", fontsize=7)
    axes[1].axvline(0.5, color="black", ls="--", lw=0.8, label="chance")
    axes[1].set_xlim(0.4, 1.12)
    axes[1].set_xlabel("validation AUC (logistic regression)")
    axes[1].set_title("How much each family of trivial cues separates the classes", fontsize=10)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "shortcut_check.png", dpi=140)
    plt.close(fig)
    best = table.sort_values("val_auc", ascending=False).iloc[0]
    log_experiment(title="shortcut check (trivial statistics, no voice content)", model="logreg / GBM on call statistics",
                   dataset_fraction=1.0, segmentation="n/a", selected_layer="n/a", classifier="logreg C=1 balanced; GBM 150x depth-2",
                   hyperparameters="default", device="cpu", runtime_s="seconds",
                   best=f"{best['feature_set']}/{best['model']}: val AUC {best['val_auc']:.3f}, acc {best['val_accuracy']:.3f}",
                   notes="NOT a production model; measures dataset shortcuts")
    print(f"\nSaved {OUT / 'results.json'}, reports/shortcut_check.csv, figures/shortcut_check.png")


if __name__ == "__main__":
    main()
