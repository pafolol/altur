"""
CALIBRATION - is the probability the model reports believable?

A classifier can be RIGHT (accuracy) and still be badly CALIBRATED: "0.99" should be wrong about 1% of the
time. The challenge uses `confidence` to break ties and "reward well-calibrated systems", so we check.

The raw probability sigmoid(mean chunk log-odds) is compared with
  * Platt scaling      p = sigmoid(a * score + b)      (2 parameters)
  * temperature scaling p = sigmoid(score / T)          (1 parameter)
Both are fitted on VALIDATION calls (the train scores are overfit and useless for this). To measure them
honestly they are evaluated OUT-OF-FOLD with 5-fold cross-validation on the validation set; the final
calibrator used at inference is then refitted on all validation calls.

Run:  python src/calibrate.py --backbone wavlm          (or all)
Outputs: models/<backbone>/calibration_{mlp,logreg}.json, outputs/calibration/<backbone>.json,
         outputs/figures/calibration_<backbone>.png
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold

import _bootstrap  # noqa: F401
import config
from metrics import calibration_metrics, decision_metrics, log_experiment, save_json, sigmoid


def fit_platt(scores, y):
    """Minimise log-loss over (a, b) for p = sigmoid(a*s + b)."""
    def loss(ab):
        p = np.clip(sigmoid(ab[0] * scores + ab[1]), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    res = minimize(loss, x0=[1.0, 0.0], method="Nelder-Mead", options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 5000})
    return {"method": "platt", "a": float(res.x[0]), "b": float(res.x[1])}


def fit_temperature(scores, y):
    def loss(t):
        p = np.clip(sigmoid(scores / t), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    ts = np.exp(np.linspace(np.log(0.05), np.log(50), 400))
    t = float(ts[np.argmin([loss(t) for t in ts])])
    return {"method": "temperature", "a": 1.0 / t, "b": 0.0, "temperature": t}


def apply(cal, scores):
    return sigmoid(cal["a"] * np.asarray(scores, dtype=float) + cal["b"])


def cross_validated(scores, y, fitter, n_splits=5, groups=None):
    """Out-of-fold calibrated probabilities on the validation set (grouped by call when several conditions per call)."""
    probs = np.zeros(len(y))
    if groups is None:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED).split(scores.reshape(-1, 1), y)
    else:
        from sklearn.model_selection import GroupKFold
        splitter = GroupKFold(n_splits=n_splits).split(scores.reshape(-1, 1), y, groups)
    for tr, te in splitter:
        cal = fitter(scores[tr], y[tr])
        probs[te] = apply(cal, scores[te])
    return probs


def summarize(y, probs):
    m = calibration_metrics(y, probs)
    d = decision_metrics(y, np.log(np.clip(probs, 1e-7, 1 - 1e-7) / np.clip(1 - probs, 1e-7, 1)), 0.0)
    return {"brier": m["brier"], "log_loss": m["log_loss"], "ece": m["ece"], "accuracy": d["accuracy"],
            "balanced_accuracy": d["balanced_accuracy"], "mean_confidence": float(np.maximum(probs, 1 - probs).mean()),
            "bins": m["bins"]}


def plot(report, backbone, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, clf in zip(axes, ("mlp", "logreg")):
        r = report[clf]
        for key, color, label in (("raw", "#7f7f7f", "raw sigmoid(score)"), ("platt_cv", "#d62728", "Platt (out-of-fold)"),
                                  ("temperature_cv", "#1f77b4", "temperature (out-of-fold)")):
            bins = [b for b in r[key]["bins"] if b["n"] > 0]
            ax.plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-", color=color, ms=4,
                    label=f"{label}: ECE {r[key]['ece']:.3f}, Brier {r[key]['brier']:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
        ax.set_xlabel("predicted probability of synthetic")
        ax.set_ylabel("observed fraction synthetic")
        ax.set_title(f"{clf.upper()}: reliability diagram ({r['fit_on'][:60]})", fontsize=8.5)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Calibration - {config.BACKBONES[backbone]['display']}")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(backbone):
    pred = pd.read_csv(config.PREDICTIONS_DIR / f"{backbone}_val.csv")
    y = pred["y"].to_numpy()
    stress_file = config.PREDICTIONS_DIR / f"{backbone}_stress_val.csv"
    stress = pd.read_csv(stress_file) if stress_file.exists() else None
    report = {"backbone": backbone}
    for clf in ("mlp", "logreg"):
        s = pred[f"score_{clf}"].to_numpy(dtype=float)
        r = {"n": int(len(y)), "raw_clean": summarize(y, sigmoid(s))}
        if clf == "mlp" and stress is not None:
            # The clean validation split contains no errors, so a calibrator fitted on it can only sharpen the
            # probabilities without limit (temperature -> 0). The stress predictions (same calls under the channel
            # perturbations of Part 6b) contain real mistakes and score drift: fit and evaluate there, grouped by call.
            ys, ss, groups = stress["y"].to_numpy(), stress["score"].to_numpy(dtype=float), stress["anon_id"].to_numpy()
            clean_mask = (stress["condition"] == "clean").to_numpy()
            r["fit_on"] = f"validation under {stress['condition'].nunique()} conditions ({len(ys)} call-conditions)"
            r["n_fit"] = int(len(ys))
            r["raw"] = summarize(ys, sigmoid(ss))
            p_platt, p_temp = cross_validated(ss, ys, fit_platt, groups=groups), cross_validated(ss, ys, fit_temperature, groups=groups)
            r["platt_cv"], r["temperature_cv"] = summarize(ys, p_platt), summarize(ys, p_temp)
            r["platt_cv_clean"], r["temperature_cv_clean"] = summarize(ys[clean_mask], p_platt[clean_mask]), summarize(ys[clean_mask], p_temp[clean_mask])
            platt, temp = fit_platt(ss, ys), fit_temperature(ss, ys)
            n_fit = int(len(ys))
        else:
            r["fit_on"] = "clean validation (no stress predictions for this classifier); slope capped at 3"
            r["n_fit"] = int(len(y))
            r["raw"] = r["raw_clean"]
            r["platt_cv"] = summarize(y, cross_validated(s, y, fit_platt))
            r["temperature_cv"] = summarize(y, cross_validated(s, y, fit_temperature))
            platt, temp = fit_platt(s, y), fit_temperature(s, y)
            for c in (platt, temp):
                c["a"] = float(min(c["a"], 3.0))
            n_fit = int(len(y))
        r["platt_final"], r["temperature_final"] = platt, temp
        # deploy the method with the better out-of-fold log-loss (both are tiny models; ties -> temperature)
        chosen = platt if r["platt_cv"]["log_loss"] < r["temperature_cv"]["log_loss"] - 1e-4 else temp
        chosen = dict(chosen, fit_on=r["fit_on"], n=n_fit, classifier=clf,
                      cv={k: r[f"{chosen['method']}_cv"][k] for k in ("brier", "log_loss", "ece", "accuracy")})
        save_json(chosen, config.MODELS_DIR / backbone / f"calibration_{clf}.json")
        r["chosen"] = chosen["method"]
        report[clf] = r
        print(f"[{backbone}] {clf:<6} ({r['fit_on']}) raw: Brier {r['raw']['brier']:.4f} logloss {r['raw']['log_loss']:.4f} ECE {r['raw']['ece']:.3f} "
              f"acc {r['raw']['accuracy']:.3f} | Platt(cv): Brier {r['platt_cv']['brier']:.4f} logloss {r['platt_cv']['log_loss']:.4f} "
              f"ECE {r['platt_cv']['ece']:.3f} acc {r['platt_cv']['accuracy']:.3f} | temp(cv): Brier {r['temperature_cv']['brier']:.4f} "
              f"ECE {r['temperature_cv']['ece']:.3f}  -> deploy {chosen['method']} (a={chosen['a']:.3f}, b={chosen['b']:.3f})")
    out = config.OUTPUTS_DIR / "calibration"
    save_json(report, out / f"{backbone}.json")
    plot(report, backbone, config.FIGURES_DIR / f"calibration_{backbone}.png")
    log_experiment(title=f"calibration - {backbone}", model=config.BACKBONES[backbone]["hf_name"], dataset_fraction="val only",
                   segmentation="as trained", selected_layer="as trained", classifier="Platt / temperature on call score",
                   hyperparameters="5-fold CV on validation", device="cpu", runtime_s="seconds",
                   mlp=f"raw Brier {report['mlp']['raw']['brier']:.4f} -> {report['mlp']['chosen']} Brier {report['mlp'][report['mlp']['chosen'] + '_cv']['brier']:.4f}",
                   notes="calibrator refitted on all validation calls for deployment")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all", choices=["all"] + config.BACKBONE_ORDER)
    args = parser.parse_args()
    keys = config.BACKBONE_ORDER if args.backbone == "all" else [args.backbone]
    for key in keys:
        if (config.PREDICTIONS_DIR / f"{key}_val.csv").exists():
            run(key)


if __name__ == "__main__":
    main()
