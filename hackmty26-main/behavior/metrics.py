"""Positive = synthetic (1). Never silently choose a threshold from evaluated labels."""

import numpy as np
from sklearn import metrics as skm


def reliability(y, p, bins=8):
    """Equal-width probability calibration: compare mean p to synthetic frequency."""
    y, p = np.asarray(y), np.asarray(p)
    ids = np.minimum((p * bins).astype(int), bins - 1)
    result = []
    for b in range(bins):
        mask = ids == b
        if mask.any():
            result.append({"bin": b, "n": int(mask.sum()), "mean_probability": float(p[mask].mean()),
                           "synthetic_frequency": float(y[mask].mean())})
    ece = sum(r["n"] * abs(r["mean_probability"] - r["synthetic_frequency"]) for r in result) / len(y)
    return result, float(ece)


def evaluate(y, probabilities, threshold=.5):
    y, p = np.asarray(y, dtype=int), np.asarray(probabilities, dtype=float)
    if len(y) == 0 or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probability/label inputs")
    pred = p >= threshold
    tn, fp, fn, tp = skm.confusion_matrix(y, pred, labels=[0, 1]).ravel()
    curve, ece = reliability(y, p)
    two_classes = len(np.unique(y)) == 2
    eer = auc = ap = None
    if two_classes:
        fpr, tpr, _ = skm.roc_curve(y, p)
        fnr = 1 - tpr
        # Interpolate the crossing, including tied-score dummy ROC diagonal.
        difference = fpr - fnr
        j = int(np.flatnonzero(difference >= 0)[0])
        if j == 0:
            eer = float(fpr[0])
        else:
            w = -difference[j - 1] / (difference[j] - difference[j - 1])
            eer = float(fpr[j - 1] + w * (fpr[j] - fpr[j - 1]))
        auc = float(skm.roc_auc_score(y, p))
        ap = float(skm.average_precision_score(y, p))
    return {
        "n": len(y), "threshold": threshold,
        "accuracy": float(skm.accuracy_score(y, pred)),
        "balanced_accuracy": float(skm.balanced_accuracy_score(y, pred)),
        "precision": float(skm.precision_score(y, pred, zero_division=0)),
        "recall": float(skm.recall_score(y, pred, zero_division=0)),
        "f1": float(skm.f1_score(y, pred, zero_division=0)),
        "roc_auc": auc, "pr_auc_ap": ap, "eer": eer,
        "fpr": float(fp / (fp + tn)) if fp + tn else None,
        "fnr": float(fn / (fn + tp)) if fn + tp else None,
        "brier": float(skm.brier_score_loss(y, p)),
        "log_loss": float(skm.log_loss(y, p, labels=[0, 1])),
        "ece_8_bins": ece, "reliability": curve,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def bootstrap_auc_difference(y, a, b, seed=2026, repeats=1000):
    """Paired call-level bootstrap; not a speaker-cluster confidence interval."""
    y, a, b = map(np.asarray, (y, a, b))
    rng = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        ids = rng.integers(0, len(y), len(y))
        if len(np.unique(y[ids])) == 2:
            differences.append(skm.roc_auc_score(y[ids], a[ids]) - skm.roc_auc_score(y[ids], b[ids]))
    return {"delta_auc": float(skm.roc_auc_score(y, a) - skm.roc_auc_score(y, b)),
            "ci95_call_bootstrap": np.quantile(differences, [.025, .975]).tolist(), "repeats": repeats}
