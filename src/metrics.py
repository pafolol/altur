"""
Evaluation metrics, identical for every model.

CONVENTION
  * class 1 = SYNTHETIC (positive), class 0 = HUMAN
  * every model produces one SCORE per call = log-odds that the caller is synthetic
    (0 -> 50/50, +3 -> ~95% synthetic, -3 -> ~95% human); probability = sigmoid(score)
  * decision: synthetic if probability >= 0.5 (i.e. score >= 0), unless a threshold is given

PRIMARY CHALLENGE METRIC
  The judges call POST /detect and score the boolean `is_synthetic`; the PDF names no formula, so the
  decision ACCURACY on the hidden set is what we optimise (balanced accuracy is reported next to it
  because the validation split is nearly balanced but the hidden set may not be).
  `confidence` is used "to break ties and reward calibration" -> Brier score / log-loss / ECE.

SECONDARY DIAGNOSTIC METRICS
  ROC AUC and EER are threshold-free measures of how well the scores RANK calls; F1, precision, recall,
  FAR (synthetic accepted as human) and FRR (human rejected as synthetic) describe the decision.
"""
import json

import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

import _bootstrap  # noqa: F401
import config

PRIMARY_METRIC = "accuracy"


def sigmoid(x):
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def compute_eer(labels, scores):
    """(EER, score threshold at which FAR == FRR). Robust to ties and constant scores."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if len(np.unique(labels)) < 2 or not np.isfinite(scores).all():
        return float("nan"), float("nan")
    fpr, tpr, thr = roc_curve(labels, scores, pos_label=1, drop_intermediate=False)
    far, frr = 1.0 - tpr, fpr  # FAR = synthetic missed, FRR = human rejected
    i = int(np.argmin(np.abs(far - frr)))
    return float((far[i] + frr[i]) / 2), float(thr[i]) if np.isfinite(thr[i]) else float(np.max(scores))


def auc(labels, scores):
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def decision_metrics(labels, scores, threshold=0.0):
    labels = np.asarray(labels)
    preds = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    tnr = tn / (tn + fp) if tn + fp else 0.0
    return {
        "accuracy": float((tp + tn) / len(labels)),
        "balanced_accuracy": float((recall + tnr) / 2),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(2 * precision * recall / (precision + recall)) if precision + recall else 0.0,
        "far": float(fn / (fn + tp)) if fn + tp else 0.0,
        "frr": float(fp / (fp + tn)) if fp + tn else 0.0,
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def calibration_metrics(labels, probs, n_bins=10):
    """Brier score, log-loss and expected calibration error of a probability of 'synthetic'."""
    labels, probs = np.asarray(labels, dtype=float), np.clip(np.asarray(probs, dtype=float), 1e-6, 1 - 1e-6)
    brier = float(np.mean((probs - labels) ** 2))
    logloss = float(-np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs)))
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.any():
            conf, acc = float(probs[m].mean()), float(labels[m].mean())
            ece += m.mean() * abs(conf - acc)
            bins.append({"lo": float(lo), "hi": float(hi), "n": int(m.sum()), "confidence": conf, "accuracy": acc})
    return {"brier": brier, "log_loss": logloss, "ece": float(ece), "bins": bins}


def evaluate_scores(labels, scores, threshold=0.0):
    """Every metric for one set of call-level log-odds scores."""
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    out = decision_metrics(labels, scores, threshold)
    out["auc"] = auc(labels, scores)
    out["eer"], out["eer_threshold"] = compute_eer(labels, scores)
    out["threshold"] = float(threshold)
    out["n_calls"] = int(len(labels))
    out["n_human"], out["n_synthetic"] = int((labels == 0).sum()), int((labels == 1).sum())
    cal = calibration_metrics(labels, sigmoid(scores))
    out.update({k: cal[k] for k in ("brier", "log_loss", "ece")})
    return out


def format_metrics(m):
    return (f"acc={m['accuracy']:.4f} bal_acc={m['balanced_accuracy']:.4f} f1={m['f1']:.4f} "
            f"AUC={m['auc']:.4f} EER={m['eer'] * 100:.2f}% FAR={m['far']:.3f} FRR={m['frr']:.3f} "
            f"brier={m['brier']:.4f} (n={m['n_calls']})")


def aggregate_chunk_scores(chunk_scores, method=config.CHUNK_AGGREGATION):
    """Combine the log-odds of every chunk of a call into one call-level log-odds score."""
    s = np.asarray(chunk_scores, dtype=float)
    if len(s) == 0:
        return 0.0
    if method == "mean":
        return float(s.mean())
    if method == "median":
        return float(np.median(s))
    if method == "max":
        return float(s.max())
    if method == "min":
        return float(s.min())
    if method == "top25_mean":  # mean of the 25% most synthetic-looking chunks (at least one)
        k = max(1, int(np.ceil(0.25 * len(s))))
        return float(np.sort(s)[-k:].mean())
    if method == "trimmed_mean":  # drop the 10% most extreme values on each side
        k = int(0.1 * len(s))
        return float(np.sort(s)[k:len(s) - k].mean()) if len(s) > 2 * k else float(s.mean())
    if method == "mean_prob":  # average probability, converted back to log-odds
        p = np.clip(sigmoid(s).mean(), 1e-6, 1 - 1e-6)
        return float(np.log(p / (1 - p)))
    if method == "duration_weighted":
        raise ValueError("duration_weighted needs weights; use aggregate_weighted()")
    raise ValueError(f"unknown aggregation {method}")


def aggregate_weighted(chunk_scores, weights):
    s, w = np.asarray(chunk_scores, dtype=float), np.asarray(weights, dtype=float)
    return float((s * w).sum() / max(w.sum(), 1e-9))


AGGREGATIONS = ["mean", "median", "max", "top25_mean", "trimmed_mean", "mean_prob", "duration_weighted"]


def save_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def log_experiment(**fields):
    """Append one experiment record to reports/EXPERIMENT_LOG.md (never overwrites earlier records)."""
    import datetime
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = config.EXPERIMENT_LOG
    if not path.exists():
        path.write_text("# Experiment log - Altur acoustic detector\n\nOne record per experiment, appended by the "
                        "scripts (never edited by hand; newest at the bottom).\n\n")
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [f"## {ts} - {fields.pop('title', 'experiment')}", ""]
    for k, v in fields.items():
        if isinstance(v, float):
            v = f"{v:.4f}"
        lines.append(f"- **{k}**: {v}")
    lines.append("")
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
