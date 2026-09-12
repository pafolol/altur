"""Generate reproducible aggregate plots and locally anonymized error timelines."""

import json
import os
import textwrap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve
from behavior.config import ROOT
from behavior.features import extract_features
from behavior.turns import load_turns, normalize_turns
from .common import write_json

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#2471a3", "#c0392b"]
FIGURES = ROOT / "reports/figures"


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=170, bbox_inches="tight")
    plt.close(fig)


def error_analysis(df, artifact):
    val = df[df.split == "val"]
    errors = val[(val.probability >= .5) != (val.y == 1)].copy()
    errors["error_type"] = np.where(errors.y == 0, "FP", "FN")
    keys = ["duration_s", "interruption_count", "barge_in_count", "response_latency_count",
            "response_latency_mean", "response_latency_std", "barge_overlap_mean",
            "caller_turn_mean", "extended_gap_caller_fraction", "quality"]
    summaries = []
    mapping = []
    for i, row in enumerate(errors.itertuples()):
        alias = f"{row.error_type}-{i + 1:02d}"
        f = errors.loc[errors.anon_id == row.anon_id].iloc[0]
        x = f[artifact["feature_names"]].to_numpy(dtype=float)
        missing = np.isnan(x)
        imputed = np.where(missing, artifact["impute"], x)
        expanded = np.r_[imputed, missing[artifact["missing_indices"]]]
        z = np.clip((expanded - artifact["mean"]) / artifact["scale"], -6, 6)
        expanded_names = artifact["feature_names"] + [artifact["feature_names"][j] + "__missing" for j in artifact["missing_indices"]]
        contributions = z * artifact["coef"]
        order = np.argsort(contributions * (1 if row.error_type == "FP" else -1))[::-1][:4]
        reference = normalize_turns(load_turns(ROOT / "turns" / f"{row.anon_id}.json"), row.duration_s)
        generated = load_turns(ROOT / "artifacts/vad_turns" / f"{row.anon_id}.json")
        reference_features, _ = extract_features(reference, row.duration_s)
        fig, ax = plt.subplots(figsize=(13, 3.5))
        for channel in (0, 1):
            for offset, turns, label in ((0, reference, "Organizer"), (.38, generated, "Silero")):
                segments = [(t.start, t.duration) for t in turns if t.channel == channel]
                ax.broken_barh(segments, (channel + offset, .26), facecolors=COLORS[channel],
                              alpha=1 if offset == 0 else .55, label=f"{label} {'caller' if channel == 0 else 'agent'}")
        ax.set(yticks=[.3, 1.3], yticklabels=["Caller (ch0)", "Agent (ch1)"], xlabel="Time since call start (s)",
               ylabel="Participant / segmentation", title=f"{alias}: {'human flagged synthetic' if row.error_type == 'FP' else 'synthetic missed'} — temporal evidence only",
               xlim=(0, row.duration_s))
        ax.legend(loc="upper center", bbox_to_anchor=(.5, -.2), ncol=4)
        save(fig, f"error_{alias}.png")
        summaries.append({"alias": alias, "type": row.error_type, "synthetic_probability": row.probability,
                          "measurements": {k: f[k] for k in keys},
                          "largest_wrong_direction_contributions": [{"feature": expanded_names[j], "standardized_logit_contribution": contributions[j]} for j in order],
                          "organizer_event_counts": {k: reference_features[k] for k in ("interruption_count", "barge_in_count", "response_latency_count")},
                          "figure": f"figures/error_{alias}.png"})
        mapping.append({"alias": alias, "anon_id": row.anon_id})
    write_json(ROOT / "reports/private/error_examples.json", summaries)
    write_json(ROOT / "reports/private/error_id_mapping.json", mapping)
    write_json(ROOT / "reports/metrics/error_summary.json", {
        "counts": errors.error_type.value_counts().to_dict(),
        "errors_median": errors[keys].median().to_dict(),
        "correct_median": val.loc[~val.anon_id.isin(errors.anon_id), keys].median().to_dict(),
        "manual_listening_performed": False,
        "method": "All full-call validation errors; feature contributions and both local timelines; no transcripts or speaker identification"})


def main():
    artifact = json.loads((ROOT / "artifacts/models/behavior.json").read_text())
    df = pd.read_csv(ROOT / "artifacts/features/vad.csv")
    pred = pd.read_csv(ROOT / "reports/private/final_predictions.csv")
    df = df.merge(pred[["anon_id", "probability", "raw_probability"]], on="anon_id", validate="one_to_one")
    val, train = df[df.split == "val"], df[df.split == "train"]
    final = json.loads((ROOT / "reports/metrics/final_model.json").read_text())
    families = json.loads((ROOT / "reports/metrics/final_family_ablations.json").read_text())
    log = pd.read_csv(ROOT / "reports/experiments.csv")

    # Distributions use one statistic per call, avoiding event-rich calls dominating.
    specs = [
        ("response_latency_mean", "Mean normal response latency", "Latency (s)"),
        ("interruption_stop_latency_mean", "Mean caller stop latency after agent onset", "Latency (s)"),
        ("barge_overlap_mean", "Mean caller barge-in overlap", "Overlap duration (s)"),
        ("response_latency_std", "Within-call response variability", "Sample standard deviation (s)"),
        ("barge_overlap_cv", "Within-call barge-in variability", "Coefficient of variation (unitless)"),
        ("interruption_count", "Detected agent interruptions per call", "Event count"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (name, title, unit) in zip(axes.flat, specs):
        values = train[name].dropna()
        edges = np.histogram_bin_edges(values, bins=18) if len(values) else [0, 1]
        for y, label in ((0, "Human"), (1, "Synthetic")):
            data = train.loc[train.y == y, name].dropna()
            ax.hist(data, bins=edges, alpha=.55, color=COLORS[y], label=f"{label} (n={len(data)})")
        ax.set(title=title + "\n(training calls)", xlabel=unit, ylabel="Calls")
        ax.legend(fontsize=8)
    save(fig, "behavior_distributions.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name, title in zip(axes, ("duration_s", "leading_silence_s", "trailing_silence_s"),
                               ("WAV duration", "Leading silence", "Trailing silence")):
        for y, label in ((0, "Human"), (1, "Synthetic")):
            ax.hist(train.loc[train.y == y, name], bins=18, alpha=.55, color=COLORS[y], label=label)
        ax.set(title=f"Shortcut audit: {title} (train)", xlabel="Duration (s)", ylabel="Calls")
        ax.legend()
    save(fig, "shortcut_distributions.png")

    vad_preds = pd.read_csv(ROOT / "artifacts/models/vad/predictions.csv").query("split == 'val'")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, p in (("Final Behaviour", val.probability), ("Global timing shortcut", vad_preds.B_shortcut)):
        fpr, tpr, _ = roc_curve(val.y, p)
        precision, recall, _ = precision_recall_curve(val.y, p)
        axes[0].plot(fpr, tpr, label=label)
        axes[1].plot(recall, precision, label=label)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=.4, label="Chance")
    axes[1].axhline(val.y.mean(), color="gray", ls="--", label="Synthetic prevalence")
    axes[0].set(title="Official validation ROC", xlabel="Human false-positive rate", ylabel="Synthetic recall", xlim=(0, 1), ylim=(0, 1.02))
    axes[1].set(title="Official validation precision–recall", xlabel="Synthetic recall", ylabel="Synthetic precision", xlim=(0, 1), ylim=(0, 1.02))
    for ax in axes:
        ax.legend()
    save(fig, "roc_pr.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, metrics in (("Uncalibrated", final["raw_val"]), ("Train-OOF calibrated", final["calibrated_val"])):
        r = metrics["reliability"]
        axes[0].plot([a["mean_probability"] for a in r], [a["synthetic_frequency"] for a in r], "o-", label=label)
        if label.startswith("Train"):
            for a in r:
                axes[0].annotate(str(a["n"]), (a["mean_probability"], a["synthetic_frequency"]), fontsize=8)
    axes[0].plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    axes[0].set(title="Validation reliability (8 bins; labels=n)", xlabel="Mean predicted P(synthetic)", ylabel="Observed synthetic fraction", xlim=(0, 1), ylim=(-.05, 1.05))
    for y, label in ((0, "Human"), (1, "Synthetic")):
        axes[1].hist(val.loc[val.y == y, "probability"], bins=np.linspace(0, 1, 16), color=COLORS[y], alpha=.6, label=label)
    axes[1].set(title="Validation calibrated score distribution", xlabel="Predicted P(synthetic)", ylabel="Calls")
    for ax in axes:
        ax.legend()
    save(fig, "calibration_scores.png")

    fig, ax = plt.subplots(figsize=(5, 4.5))
    cm = np.array(final["val"]["confusion_matrix"])
    image = ax.imshow(cm, cmap="Blues", vmin=0)
    for (i, j), n in np.ndenumerate(cm):
        ax.text(j, i, f"{n}\n{n / cm[i].sum():.1%}", ha="center", va="center", color="white" if n > cm.max()/2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Human", "Synthetic"], yticklabels=["Human", "Synthetic"],
           title="Official validation confusion matrix\nCounts and within-true-class percentages", xlabel="Predicted class (threshold 0.5)", ylabel="True class")
    fig.colorbar(image, ax=ax, label="Calls")
    save(fig, "confusion_matrix.png")

    names = artifact["feature_names"]
    corr = train[names].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(13, 12))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="coolwarm")
    labels = [n.replace("interruption_", "int_").replace("latency", "lat.") for n in names]
    ax.set(xticks=range(len(names)), yticks=range(len(names)), xticklabels=labels, yticklabels=labels,
           title="Training-only Spearman feature correlation\nPairwise observed measurements", xlabel="Feature", ylabel="Feature")
    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    fig.colorbar(im, ax=ax, label="Spearman correlation (unitless)", fraction=.03)
    save(fig, "feature_correlations.png")

    all_names = names + [names[j] + " [missing]" for j in artifact["missing_indices"]]
    coefs = np.array(artifact["coef"])
    ix = np.argsort(np.abs(coefs))[-18:]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.barh(range(len(ix)), coefs[ix], color=[COLORS[int(v > 0)] for v in coefs[ix]])
    ax.set(yticks=range(len(ix)), yticklabels=[all_names[j] for j in ix], xlabel="Weight per training-standardized feature (log odds)",
           ylabel="Feature or missingness indicator", title="Largest fitted logistic weights\nRed pushes synthetic, blue pushes human; correlated features share credit")
    ax.axvline(0, color="black", lw=.5)
    save(fig, "logistic_coefficients.png")

    core_ids = ["A_dummy", "B_shortcut", "C_temporal", "D_reaction", "E_consistency"]
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, source in enumerate(("organizer", "vad")):
        sub = log[(log.source == source) & log.experiment.isin(core_ids)].set_index("experiment").loc[core_ids]
        ax.bar(np.arange(len(core_ids)) + (i - .5) * .35, sub.val_roc_auc, width=.35, label=source)
    ax.set(xticks=range(len(core_ids)), xticklabels=["Dummy", "Global timing", "Temporal", "+ Reaction", "+ Consistency"],
           ylim=(0, 1.05), xlabel="Feature ladder (same logistic C=0.1 except dummy)", ylabel="Validation ROC-AUC", title="Organizer-turn vs WAV-derived feature ladder")
    ax.legend()
    save(fig, "ablation_ladder.png")

    labels = ["Final"] + [n.replace("without_", "− ").replace("_", " ") for n in families]
    fig, ax = plt.subplots(figsize=(11, 6))
    aucs = [final["val"]["roc_auc"]] + [v["metrics"]["roc_auc"] for v in families.values()]
    bas = [final["val"]["balanced_accuracy"]] + [v["metrics"]["balanced_accuracy"] for v in families.values()]
    ax.plot(aucs, "o-", label="ROC-AUC")
    ax.plot(bas, "s-", label="Balanced accuracy at 0.5")
    ax.set(xticks=range(len(labels)), xticklabels=labels, ylim=(.5, 1.02), xlabel="Feature family removed and model retrained on train",
           ylabel="Official validation score", title="Final-model family ablation (calibration fitted on train only)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    ax.legend()
    save(fig, "final_family_ablations.png")

    prefixes = json.loads((ROOT / "reports/metrics/prefixes.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["15s", "30s", "60s", "Full"]
    axes[0].plot(labels, [v["metrics"]["roc_auc"] for v in prefixes.values()], "o-", label="ROC-AUC")
    axes[0].plot(labels, [v["metrics"]["balanced_accuracy"] for v in prefixes.values()], "s-", label="Balanced accuracy")
    axes[0].set(ylabel="Validation score", ylim=(0, 1.02), title="Same frozen model on actual prefixes")
    axes[1].plot(labels, [v["mean_quality"] for v in prefixes.values()], "o-", label="Heuristic quality")
    axes[1].set(ylabel="Mean evidence quality (0–1)", ylim=(0, 1), title="Short clips provide little evidence")
    axes[2].plot(labels, [v["latency_p50_ms"] for v in prefixes.values()], "o-", label="p50")
    axes[2].plot(labels, [v["latency_p95_ms"] for v in prefixes.values()], "s-", label="p95")
    axes[2].set(ylabel="Offline inference latency (ms)", title="CPU latency; excludes network")
    for ax in axes:
        ax.set_xlabel("Available audio prefix")
        ax.legend()
    save(fig, "prefix_performance.png")

    comparison = pd.read_csv(ROOT / "reports/private/vad_comparison.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ch, label in ((0, "Caller"), (1, "Agent")):
        axes[0].hist(comparison[f"ch{ch}_speech_iou"], bins=np.linspace(0, 1, 21), alpha=.6, label=label)
        axes[1].scatter(comparison[f"ch{ch}_reference_segments"], comparison[f"ch{ch}_generated_segments"], s=12, alpha=.4, label=label)
    axes[1].plot([0, 70], [0, 70], "k--", label="Equal counts")
    for key, label in (("interruption_count_delta", "Agent interruptions"), ("barge_in_count_delta", "Caller barge-ins")):
        axes[2].hist(comparison[key], bins=20, alpha=.6, label=label)
    axes[0].set(title="VAD vs automatic organizer reference", xlabel="Speech-mask IoU (0–1)", ylabel="Calls")
    axes[1].set(title="Segmentation granularity", xlabel="Organizer segments (count)", ylabel="Silero segments (count)")
    axes[2].set(title="Directional event count differences", xlabel="Silero minus organizer (events/call)", ylabel="Calls")
    for ax in axes:
        ax.legend(fontsize=8)
    save(fig, "vad_agreement.png")

    rob = json.loads((ROOT / "reports/metrics/robustness.json").read_text())
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ["\n".join(textwrap.wrap(k.replace("_", " "), 18)) for k in rob]
    ax.bar(labels, [v["mean_absolute_probability_change"] for v in rob.values()])
    ax.set(title="Sensitivity probes: probability changes from original\nAudio perturbations n=24, threshold/jitter n=71", xlabel="Fixed perturbation (no retuning)", ylabel="Mean absolute change in P(synthetic)")
    save(fig, "robustness.png")
    error_analysis(df, artifact)
    print(f"Generated {len(list(FIGURES.glob('*.png')))} local figures and complete validation error analysis.")


if __name__ == "__main__":
    main()
