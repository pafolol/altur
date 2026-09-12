"""
STAGE 4 - THE CONTROLLED COMPARISON  (run after extract_embeddings + train_classifier [+ calibrate] for every backbone)

Puts the three backbones side by side with the SAME data, segmentation, classifier and metrics:
  * validation metrics (primary: decision accuracy; diagnostics: balanced accuracy, F1, AUC, EER, Brier)
  * resources: parameters, weights on disk, embedding dimension, extraction time, realtime factor,
    peak VRAM, classifier training time, cached embedding size
  * measured end-to-end inference latency per call (decode -> VAD -> backbone -> classifier) on the GPU
    and, optionally, on the CPU
  * figures: layer probes, metric bars, ROC curves, confusion matrices, resources, aggregation heatmap
  * the WINNER, chosen by validation AUC of the MLP (documented tie-breaks), written to reports/best_model.json

Run:  python src/benchmark.py [--latency-calls 10] [--cpu]
Outputs: reports/benchmark.csv, reports/model_comparison.csv, reports/best_model.json, outputs/figures/bench_*.png
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

import _bootstrap  # noqa: F401
import config
import dataset
from metrics import log_experiment, save_json

R, O, F = config.REPORTS_DIR, config.OUTPUTS_DIR, config.FIGURES_DIR
COLORS = {"wavlm": "#1f77b4", "wav2vec2_spanish": "#ff7f0e", "xlsr": "#2ca02c"}


def load(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def gather():
    rows = {}
    for key in config.BACKBONE_ORDER:
        res = load(O / "classifier" / key / "results.json")
        ext = load(O / "embeddings" / key / "extraction_stats.json")
        if res is None or ext is None:
            print(f"[{key}] missing results/extraction stats -> skipped")
            continue
        rows[key] = {"results": res, "extraction": ext, "calibration": load(O / "calibration" / f"{key}.json", {}),
                     "pred_val": pd.read_csv(config.PREDICTIONS_DIR / f"{key}_val.csv"),
                     "stress": load(O / "stress" / key / "mlp_stress.json"), "stress_probe": load(O / "stress" / key / "probe_stress.json")}
    return rows


def measure_latency(key, calls, device, cpu=False):
    from predict import AcousticDetector
    det = AcousticDetector(key, "mlp", device=device, verbose=False)
    det.predict_wav(calls[0])  # warm-up
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    per_call = []
    for p in calls:
        t0 = time.perf_counter()
        r = det.predict_wav(p)
        per_call.append({"total_s": time.perf_counter() - t0, **{k: v for k, v in r["timings"].items()},
                         "n_chunks": r["n_chunks"], "duration_s": r["duration_s"]})
    df = pd.DataFrame(per_call)
    out = {f"{'cpu' if cpu else 'gpu'}_latency_median_s": float(df["total_s"].median()),
           f"{'cpu' if cpu else 'gpu'}_latency_mean_s": float(df["total_s"].mean()),
           f"{'cpu' if cpu else 'gpu'}_latency_max_s": float(df["total_s"].max()),
           f"{'cpu' if cpu else 'gpu'}_backbone_median_s": float(df["backbone_s"].median()),
           f"{'cpu' if cpu else 'gpu'}_segmentation_median_s": float(df["segmentation_s"].median()),
           f"{'cpu' if cpu else 'gpu'}_ms_per_chunk": float(1000 * df["backbone_s"].sum() / max(df["n_chunks"].sum(), 1)),
           "load_s": det.load_s}
    if device.type == "cuda":
        out["inference_peak_vram_mb"] = torch.cuda.max_memory_allocated() / 1e6
    del det
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- figures
def fig_layer_probes(rows):
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for key, r in rows.items():
        probe = r["results"]["probe"]
        n = len(probe) - 1
        x = [p["layer"] / n for p in probe]
        ax.plot(x, [p["val"]["auc"] for p in probe], "o-", ms=4, color=COLORS[key],
                label=f"{config.BACKBONES[key]['display']} ({n} layers) - chosen layer {r['results']['layer']}")
        ax.scatter([r["results"]["layer"] / n], [probe[r["results"]["layer"]]["val"]["auc"]], s=140, facecolors="none",
                   edgecolors=COLORS[key], linewidths=2)
    ax.set_xlabel("relative depth (0 = CNN feature encoder, 1 = last transformer layer)")
    ax.set_ylabel("validation call-level AUC (logistic regression probe)")
    ax.set_title("Where does each frozen backbone keep the human/synthetic evidence?")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    lo = min(min(p["val"]["auc"] for p in r["results"]["probe"]) for r in rows.values())
    ax.set_ylim(max(0.4, lo - 0.03), 1.005)
    fig.tight_layout()
    fig.savefig(F / "bench_layer_probes.png", dpi=140)
    plt.close(fig)


def fig_metric_bars(table):
    metrics = [("val_accuracy", "accuracy (PRIMARY)"), ("val_balanced_accuracy", "balanced accuracy"), ("val_f1", "F1"),
               ("val_auc", "ROC AUC"), ("val_one_minus_eer", "1 - EER")]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(metrics))
    w = 0.8 / len(table)
    for i, (_, row) in enumerate(table.iterrows()):
        vals = [row[m] for m, _ in metrics]
        bars = ax.bar(x + (i - (len(table) - 1) / 2) * w, vals, w, color=COLORS[row["backbone"]], label=row["model"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [name for _, name in metrics])
    ax.set_ylim(min(0.5, table[[m for m, _ in metrics]].min().min() - 0.05), 1.06)
    ax.set_ylabel("validation (71 calls, unseen speakers)")
    ax.set_title("Frozen backbone + MLP on the validation split: same chunks, same classifier, same seed")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "bench_validation_metrics.png", dpi=140)
    plt.close(fig)


def fig_roc(rows):
    fig, ax = plt.subplots(figsize=(5.4, 5))
    for key, r in rows.items():
        p = r["pred_val"]
        fpr, tpr, _ = roc_curve(p["y"], p["score_mlp"])
        ax.plot(fpr, tpr, color=COLORS[key], lw=1.8,
                label=f"{config.BACKBONES[key]['display']}  AUC {r['results']['mlp']['val']['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate (humans flagged as synthetic)")
    ax.set_ylabel("true positive rate (synthetic callers caught)")
    ax.set_title("ROC curves on validation (MLP, chosen layer)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(F / "bench_roc.png", dpi=140)
    plt.close(fig)


def fig_confusions(rows):
    fig, axes = plt.subplots(1, len(rows), figsize=(4.2 * len(rows), 4))
    for ax, (key, r) in zip(np.atleast_1d(axes), rows.items()):
        m = r["results"]["mlp"]["val"]
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
        names = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                share = cm[i, j] / cm[i].sum()
                ax.text(j, i, f"{names[i][j]}\n{cm[i, j]}\n({share:.0%} of row)", ha="center", va="center",
                        color="white" if share > 0.5 else "black", fontsize=9)
        ax.set_xticks([0, 1], ["pred HUMAN", "pred SYNTHETIC"], fontsize=8)
        ax.set_yticks([0, 1], ["true HUMAN", "true SYNTHETIC"], fontsize=8)
        ax.set_title(f"{config.BACKBONES[key]['display']}\nacc {m['accuracy']:.3f}, F1 {m['f1']:.3f}", fontsize=9)
    fig.suptitle("Validation confusion matrices (MLP, threshold p = 0.5)")
    fig.tight_layout()
    fig.savefig(F / "bench_confusion.png", dpi=140)
    plt.close(fig)


def fig_resources(table):
    panels = [("params_m", "backbone parameters (M)"), ("extraction_backbone_s", "embedding extraction, backbone time (s)"),
              ("gpu_latency_median_s", "inference latency per call, GPU (s)"), ("extraction_peak_vram_mb", "peak VRAM during extraction (MB)"),
              ("weights_mb", "weights on disk (MB)"), ("cache_mb", "cached embeddings, all layers (MB)")]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6))
    for ax, (col, title) in zip(axes.ravel(), panels):
        vals = table[col].to_numpy(float)
        bars = ax.bar(table["model"], vals, color=[COLORS[b] for b in table["backbone"]])
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.0f}" if v >= 10 else f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_title(title, fontsize=9.5)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Cost of each backbone (measured on this machine)")
    fig.tight_layout()
    fig.savefig(F / "bench_resources.png", dpi=140)
    plt.close(fig)


def fig_aggregation(rows):
    methods = list(next(iter(rows.values()))["results"]["aggregation"]["mlp"].keys())
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for ax, metric in zip(axes, ("auc", "accuracy")):
        M = np.array([[r["results"]["aggregation"]["mlp"][m][metric] for m in methods] for r in rows.values()])
        im = ax.imshow(M, cmap="YlGn", vmin=max(0.5, M.min() - 0.02), vmax=1.0, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8)
        ax.set_xticks(range(len(methods)), methods, rotation=25, fontsize=8)
        ax.set_yticks(range(len(rows)), [config.BACKBONES[k]["display"] for k in rows], fontsize=8)
        ax.set_title(f"validation {metric} per call-level aggregation (MLP)", fontsize=9.5)
    fig.suptitle("How chunk scores are combined into a call score ('mean' was fixed in advance; the rest is diagnostic)")
    fig.tight_layout()
    fig.savefig(F / "bench_aggregation.png", dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latency-calls", type=int, default=10)
    parser.add_argument("--cpu", action="store_true", help="also measure CPU-only inference latency (3 calls)")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()
    rows = gather()
    if not rows:
        raise SystemExit("nothing to benchmark")
    device = config.get_device()
    val = dataset.load_split("val")
    calls = val.sample(n=min(args.latency_calls, len(val)), random_state=config.SEED)["path"].tolist()

    table, comparison = [], []
    for key, r in rows.items():
        res, ext = r["results"], r["extraction"]
        lat = {}
        if not args.skip_latency:
            lat = measure_latency(key, calls, device)
            if args.cpu:
                lat.update(measure_latency(key, calls[:3], torch.device("cpu"), cpu=True))
            print(f"[{key}] GPU latency per call: median {lat['gpu_latency_median_s'] * 1000:.0f} ms "
                  f"(backbone {lat['gpu_backbone_median_s'] * 1000:.0f} ms, {lat['gpu_ms_per_chunk']:.1f} ms/chunk)"
                  + (f", CPU median {lat['cpu_latency_median_s']:.2f} s" if args.cpu else ""))
        m = res["mlp"]["val"]
        cal = r["calibration"].get("mlp", {})
        row = {"backbone": key, "model": config.BACKBONES[key]["display"], "hf_name": ext["hf_name"],
               "params_m": ext["params_m"], "weights_mb": ext.get("weights_mb", np.nan), "embedding_dim": ext["hidden_size"],
               "n_layers": ext["num_hidden_layers"], "selected_layer": res["layer"],
               "extraction_total_s": ext["total_s"], "extraction_backbone_s": sum(s["backbone_s"] for s in ext["splits"].values()),
               "realtime_factor": ext["backbone_realtime_factor"], "extraction_peak_vram_mb": ext.get("peak_vram_allocated_mb", np.nan),
               "cache_mb": sum(s["cache_mb"] for s in ext["splits"].values()),
               "probe_time_s": res["probe_time_s"], "mlp_train_time_s": res["mlp_train_time_s"], "mlp_best_epoch": res["mlp_best_epoch"],
               "val_accuracy": m["accuracy"], "val_balanced_accuracy": m["balanced_accuracy"], "val_f1": m["f1"],
               "val_precision": m["precision"], "val_recall": m["recall"], "val_auc": m["auc"], "val_eer": m["eer"],
               "val_one_minus_eer": 1 - m["eer"], "val_far": m["far"], "val_frr": m["frr"], "val_brier_raw": m["brier"],
               "val_brier_calibrated_cv": cal.get(cal.get("chosen", "platt") + "_cv", {}).get("brier", np.nan),
               "val_ece_raw": m["ece"], "val_ece_calibrated_cv": cal.get(cal.get("chosen", "platt") + "_cv", {}).get("ece", np.nan),
               "train_accuracy": res["mlp"]["train"]["accuracy"], "train_auc": res["mlp"]["train"]["auc"],
               "logreg_val_auc": res["logreg"]["val"]["auc"], "logreg_val_accuracy": res["logreg"]["val"]["accuracy"],
               "logreg_val_eer": res["logreg"]["val"]["eer"], **lat}
        st = r["stress"]
        if st:
            row.update({"stress_mean_auc": st["mlp_mean_auc_stressed"], "stress_mean_accuracy": st["mlp_mean_accuracy_stressed"],
                        "stress_worst_auc": st["mlp_worst_auc"], "stress_worst_condition": st["mlp_worst_condition"],
                        "stress_mean_auc_logreg": st["logreg_mean_auc_stressed"],
                        **{f"stress_{c}_auc": st["conditions"][c]["mlp"]["auc"] for c in st["conditions"]},
                        **{f"stress_{c}_acc": st["conditions"][c]["mlp"]["accuracy"] for c in st["conditions"]}})
        else:
            row.update({"stress_mean_auc": np.nan, "stress_mean_accuracy": np.nan, "stress_worst_auc": np.nan, "stress_worst_condition": ""})
        table.append(row)
        for clf, label in (("logreg", "logistic regression"), ("mlp", "MLP"), ("call_embedding_logreg", "call-embedding logreg")):
            for split in ("train", "val"):
                mm = res[clf][split]
                comparison.append({"backbone": key, "model": config.BACKBONES[key]["display"], "layer": res["layer"],
                                   "classifier": label, "split": split, **{k: mm[k] for k in
                                   ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "auc", "eer", "far", "frr", "brier", "n_calls")}})
    table = pd.DataFrame(table)
    comparison = pd.DataFrame(comparison)
    shortcut = R / "shortcut_check.csv"
    if shortcut.exists():
        s = pd.read_csv(shortcut)
        best = s.sort_values("val_auc", ascending=False).iloc[0]
        comparison.loc[len(comparison)] = {"backbone": "shortcut", "model": f"shortcut ({best['feature_set']}, {best['model']})",
                                           "layer": "-", "classifier": "trivial statistics", "split": "val",
                                           "accuracy": best["val_accuracy"], "balanced_accuracy": best["val_balanced_accuracy"],
                                           "f1": best["val_f1"], "auc": best["val_auc"], "eer": best["val_eer"], "n_calls": 71}
    R.mkdir(parents=True, exist_ok=True)
    table.to_csv(R / "benchmark.csv", index=False)
    comparison.to_csv(R / "model_comparison.csv", index=False)

    # winner. The clean validation split is saturated (every backbone reaches AUC ~1.0), so the selection score is the
    # MEAN VALIDATION AUC UNDER THE STRESS CONDITIONS (test-time perturbations of the caller channel; stress_test.py).
    # Ties (< 0.005) are broken by stressed accuracy, then clean EER, then lower latency. Without stress results the
    # rule falls back to clean validation AUC.
    use_stress = table["stress_mean_auc"].notna().all()
    key_col = "stress_mean_auc" if use_stress else "val_auc"
    ranked = table.sort_values([key_col, "val_eer", "val_accuracy"], ascending=[False, True, False]).reset_index(drop=True)
    top = ranked.iloc[0]
    contenders = ranked[ranked[key_col] >= top[key_col] - 0.005]
    if len(contenders) > 1:
        sort_cols = ["stress_mean_accuracy", "val_eer"] if use_stress else ["val_eer", "val_accuracy"]
        asc = [False, True] if use_stress else [True, False]
        if "gpu_latency_median_s" in contenders:
            sort_cols, asc = sort_cols + ["gpu_latency_median_s"], asc + [True]
        contenders = contenders.sort_values(sort_cols, ascending=asc)
        top = contenders.iloc[0]
    n_cond = len(rows[top["backbone"]]["stress"]["conditions"]) - 1 if use_stress else 0
    if use_stress:
        reason = (f"highest mean validation AUC under the {n_cond} stress conditions ({top['stress_mean_auc']:.4f}; stressed accuracy "
                  f"{top['stress_mean_accuracy']:.3f}, worst condition {top['stress_worst_condition']} AUC {top['stress_worst_auc']:.3f}); "
                  f"clean validation accuracy {top['val_accuracy']:.3f} / AUC {top['val_auc']:.4f} - the clean split is saturated for every "
                  f"backbone and cannot rank them; {len(contenders)} model(s) within 0.005 -> tie-break by stressed accuracy, EER, latency")
    else:
        reason = (f"highest validation AUC ({top['val_auc']:.4f}); EER {top['val_eer'] * 100:.2f}%, accuracy {top['val_accuracy']:.3f}; "
                  f"{len(contenders)} model(s) within 0.005 AUC -> tie-break by EER, accuracy, latency")
    best = {"backbone": top["backbone"], "classifier": "mlp", "layer": int(top["selected_layer"]), "reason": reason,
            "val_metrics": {k: float(top[k]) for k in ("val_accuracy", "val_balanced_accuracy", "val_f1", "val_auc", "val_eer")},
            "stress_metrics": ({k: float(top[k]) for k in ("stress_mean_auc", "stress_mean_accuracy", "stress_worst_auc")} |
                               {"stress_worst_condition": str(top["stress_worst_condition"])}) if use_stress else {},
            "gpu_latency_median_s": float(top.get("gpu_latency_median_s", np.nan)),
            "selection_rule": "mean validation AUC over stress conditions (MLP)" if use_stress else "validation AUC (MLP)"}
    save_json(best, R / "best_model.json")

    fig_layer_probes(rows)
    fig_metric_bars(table)
    fig_roc(rows)
    fig_confusions(rows)
    fig_resources(table)
    fig_aggregation(rows)
    print("\nBENCHMARK (validation, MLP on the chosen layer)")
    cols = ["model", "selected_layer", "params_m", "val_accuracy", "val_balanced_accuracy", "val_f1", "val_auc", "val_eer",
            "stress_mean_auc", "stress_mean_accuracy", "stress_worst_auc", "stress_worst_condition", "gpu_latency_median_s", "extraction_peak_vram_mb"]
    print(table[[c for c in cols if c in table]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nWINNER: {config.BACKBONES[best['backbone']]['display']} layer {best['layer']} + MLP  ({reason})")
    log_experiment(title="benchmark: three frozen backbones", model=", ".join(table["hf_name"]), dataset_fraction=1.0,
                   segmentation="shared", selected_layer=dict(zip(table["backbone"], table["selected_layer"])),
                   classifier="MLP (same for all)", hyperparameters="see train_classifier", device=str(device),
                   runtime_s="n/a", winner=f"{best['backbone']} layer {best['layer']}", reason=reason,
                   val_auc=dict(zip(table["backbone"], table["val_auc"].round(4))),
                   val_accuracy=dict(zip(table["backbone"], table["val_accuracy"].round(4))),
                   stress_mean_auc=dict(zip(table["backbone"], table["stress_mean_auc"].round(4))))


if __name__ == "__main__":
    main()
