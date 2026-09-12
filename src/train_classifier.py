"""
STAGE 3 - SMALL CLASSIFIERS ON THE CACHED EMBEDDINGS  (the same recipe for every backbone)

  1. LAYER PROBE: one logistic regression per hidden layer, trained on TRAIN chunks, scored on VAL calls
     (chunk log-odds averaged per call). Which layer of the frozen backbone separates human from synthetic best?
  2. The layer is chosen on VALIDATION ONLY (highest call-level AUC; ties -> lower EER -> lower layer).
  3. On that layer: logistic regression (linear probe) and a small MLP
        Linear(dim, 256) -> ReLU -> Dropout(0.3) -> Linear(256, 2)
     trained on chunks with class-weighted cross-entropy, AdamW, early stopping on validation loss.
  4. Call-level aggregation: the pre-registered default is the MEAN of the chunk log-odds; the other
     aggregations are reported as diagnostics (choosing one on validation would be mild peeking).
  5. A "call-level embedding" variant (average the chunk embeddings first, then classify the call) is also
     reported for comparison.

Run:  python src/train_classifier.py --backbone wavlm            (or wav2vec2_spanish | xlsr | all)
      python src/train_classifier.py --backbone all --pooling meanstd      (diagnostic: mean+std pooling)
Outputs: outputs/classifier/<backbone>[_meanstd]/{layer_probe.json, results.json, mlp_history.json, *.png}
         outputs/predictions/<backbone>_{train,val}.csv  (one row per call, scores of both classifiers)
         models/<backbone>/{logreg.joblib, mlp.pt, meta.json}
"""
import argparse
import copy
import time

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import _bootstrap  # noqa: F401
import config
from metrics import (AGGREGATIONS, aggregate_chunk_scores, aggregate_weighted, auc, compute_eer, evaluate_scores,
                     format_metrics, log_experiment, save_json)


# ----------------------------------------------------------------------------- data
def load_embeddings(backbone, split, pooling="mean"):
    """index DataFrame (one row per chunk) + array (n_chunks, n_layers+1, feat_dim) float32."""
    d = config.EMBEDDINGS_DIR / backbone
    index = pd.read_csv(d / f"{split}_index.csv")
    mean = np.load(d / f"{split}_mean.npy").astype(np.float32)
    if pooling == "mean":
        return index, mean
    std = np.load(d / f"{split}_std.npy").astype(np.float32)
    return index, np.concatenate([mean, std], axis=2)


def call_table(index, chunk_scores, method=config.CHUNK_AGGREGATION):
    """Chunk-level scores -> one row per call: anon_id, y, score, n_chunks (aggregation of the log-odds)."""
    df = index[["anon_id", "y", "duration_s"]].copy()
    df["chunk_score"] = np.asarray(chunk_scores, dtype=float)
    rows = []
    for anon_id, g in df.groupby("anon_id", sort=True):
        if method == "duration_weighted":
            score = aggregate_weighted(g["chunk_score"], g["duration_s"])
        else:
            score = aggregate_chunk_scores(g["chunk_score"].to_numpy(), method)
        rows.append({"anon_id": anon_id, "y": int(g["y"].iloc[0]), "score": score, "n_chunks": len(g),
                     "chunk_scores": g["chunk_score"].round(3).tolist()})
    return pd.DataFrame(rows)


def make_logreg():
    return make_pipeline(StandardScaler(), LogisticRegression(C=config.LOGREG_C, class_weight="balanced", max_iter=5000))


# ----------------------------------------------------------------------------- MLP
class MLP(nn.Module):
    """dim -> 256 -> 2, with train-set standardisation stored as buffers (saved with the weights)."""

    def __init__(self, dim, hidden=config.MLP_HIDDEN, dropout=config.MLP_DROPOUT, mean=None, std=None):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(dim) if std is None else torch.as_tensor(std, dtype=torch.float32))
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def scores(self, x):
        """log-odds of synthetic for a (n, dim) tensor (eval mode, no gradients)."""
        self.eval()
        with torch.no_grad():
            out = []
            for i in range(0, len(x), 2048):
                logits = self(x[i:i + 2048])
                out.append(logits[:, 1] - logits[:, 0])
            return torch.cat(out).cpu().numpy()


def class_weights(y):
    counts = np.bincount(y, minlength=2)
    return torch.tensor(len(y) / (2 * counts), dtype=torch.float32)


def train_mlp(Xtr, ytr, Xval, yval, val_index, device, verbose=True):
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    Xtr_t, ytr_t = torch.tensor(Xtr, device=device), torch.tensor(ytr, device=device)
    Xval_t, yval_t = torch.tensor(Xval, device=device), torch.tensor(yval, device=device)
    model = MLP(Xtr.shape[1], mean=Xtr.mean(0), std=Xtr.std(0) + 1e-6).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr).to(device))
    val_loss_fn = nn.CrossEntropyLoss(weight=class_weights(yval).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.MLP_LR, weight_decay=config.MLP_WEIGHT_DECAY)
    gen = torch.Generator(device=device).manual_seed(config.SEED)
    history, best_state, best_loss, best_epoch, bad = [], None, float("inf"), 0, 0
    for epoch in range(1, config.MLP_EPOCHS + 1):
        model.train()
        order = torch.randperm(len(Xtr_t), device=device, generator=gen)
        for start in range(0, len(order), config.MLP_BATCH_SIZE):
            idx = order[start:start + config.MLP_BATCH_SIZE]
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            tr_logits, va_logits = model(Xtr_t), model(Xval_t)
            tr_loss, va_loss = loss_fn(tr_logits, ytr_t).item(), val_loss_fn(va_logits, yval_t).item()
            tr_acc = (tr_logits.argmax(1) == ytr_t).float().mean().item()
            va_acc = (va_logits.argmax(1) == yval_t).float().mean().item()
            va_scores = (va_logits[:, 1] - va_logits[:, 0]).cpu().numpy()
        calls = call_table(val_index, va_scores)
        h = {"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss, "train_chunk_acc": tr_acc,
             "val_chunk_acc": va_acc, "val_call_auc": auc(calls["y"], calls["score"]),
             "val_call_eer": compute_eer(calls["y"], calls["score"])[0],
             "val_call_acc": float(((calls["score"] >= 0).astype(int) == calls["y"]).mean())}
        history.append(h)
        if verbose:
            print(f"  epoch {epoch:3d}  train loss {tr_loss:.4f}  val loss {va_loss:.4f}  chunk acc {tr_acc:.3f}/{va_acc:.3f}"
                  f"  val call AUC {h['val_call_auc']:.4f}  EER {h['val_call_eer'] * 100:.2f}%  acc {h['val_call_acc']:.3f}")
        if va_loss < best_loss:
            best_loss, best_state, best_epoch, bad = va_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= config.MLP_PATIENCE:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch


# ----------------------------------------------------------------------------- plots
def plot_layer_probe(probe, best_layer, backbone, path):
    layers = [p["layer"] for p in probe]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].plot(layers, [p["val"]["auc"] for p in probe], "o-", label="validation AUC (call level)")
    axes[0].plot(layers, [p["train"]["auc"] for p in probe], "s--", alpha=0.5, label="train AUC (fit data; optimistic)")
    axes[0].set_ylabel("ROC AUC")
    axes[0].set_ylim(min(0.5, min(p["val"]["auc"] for p in probe) - 0.02), 1.005)
    axes[1].plot(layers, [p["val"]["eer"] * 100 for p in probe], "o-", color="#d62728", label="validation EER (%)")
    axes[1].plot(layers, [p["val"]["accuracy"] * 100 for p in probe], "^-", color="#2ca02c", label="validation accuracy (%)")
    axes[1].set_ylabel("%")
    for ax in axes:
        ax.axvline(best_layer, color="gray", ls=":", label=f"chosen layer = {best_layer}")
        ax.set_xticks(layers, ["CNN\n(0)"] + [str(l) for l in layers[1:]], fontsize=7)
        ax.set_xlabel("hidden layer (0 = CNN feature encoder output, 1..N = transformer layers)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Layer probe - {config.BACKBONES[backbone]['display']}: logistic regression on each frozen layer")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_mlp_history(history, best_epoch, backbone, path):
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", ms=3, label="train loss (chunks)")
    axes[0].plot(ep, [h["val_loss"] for h in history], "o-", ms=3, label="validation loss (chunks)")
    axes[0].axvline(best_epoch, color="gray", ls="--", label=f"kept epoch {best_epoch} (lowest val loss)")
    axes[0].set_title("loss")
    axes[1].plot(ep, [h["train_chunk_acc"] for h in history], "o-", ms=3, label="train chunk accuracy")
    axes[1].plot(ep, [h["val_chunk_acc"] for h in history], "o-", ms=3, label="validation chunk accuracy")
    axes[1].plot(ep, [h["val_call_acc"] for h in history], "^-", ms=3, label="validation CALL accuracy")
    axes[1].set_title("accuracy")
    axes[2].plot(ep, [h["val_call_auc"] for h in history], "o-", ms=3, color="black", label="validation call AUC")
    axes[2].plot(ep, [1 - h["val_call_eer"] for h in history], "s-", ms=3, color="#d62728", label="1 - validation call EER")
    axes[2].set_title("validation ranking quality")
    for ax in axes:
        ax.axvline(best_epoch, color="gray", ls="--")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"MLP on frozen {config.BACKBONES[backbone]['display']} embeddings: train vs validation")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_score_hist(calls, backbone, path, title):
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    lo, hi = calls["score"].min() - 0.5, calls["score"].max() + 0.5
    bins = np.linspace(lo, hi, 40)
    for y, name, color in [(0, "human", "#1f77b4"), (1, "synthetic", "#d62728")]:
        ax.hist(calls.loc[calls["y"] == y, "score"], bins=bins, alpha=0.6, color=color, label=name)
    ax.axvline(0, color="black", ls="--", label="decision threshold (p = 0.5)")
    ax.set_xlabel("call score (mean chunk log-odds of synthetic)")
    ax.set_ylabel("calls")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def run(backbone, pooling, forced_layer, device):
    tag = backbone if pooling == "mean" else f"{backbone}_{pooling}"
    out = config.OUTPUTS_DIR / "classifier" / tag
    out.mkdir(parents=True, exist_ok=True)
    model_dir = config.MODELS_DIR / tag
    model_dir.mkdir(parents=True, exist_ok=True)
    idx_tr, E_tr = load_embeddings(backbone, "train", pooling)
    idx_va, E_va = load_embeddings(backbone, "val", pooling)
    ytr, yva = idx_tr["y"].to_numpy(), idx_va["y"].to_numpy()
    n_layers = E_tr.shape[1]
    print(f"\n=== {config.BACKBONES[backbone]['display']} ({pooling} pooling): train {E_tr.shape}, val {E_va.shape}, "
          f"{idx_tr['anon_id'].nunique()} / {idx_va['anon_id'].nunique()} calls")
    results = {"backbone": backbone, "pooling": pooling, "n_layers": n_layers, "feat_dim": int(E_tr.shape[2]),
               "n_train_chunks": int(len(ytr)), "n_val_chunks": int(len(yva)),
               "n_train_calls": int(idx_tr["anon_id"].nunique()), "n_val_calls": int(idx_va["anon_id"].nunique())}

    # 1) layer probe ---------------------------------------------------------------------------------
    t0 = time.time()
    probe = []
    for layer in range(n_layers):
        clf = make_logreg().fit(E_tr[:, layer], ytr)
        tr_calls = call_table(idx_tr, clf.decision_function(E_tr[:, layer]))
        va_calls = call_table(idx_va, clf.decision_function(E_va[:, layer]))
        va_chunk_auc = auc(yva, clf.decision_function(E_va[:, layer]))
        probe.append({"layer": layer, "train": evaluate_scores(tr_calls["y"], tr_calls["score"]),
                      "val": evaluate_scores(va_calls["y"], va_calls["score"]), "val_chunk_auc": va_chunk_auc})
        v = probe[-1]["val"]
        print(f"  layer {layer:2d}: val AUC {v['auc']:.4f}  EER {v['eer'] * 100:5.2f}%  acc {v['accuracy']:.3f}  "
              f"(chunk AUC {va_chunk_auc:.4f})   train AUC {probe[-1]['train']['auc']:.4f}")
    results["probe_time_s"] = time.time() - t0
    if forced_layer is None:
        best = max(probe, key=lambda p: (round(p["val"]["auc"], 4), -round(p["val"]["eer"], 4), -p["layer"]))
        best_layer = best["layer"]
    else:
        best_layer = forced_layer
    results["layer"] = best_layer
    results["probe"] = probe
    save_json(probe, out / "layer_probe.json")
    plot_layer_probe(probe, best_layer, backbone, out / "layer_probe.png")
    print(f"  -> chosen layer {best_layer} (validation AUC {probe[best_layer]['val']['auc']:.4f})")
    Xtr, Xva = E_tr[:, best_layer], E_va[:, best_layer]

    # 2) logistic regression on the chosen layer ----------------------------------------------------
    t0 = time.time()
    logreg = make_logreg().fit(Xtr, ytr)
    results["logreg_train_time_s"] = time.time() - t0
    lr_tr, lr_va = logreg.decision_function(Xtr), logreg.decision_function(Xva)
    calls_lr = {"train": call_table(idx_tr, lr_tr), "val": call_table(idx_va, lr_va)}
    results["logreg"] = {s: evaluate_scores(c["y"], c["score"]) for s, c in calls_lr.items()}
    results["logreg"]["val_chunk"] = evaluate_scores(yva, lr_va)
    print(f"  logreg   val: {format_metrics(results['logreg']['val'])}")
    joblib.dump(logreg, model_dir / "logreg.joblib")

    # 3) MLP -----------------------------------------------------------------------------------------
    t0 = time.time()
    mlp, history, best_epoch = train_mlp(Xtr, ytr, Xva, yva, idx_va, device, verbose=False)
    results["mlp_train_time_s"] = time.time() - t0
    results["mlp_best_epoch"] = best_epoch
    results["mlp_epochs_run"] = len(history)
    results["mlp_params"] = int(sum(p.numel() for p in mlp.parameters()))
    save_json(history, out / "mlp_history.json")
    plot_mlp_history(history, best_epoch, backbone, out / "mlp_curves.png")
    mlp_tr = mlp.scores(torch.tensor(Xtr, device=device))
    mlp_va = mlp.scores(torch.tensor(Xva, device=device))
    calls_mlp = {"train": call_table(idx_tr, mlp_tr), "val": call_table(idx_va, mlp_va)}
    results["mlp"] = {s: evaluate_scores(c["y"], c["score"]) for s, c in calls_mlp.items()}
    results["mlp"]["val_chunk"] = evaluate_scores(yva, mlp_va)
    print(f"  MLP      val: {format_metrics(results['mlp']['val'])}   (best epoch {best_epoch}, {results['mlp_train_time_s']:.1f}s)")
    torch.save({"state_dict": mlp.state_dict(), "dim": int(Xtr.shape[1]), "hidden": config.MLP_HIDDEN,
                "dropout": config.MLP_DROPOUT}, model_dir / "mlp.pt")
    plot_score_hist(calls_mlp["val"], backbone, out / "val_score_hist.png",
                    f"{config.BACKBONES[backbone]['display']} layer {best_layer} + MLP: validation call scores")

    # 4) aggregation diagnostics (validation) -------------------------------------------------------
    results["aggregation"] = {}
    for name, chunk_scores in (("logreg", lr_va), ("mlp", mlp_va)):
        results["aggregation"][name] = {}
        for method in AGGREGATIONS:
            c = call_table(idx_va, chunk_scores, method)
            m = evaluate_scores(c["y"], c["score"])
            results["aggregation"][name][method] = {k: m[k] for k in ("auc", "eer", "accuracy", "balanced_accuracy", "f1", "brier")}
    agg = results["aggregation"]["mlp"]
    print("  aggregation (MLP, val): " + "  ".join(f"{k}: AUC {v['auc']:.3f}/acc {v['accuracy']:.3f}" for k, v in agg.items()))

    # 5) call-level embedding variant ---------------------------------------------------------------
    def call_embeddings(index, X):
        g = index.groupby("anon_id", sort=True).indices
        ids = sorted(g)
        return ids, np.stack([X[g[i]].mean(0) for i in ids]), np.array([int(index["y"].iloc[g[i][0]]) for i in ids])
    ids_tr, Ctr, cy_tr = call_embeddings(idx_tr, Xtr)
    ids_va, Cva, cy_va = call_embeddings(idx_va, Xva)
    clf_c = make_logreg().fit(Ctr, cy_tr)
    results["call_embedding_logreg"] = {"train": evaluate_scores(cy_tr, clf_c.decision_function(Ctr)),
                                        "val": evaluate_scores(cy_va, clf_c.decision_function(Cva))}
    print(f"  call-embedding logreg val: {format_metrics(results['call_embedding_logreg']['val'])}")

    # 6) predictions + metadata ----------------------------------------------------------------------
    for split, c_lr, c_mlp in (("train", calls_lr["train"], calls_mlp["train"]), ("val", calls_lr["val"], calls_mlp["val"])):
        pred = c_mlp.rename(columns={"score": "score_mlp", "chunk_scores": "chunk_scores_mlp"})
        pred["score_logreg"] = c_lr["score"].to_numpy()
        pred["prob_mlp"] = 1 / (1 + np.exp(-pred["score_mlp"]))
        pred["pred_mlp"] = (pred["score_mlp"] >= 0).astype(int)
        pred["backbone"], pred["layer"], pred["pooling"] = backbone, best_layer, pooling
        pred.to_csv(config.PREDICTIONS_DIR / f"{tag}_{split}.csv", index=False)
    meta = {"backbone": backbone, "hf_name": config.BACKBONES[backbone]["hf_name"], "layer": best_layer,
            "pooling": pooling, "aggregation": config.CHUNK_AGGREGATION, "chunk_seconds": config.CHUNK_SECONDS,
            "target_rms_db": config.TARGET_RMS_DB, "mlp_best_epoch": best_epoch, "seed": config.SEED,
            "val_metrics_mlp": results["mlp"]["val"], "val_metrics_logreg": results["logreg"]["val"]}
    save_json(meta, model_dir / "meta.json")
    save_json(results, out / "results.json")
    log_experiment(title=f"frozen {backbone} ({pooling} pooling) - layer probe + logreg + MLP",
                   model=config.BACKBONES[backbone]["hf_name"], dataset_fraction=1.0,
                   segmentation=f"VAD + {config.CHUNK_SECONDS:.0f}s chunks, RMS {config.TARGET_RMS_DB} dB",
                   selected_layer=best_layer, classifier=f"logreg C={config.LOGREG_C}; MLP {Xtr.shape[1]}-{config.MLP_HIDDEN}-2",
                   hyperparameters=f"lr={config.MLP_LR}, wd={config.MLP_WEIGHT_DECAY}, batch={config.MLP_BATCH_SIZE}, "
                                   f"dropout={config.MLP_DROPOUT}, patience={config.MLP_PATIENCE}, seed={config.SEED}",
                   device=str(device), runtime_s=round(results["probe_time_s"] + results["mlp_train_time_s"], 1),
                   val_logreg=format_metrics(results["logreg"]["val"]), val_mlp=format_metrics(results["mlp"]["val"]),
                   notes=f"MLP best epoch {best_epoch}; aggregation = mean of chunk log-odds")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all", choices=["all"] + config.BACKBONE_ORDER)
    parser.add_argument("--pooling", default="mean", choices=["mean", "meanstd"])
    parser.add_argument("--layer", type=int, default=None, help="force a layer instead of probing")
    args = parser.parse_args()
    config.set_seed()
    config.ensure_dirs()
    device = config.get_device()
    keys = config.BACKBONE_ORDER if args.backbone == "all" else [args.backbone]
    for key in keys:
        if not (config.EMBEDDINGS_DIR / key / "train_mean.npy").exists():
            print(f"[{key}] no cached embeddings -> run extract_embeddings.py --backbone {key} first")
            continue
        run(key, args.pooling, args.layer, device)


if __name__ == "__main__":
    main()
