"""
SANITY CHECKS - can a 100% validation score be trusted?

Four tests on the cached embeddings of the winning backbone (nothing is re-extracted):

  1. LABEL PERMUTATION. Train the classifier on SHUFFLED training labels and score validation. If the pipeline
     leaked anything (validation data inside training, labels encoded in the features, a metric bug) the shuffled
     model would still score high. An honest pipeline gives chance (AUC ~0.5).
  2. HOW LITTLE AUDIO IS ENOUGH. Score every validation call from ONE chunk only (its first / a random chunk) and
     from progressively fewer chunks. If one 1-4 s chunk already gives ~100%, the cue is gross, not subtle.
  3. HOW MANY DISTINCT VOICES. Cluster the call-level embeddings of each class (agglomerative, cosine) and report
     the silhouette-best number of clusters: an estimate of how many TTS voices / human speakers the data contains,
     and whether validation synthetic calls fall into clusters that also exist in training.
  4. LEAVE-ONE-VOICE-OUT. For every synthetic voice cluster: remove ALL its calls from training, retrain, and test on
     the removed calls. This is the judges' scenario ("voices that appear in neither split"). If accuracy collapses,
     the model memorises voices; if it stays high, the evidence transfers across voices.
  4b. NEAREST-NEIGHBOUR SIMILARITY. Cosine similarity of every validation call to its nearest training call of the
     same class: much higher for synthetic than for human calls = the same voices recur.

Run:  python src/sanity_checks.py [--backbone wav2vec2_spanish]
Outputs: outputs/sanity/<backbone>/results.json, outputs/figures/sanity_<backbone>.png
"""
import argparse
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

import _bootstrap  # noqa: F401
import config
from metrics import evaluate_scores, log_experiment, save_json
from train_classifier import call_table, load_embeddings, make_logreg


def call_level(index, X):
    """Mean embedding per call (L2-normalised) + labels + ids, in sorted anon_id order."""
    g = index.groupby("anon_id", sort=True).indices
    ids = sorted(g)
    C = np.stack([X[g[i]].mean(0) for i in ids])
    C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-9)
    y = np.array([int(index["y"].iloc[g[i][0]]) for i in ids])
    return ids, C, y


def fit_score(Xtr, ytr, idx_va, Xva):
    clf = make_logreg().fit(Xtr, ytr)
    calls = call_table(idx_va, clf.decision_function(Xva))
    return evaluate_scores(calls["y"], calls["score"]), calls


def permutation_test(Xtr, ytr, idx_va, Xva, n=5):
    rng = np.random.default_rng(config.SEED)
    out = []
    for i in range(n):
        yp = rng.permutation(ytr)
        m, _ = fit_score(Xtr, yp, idx_va, Xva)
        out.append({"auc": m["auc"], "accuracy": m["accuracy"]})
        print(f"  permutation {i + 1}: val AUC {m['auc']:.3f}  accuracy {m['accuracy']:.3f}")
    return out


def few_chunks_test(Xtr, ytr, idx_va, Xva):
    clf = make_logreg().fit(Xtr, ytr)
    s = clf.decision_function(Xva)
    df = idx_va[["anon_id", "y", "chunk", "duration_s"]].copy()
    df["score"] = s
    rng = np.random.default_rng(config.SEED)
    res = {}
    for name in ("first_chunk", "random_chunk", "first_3_chunks", "all_chunks"):
        rows = []
        for aid, g in df.groupby("anon_id", sort=True):
            g = g.sort_values("chunk")
            if name == "first_chunk":
                sc = g["score"].iloc[0]
            elif name == "random_chunk":
                sc = g["score"].iloc[rng.integers(0, len(g))]
            elif name == "first_3_chunks":
                sc = g["score"].iloc[:3].mean()
            else:
                sc = g["score"].mean()
            rows.append({"y": int(g["y"].iloc[0]), "score": sc})
        r = pd.DataFrame(rows)
        m = evaluate_scores(r["y"], r["score"])
        res[name] = {"auc": m["auc"], "accuracy": m["accuracy"]}
        print(f"  {name:<15} val AUC {m['auc']:.4f}  accuracy {m['accuracy']:.3f}")
    # chunk-level: single chunks judged alone, by duration bucket
    m_chunk = evaluate_scores(df["y"], df["score"])
    short = df[df["duration_s"] < 2.0]
    m_short = evaluate_scores(short["y"], short["score"]) if short["y"].nunique() == 2 else None
    res["single_chunk_all"] = {"auc": m_chunk["auc"], "accuracy": m_chunk["accuracy"], "n": int(len(df))}
    if m_short:
        res["single_chunk_under_2s"] = {"auc": m_short["auc"], "accuracy": m_short["accuracy"], "n": int(len(short))}
        print(f"  single chunks: all {m_chunk['accuracy']:.3f} acc (n={len(df)}), shorter than 2 s {m_short['accuracy']:.3f} acc (n={len(short)})")
    return res


def cluster_voices(C, ids, split_of, max_k=20):
    """Agglomerative clustering (cosine, average linkage); k chosen by silhouette. Returns labels and diagnostics."""
    best = None
    scores = {}
    for k in range(2, min(max_k, len(C) - 1) + 1):
        lab = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average").fit_predict(C)
        sil = silhouette_score(C, lab, metric="cosine")
        scores[k] = sil
        if best is None or sil > best[1]:
            best = (k, sil, lab)
    k, sil, lab = best
    table = pd.DataFrame({"id": ids, "cluster": lab, "split": [split_of[i] for i in ids]})
    per = table.groupby("cluster")["split"].value_counts().unstack(fill_value=0)
    return lab, {"best_k": int(k), "silhouette": float(sil), "silhouette_per_k": {int(a): float(b) for a, b in scores.items()},
                 "clusters": {int(c): {"train": int(per.loc[c].get("train", 0)), "val": int(per.loc[c].get("val", 0))} for c in per.index}}


def leave_one_cluster_out(idx_tr, Xtr, idx_va, Xva, cluster_of, y_class, name):
    """Hold out one cluster of class `y_class` at a time (its calls from train AND val), train on the rest of train,
    test on held-out calls + the opposite-class validation calls."""
    idx_all = pd.concat([idx_tr.assign(split="train"), idx_va.assign(split="val")], ignore_index=True)
    X_all = np.concatenate([Xtr, Xva])
    idx_all["cluster"] = idx_all["anon_id"].map(cluster_of)
    results = []
    for c in sorted(set(v for v in cluster_of.values())):
        held = (idx_all["cluster"] == c) & (idx_all["y"] == y_class)
        train_mask = (idx_all["split"] == "train") & ~held
        other_val = (idx_all["split"] == "val") & (idx_all["y"] != y_class)
        test_mask = held | other_val
        clf = make_logreg().fit(X_all[train_mask.to_numpy()], idx_all.loc[train_mask, "y"].to_numpy())
        te = idx_all[test_mask].reset_index(drop=True)
        calls = call_table(te, clf.decision_function(X_all[test_mask.to_numpy()]))
        m = evaluate_scores(calls["y"], calls["score"])
        held_calls = calls[calls["y"] == y_class]
        held_acc = float(((held_calls["score"] >= 0).astype(int) == y_class).mean())
        n_held = int(held_calls.shape[0])
        results.append({"cluster": int(c), "n_held_out_calls": n_held, "held_out_accuracy": held_acc, "auc": m["auc"], "accuracy": m["accuracy"]})
        print(f"  {name} cluster {c:2d}: {n_held:3d} calls held out -> recognised {held_acc:.3f}   (AUC vs val other class {m['auc']:.3f})")
    return results


def nearest_neighbour(Ctr, ytr, Cva, yva):
    out = {}
    for y, label in ((0, "human"), (1, "synthetic")):
        S = Cva[yva == y] @ Ctr[ytr == y].T
        nn = S.max(axis=1)
        out[label] = {"median_nn_similarity": float(np.median(nn)), "min": float(nn.min()), "max": float(nn.max())}
        print(f"  {label:<10} val->train nearest-neighbour cosine: median {np.median(nn):.3f} (min {nn.min():.3f}, max {nn.max():.3f})")
    # within-train same-class similarity for reference (excluding self)
    for y, label in ((0, "human"), (1, "synthetic")):
        S = Ctr[ytr == y] @ Ctr[ytr == y].T
        np.fill_diagonal(S, -1)
        out[label]["train_within_class_nn_median"] = float(np.median(S.max(axis=1)))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()
    config.set_seed()
    backbone = args.backbone or json.loads((config.REPORTS_DIR / "best_model.json").read_text())["backbone"]
    layer = int(json.loads((config.MODELS_DIR / backbone / "meta.json").read_text())["layer"])
    out = config.OUTPUTS_DIR / "sanity" / backbone
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    idx_tr, E_tr = load_embeddings(backbone, "train")
    idx_va, E_va = load_embeddings(backbone, "val")
    Xtr, Xva = E_tr[:, layer], E_va[:, layer]
    ytr = idx_tr["y"].to_numpy()
    print(f"=== {config.BACKBONES[backbone]['display']} layer {layer}: {len(ytr)} train chunks, {len(idx_va)} val chunks")
    res = {"backbone": backbone, "layer": layer}

    print("\n1) label permutation test (should be ~0.5 AUC)")
    res["permutation"] = permutation_test(Xtr, ytr, idx_va, Xva)

    print("\n2) how little audio is enough (logistic regression, validation)")
    res["few_chunks"] = few_chunks_test(Xtr, ytr, idx_va, Xva)

    print("\n3) how many distinct voices (clustering of call embeddings)")
    ids_tr, Ctr, cy_tr = call_level(idx_tr, Xtr)
    ids_va, Cva, cy_va = call_level(idx_va, Xva)
    ids_all = ids_tr + ids_va
    C_all, y_all = np.concatenate([Ctr, Cva]), np.concatenate([cy_tr, cy_va])
    split_of = {i: "train" for i in ids_tr} | {i: "val" for i in ids_va}
    res["clusters"] = {}
    cluster_of = {}
    for y, label in ((1, "synthetic"), (0, "human")):
        sel = y_all == y
        lab, diag = cluster_voices(C_all[sel], [i for i, s in zip(ids_all, sel) if s], split_of)
        res["clusters"][label] = diag
        cluster_of[label] = {i: int(c) for i, c in zip([i for i, s in zip(ids_all, sel) if s], lab)}
        shared = sum(1 for c, v in diag["clusters"].items() if v["train"] > 0 and v["val"] > 0)
        print(f"  {label:<10}: best k = {diag['best_k']} clusters (silhouette {diag['silhouette']:.3f}); "
              f"{shared} of {diag['best_k']} clusters contain both train and val calls")
        for c, v in diag["clusters"].items():
            print(f"     cluster {c:2d}: train {v['train']:3d}  val {v['val']:3d}")

    print("\n4) leave-one-voice-cluster-out (synthetic)")
    res["loo_synthetic"] = leave_one_cluster_out(idx_tr, Xtr, idx_va, Xva, cluster_of["synthetic"], 1, "synthetic")
    print("   leave-one-speaker-cluster-out (human)")
    res["loo_human"] = leave_one_cluster_out(idx_tr, Xtr, idx_va, Xva, cluster_of["human"], 0, "human")
    for key in ("loo_synthetic", "loo_human"):
        accs = [r["held_out_accuracy"] for r in res[key]]
        w = np.array([r["n_held_out_calls"] for r in res[key]])
        res[key + "_summary"] = {"min_held_out_accuracy": float(min(accs)), "call_weighted_accuracy": float(np.average(accs, weights=w)),
                                 "clusters_below_0.9": int(sum(a < 0.9 for a in accs))}
        print(f"  -> {key}: call-weighted held-out accuracy {res[key + '_summary']['call_weighted_accuracy']:.3f}, "
              f"worst cluster {min(accs):.3f}, clusters below 0.9: {res[key + '_summary']['clusters_below_0.9']}")

    print("\n4b) nearest-neighbour similarity val -> train")
    res["nearest_neighbour"] = nearest_neighbour(Ctr, cy_tr, Cva, cy_va)

    print("\n5) provenance: what the classifier trained on vs what it was scored on")
    manifest = pd.read_csv(config.MANIFEST)
    tr_ids, va_ids = set(idx_tr["anon_id"]), set(idx_va["anon_id"])
    res["provenance"] = {"train_calls": len(tr_ids), "val_calls": len(va_ids), "overlap": len(tr_ids & va_ids),
                         "val_equals_manifest_val": va_ids == set(manifest[manifest["split"] == "val"]["anon_id"]),
                         "train_equals_manifest_train": tr_ids == set(manifest[manifest["split"] == "train"]["anon_id"])}
    print(f"  train {len(tr_ids)} calls, val {len(va_ids)} calls, overlap {len(tr_ids & va_ids)}; splits identical to the official manifest: "
          f"{res['provenance']['val_equals_manifest_val'] and res['provenance']['train_equals_manifest_train']}")

    print("\n6) tiny training set: 5 human + 5 synthetic training calls -> logistic regression -> all 71 validation calls")
    rng = np.random.default_rng(0)
    h_ids = sorted(set(idx_tr.loc[idx_tr["y"] == 0, "anon_id"])); s_ids = sorted(set(idx_tr.loc[idx_tr["y"] == 1, "anon_id"]))
    tiny = []
    for rep in range(10):
        ids = list(rng.choice(h_ids, 5, replace=False)) + list(rng.choice(s_ids, 5, replace=False))
        sub = idx_tr["anon_id"].isin(ids).to_numpy()
        m, _ = fit_score(Xtr[sub], ytr[sub], idx_va, Xva)
        tiny.append({"accuracy": m["accuracy"], "auc": m["auc"]})
    res["tiny_train"] = {"n_train_calls": 10, "repeats": 10, "accuracies": [t["accuracy"] for t in tiny],
                         "mean_accuracy": float(np.mean([t["accuracy"] for t in tiny])), "mean_auc": float(np.mean([t["auc"] for t in tiny]))}
    print(f"  10 random draws: val accuracy {[round(t['accuracy'], 3) for t in tiny]} (mean {res['tiny_train']['mean_accuracy']:.3f}, mean AUC {res['tiny_train']['mean_auc']:.3f})")

    print("\n7) reversed: train on the 71 validation calls only, score the 282 training calls")
    clf = make_logreg().fit(Xva, idx_va["y"].to_numpy())
    calls = call_table(idx_tr, clf.decision_function(Xtr))
    m = evaluate_scores(calls["y"], calls["score"])
    res["reversed"] = {"accuracy": m["accuracy"], "auc": m["auc"], "errors": int(m["fp"] + m["fn"]), "n": int(m["n_calls"])}
    print(f"  train-set accuracy {m['accuracy']:.3f}, AUC {m['auc']:.3f} ({res['reversed']['errors']} errors of {m['n_calls']})")

    res["runtime_s"] = time.time() - t0
    save_json(res, out / "results.json")

    # figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    p = res["permutation"]
    axes[0].bar(["real labels"] + [f"shuffled {i + 1}" for i in range(len(p))], [1.0] + [x["auc"] for x in p],
                color=["#2ca02c"] + ["#7f7f7f"] * len(p))
    axes[0].axhline(0.5, color="black", ls="--", lw=0.8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("validation AUC")
    axes[0].set_title("1) label permutation: shuffled labels must give chance", fontsize=9)
    fc = res["few_chunks"]
    names = ["first_chunk", "random_chunk", "first_3_chunks", "all_chunks"]
    axes[1].bar(names, [fc[n]["accuracy"] for n in names], color="#1f77b4")
    axes[1].set_ylim(0.5, 1.02)
    axes[1].set_ylabel("validation accuracy")
    axes[1].set_title("2) accuracy from little audio (one 1-4 s chunk vs whole call)", fontsize=9)
    axes[1].tick_params(axis="x", labelsize=8)
    for key, color, label in (("loo_synthetic", "#d62728", "synthetic voice clusters"), ("loo_human", "#1f77b4", "human speaker clusters")):
        r = res[key]
        axes[2].scatter([x["n_held_out_calls"] for x in r], [x["held_out_accuracy"] for x in r], color=color, label=label, alpha=0.8)
    axes[2].set_xlabel("calls in the held-out cluster")
    axes[2].set_ylabel("held-out calls recognised")
    axes[2].set_ylim(-0.02, 1.05)
    axes[2].set_title("4) leave-one-voice-cluster-out: unseen voices", fontsize=9)
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle(f"Sanity checks on the 100% validation score - {config.BACKBONES[backbone]['display']} layer {layer}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"sanity_{backbone}.png", dpi=140)
    plt.close(fig)
    log_experiment(title=f"sanity checks - {backbone}", model=config.BACKBONES[backbone]["hf_name"], dataset_fraction=1.0,
                   segmentation="cached embeddings", selected_layer=layer, classifier="logreg", hyperparameters="permutation x5; clustering cosine/average",
                   device="cpu", runtime_s=round(res["runtime_s"]),
                   permutation_auc=[round(x["auc"], 3) for x in p],
                   one_chunk_accuracy=round(fc["first_chunk"]["accuracy"], 3),
                   voices=f"synthetic k={res['clusters']['synthetic']['best_k']}, human k={res['clusters']['human']['best_k']}",
                   loo_synthetic=res["loo_synthetic_summary"], loo_human=res["loo_human_summary"])
    print(f"\nSaved {out / 'results.json'} and figures/sanity_{backbone}.png ({res['runtime_s']:.0f}s)")


if __name__ == "__main__":
    main()
