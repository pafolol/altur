"""
VOICE IDENTITY - how many distinct voices does the dataset contain, and does the detector survive an unseen voice?

Uses a SPEAKER-VERIFICATION model (microsoft/wavlm-base-plus-sv: WavLM + x-vector head, trained on VoxCeleb to tell
speakers apart) purely as a measuring instrument. Its embeddings encode WHO is speaking, unlike the detector's
embeddings, which encode human-vs-synthetic. Steps:

  1. one x-vector per call (mean of up to 8 caller chunks, L2-normalised)
  2. cosine-similarity matrix; agglomerative clustering at several similarity thresholds -> voice clusters
     (the model card's VoxCeleb verification threshold is 0.86 cosine; telephone audio is harder, so a range is shown)
  3. per class: number of voice clusters, calls per cluster, and how many validation calls share a voice with training
  4. LEAVE-ONE-VOICE-OUT on the detector: for every synthetic voice cluster with >= 5 calls, remove all its calls from
     training, retrain the detector's logistic regression on the winner's frozen embeddings, and test on the removed
     calls. This is the judges' "voices that appear in neither split" scenario.

Run:  python src/voice_identity.py [--backbone wav2vec2_spanish] [--threshold 0.7]
Outputs: outputs/voice_identity/{xvectors.npy, calls.csv, results.json}, outputs/figures/voice_identity.png
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
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from metrics import evaluate_scores, log_experiment, save_json
from train_classifier import call_table, load_embeddings, make_logreg

SV_MODEL = "microsoft/wavlm-base-plus-sv"
OUT = config.OUTPUTS_DIR / "voice_identity"


@torch.inference_mode()
def extract_xvectors(device, max_chunks=8):
    from transformers import AutoFeatureExtractor, WavLMForXVector
    fe = AutoFeatureExtractor.from_pretrained(SV_MODEL)
    model = WavLMForXVector.from_pretrained(SV_MODEL).to(device).eval()
    df = dataset.load_manifest()
    vecs, rows = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc="x-vectors", unit="call"):
        stereo, sr = audio.read_wav(row.path)
        chunks, spans = audio.caller_chunks(stereo, sr)
        order = np.argsort([-len(c) for c in chunks])[:max_chunks]  # the longest chunks carry the most voice
        embs = []
        for i in order:
            x = torch.from_numpy(audio.normalize_for_model(chunks[i], bool(fe.do_normalize)))[None].to(device)
            e = model(x).embeddings[0].float()
            embs.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
        v = np.mean(embs, axis=0)
        vecs.append(v / (np.linalg.norm(v) + 1e-9))
        rows.append({"anon_id": row.anon_id, "label": row.label, "split": row.split, "y": row.y, "n_chunks_used": len(order)})
    return np.stack(vecs), pd.DataFrame(rows)


def cluster_at(V, threshold):
    """Agglomerative clustering with average linkage: two calls join a cluster if their average cosine >= threshold."""
    return AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - threshold).fit_predict(V)


def describe_clusters(labels, calls):
    d = calls.assign(cluster=labels)
    per = d.groupby("cluster").agg(n=("anon_id", "size"), train=("split", lambda s: int((s == "train").sum())),
                                   val=("split", lambda s: int((s == "val").sum()))).sort_values("n", ascending=False)
    shared = per[(per["train"] > 0) & (per["val"] > 0)]
    val_calls_with_seen_voice = int(shared["val"].sum())
    return {"n_clusters": int(len(per)), "n_singletons": int((per["n"] == 1).sum()), "largest_cluster": int(per["n"].max()),
            "clusters_with_train_and_val": int(len(shared)), "val_calls_whose_voice_is_in_train": val_calls_with_seen_voice,
            "val_calls_total": int(d["split"].eq("val").sum()), "sizes": per["n"].tolist()[:15]}


def leave_one_voice_out(labels, calls, backbone, layer, min_calls=5):
    idx_tr, E_tr = load_embeddings(backbone, "train")
    idx_va, E_va = load_embeddings(backbone, "val")
    idx_all = pd.concat([idx_tr.assign(split="train"), idx_va.assign(split="val")], ignore_index=True)
    X_all = np.concatenate([E_tr[:, layer], E_va[:, layer]])
    voice_of = dict(zip(calls["anon_id"], labels))
    idx_all["voice"] = idx_all["anon_id"].map(voice_of)
    results = []
    for c in sorted(set(labels)):
        member_calls = calls[(labels == c) & (calls["label"] == "synthetic")]["anon_id"]
        if len(member_calls) < min_calls:
            continue
        held = idx_all["anon_id"].isin(member_calls)
        train_mask = (idx_all["split"] == "train") & ~held
        test_mask = held | ((idx_all["split"] == "val") & (idx_all["y"] == 0))
        clf = make_logreg().fit(X_all[train_mask.to_numpy()], idx_all.loc[train_mask, "y"].to_numpy())
        te = idx_all[test_mask].reset_index(drop=True)
        calls_te = call_table(te, clf.decision_function(X_all[test_mask.to_numpy()]))
        m = evaluate_scores(calls_te["y"], calls_te["score"])
        held_calls = calls_te[calls_te["y"] == 1]
        results.append({"voice": int(c), "n_held_out_calls": int(len(held_calls)),
                        "n_train_synthetic_left": int(((idx_all["split"] == "train") & (idx_all["y"] == 1) & ~held)["anon_id"].nunique()) if False else int(idx_all.loc[train_mask & (idx_all["y"] == 1), "anon_id"].nunique()),
                        "held_out_recognised": float(((held_calls["score"] >= 0)).mean()), "auc": m["auc"], "accuracy": m["accuracy"],
                        "human_val_frr": m["frr"]})
        r = results[-1]
        print(f"  voice {c:3d}: {r['n_held_out_calls']:3d} calls held out (train keeps {r['n_train_synthetic_left']} synthetic calls) "
              f"-> recognised {r['held_out_recognised']:.3f}, AUC {r['auc']:.3f}, human FRR {r['human_val_frr']:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--threshold", type=float, default=0.70, help="cosine threshold used for the leave-one-voice-out test")
    args = parser.parse_args()
    config.set_seed()
    OUT.mkdir(parents=True, exist_ok=True)
    device = config.get_device()
    backbone = args.backbone or json.loads((config.REPORTS_DIR / "best_model.json").read_text())["backbone"]
    layer = int(json.loads((config.MODELS_DIR / backbone / "meta.json").read_text())["layer"])
    t0 = time.time()
    if (OUT / "xvectors.npy").exists():
        V, calls = np.load(OUT / "xvectors.npy"), pd.read_csv(OUT / "calls.csv")
    else:
        V, calls = extract_xvectors(device)
        np.save(OUT / "xvectors.npy", V)
        calls.to_csv(OUT / "calls.csv", index=False)
    print(f"x-vectors: {V.shape} ({time.time() - t0:.0f}s)")

    S = V @ V.T
    np.fill_diagonal(S, np.nan)
    res = {"backbone": backbone, "layer": layer, "sv_model": SV_MODEL}
    for y, label in ((0, "human"), (1, "synthetic")):
        sel = (calls["y"] == y).to_numpy()
        within = S[np.ix_(sel, sel)]
        res[f"{label}_nn_similarity_median"] = float(np.nanmedian(np.nanmax(within, axis=1)))
        res[f"{label}_pair_similarity_median"] = float(np.nanmedian(within))
    cross = S[np.ix_((calls["y"] == 0).to_numpy(), (calls["y"] == 1).to_numpy())]
    res["human_vs_synthetic_pair_similarity_median"] = float(np.nanmedian(cross))
    print(f"nearest-neighbour cosine: human {res['human_nn_similarity_median']:.3f}, synthetic {res['synthetic_nn_similarity_median']:.3f}; "
          f"median pair similarity within human {res['human_pair_similarity_median']:.3f}, within synthetic {res['synthetic_pair_similarity_median']:.3f}, across {res['human_vs_synthetic_pair_similarity_median']:.3f}")

    res["clusters"] = {}
    print("\nvoice clusters per cosine threshold (average linkage):")
    for thr in (0.6, 0.65, 0.7, 0.75, 0.8, 0.86):
        res["clusters"][str(thr)] = {}
        for y, label in ((1, "synthetic"), (0, "human")):
            sel = (calls["y"] == y).to_numpy()
            lab = cluster_at(V[sel], thr)
            d = describe_clusters(lab, calls[sel].reset_index(drop=True))
            res["clusters"][str(thr)][label] = d
            print(f"  thr {thr:.2f} {label:<10}: {d['n_clusters']:3d} clusters ({d['n_singletons']} singletons, largest {d['largest_cluster']}); "
                  f"{d['val_calls_whose_voice_is_in_train']}/{d['val_calls_total']} val calls have a voice also in train; sizes {d['sizes']}")

    print(f"\nleave-one-voice-out on the detector (synthetic voices at threshold {args.threshold}, clusters with >= 5 calls):")
    sel = (calls["y"] == 1).to_numpy()
    lab_syn = cluster_at(V[sel], args.threshold)
    labels_full = np.full(len(calls), -1)
    labels_full[sel] = lab_syn
    res["loo"] = leave_one_voice_out(labels_full, calls, backbone, layer)
    if res["loo"]:
        w = np.array([r["n_held_out_calls"] for r in res["loo"]])
        acc = np.array([r["held_out_recognised"] for r in res["loo"]])
        res["loo_summary"] = {"threshold": args.threshold, "voices_tested": len(res["loo"]), "calls_tested": int(w.sum()),
                              "call_weighted_recognised": float(np.average(acc, weights=w)), "min_recognised": float(acc.min()),
                              "mean_auc": float(np.mean([r["auc"] for r in res["loo"]]))}
        print(f"  -> {res['loo_summary']['voices_tested']} voices / {res['loo_summary']['calls_tested']} calls: unseen-voice recall "
              f"{res['loo_summary']['call_weighted_recognised']:.3f} (worst voice {res['loo_summary']['min_recognised']:.3f}), mean AUC {res['loo_summary']['mean_auc']:.3f}")
    res["runtime_s"] = time.time() - t0
    save_json(res, OUT / "results.json")

    # figure: similarity histograms + clusters vs threshold + leave-one-voice-out
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hs = S[np.ix_((calls["y"] == 0).to_numpy(), (calls["y"] == 0).to_numpy())]
    ss = S[np.ix_(sel, sel)]
    bins = np.linspace(-0.2, 1, 50)
    axes[0].hist(hs[~np.isnan(hs)], bins=bins, alpha=0.5, density=True, color="#1f77b4", label="human-human pairs")
    axes[0].hist(ss[~np.isnan(ss)], bins=bins, alpha=0.5, density=True, color="#d62728", label="synthetic-synthetic pairs")
    axes[0].hist(cross[~np.isnan(cross)], bins=bins, alpha=0.4, density=True, color="#7f7f7f", label="human-synthetic pairs")
    axes[0].set_xlabel("cosine similarity of speaker x-vectors")
    axes[0].set_title("how alike are the voices? (all call pairs)", fontsize=9)
    axes[0].legend(fontsize=7)
    thrs = [float(t) for t in res["clusters"]]
    axes[1].plot(thrs, [res["clusters"][str(t)]["synthetic"]["n_clusters"] for t in thrs], "o-", color="#d62728", label="synthetic voice clusters")
    axes[1].plot(thrs, [res["clusters"][str(t)]["human"]["n_clusters"] for t in thrs], "o-", color="#1f77b4", label="human speaker clusters")
    axes[1].set_xlabel("cosine threshold to join a cluster")
    axes[1].set_ylabel("number of clusters")
    axes[1].set_title("estimated number of distinct voices (203 synthetic / 150 human calls)", fontsize=9)
    axes[1].legend(fontsize=7)
    if res["loo"]:
        axes[2].bar([str(r["voice"]) for r in res["loo"]], [r["held_out_recognised"] for r in res["loo"]], color="#d62728")
        for i, r in enumerate(res["loo"]):
            axes[2].text(i, r["held_out_recognised"] + 0.01, f"n={r['n_held_out_calls']}", ha="center", fontsize=7)
        axes[2].set_ylim(0, 1.08)
        axes[2].set_xlabel(f"synthetic voice cluster held out (threshold {args.threshold})")
        axes[2].set_ylabel("held-out calls recognised as synthetic")
        axes[2].set_title("leave-one-voice-out: detector on voices it never saw", fontsize=9)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Voice identity check with a speaker-verification model (WavLM x-vectors)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "voice_identity.png", dpi=140)
    plt.close(fig)
    log_experiment(title="voice identity + leave-one-voice-out", model=f"{SV_MODEL} (measuring) / {backbone} L{layer} (detector)",
                   dataset_fraction=1.0, segmentation="caller chunks", selected_layer=layer, classifier="logreg",
                   hyperparameters=f"cosine threshold {args.threshold}", device=str(device), runtime_s=round(res["runtime_s"]),
                   clusters=json.dumps({t: {k: v["n_clusters"] for k, v in res["clusters"][t].items()} for t in res["clusters"]}),
                   loo=res.get("loo_summary"))
    print(f"\nSaved {OUT / 'results.json'} and figures/voice_identity.png")


if __name__ == "__main__":
    main()
