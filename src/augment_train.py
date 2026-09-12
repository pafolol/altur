"""
OPTIONAL STAGE - AUGMENTATION EXPERIMENT on the winning backbone (model-specific optimisation, labelled as such).

The stress test showed that under channel changes the SCORE RANKING survives (AUC stays ~0.99) but the fixed
p = 0.5 decision threshold does not (accuracy drops). Can training the same MLP on perturbed copies of the
training chunks make the decision boundary robust?

Design (honest about what is "seen"):
  * SEEN perturbation families used for training: gain, additive noise (white 20/10 dB, pink 10 dB), mu-law codec.
  * UNSEEN families never used for training: low-pass 3.4/3.0 kHz, PSTN band-pass, reverberation, spectral tilt.
  * The train chunks are re-extracted (robust layer only) under each seen perturbation; the MLP is trained on
    clean + perturbed chunks with the SAME architecture, optimiser, seed and early-stopping rule as the baseline.
  * Evaluation: the cached validation stress embeddings of Part 6b - every condition, seen and unseen.
  * Nothing in the backbone changes; the perturbations are applied to the 8 kHz caller channel before the pipeline.

Run:  python src/augment_train.py [--backbone <winner>]
Outputs: outputs/embeddings/<backbone>/augment/<cond>_train_mean_L<layer>.npy, outputs/augmentation/<backbone>/results.json,
         models/<backbone>/mlp_augmented.pt, outputs/figures/augmentation_<backbone>.png
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
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset
import stress_test
from metrics import evaluate_scores, log_experiment, save_json
from train_classifier import call_table, load_embeddings, train_mlp

SEEN = ["gain_-12dB", "white_snr20", "white_snr10", "pink_snr10", "mulaw_codec"]
UNSEEN = ["lowpass_3400", "lowpass_3000", "pstn_300-3400", "reverb_0.3s", "tilt_-3dB_oct"]


def extract_train(backbone, layer, conditions, device):
    from extract_embeddings import embed_chunks, load_backbone
    out = config.EMBEDDINGS_DIR / backbone / "augment"
    out.mkdir(parents=True, exist_ok=True)
    todo = [c for c in conditions if not (out / f"{c}_train_mean_L{layer}.npy").exists()]
    if not todo:
        return
    model, do_normalize, _ = load_backbone(backbone, device)
    df = dataset.load_split("train")
    for cond in todo:
        t0 = time.time()
        feats, index = [], []
        for row in tqdm(df.itertuples(), total=len(df), desc=f"[{backbone}] train {cond}", unit="call", leave=False):
            stereo, sr = audio.read_wav(row.path)
            stereo = stereo.copy()
            stereo[:, config.CALLER_CHANNEL] = stress_test.CONDITIONS[cond](stereo[:, config.CALLER_CHANNEL])
            chunks, spans = audio.caller_chunks(stereo, sr)
            if not chunks:
                continue
            m, _ = embed_chunks(model, chunks, do_normalize, device)
            feats.append(m[:, layer].astype(np.float16))
            index += [{"anon_id": row.anon_id, "y": row.y, "chunk": k, "start_s": s, "end_s": e, "duration_s": e - s}
                      for k, (s, e) in enumerate(spans)]
        np.save(out / f"{cond}_train_mean_L{layer}.npy", np.concatenate(feats))
        pd.DataFrame(index).to_csv(out / f"{cond}_train_index.csv", index=False)
        print(f"[{backbone}] train {cond}: {len(index)} chunks in {time.time() - t0:.0f}s")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def evaluate_under_stress(mlp, backbone, layer, device):
    res = {}
    for cond in stress_test.CONDITIONS:
        idx, E = stress_test.load_stress(backbone, cond)
        calls = stress_test.complete_calls(call_table(idx, mlp.scores(torch.tensor(E[:, layer], device=device))))
        res[cond] = evaluate_scores(calls["y"], calls["score"])
    return res


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None)
    args = parser.parse_args()
    config.set_seed()
    device = config.get_device()
    best_file = config.REPORTS_DIR / "best_model.json"
    backbone = args.backbone or json.loads(best_file.read_text())["backbone"]
    meta = json.loads((config.MODELS_DIR / backbone / "meta.json").read_text())
    layer = int(meta["layer"])
    out = config.OUTPUTS_DIR / "augmentation" / backbone
    out.mkdir(parents=True, exist_ok=True)
    print(f"Augmentation experiment: {config.BACKBONES[backbone]['display']} layer {layer}; seen={SEEN}, unseen={UNSEEN}")
    t0 = time.time()
    extract_train(backbone, layer, SEEN, device)

    idx_tr, E_tr = load_embeddings(backbone, "train")
    idx_va, E_va = load_embeddings(backbone, "val")
    X_clean, y_clean = E_tr[:, layer], idx_tr["y"].to_numpy()
    Xs, ys = [X_clean], [y_clean]
    for cond in SEEN:
        d = config.EMBEDDINGS_DIR / backbone / "augment"
        Xs.append(np.load(d / f"{cond}_train_mean_L{layer}.npy").astype(np.float32))
        ys.append(pd.read_csv(d / f"{cond}_train_index.csv")["y"].to_numpy())
    X_aug, y_aug = np.concatenate(Xs), np.concatenate(ys)
    print(f"  training chunks: clean {len(y_clean)} -> clean + augmented {len(y_aug)}")

    # baseline (clean training) re-trained here with the identical recipe, so both models share the exact code path
    t1 = time.time()
    base, hist_b, ep_b = train_mlp(X_clean, y_clean, E_va[:, layer], idx_va["y"].to_numpy(), idx_va, device, verbose=False)
    aug, hist_a, ep_a = train_mlp(X_aug, y_aug, E_va[:, layer], idx_va["y"].to_numpy(), idx_va, device, verbose=False)
    train_s = time.time() - t1
    res_b, res_a = evaluate_under_stress(base, backbone, layer, device), evaluate_under_stress(aug, backbone, layer, device)
    summary = {"backbone": backbone, "layer": layer, "seen": SEEN, "unseen": UNSEEN, "n_train_clean": int(len(y_clean)),
               "n_train_augmented": int(len(y_aug)), "baseline": res_b, "augmented": res_a, "best_epoch_baseline": ep_b,
               "best_epoch_augmented": ep_a, "train_s": train_s}
    for name, res in (("baseline", res_b), ("augmented", res_a)):
        for group, conds in (("seen", SEEN), ("unseen", UNSEEN), ("all_stressed", SEEN + UNSEEN)):
            summary[f"{name}_{group}_mean_auc"] = float(np.mean([res[c]["auc"] for c in conds]))
            summary[f"{name}_{group}_mean_accuracy"] = float(np.mean([res[c]["accuracy"] for c in conds]))
    print("\n  condition          baseline AUC/acc      augmented AUC/acc")
    for c in stress_test.CONDITIONS:
        tag = "seen" if c in SEEN else ("unseen" if c in UNSEEN else "clean")
        print(f"  {c:<16} {tag:<7} {res_b[c]['auc']:.4f} / {res_b[c]['accuracy']:.3f}      {res_a[c]['auc']:.4f} / {res_a[c]['accuracy']:.3f}")
    for g in ("seen", "unseen", "all_stressed"):
        print(f"  mean {g:<12} baseline AUC {summary[f'baseline_{g}_mean_auc']:.4f} acc {summary[f'baseline_{g}_mean_accuracy']:.3f}   "
              f"augmented AUC {summary[f'augmented_{g}_mean_auc']:.4f} acc {summary[f'augmented_{g}_mean_accuracy']:.3f}")
    # deploy the augmented head only if it wins on the UNSEEN families (the honest test) without losing on clean
    wins = (summary["augmented_unseen_mean_auc"] > summary["baseline_unseen_mean_auc"] + 0.002 or
            (abs(summary["augmented_unseen_mean_auc"] - summary["baseline_unseen_mean_auc"]) <= 0.002 and
             summary["augmented_unseen_mean_accuracy"] > summary["baseline_unseen_mean_accuracy"] + 0.02))
    wins = wins and res_a["clean"]["accuracy"] >= res_b["clean"]["accuracy"] - 1e-9
    summary["winner"] = "augmented" if wins else "baseline"
    summary["rule"] = "augmented wins if its mean AUC on the UNSEEN families is higher by > 0.002 (or tied and accuracy higher by > 2 points) and clean accuracy does not drop"
    print(f"  -> {summary['winner'].upper()} ({summary['rule']})")
    torch.save({"state_dict": aug.state_dict(), "dim": int(X_aug.shape[1]), "hidden": config.MLP_HIDDEN, "dropout": config.MLP_DROPOUT,
                "augmented_with": SEEN}, config.MODELS_DIR / backbone / "mlp_augmented.pt")
    summary["runtime_s"] = time.time() - t0
    save_json(summary, out / "results.json")

    conds = list(stress_test.CONDITIONS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    x = np.arange(len(conds))
    for ax, metric in zip(axes, ("auc", "accuracy")):
        ax.bar(x - 0.2, [res_b[c][metric] for c in conds], 0.4, color="#7f7f7f", label="baseline MLP (clean training)")
        ax.bar(x + 0.2, [res_a[c][metric] for c in conds], 0.4, color="#d62728", label="augmented MLP (clean + seen perturbations)")
        for i, c in enumerate(conds):
            if c in UNSEEN:
                ax.axvspan(i - 0.5, i + 0.5, color="#fff3cd", zorder=0)
        ax.set_xticks(x, conds, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.4, 1.02)
        ax.set_ylabel(f"validation {metric}")
        ax.set_title(f"{metric} per condition (shaded = UNSEEN families)", fontsize=9.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Augmentation experiment - {config.BACKBONES[backbone]['display']} layer {layer}")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"augmentation_{backbone}.png", dpi=140)
    plt.close(fig)
    log_experiment(title=f"augmentation experiment - {backbone}", model=config.BACKBONES[backbone]["hf_name"], dataset_fraction=1.0,
                   segmentation="as pipeline; train caller channel perturbed (seen families)", selected_layer=layer,
                   classifier="MLP (same recipe), clean vs clean+augmented", hyperparameters=f"seen={SEEN}; unseen={UNSEEN}",
                   device=str(device), runtime_s=round(summary["runtime_s"]),
                   unseen_auc=f"baseline {summary['baseline_unseen_mean_auc']:.4f} -> augmented {summary['augmented_unseen_mean_auc']:.4f}",
                   unseen_acc=f"baseline {summary['baseline_unseen_mean_accuracy']:.3f} -> augmented {summary['augmented_unseen_mean_accuracy']:.3f}",
                   winner=summary["winner"])


if __name__ == "__main__":
    main()
