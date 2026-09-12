"""
ROBUST ACOUSTIC V2 - the same frozen backbone, a classifier that has seen a telephone line.

MODEL A, the Altur specialist, is not touched by this script. It stays in models/wav2vec2_spanish/ exactly as
it was benchmarked. MODEL B is written to models/robust_v2/.

THE ONE VARIABLE
  Backbone, layer stack, chunking, loudness normalisation, MLP architecture, optimiser and seed are all
  identical to the specialist. What changes is the TRAINING DOMAIN: the classifier now also sees the same
  TRAIN calls after they have been through a telephone channel. (The segmentation also changes, from the
  energy VAD to Silero - forced by the data, see extract_channel_embeddings.py. The orig_only variant below
  exists precisely so the report can price that change separately instead of crediting it to the new data.)

THE VARIANTS
    orig_only     original TRAIN only. The specialist's recipe with the robust segmentation - the control
                  that says how much of any improvement is just the VAD.
    channel_only  channel TRAIN only. The control that shows what happens if you forget the clean domain.
    mixed         both, concatenated, class-weighted loss.
    balanced      both, with every (domain, class) cell sampled equally often. The pre-registered favourite.

MODEL SELECTION
  NOT on clean Altur validation: it is saturated - every backbone and almost every layer scores 1.000 there,
  so it cannot rank anything. The criterion is the WORST call-level AUC across the two validation domains the
  model is allowed to look at (original and channel_seen). A model that is excellent on one and poor on the
  other is not the model we want in front of a bank.

  val_channel_unseen (GSM, AMR-NB, iLBC, Speex, Opus, G.722) is NEVER consulted here - not for early stopping,
  not for picking a layer, not for picking a variant. It is reported once, at the end, by evaluate_models.py.
  That is the number that means something.

Run:  python src/train_robust.py [--variants all] [--layer 5] [--probe]
Outputs: models/robust_v2/{mlp.pt, meta.json}, outputs/robust/{results.json, curves.png, layer_probe.json}
"""
import argparse
import copy
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import _bootstrap  # noqa: F401
import config
from extract_channel_embeddings import OUT as EMB
from metrics import auc, compute_eer, evaluate_scores, format_metrics, log_experiment, save_json
from train_classifier import MLP, make_logreg

MODEL_DIR = config.MODELS_DIR / "robust_v2"
RESULTS = config.OUTPUTS_DIR / "robust"
TRAIN_SETS = {"orig_only": ["train_original"],
              "channel_only": ["train_channel"],
              "mixed": ["train_original", "train_channel"],
              "balanced": ["train_original", "train_channel"]}
EVAL_SETS = ["val_original", "val_channel_seen", "val_channel_unseen"]
SELECTION_SETS = ["val_original", "val_channel_seen"]     # what model selection is allowed to see


def load_set(name, layer=None):
    """(index DataFrame, embeddings) for one extracted set. `layer` selects one hidden layer."""
    index = pd.read_csv(EMB / f"{name}_index.csv")
    emb = np.load(EMB / f"{name}_mean.npy").astype(np.float32)
    assert len(index) == len(emb), f"{name}: {len(index)} index rows vs {len(emb)} embeddings"
    return index, (emb if layer is None else emb[:, layer])


def call_table(index, chunk_scores):
    """Chunk log-odds -> one row per FILE (mean of the chunk log-odds, as the specialist aggregates)."""
    df = index[["sample_id", "call_id", "y", "domain"]].copy()
    df["chunk_score"] = np.asarray(chunk_scores, dtype=float)
    g = df.groupby("sample_id", sort=True)
    return pd.DataFrame({"sample_id": list(g.groups), "y": g["y"].first().to_numpy(),
                         "call_id": g["call_id"].first().to_numpy(), "domain": g["domain"].first().to_numpy(),
                         "score": g["chunk_score"].mean().to_numpy(), "n_chunks": g.size().to_numpy()})


def domain_class_weights(index):
    """One weight per chunk so that every (domain, class) cell is drawn equally often."""
    key = index["domain"].astype(str) + "|" + index["y"].astype(str)
    counts = key.value_counts()
    return (1.0 / counts[key].to_numpy()).astype(np.float64)


def train_one(variant, layer, train_index, Xtr, ytr, evals, device, epochs, verbose=True):
    """
    One MLP. Returns (model, history, best_epoch, best_score). Early stopping is on the WORST call-level AUC
    over SELECTION_SETS, not on validation loss: the clean split is saturated, so its loss says nothing about
    the thing we are trying to buy.
    """
    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)
    Xtr_t = torch.tensor(Xtr, device=device)
    ytr_t = torch.tensor(ytr, device=device)
    model = MLP(Xtr.shape[1], mean=Xtr.mean(0), std=Xtr.std(0) + 1e-6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.MLP_LR, weight_decay=config.MLP_WEIGHT_DECAY)
    gen = torch.Generator(device="cpu").manual_seed(config.SEED)

    if variant == "balanced":
        # Every (domain, class) cell equally likely, so 564 channel samples cannot drown 282 original ones and
        # the 60/40 class split cannot drown either. Sampling with replacement; the loss is then unweighted.
        weights = torch.tensor(domain_class_weights(train_index), dtype=torch.double)
        loss_fn = nn.CrossEntropyLoss()
    else:
        weights = None
        counts = np.bincount(ytr, minlength=2)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(len(ytr) / (2 * counts), dtype=torch.float32).to(device))

    history, best_state, best_score, best_epoch, bad = [], None, (-1.0, -1.0), 0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        if weights is None:
            order = torch.randperm(len(Xtr_t), generator=gen).to(device)
        else:
            order = torch.multinomial(weights, len(Xtr_t), replacement=True, generator=gen).to(device)
        for start in range(0, len(order), config.MLP_BATCH_SIZE):
            idx = order[start:start + config.MLP_BATCH_SIZE]
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        h = {"epoch": epoch}
        with torch.no_grad():
            h["train_loss"] = float(loss_fn(model(Xtr_t), ytr_t).item())
        for name, (idx_v, Xv) in evals.items():
            calls = call_table(idx_v, model.scores(torch.tensor(Xv, device=device)))
            h[f"{name}_auc"] = auc(calls["y"], calls["score"])
            h[f"{name}_acc"] = float(((calls["score"] >= 0).astype(int) == calls["y"]).mean())
            h[f"{name}_eer"] = compute_eer(calls["y"], calls["score"])[0]
        h["worst_auc"] = min(h[f"{n}_auc"] for n in SELECTION_SETS)
        h["worst_acc"] = min(h[f"{n}_acc"] for n in SELECTION_SETS)
        history.append(h)
        if verbose:
            print(f"  epoch {epoch:3d}  loss {h['train_loss']:.4f}  " +
                  "  ".join(f"{n.replace('val_', '')}: AUC {h[n + '_auc']:.4f}/acc {h[n + '_acc']:.3f}" for n in EVAL_SETS) +
                  f"   worst(selectable) AUC {h['worst_auc']:.4f}")
        # Tie-break the saturated AUC with the worst accuracy. The strict ">" then keeps the EARLIEST epoch
        # that reaches the best score, which is deliberate: when several epochs are indistinguishable on the
        # validation domains, the least-trained one has had the fewest opportunities to fit the training
        # domain's quirks. (The choice BETWEEN variants needs a different tie-break - see selection_score -
        # because there "earliest" has no meaning and dictionary order is not a criterion.)
        score = (round(h["worst_auc"], 5), round(h["worst_acc"], 4))
        if score > best_score:
            best_score, best_state, best_epoch, bad = score, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            bad += 1
            if bad >= config.MLP_PATIENCE:
                if verbose:
                    print(f"  early stopping at epoch {epoch} (best epoch {best_epoch})")
                break
    model.load_state_dict(best_state)
    return model, history, best_epoch, best_score


def layer_probe(train_index, Etr, ytr, evals_all, device):
    """
    Which hidden layer survives a telephone line best? The specialist picked layer 5 on the clean split, where
    every layer scores 1.000 - so the choice was never really tested. Here the criterion has teeth: the worst
    call-level AUC across the selectable validation domains.
    """
    rows = []
    for layer in range(Etr.shape[1]):
        clf = make_logreg().fit(Etr[:, layer], ytr)
        row = {"layer": layer}
        for name, (idx_v, Ev) in evals_all.items():
            calls = call_table(idx_v, clf.decision_function(Ev[:, layer]))
            row[f"{name}_auc"] = auc(calls["y"], calls["score"])
            row[f"{name}_acc"] = float(((calls["score"] >= 0).astype(int) == calls["y"]).mean())
        row["worst_auc"] = min(row[f"{n}_auc"] for n in SELECTION_SETS)
        row["worst_acc"] = min(row[f"{n}_acc"] for n in SELECTION_SETS)
        rows.append(row)
        print(f"  layer {layer:2d}: " + "  ".join(f"{n.replace('val_', '')} AUC {row[n + '_auc']:.4f}/acc {row[n + '_acc']:.3f}" for n in EVAL_SETS) +
              f"   worst(sel) AUC {row['worst_auc']:.4f} acc {row['worst_acc']:.3f}")
    return rows


def plot_curves(results, path):
    variants = [v for v in TRAIN_SETS if v in results]
    fig, axes = plt.subplots(1, len(variants), figsize=(4.6 * len(variants), 4.0), squeeze=False)
    colours = {"val_original": "#1f77b4", "val_channel_seen": "#d62728", "val_channel_unseen": "#2ca02c"}
    for ax, variant in zip(axes[0], variants):
        h = pd.DataFrame(results[variant]["history"])
        for name in EVAL_SETS:
            ax.plot(h["epoch"], h[f"{name}_acc"], "o-", ms=3, color=colours[name],
                    label=name.replace("val_", ""), ls="--" if name == "val_channel_unseen" else "-")
        ax.axvline(results[variant]["best_epoch"], color="gray", ls=":", label="kept epoch")
        ax.set_title(f"{variant}\n(trained on {', '.join(TRAIN_SETS[variant])})", fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("call accuracy")
        ax.set_ylim(0.3, 1.02)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("Robust Acoustic V2: what each training domain buys (dashed = codecs never trained on)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", default=["all"], choices=["all", *TRAIN_SETS])
    parser.add_argument("--layer", type=int, default=None, help="hidden layer (default: the probe's choice)")
    parser.add_argument("--probe", action="store_true", help="run the layer probe even if --layer is given")
    parser.add_argument("--epochs", type=int, default=config.MLP_EPOCHS)
    args = parser.parse_args()
    config.set_seed()
    device = config.get_device()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    variants = list(TRAIN_SETS) if "all" in args.variants else args.variants

    print("[data] loading cached embeddings")
    sets = {n: load_set(n) for n in ["train_original", "train_channel", *EVAL_SETS]}
    for n, (idx, emb) in sets.items():
        print(f"  {n:20s} {len(idx['sample_id'].unique()):4d} files, {len(idx):5d} chunks, shape {emb.shape}")
    n_layers = sets["train_original"][1].shape[1]

    # ------------------------------------------------------------------ layer choice
    probe_rows, layer = None, args.layer
    if args.probe or args.layer is None:
        print("\n[layer probe] logistic regression per layer, trained on original+channel TRAIN")
        idx_all = pd.concat([sets["train_original"][0], sets["train_channel"][0]], ignore_index=True)
        E_all = np.concatenate([sets["train_original"][1], sets["train_channel"][1]])
        evals_all = {n: (sets[n][0], sets[n][1]) for n in EVAL_SETS}
        probe_rows = layer_probe(idx_all, E_all, idx_all["y"].to_numpy(), evals_all, device)
        save_json(probe_rows, RESULTS / "layer_probe.json")
        if layer is None:
            # AUC alone cannot choose: layers 1-12 all reach a worst AUC of 1.0000 on the selectable domains.
            # Accuracy still discriminates, so it is the second key, and the LOWEST layer wins any remaining tie
            # - cheaper at inference, and it happens to land on the layer the specialist already uses, which
            # keeps the comparison between the two models a single-variable one.
            best = max(probe_rows, key=lambda r: (round(r["worst_auc"], 4), round(r["worst_acc"], 4), -r["layer"]))
            layer = best["layer"]
            print(f"  -> layer {layer} (worst selectable AUC {best['worst_auc']:.4f}, accuracy {best['worst_acc']:.3f})")

    evals = {n: (sets[n][0], sets[n][1][:, layer]) for n in EVAL_SETS}

    # ------------------------------------------------------------------ variants
    results = {}
    for variant in variants:
        names = TRAIN_SETS[variant]
        idx = pd.concat([sets[n][0] for n in names], ignore_index=True)
        X = np.concatenate([sets[n][1][:, layer] for n in names])
        y = idx["y"].to_numpy()
        print(f"\n=== {variant}: {len(idx)} chunks from {names}")
        print("   " + idx.groupby(["domain", "label"]).size().to_string().replace("\n", "\n   "))
        t0 = time.time()
        model, history, best_epoch, best_score = train_one(variant, layer, idx, X, y, evals, device, args.epochs)
        runtime = time.time() - t0
        final = {}
        for name, (idx_v, Xv) in evals.items():
            calls = call_table(idx_v, model.scores(torch.tensor(Xv, device=device)))
            final[name] = evaluate_scores(calls["y"], calls["score"])
            calls.to_csv(RESULTS / f"scores_{variant}_{name}.csv", index=False)
        results[variant] = {"train_sets": names, "layer": layer, "n_train_chunks": int(len(idx)),
                            "best_epoch": best_epoch, "worst_selectable_auc": best_score[0],
                            "worst_selectable_acc": best_score[1],
                            "runtime_s": runtime, "history": history, "metrics": final}
        print(f"  kept epoch {best_epoch} ({runtime:.0f}s)")
        for name in EVAL_SETS:
            print(f"    {name:22s} {format_metrics(final[name])}")
        torch.save({"state_dict": model.state_dict(), "dim": int(X.shape[1]), "hidden": config.MLP_HIDDEN,
                    "dropout": config.MLP_DROPOUT}, MODEL_DIR / f"mlp_{variant}.pt")
        # Every variant is also written as a loadable model directory, so evaluate_models.py can put the
        # controls through exactly the same end-to-end path as the two deployed models. orig_only in
        # particular is the control that separates "we fixed the segmenter" from "we added channel data":
        # it is the specialist's training data with the specialist's layer, re-fitted on Silero-segmented
        # chunks. Swapping the VAD at inference only (specialist_silero) is a different, weaker control,
        # because that MLP was fitted on chunks the energy detector chose.
        variant_dir = config.MODELS_DIR / f"robust_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "dim": int(X.shape[1]), "hidden": config.MLP_HIDDEN,
                    "dropout": config.MLP_DROPOUT}, variant_dir / "mlp.pt")
        save_json({"backbone": "wav2vec2_spanish", "hf_name": config.BACKBONES["wav2vec2_spanish"]["hf_name"],
                   "layer": layer, "pooling": "mean", "aggregation": config.CHUNK_AGGREGATION, "vad": "silero",
                   "chunk_seconds": config.CHUNK_SECONDS, "target_rms_db": config.TARGET_RMS_DB,
                   "seed": config.SEED, "variant": variant, "train_sets": names,
                   "display": f"control: {variant} (Wav2Vec2 Spanish, frozen)",
                   "role": "control" if variant != "balanced" else "candidate",
                   "val_metrics": final}, variant_dir / "meta.json")

    # ------------------------------------------------------------------ pick and deploy Model B
    def selection_score(r):
        """
        Worst AUC, then worst ACCURACY, then worst BRIER across the selectable domains (lower Brier is better,
        hence the minus sign).

        Two tie-breaks, because both of the obvious criteria saturate. At layer 5 every variant that sees any
        channel data reaches AUC 1.0000 AND accuracy 1.000 on both selectable domains - the evaluation simply
        runs out of resolution. Brier score still has some: it is a proper scoring rule, so it keeps measuring
        how much margin a model has left once it has stopped making mistakes.

        Without a tie-break at all the winner would be decided by dictionary order, which handed the title to
        `channel_only` - the one variant that never sees the clean domain, and exactly what we were asked not
        to build. The report states plainly that the measurement could not separate the top three.
        """
        return (round(min(r["metrics"][n]["auc"] for n in SELECTION_SETS), 4),
                round(min(r["metrics"][n]["accuracy"] for n in SELECTION_SETS), 4),
                -round(max(r["metrics"][n]["brier"] for n in SELECTION_SETS), 5))

    ranked = sorted(results.items(), key=lambda kv: selection_score(kv[1]), reverse=True)
    winner, best = ranked[0]
    print("\n" + "=" * 96)
    print(f"{'variant':<14}" + "".join(f"{n.replace('val_', ''):>22}" for n in EVAL_SETS) + f"{'worst AUC':>11}{'worst acc':>11}{'worst Brier':>11}")
    for name, r in results.items():
        line = f"{name:<14}" + "".join(f"{r['metrics'][n]['accuracy']:>13.3f} /{r['metrics'][n]['auc']:>7.4f}" for n in EVAL_SETS)
        w_auc, w_acc, neg_brier = selection_score(r)
        print(line + f"{w_auc:>11.4f}{w_acc:>11.3f}{-neg_brier:>11.4f}" + ("   <- Model B" if name == winner else ""))
    print("=" * 96)
    print("(accuracy / AUC; channel_unseen was not used to choose anything)")

    import shutil
    shutil.copy(MODEL_DIR / f"mlp_{winner}.pt", MODEL_DIR / "mlp.pt")

    # ------------------------------------------------------------------ calibration of Model B
    # Fitted on the two validation domains the model is allowed to see, pooled, and measured out-of-fold with
    # the folds GROUPED BY CALL so a call's original and its channel transform never straddle a fold (they
    # share a speaker; splitting them would make the calibrator look better than it is). channel_unseen is
    # excluded here too - a calibrator is a fitted model like any other.
    from calibrate import apply as apply_cal, cross_validated, fit_platt, fit_temperature, summarize
    cal_calls = pd.concat([pd.read_csv(RESULTS / f"scores_{winner}_{n}.csv") for n in SELECTION_SETS])
    cy, cs, groups = cal_calls["y"].to_numpy(), cal_calls["score"].to_numpy(dtype=float), cal_calls["call_id"].to_numpy()
    cal_report = {"fit_on": " + ".join(SELECTION_SETS), "n": int(len(cy)),
                  "raw": summarize(cy, 1 / (1 + np.exp(-cs))),
                  "platt_cv": summarize(cy, cross_validated(cs, cy, fit_platt, groups=groups)),
                  "temperature_cv": summarize(cy, cross_validated(cs, cy, fit_temperature, groups=groups))}
    platt, temp = fit_platt(cs, cy), fit_temperature(cs, cy)
    # If the fit set contains no mistakes, the likelihood has no maximum: the calibrator can sharpen without
    # limit (temperature -> 0) and reports 0.9999 on everything, which is not calibration, it is bravado.
    # calibrate.py caps the slope at 3 in exactly this situation, and so do we.
    separable = bool(((cs >= 0).astype(int) == cy).all())
    cal_report["fit_set_separable"] = separable
    if separable:
        for c in (platt, temp):
            c["a"] = float(min(c["a"], 3.0))
        cal_report["slope_capped_at"] = 3.0
    chosen = platt if cal_report["platt_cv"]["log_loss"] < cal_report["temperature_cv"]["log_loss"] - 1e-4 else temp
    chosen = dict(chosen, fit_on=cal_report["fit_on"], n=cal_report["n"], classifier="mlp",
                  cv={k: cal_report[f"{chosen['method']}_cv"][k] for k in ("brier", "log_loss", "ece", "accuracy")})
    save_json(chosen, MODEL_DIR / "calibration_mlp.json")
    save_json(cal_report, RESULTS / "calibration.json")
    print(f"\n[calibration] raw Brier {cal_report['raw']['brier']:.4f} / ECE {cal_report['raw']['ece']:.3f}  ->  "
          f"{chosen['method']} Brier {chosen['cv']['brier']:.4f} / ECE {chosen['cv']['ece']:.3f} "
          f"(a={chosen['a']:.3f}, b={chosen['b']:.3f}, fitted on {cal_report['n']} calls)")
    meta = {"backbone": "wav2vec2_spanish", "hf_name": config.BACKBONES["wav2vec2_spanish"]["hf_name"],
            "layer": layer, "pooling": "mean", "aggregation": config.CHUNK_AGGREGATION, "vad": "silero",
            "chunk_seconds": config.CHUNK_SECONDS, "target_rms_db": config.TARGET_RMS_DB, "seed": config.SEED,
            "variant": winner, "train_sets": TRAIN_SETS[winner], "best_epoch": best["best_epoch"],
            "selection_rule": "worst call-level AUC over " + ", ".join(SELECTION_SETS),
            "display": "Robust Acoustic V2 (Wav2Vec2 Spanish, frozen)",
            "val_metrics": {n: best["metrics"][n] for n in EVAL_SETS}}
    save_json(meta, MODEL_DIR / "meta.json")
    save_json({"layer": layer, "winner": winner, "variants": results, "layer_probe": probe_rows},
              RESULTS / "results.json")
    plot_curves(results, config.FIGURES_DIR / "robust_v2_training.png")
    for variant, r in results.items():
        log_experiment(title=f"Robust Acoustic V2 - {variant}",
                       model=config.BACKBONES["wav2vec2_spanish"]["hf_name"] + " (frozen)",
                       training_domains=", ".join(r["train_sets"]), n_train_chunks=r["n_train_chunks"],
                       selected_layer=layer, segmentation="silero VAD + 4s chunks, RMS -26 dB",
                       classifier=f"MLP {sets['train_original'][1].shape[2]}-{config.MLP_HIDDEN}-2",
                       hyperparameters=f"lr={config.MLP_LR}, wd={config.MLP_WEIGHT_DECAY}, batch={config.MLP_BATCH_SIZE}, "
                                       f"dropout={config.MLP_DROPOUT}, seed={config.SEED}, best epoch {r['best_epoch']}",
                       device=str(device), runtime_s=round(r["runtime_s"], 1),
                       val_original=format_metrics(r["metrics"]["val_original"]),
                       val_channel_seen=format_metrics(r["metrics"]["val_channel_seen"]),
                       val_channel_unseen=format_metrics(r["metrics"]["val_channel_unseen"]),
                       notes=("DEPLOYED as Model B" if variant == winner else "control") +
                             "; channel_unseen never used for selection")
    print(f"\nModel B -> {MODEL_DIR}  (variant '{winner}', layer {layer})")
    print(f"results -> {RESULTS / 'results.json'}")


if __name__ == "__main__":
    main()
