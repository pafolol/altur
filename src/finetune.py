"""
OPTIONAL STAGE - CONSERVATIVE FINE-TUNING of the best frozen backbone.

Only run after the frozen benchmark, on its winner. Design:
  * the backbone is TRUNCATED after the layer the frozen experiment selected (deeper layers are unused);
    the head is the same MLP architecture, WARM-STARTED from the frozen-embedding MLP
  * phase 1: backbone frozen, head trained end-to-end on random crops (sanity: should match the frozen result)
  * phase 2: the last N transformer layers (below the head) are unfrozen with a tiny learning rate, warm-up +
    linear decay, gradient clipping, early stopping; the checkpoint with the best VALIDATION call-level AUC is kept
  * the same chunks, aggregation and metrics as the frozen pipeline -> a fair frozen-vs-fine-tuned comparison
  * crash-safe: the training state is saved after every epoch (--resume continues)

Run:  python src/finetune.py [--backbone wavlm] [--unfreeze 4] [--head-epochs 3] [--unfrozen-epochs 8]
      python src/finetune.py --fraction 0.1 --head-epochs 1 --unfrozen-epochs 1      (smoke test)
Outputs: models/<backbone>_finetuned/{backbone/, head.pt, meta.json}, outputs/finetune/<backbone>/{history.json, curves.png, results.json}
"""
import argparse
import copy
import json
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModel, get_linear_schedule_with_warmup

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from metrics import evaluate_scores, format_metrics, log_experiment, save_json
from train_classifier import MLP, call_table, class_weights


def load_chunks(split, fraction):
    """All caller chunks of a split, kept in RAM as float16 (train: ~4k chunks x 4 s = ~0.5 GB)."""
    df = dataset.load_split(split, fraction)
    chunks, index = [], []
    for row in tqdm(df.itertuples(), total=len(df), desc=f"chunks [{split}]", unit="call", leave=False):
        stereo, sr = audio.read_wav(row.path)
        cs, spans = audio.caller_chunks(stereo, sr)
        for k, (c, (s, e)) in enumerate(zip(cs, spans)):
            chunks.append(c.astype(np.float16))
            index.append({"anon_id": row.anon_id, "y": row.y, "label": row.label, "chunk": k, "start_s": s, "end_s": e,
                          "duration_s": e - s})
    return chunks, pd.DataFrame(index)


class FineTuned(nn.Module):
    def __init__(self, backbone, layer, head, do_normalize):
        super().__init__()
        self.backbone, self.layer, self.head, self.do_normalize = backbone, layer, head, do_normalize

    def forward(self, x):  # x: (batch, samples) float32, already RMS-normalised
        if self.do_normalize:
            x = (x - x.mean(dim=1, keepdim=True)) / torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-7)
        h = self.backbone(x, output_hidden_states=True).hidden_states[self.layer]
        return self.head(h.float().mean(dim=1))


def set_trainable(model, unfreeze_last_n):
    for p in model.backbone.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if unfreeze_last_n > 0:
        for layer in model.backbone.encoder.layers[-unfreeze_last_n:]:
            for p in layer.parameters():
                p.requires_grad = True
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"  trainable parameters: {n_tr:,} of {n_all:,} ({n_tr / n_all:.1%})")
    return n_tr


def train_batches(chunks, y, batch_size, rng):
    """Batches of similar-length chunks, each randomly cropped to the shortest chunk of the batch (no padding)."""
    order = np.argsort([len(c) for c in chunks], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    rng.shuffle(batches)
    for ids in batches:
        L = min(len(chunks[i]) for i in ids)
        x = np.stack([chunks[i][(o := rng.integers(0, len(chunks[i]) - L + 1)):o + L].astype(np.float32) for i in ids])
        yield torch.from_numpy(x), torch.tensor(y[ids])


@torch.inference_mode()
def score_all(model, chunks, device, batch_size=16):
    model.eval()
    by_len = defaultdict(list)
    for i, c in enumerate(chunks):
        by_len[len(c)].append(i)
    scores = np.zeros(len(chunks))
    for L, idx in by_len.items():
        for b in range(0, len(idx), batch_size):
            ids = idx[b:b + batch_size]
            x = torch.from_numpy(np.stack([chunks[i].astype(np.float32) for i in ids])).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits = model(x).float()
            scores[ids] = (logits[:, 1] - logits[:, 0]).cpu().numpy()
    return scores


def run_epoch(model, chunks, y, optimizer, scheduler, loss_fn, device, batch_size, rng, desc):
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    total, n = 0.0, 0
    for x, yb in tqdm(list(train_batches(chunks, y, batch_size, rng)), desc=desc, leave=False):
        x, yb = x.to(device), yb.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(x)
        loss = loss_fn(logits.float(), yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config.FT_MAX_GRAD_NORM)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total += loss.item() * len(yb)
        n += len(yb)
    return total / max(n, 1)


def evaluate(model, chunks, index, device, loss_fn):
    scores = score_all(model, chunks, device)
    y = index["y"].to_numpy()
    logits = torch.tensor(np.stack([-scores / 2, scores / 2], axis=1), dtype=torch.float32)
    loss = loss_fn(logits, torch.tensor(y)).item()
    calls = call_table(index, scores)
    return scores, calls, evaluate_scores(calls["y"], calls["score"]), loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=None)
    parser.add_argument("--fraction", type=float, default=1.0)
    parser.add_argument("--unfreeze", type=int, default=config.FT_UNFREEZE_LAST_N_LAYERS)
    parser.add_argument("--head-epochs", type=int, default=config.FT_HEAD_EPOCHS)
    parser.add_argument("--unfrozen-epochs", type=int, default=config.FT_UNFROZEN_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.FT_BATCH_SIZE)
    parser.add_argument("--backbone-lr", type=float, default=config.FT_BACKBONE_LR)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config.set_seed()
    device = config.get_device()
    best_file = config.REPORTS_DIR / "best_model.json"
    backbone = args.backbone or (json.loads(best_file.read_text())["backbone"] if best_file.exists() else "wavlm")
    meta = json.loads((config.MODELS_DIR / backbone / "meta.json").read_text())
    layer = int(meta["layer"])
    out = config.OUTPUTS_DIR / "finetune" / backbone
    out.mkdir(parents=True, exist_ok=True)
    model_dir = config.MODELS_DIR / f"{backbone}_finetuned"
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Fine-tuning {config.BACKBONES[backbone]['display']} truncated at layer {layer}, unfreezing last {args.unfreeze} kept layers")

    t0 = time.time()
    tr_chunks, tr_index = load_chunks("train", args.fraction)
    va_chunks, va_index = load_chunks("val", args.fraction)
    ytr = tr_index["y"].to_numpy()
    print(f"  chunks: train {len(tr_chunks)}, val {len(va_chunks)} (loaded in {time.time() - t0:.0f}s)")

    name = config.BACKBONES[backbone]["hf_name"]
    do_normalize = bool(AutoFeatureExtractor.from_pretrained(name).do_normalize)
    bb = AutoModel.from_pretrained(name)
    keep = min(layer + 1, bb.config.num_hidden_layers)
    bb.encoder.layers = bb.encoder.layers[:keep]
    bb.config.num_hidden_layers = keep
    if hasattr(bb.config, "layerdrop"):
        bb.config.layerdrop = 0.0
    if hasattr(bb.config, "mask_time_prob"):
        bb.config.mask_time_prob = 0.0  # SpecAugment masking off: it would hide the artefacts we want to detect
    ckpt = torch.load(config.MODELS_DIR / backbone / "mlp.pt", map_location="cpu")
    head = MLP(ckpt["dim"], ckpt["hidden"], ckpt["dropout"])
    head.load_state_dict(ckpt["state_dict"])  # warm start from the frozen-embedding MLP
    model = FineTuned(bb, layer, head, do_normalize).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr).to(device))
    val_loss_fn = nn.CrossEntropyLoss(weight=class_weights(va_index["y"].to_numpy()))
    rng = np.random.default_rng(config.SEED)

    history, best = [], {"auc": -1, "loss": float("inf"), "epoch": 0, "state": None}
    state_file = out / "resume_checkpoint.pt"
    start_epoch = 1
    if args.resume and state_file.exists():
        st = torch.load(state_file, map_location=device)
        model.load_state_dict(st["model"])
        history, best, start_epoch = st["history"], st["best"], st["epoch"] + 1
        print(f"  resumed after epoch {st['epoch']}")

    # baseline: frozen backbone + warm-started head (should reproduce the frozen result up to bf16 noise)
    _, _, m0, l0 = evaluate(model, va_chunks, va_index, device, val_loss_fn)
    print(f"  epoch 0 (frozen backbone, frozen-trained head): val {format_metrics(m0)}")
    if not history:
        history.append({"epoch": 0, "phase": "frozen", "train_loss": float("nan"), "val_loss": l0, **{f"val_{k}": m0[k] for k in ("auc", "eer", "accuracy", "f1")}})

    total_epochs = args.head_epochs + args.unfrozen_epochs
    bad = 0
    for epoch in range(start_epoch, total_epochs + 1):
        phase = "head" if epoch <= args.head_epochs else "unfrozen"
        if epoch == 1 or epoch == args.head_epochs + 1:  # (re)build the optimizer at each phase start
            n_unfreeze = 0 if phase == "head" else args.unfreeze
            set_trainable(model, n_unfreeze)
            if phase == "head":
                optimizer = torch.optim.AdamW(model.head.parameters(), lr=config.FT_HEAD_LR / 10, weight_decay=config.FT_WEIGHT_DECAY)
                scheduler = None
            else:
                body = [p for n, p in model.backbone.named_parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": config.FT_HEAD_LR / 10},
                                               {"params": body, "lr": args.backbone_lr}], weight_decay=config.FT_WEIGHT_DECAY)
                steps = args.unfrozen_epochs * int(np.ceil(len(tr_chunks) / args.batch_size))
                scheduler = get_linear_schedule_with_warmup(optimizer, int(config.FT_WARMUP_FRACTION * steps), steps)
        t1 = time.time()
        tr_loss = run_epoch(model, tr_chunks, ytr, optimizer, scheduler, loss_fn, device, args.batch_size, rng,
                            f"epoch {epoch}/{total_epochs} [{phase}]")
        _, va_calls, m, va_loss = evaluate(model, va_chunks, va_index, device, val_loss_fn)
        h = {"epoch": epoch, "phase": phase, "train_loss": tr_loss, "val_loss": va_loss, "epoch_s": time.time() - t1,
             **{f"val_{k}": m[k] for k in ("auc", "eer", "accuracy", "f1")}}
        history.append(h)
        print(f"  epoch {epoch:2d} [{phase:8s}] train loss {tr_loss:.4f}  val loss {va_loss:.4f}  val {format_metrics(m)}  ({h['epoch_s']:.0f}s)")
        if phase == "unfrozen":
            if (m["auc"], -va_loss) > (best["auc"], -best["loss"]):
                best = {"auc": m["auc"], "loss": va_loss, "epoch": epoch, "state": copy.deepcopy(model.state_dict()), "metrics": m}
                bad = 0
            else:
                bad += 1
                if bad >= config.FT_PATIENCE:
                    print(f"  early stopping: no validation improvement for {config.FT_PATIENCE} epochs")
                    break
        torch.save({"model": model.state_dict(), "history": history, "best": best, "epoch": epoch}, state_file)

    if best["state"] is None:  # no unfrozen epochs ran
        best = {"auc": history[-1]["val_auc"], "loss": history[-1]["val_loss"], "epoch": history[-1]["epoch"],
                "state": copy.deepcopy(model.state_dict())}
    model.load_state_dict(best["state"])
    va_scores, va_calls, m_ft, _ = evaluate(model, va_chunks, va_index, device, val_loss_fn)
    _, tr_calls, m_tr, _ = evaluate(model, tr_chunks, tr_index, device, loss_fn.cpu())
    frozen = meta["val_metrics_mlp"]
    print(f"\nFROZEN   val: {format_metrics(frozen)}")
    print(f"FINETUNED val: {format_metrics(m_ft)}   (best epoch {best['epoch']})")

    # the clean split is saturated -> compare under the same stress conditions as the frozen benchmark
    stress_ft, frozen_stress = {}, None
    frozen_stress_file = config.OUTPUTS_DIR / "stress" / backbone / "mlp_stress.json"
    if frozen_stress_file.exists() and args.fraction == 1.0:
        import stress_test
        frozen_stress = json.loads(frozen_stress_file.read_text())
        val_df = dataset.load_split("val")
        print("\nfine-tuned model under stress:")
        for cond, fn in stress_test.CONDITIONS.items():
            chunks, index = [], []
            for row in val_df.itertuples():
                stereo, sr = audio.read_wav(row.path)
                stereo = stereo.copy()
                stereo[:, config.CALLER_CHANNEL] = fn(stereo[:, config.CALLER_CHANNEL])
                cs, spans = audio.caller_chunks(stereo, sr)
                chunks += [c.astype(np.float16) for c in cs]
                index += [{"anon_id": row.anon_id, "y": row.y, "duration_s": e - s0} for s0, e in spans]
            idx = pd.DataFrame(index)
            calls = stress_test.complete_calls(call_table(idx, score_all(model, chunks, device)))
            stress_ft[cond] = evaluate_scores(calls["y"], calls["score"])
            fz = frozen_stress["conditions"][cond]["mlp"]
            print(f"  {cond:<16} fine-tuned AUC {stress_ft[cond]['auc']:.4f} acc {stress_ft[cond]['accuracy']:.3f}   |   frozen AUC {fz['auc']:.4f} acc {fz['accuracy']:.3f}")
        stressed = [c for c in stress_ft if c != "clean"]
        ft_mean = float(np.mean([stress_ft[c]["auc"] for c in stressed]))
        fz_mean = frozen_stress["mlp_mean_auc_stressed"]
        print(f"  mean stressed AUC: fine-tuned {ft_mean:.4f} vs frozen {fz_mean:.4f}")
        winner = "finetuned" if ft_mean > fz_mean + 0.005 else "frozen"
        rule = f"mean stressed validation AUC (fine-tuned {ft_mean:.4f} vs frozen {fz_mean:.4f}; fine-tuned must win by > 0.005)"
    else:
        winner = "finetuned" if (round(m_ft["auc"], 4), -round(m_ft["eer"], 4)) > (round(frozen["auc"], 4), -round(frozen["eer"], 4)) else "frozen"
        rule = "clean validation AUC / EER"
    print(f"-> {winner.upper()} wins ({rule})")

    model.backbone.save_pretrained(model_dir / "backbone")
    torch.save({"state_dict": model.head.state_dict(), "dim": ckpt["dim"], "hidden": ckpt["hidden"], "dropout": ckpt["dropout"]},
               model_dir / "head.pt")
    save_json({"backbone": backbone, "hf_name": name, "layer": layer, "kept_layers": keep, "unfrozen_layers": args.unfreeze,
               "pooling": "mean", "aggregation": config.CHUNK_AGGREGATION, "do_normalize": do_normalize, "best_epoch": best["epoch"],
               "val_metrics": m_ft, "frozen_val_metrics": frozen, "winner": winner, "fraction": args.fraction,
               "backbone_lr": args.backbone_lr, "batch_size": args.batch_size}, model_dir / "meta.json")
    save_json(history, out / "history.json")
    save_json({"frozen": frozen, "finetuned": {"train": m_tr, "val": m_ft}, "best_epoch": best["epoch"], "winner": winner,
               "rule": rule, "finetuned_stress": stress_ft,
               "finetuned_mean_auc_stressed": float(np.mean([v["auc"] for c, v in stress_ft.items() if c != "clean"])) if stress_ft else None,
               "frozen_mean_auc_stressed": frozen_stress["mlp_mean_auc_stressed"] if frozen_stress else None,
               "history": history, "runtime_s": time.time() - t0}, out / "results.json")
    pred = va_calls.rename(columns={"score": "score_finetuned"})
    pred.to_csv(config.PREDICTIONS_DIR / f"{backbone}_finetuned_val.csv", index=False)

    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    axes[0].plot(ep, [h["train_loss"] for h in history], "o-", ms=3, label="train loss")
    axes[0].plot(ep, [h["val_loss"] for h in history], "o-", ms=3, label="validation loss")
    axes[0].set_title("loss")
    axes[1].plot(ep, [h["val_auc"] for h in history], "o-", ms=3, color="black", label="validation AUC")
    axes[1].axhline(frozen["auc"], color="gray", ls="--", label=f"frozen MLP AUC {frozen['auc']:.3f}")
    axes[1].set_title("validation AUC")
    axes[2].plot(ep, [h["val_eer"] * 100 for h in history], "o-", ms=3, color="#d62728", label="validation EER (%)")
    axes[2].axhline(frozen["eer"] * 100, color="gray", ls="--", label=f"frozen MLP EER {frozen['eer'] * 100:.2f}%")
    axes[2].set_title("validation EER")
    for ax in axes:
        ax.axvline(args.head_epochs + 0.5, color="purple", ls=":", label="layers unfrozen")
        ax.axvline(best["epoch"], color="green", ls="--", lw=0.8, label=f"kept epoch {best['epoch']}")
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle(f"Fine-tuning {config.BACKBONES[backbone]['display']} (layer {layer}, last {args.unfreeze} kept layers unfrozen)")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / f"finetune_{backbone}.png", dpi=140)
    plt.close(fig)
    log_experiment(title=f"fine-tuning {backbone}", model=name, dataset_fraction=args.fraction,
                   segmentation="as frozen pipeline", selected_layer=layer, classifier="MLP head (warm start)",
                   hyperparameters=f"unfreeze={args.unfreeze}, head_epochs={args.head_epochs}, unfrozen_epochs={args.unfrozen_epochs}, "
                                   f"batch={args.batch_size}, backbone_lr={args.backbone_lr}, head_lr={config.FT_HEAD_LR / 10}",
                   device=str(device), runtime_s=round(time.time() - t0), frozen_val=format_metrics(frozen),
                   finetuned_val=format_metrics(m_ft), winner=winner, best_epoch=best["epoch"], rule=rule)


if __name__ == "__main__":
    main()
