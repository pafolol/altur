"""
STAGE 2 - FROZEN BACKBONE -> CACHED EMBEDDINGS  (run once per backbone)

    call WAV -> caller channel -> VAD -> chunks (<= 4 s) -> 16 kHz -> frozen backbone
             -> hidden states of EVERY layer -> mean (and std) over time -> saved to disk

The backbone weights never change here (no gradients, eval mode). Everything downstream
(train_classifier.py, benchmark.py) reads the cached arrays, so trying a new classifier costs seconds.

Fairness: the three backbones see exactly the same chunks (same VAD, same chunking, same loudness
normalisation, same resampling). The only per-model difference is the model's own pretrained input
convention (do_normalize in its preprocessor config), which is part of the model, not of our pipeline.

Chunks are processed WITHOUT padding (chunks of equal length are batched together), so no attention-mask
or group-norm-padding subtleties can bias one architecture against another.

Run:  python src/extract_embeddings.py --backbone wavlm            (or wav2vec2_spanish | xlsr | all)
      python src/extract_embeddings.py --backbone all --fraction 0.05   (smoke test)
Outputs: outputs/embeddings/<backbone>/{train,val}_mean.npy  (n_chunks, n_layers+1, dim)  float16
                                       {train,val}_std.npy   same shape
                                       {train,val}_index.csv  one row per chunk (anon_id, y, start_s, end_s, ...)
                                       extraction_stats.json  timings, VRAM, parameter count, cache size
"""
import argparse
import json
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from metrics import log_experiment, save_json


def load_backbone(key, device):
    """Frozen pretrained model + its input convention. Returns (model, do_normalize, info dict)."""
    from transformers import AutoConfig, AutoFeatureExtractor, AutoModel
    name = config.BACKBONES[key]["hf_name"]
    fe = AutoFeatureExtractor.from_pretrained(name)
    model = AutoModel.from_pretrained(name).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False
    cfg = AutoConfig.from_pretrained(name)
    n_params = sum(p.numel() for p in model.parameters())
    info = {"backbone": key, "hf_name": name, "params_m": n_params / 1e6, "hidden_size": cfg.hidden_size,
            "num_hidden_layers": cfg.num_hidden_layers, "do_normalize": bool(fe.do_normalize),
            "feat_extract_norm": getattr(cfg, "feat_extract_norm", None),
            "weights_mb": _weights_size_mb(name)}
    return model, bool(fe.do_normalize), info


def _weights_size_mb(name):
    try:
        from huggingface_hub import scan_cache_dir
        for repo in scan_cache_dir().repos:
            if repo.repo_id == name:
                return repo.size_on_disk / 1e6
    except Exception:
        pass
    return float("nan")


@torch.inference_mode()
def embed_chunks(model, chunks, do_normalize, device, batch_size=16):
    """
    list of 16 kHz float32 chunks -> (mean, std) arrays of shape (n_chunks, n_layers+1, dim).
    Chunks of identical length are batched (no padding); output order == input order.
    """
    by_len = defaultdict(list)
    for i, c in enumerate(chunks):
        by_len[len(c)].append(i)
    n_layers, dim = model.config.num_hidden_layers + 1, model.config.hidden_size
    mean = np.zeros((len(chunks), n_layers, dim), dtype=np.float32)
    std = np.zeros_like(mean)
    for length, idx in by_len.items():
        for b in range(0, len(idx), batch_size):
            ids = idx[b:b + batch_size]
            x = np.stack([audio.normalize_for_model(chunks[i], do_normalize) for i in ids])
            x = torch.from_numpy(x).to(device)
            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                hidden = model(x, output_hidden_states=True).hidden_states  # (n_layers+1) x (batch, frames, dim)
            h = torch.stack([t.float() for t in hidden], dim=1)             # (batch, n_layers+1, frames, dim)
            mean[ids] = h.mean(dim=2).cpu().numpy()
            std[ids] = h.std(dim=2).cpu().numpy()
    return mean, std


def extract_split(model, do_normalize, split, df, device, out_dir):
    means, stds, index = [], [], []
    t_audio = t_model = 0.0
    audio_seconds = 0.0
    for row in tqdm(df.itertuples(), total=len(df), desc=f"[{split}]", unit="call"):
        t0 = time.perf_counter()
        stereo, sr = audio.read_wav(row.path)
        chunks, spans = audio.caller_chunks(stereo, sr)
        t_audio += time.perf_counter() - t0
        if not chunks:
            print(f"  [warning] {row.anon_id}: no speech chunks found")
            continue
        t0 = time.perf_counter()
        m, s = embed_chunks(model, chunks, do_normalize, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        means.append(m.astype(np.float16))
        stds.append(s.astype(np.float16))
        for k, (start, end) in enumerate(spans):
            index.append({"anon_id": row.anon_id, "y": row.y, "label": row.label, "chunk": k,
                          "start_s": start, "end_s": end, "duration_s": end - start})
            audio_seconds += end - start
    mean = np.concatenate(means)
    std = np.concatenate(stds)
    np.save(out_dir / f"{split}_mean.npy", mean)
    np.save(out_dir / f"{split}_std.npy", std)
    pd.DataFrame(index).to_csv(out_dir / f"{split}_index.csv", index=False)
    return {"n_calls": len(df), "n_chunks": int(len(index)), "chunk_hours": audio_seconds / 3600,
            "audio_prep_s": t_audio, "backbone_s": t_model, "shape": list(mean.shape),
            "cache_mb": (mean.nbytes + std.nbytes) / 1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="all", choices=["all"] + config.BACKBONE_ORDER)
    parser.add_argument("--fraction", type=float, default=1.0, help="stratified subset of each split (smoke test)")
    parser.add_argument("--splits", nargs="+", default=config.SPLITS)
    parser.add_argument("--force", action="store_true", help="recompute even if the cache exists")
    args = parser.parse_args()
    config.set_seed()
    device = config.get_device()
    keys = config.BACKBONE_ORDER if args.backbone == "all" else [args.backbone]
    for key in keys:
        out_dir = config.EMBEDDINGS_DIR / key
        out_dir.mkdir(parents=True, exist_ok=True)
        stats_path = out_dir / "extraction_stats.json"
        if stats_path.exists() and not args.force and args.fraction == 1.0:
            print(f"[{key}] cache exists ({stats_path}) -> skipping (use --force to recompute)")
            continue
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, do_normalize, info = load_backbone(key, device)
        info["load_s"] = time.time() - t0
        print(f"[{key}] {info['hf_name']}: {info['params_m']:.1f}M params, {info['num_hidden_layers']} layers x "
              f"{info['hidden_size']} dims, do_normalize={do_normalize}, frozen")
        # warm-up (cuDNN autotuning, allocator) so timings reflect steady state
        embed_chunks(model, [np.zeros(config.MODEL_SAMPLE_RATE * 4, np.float32)], do_normalize, device)
        info["splits"] = {}
        for split in args.splits:
            df = dataset.load_split(split, args.fraction)
            info["splits"][split] = extract_split(model, do_normalize, split, df, device, out_dir)
            s = info["splits"][split]
            print(f"  [{split}] {s['n_calls']} calls -> {s['n_chunks']} chunks ({s['chunk_hours']:.2f} h): "
                  f"audio prep {s['audio_prep_s']:.1f}s, backbone {s['backbone_s']:.1f}s, cache {s['cache_mb']:.0f} MB")
        info["total_s"] = time.time() - t0
        info["fraction"] = args.fraction
        info["chunk_seconds"] = config.CHUNK_SECONDS
        info["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        if device.type == "cuda":
            info["peak_vram_allocated_mb"] = torch.cuda.max_memory_allocated() / 1e6
            info["peak_vram_reserved_mb"] = torch.cuda.max_memory_reserved() / 1e6
        total_hours = sum(s["chunk_hours"] for s in info["splits"].values())
        total_backbone = sum(s["backbone_s"] for s in info["splits"].values())
        info["backbone_realtime_factor"] = total_hours * 3600 / max(total_backbone, 1e-9)
        save_json(info, stats_path)
        log_experiment(title=f"embedding extraction - {key}", model=info["hf_name"], fraction=args.fraction,
                       segmentation=f"VAD + {config.CHUNK_SECONDS:.0f}s chunks", device=info["device"],
                       chunks={s: info["splits"][s]["n_chunks"] for s in info["splits"]},
                       runtime_s=round(info["total_s"], 1), backbone_s=round(total_backbone, 1),
                       peak_vram_mb=round(info.get("peak_vram_allocated_mb", 0)),
                       realtime_factor=round(info["backbone_realtime_factor"], 1),
                       notes="frozen backbone, mean+std pooling of every hidden layer, fp16 cache")
        print(f"[{key}] done in {info['total_s']:.0f}s; peak VRAM {info.get('peak_vram_allocated_mb', 0):.0f} MB; "
              f"{info['backbone_realtime_factor']:.0f}x realtime")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
