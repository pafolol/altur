"""
FROZEN BACKBONE -> CACHED EMBEDDINGS, for the original and the telephone-channel audio alike.

Same backbone, same layer stack, same chunking, same loudness normalisation as the deployed specialist
(extract_embeddings.py) - the experimental variable is the TRAINING DOMAIN, not the feature extractor. The
one deliberate difference is the voice-activity detector:

    the specialist segments with the energy VAD (config.VAD_METHOD), which assumes the caller channel is
    digitally silent between turns. It is, in the Altur recordings. It is not, on a telephone line: AGC and
    comfort noise lift the noise floor by ~45 dB, the dynamic range collapses from 56 dB to 10 dB, and the
    detector returns NOTHING. Four of the ten pilot calls produced zero chunks.

So everything here is segmented with Silero VAD, a small neural detector that does not depend on the noise
floor. The report keeps the two effects apart by scoring the specialist under BOTH detectors, so the gain
attributable to the segmentation fix is never quietly credited to the new training data.

THE SETS (a set is a split x domain pair; the split is never crossed)
    train_original        the official TRAIN calls, untouched            domain = altur_original
    train_channel         the SEEN-family transforms of those same calls domain = channel_seen
    val_original          the official VAL calls, untouched              domain = altur_original
    val_channel_seen      SEEN-family transforms of the VAL calls        domain = channel_seen
    val_channel_unseen    UNSEEN-family transforms of the VAL calls      domain = channel_unseen

Only the two train_* sets may ever reach a gradient.

Run:  python src/extract_channel_embeddings.py --sets all [--vad silero] [--limit 20]
Outputs: outputs/embeddings/robust/<set>_{mean.npy,index.csv}, extraction_stats.json
"""
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import _bootstrap  # noqa: F401
import audio
import config
import dataset
from build_channel_dataset import MANIFESTS
from extract_embeddings import embed_chunks, load_backbone
from metrics import log_experiment, save_json

OUT = config.EMBEDDINGS_DIR / "robust"
SETS = {
    "train_original":     {"split": "train", "family": None,     "domain": "altur_original"},
    "train_channel":      {"split": "train", "family": "seen",   "domain": "channel_seen"},
    "val_original":       {"split": "val",   "family": None,     "domain": "altur_original"},
    "val_channel_seen":   {"split": "val",   "family": "seen",   "domain": "channel_seen"},
    "val_channel_unseen": {"split": "val",   "family": "unseen", "domain": "channel_unseen"},
}
TRAINABLE = {"train_original", "train_channel"}


def set_rows(name, limit=0):
    """One row per audio file in a set: sample_id, call_id, label, y, path, domain, split."""
    spec = SETS[name]
    if spec["family"] is None:
        df = dataset.load_split(spec["split"])
        rows = pd.DataFrame({"sample_id": df["anon_id"], "call_id": df["anon_id"], "label": df["label"],
                             "y": df["y"], "path": df["path"]})
    else:
        man = pd.read_csv(MANIFESTS / "all.csv")
        man = man[(man["status"] == "ok") & (man["split"] == spec["split"]) & (man["family"] == spec["family"])]
        if man.empty:
            raise SystemExit(f"no samples for {name} - run build_channel_dataset.py first")
        rows = pd.DataFrame({"sample_id": man["sample_id"], "call_id": man["call_id"], "label": man["label"],
                             "y": man["y"], "path": man["channel_path"]})
    rows = rows.assign(domain=spec["domain"], split=spec["split"]).sort_values("sample_id").reset_index(drop=True)
    # A set that is allowed to train must not contain a validation call. Cheap to check, catastrophic to miss.
    official = dataset.load_manifest().set_index("anon_id")["split"]
    wrong = [c for c in rows["call_id"].unique() if official.get(c) != spec["split"]]
    assert not wrong, f"{name} contains calls from the wrong split: {wrong[:5]}"
    if limit:
        rows = pd.concat([g.head(max(1, limit // 2)) for _, g in rows.groupby("label")]).reset_index(drop=True)
    return rows


def extract_set(model, do_normalize, name, rows, device, vad):
    means, index = [], []
    t_audio = t_model = 0.0
    empty = 0
    for r in tqdm(rows.itertuples(), total=len(rows), desc=f"[{name}]", unit="file"):
        t0 = time.perf_counter()
        wave, sr = audio.read_wav(r.path)
        chunks, spans = audio.caller_chunks(wave, sr, vad_method=vad)
        t_audio += time.perf_counter() - t0
        if not chunks:
            empty += 1
            continue
        t0 = time.perf_counter()
        m, _ = embed_chunks(model, chunks, do_normalize, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_model += time.perf_counter() - t0
        means.append(m.astype(np.float16))
        for k, (start, end) in enumerate(spans):
            index.append({"sample_id": r.sample_id, "call_id": r.call_id, "y": int(r.y), "label": r.label,
                          "domain": r.domain, "split": r.split, "chunk": k,
                          "start_s": start, "end_s": end, "duration_s": end - start})
    if not means:
        raise SystemExit(f"{name}: every file produced zero chunks")
    mean = np.concatenate(means)
    OUT.mkdir(parents=True, exist_ok=True)
    np.save(OUT / f"{name}_mean.npy", mean)
    pd.DataFrame(index).to_csv(OUT / f"{name}_index.csv", index=False)
    return {"n_files": len(rows), "n_chunks": int(len(index)), "files_without_speech": empty,
            "chunk_hours": sum(i["duration_s"] for i in index) / 3600, "shape": list(mean.shape),
            "audio_prep_s": t_audio, "backbone_s": t_model, "cache_mb": mean.nbytes / 1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sets", nargs="+", default=["all"], choices=["all", *SETS])
    parser.add_argument("--backbone", default="wav2vec2_spanish", choices=config.BACKBONE_ORDER)
    parser.add_argument("--vad", default="silero", choices=["silero", "energy"])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config.set_seed()
    names = list(SETS) if "all" in args.sets else args.sets
    device = config.get_device()

    todo = [n for n in names if args.force or args.limit or not (OUT / f"{n}_mean.npy").exists()]
    for n in set(names) - set(todo):
        print(f"[{n}] cache exists -> skipping (use --force to recompute)")
    if not todo:
        return

    model, do_normalize, info = load_backbone(args.backbone, device)
    print(f"[backbone] {info['hf_name']}: {info['params_m']:.1f}M params, {info['num_hidden_layers']} layers, "
          f"do_normalize={do_normalize}, frozen  |  VAD = {args.vad}")
    embed_chunks(model, [np.zeros(config.MODEL_SAMPLE_RATE * 4, np.float32)], do_normalize, device)  # warm-up

    stats = {"backbone": args.backbone, "hf_name": info["hf_name"], "vad": args.vad,
             "chunk_seconds": config.CHUNK_SECONDS, "target_rms_db": config.TARGET_RMS_DB, "sets": {}}
    t_all = time.time()
    for name in todo:
        rows = set_rows(name, args.limit)
        s = extract_set(model, do_normalize, name, rows, device, args.vad)
        stats["sets"][name] = s
        print(f"  [{name}] {s['n_files']} files -> {s['n_chunks']} chunks ({s['chunk_hours']:.2f} h), "
              f"{s['files_without_speech']} without speech, {s['cache_mb']:.0f} MB, "
              f"prep {s['audio_prep_s']:.0f}s + backbone {s['backbone_s']:.0f}s")
    stats["total_s"] = time.time() - t_all
    stats["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    if not args.limit:
        save_json(stats, OUT / "extraction_stats.json")
    log_experiment(title="embedding extraction - original + telephone channel", model=info["hf_name"],
                   segmentation=f"{args.vad} VAD + {config.CHUNK_SECONDS:.0f}s chunks, RMS {config.TARGET_RMS_DB} dB",
                   device=stats["device"], sets={k: v["n_chunks"] for k, v in stats["sets"].items()},
                   runtime_s=round(stats["total_s"], 1),
                   notes="frozen backbone, mean pooling of every hidden layer; train sets are the only ones "
                         "that may be trained on")
    print(f"\n[done] {time.time() - t_all:.0f}s -> {OUT}")


if __name__ == "__main__":
    main()
