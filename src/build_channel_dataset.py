"""
THE DATASET FACTORY - the official Altur calls, pushed through telephone lines.

    hackmty26/<id>.wav (stereo 8 kHz)
      -> channel 0 (caller), FULL TIMELINE including every silence
      -> phone_channel.apply_channel(): real codec(s), 20 ms RTP framing, loss + concealment, drift,
         comfort noise, line noise, AGC
      -> channel_dataset/<split>/audio/<id>__<family><variant>.wav      (8 kHz mono PCM16)
      -> channel_dataset/reconstructed_stereo/<split>/<id>__...wav      (ch0 transformed, ch1 ORIGINAL agent)

THE SPLIT RULE, WHICH IS THE WHOLE POINT
    train call -> transformed sample lands in TRAIN.   val call -> transformed sample lands in VAL.
  A transformed validation call is still a validation call: it shares a speaker, a script and a recording
  session with its original, so letting one into training would leak the validation set into the model
  through the back door and every number after that would be a lie. The split is read from the official
  manifest, carried through untouched, and asserted on write; tests/test_channel_dataset.py checks that no
  call id appears in two splits and that every transformed row still carries its original label.

FAMILIES
    seen    G.711 mu-law/A-law, G.726 - generated for TRAIN and VAL. May be trained on.
    unseen  GSM, AMR-NB, iLBC, Speex, Opus, G.722 - generated for VAL ONLY, and never for train, so there is
            no file on disk that could accidentally be trained on. This is the generalisation test: a channel
            the model has never met.

REPRODUCIBLE AND RESUMABLE
  The channel of a sample is drawn from a generator seeded with (call id, family, variant), so re-running the
  factory reproduces the same dataset bit for bit, and a sample whose WAV and metadata already exist is
  skipped. A crash half way through costs nothing.

Run:  python src/build_channel_dataset.py --split all                 (the full dataset)
      python src/build_channel_dataset.py --split val --limit 10      (a quick look)
      python src/build_channel_dataset.py --split all --force         (regenerate everything)
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

import _bootstrap  # noqa: F401
import audio
import config
import dataset
import phone_channel as pc

ROOT = config.PROJECT_ROOT.parent / "channel_dataset"
MANIFESTS = ROOT / "manifests"
LOGS = ROOT / "logs"

# How many transformed copies of each call, per split and family. TRAIN gets two draws from the seen family so
# the model sees the same voice under two different lines (that is the point of domain randomisation); VAL gets
# one draw per family, because an evaluation set has to be a fixed target, not a distribution we keep sampling.
PLAN = {
    ("train", "seen"): 2,
    ("train", "unseen"): 0,      # deliberately empty: nothing unseen may ever exist on the training side
    ("val", "seen"): 1,
    ("val", "unseen"): 1,
}


def sample_seed(anon_id, family, variant):
    """A stable 64-bit seed from the identity of the sample (hash() is salted per process and useless here)."""
    import hashlib
    h = hashlib.sha256(f"{anon_id}|{family}|{variant}".encode()).digest()
    return int.from_bytes(h[:8], "big")


def sample_name(anon_id, family, variant):
    return f"{anon_id}__{family}{variant}"


def paths_for(split, anon_id, family, variant):
    name = sample_name(anon_id, family, variant)
    return (ROOT / split / "audio" / f"{name}.wav",
            ROOT / "reconstructed_stereo" / split / f"{name}.wav")


def transform_one(job):
    """
    One (call, family, variant) -> two WAVs on disk + one metadata row. Runs in a worker process, so it takes
    plain data in and returns plain data out.
    """
    anon_id, label, y, split, src_path, family, variant, make_stereo = job
    mono_path, stereo_path = paths_for(split, anon_id, family, variant)
    row = {"call_id": anon_id, "sample_id": sample_name(anon_id, family, variant), "split": split,
           "label": label, "y": int(y), "family": family, "variant": int(variant),
           "original_path": str(src_path), "channel_path": str(mono_path), "stereo_path": str(stereo_path)}
    try:
        stereo, sr = audio.read_wav(src_path)
        assert sr == pc.SR, f"expected {pc.SR} Hz, got {sr}"
        caller = audio.get_channel(stereo, config.CALLER_CHANNEL)
        agent = audio.get_channel(stereo, config.AGENT_CHANNEL) if stereo.shape[1] > 1 else np.zeros_like(caller)

        # The DTX / comfort-noise stage needs to know where the speech is. It is measured on the ORIGINAL
        # caller channel, where the silences are true digital silence and any detector gets it right.
        regions = audio.detect_speech_energy(caller, sr)
        mask = pc.speech_mask_from_regions(regions, len(caller))

        seed = sample_seed(anon_id, family, variant)
        params = pc.sample_channel(np.random.default_rng(seed), family=family)
        out, info = pc.apply_channel(caller, params, mask, seed=seed % (2 ** 31))

        mono_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(mono_path, pc.float_to_pcm16(out), pc.SR, subtype="PCM_16")
        if make_stereo:
            # ch0 = the caller AFTER the telephone line, ch1 = the agent EXACTLY as the challenge recorded it.
            # Only channel 0 went through a codec; the report says so wherever this file is mentioned, so the
            # agent samples are copied as int16 straight from the source. Round-tripping them through float
            # would NOT be lossless - soundfile scales int16 by 1/32768 on the way in and float_to_pcm16
            # multiplies by 32767 on the way out, which quietly shifts every loud sample by one count.
            stereo_path.parent.mkdir(parents=True, exist_ok=True)
            source_i16, _ = sf.read(src_path, dtype="int16", always_2d=True)
            agent_i16 = source_i16[:, config.AGENT_CHANNEL] if source_i16.shape[1] > 1 else np.zeros(len(out), np.int16)
            n = min(len(out), len(agent_i16))
            sf.write(stereo_path, np.stack([pc.float_to_pcm16(out[:n]), agent_i16[:n]], axis=1),
                     pc.SR, subtype="PCM_16")
        else:
            row["stereo_path"] = ""

        row.update({k: params[k] for k in params if k != "codecs"})
        row.update({"codecs": "+".join(params["codecs"]), "n_hops": len(params["codecs"]),
                    "duration_original_s": len(caller) / sr, "duration_channel_s": len(out) / pc.SR,
                    "sample_rate": pc.SR, "seed": seed, "status": "ok",
                    "speech_s_original": float(sum(e - s for s, e in regions)),
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        row.update({k: info[k] for k in ("estimated_delay_ms", "estimated_delay_samples", "alignment_correlation",
                                         "residual_delay_ms", "residual_delay_samples", "residual_correlation",
                                         "input_rms_db", "output_rms_db", "gain_db", "realised_loss")})
    except Exception as exc:
        row.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                    "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return row


def build_jobs(splits, families, limit, variants_override, make_stereo, force):
    df = dataset.load_manifest()
    jobs, skipped = [], 0
    for split in splits:
        part = df[df["split"] == split]
        if limit:
            # a balanced subset, so a --limit run never ends up all-human or all-synthetic
            part = pd.concat([g.head(max(1, limit // 2)) for _, g in part.groupby("label")]).sort_values("anon_id")
        for family in families:
            n_variants = variants_override if variants_override is not None else PLAN.get((split, family), 0)
            for variant in range(n_variants):
                for r in part.itertuples():
                    assert r.split == split, f"split mismatch for {r.anon_id}: {r.split} != {split}"
                    mono, stereo = paths_for(split, r.anon_id, family, variant)
                    if not force and mono.exists() and (not make_stereo or stereo.exists()):
                        skipped += 1
                        continue
                    jobs.append((r.anon_id, r.label, r.y, split, r.path, family, variant, make_stereo))
    return jobs, skipped


def write_manifests():
    """Re-read every per-sample row on disk and rebuild the manifests from scratch (so they never drift)."""
    rows = []
    for p in sorted(LOGS.glob("rows_*.jsonl")):
        for line in p.read_text(encoding="utf8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop_duplicates(subset="sample_id", keep="last").sort_values("sample_id")
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    df.to_csv(MANIFESTS / "all.csv", index=False)
    ok = df[df["status"] == "ok"]
    for split in config.SPLITS:
        part = ok[ok["split"] == split]
        part.to_csv(MANIFESTS / f"{split}.csv", index=False)
        for family in ("seen", "unseen"):
            sub = part[part["family"] == family]
            if len(sub):
                sub.to_csv(MANIFESTS / f"{split}_{family}.csv", index=False)
    # The guard that matters: a call id must never appear in two splits.
    per_call = ok.groupby("call_id")["split"].nunique()
    assert (per_call <= 1).all(), f"SPLIT LEAK: {list(per_call[per_call > 1].index)}"
    return df


def main():
    parser = argparse.ArgumentParser(description="Push the Altur caller channels through telephone lines.")
    parser.add_argument("--split", default="all", choices=["all", *config.SPLITS])
    parser.add_argument("--family", default="all", choices=["all", "seen", "unseen"])
    parser.add_argument("--limit", type=int, default=0, help="only this many calls per split (balanced)")
    parser.add_argument("--variants", type=int, default=None, help="override the per-split variant count")
    # Every worker is a fresh interpreter that re-imports numpy/scipy/soundfile (spawn, on Windows), so the
    # ceiling here is memory, not cores: 22 workers exhausted the paging file on a 24-thread machine.
    parser.add_argument("--workers", type=int, default=min(8, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--no-stereo", action="store_true", help="skip the reconstructed stereo files")
    parser.add_argument("--force", action="store_true", help="regenerate samples that already exist")
    args = parser.parse_args()

    splits = config.SPLITS if args.split == "all" else [args.split]
    families = ["seen", "unseen"] if args.family == "all" else [args.family]
    make_stereo = not args.no_stereo
    for d in (MANIFESTS, LOGS):
        d.mkdir(parents=True, exist_ok=True)

    jobs, skipped = build_jobs(splits, families, args.limit, args.variants, make_stereo, args.force)
    print(f"[factory] {len(jobs)} samples to generate, {skipped} already on disk"
          f"  (splits={splits}, families={families}, workers={args.workers})")
    if not jobs:
        df = write_manifests()
        print(f"[factory] nothing to do; manifest has {len(df)} rows")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOGS / f"rows_{stamp}.jsonl"
    t0 = time.time()
    done = failed = 0
    with open(log_path, "a", encoding="utf8") as log, ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(transform_one, j): j for j in jobs}
        for fut in as_completed(futures):
            row = fut.result()
            log.write(json.dumps(row, default=float) + "\n")
            log.flush()
            done += 1
            failed += row["status"] != "ok"
            if row["status"] != "ok":
                print(f"  [FAILED] {row['sample_id']}: {row.get('error')}")
            if done % 25 == 0 or done == len(jobs):
                rate = done / max(time.time() - t0, 1e-9)
                eta = (len(jobs) - done) / max(rate, 1e-9)
                print(f"  {done}/{len(jobs)}  ({rate:.1f}/s, ETA {eta / 60:.1f} min, {failed} failed)")

    df = write_manifests()
    ok = df[df["status"] == "ok"]
    print(f"\n[factory] done in {(time.time() - t0) / 60:.1f} min: {len(ok)} samples ok, "
          f"{(df['status'] != 'ok').sum()} failed")
    print(ok.groupby(["split", "family", "label"]).size().to_string())
    print(f"\ntransformed audio: {ok['duration_channel_s'].sum() / 3600:.2f} h")
    print(f"alignment correlation: min {ok['alignment_correlation'].min():.3f}  "
          f"median {ok['alignment_correlation'].median():.3f}  max {ok['alignment_correlation'].max():.3f}")
    print(f"measured delay (ms):   min {ok['estimated_delay_ms'].min():.1f}  "
          f"median {ok['estimated_delay_ms'].median():.1f}  max {ok['estimated_delay_ms'].max():.1f}")
    print(f"manifests -> {MANIFESTS}")


if __name__ == "__main__":
    main()
