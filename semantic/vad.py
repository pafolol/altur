"""Fase 1: energy VAD (§2.5) + compaction (§2.6). Machinery, not features.

read_wav(path)            -> (x[n, ch] float32 in [-1,1], sr)
vad(x1d, sr)              -> [(start_s, end_s), ...]
compact(x1d, segs, sr)    -> (y1d, tmap)   tmap rows: (t_compact, t_orig, dur)
to_original(t, tmap)      -> t_orig for a compacted-time t
"""
import bisect, json, pathlib, sys, wave
import numpy as np

SR = 8000


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getsampwidth() == 2 and w.getframerate() == SR, (w.getsampwidth(), w.getframerate())
        ch = w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, ch)
    return x.astype(np.float32) / 32768.0, SR


def write_wav(path, x, sr=SR):
    with wave.open(path if hasattr(path, "write") else str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def _runs(mask):
    """Boolean mask -> list of (i_start, i_end_exclusive) runs of True."""
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def vad(x, sr=SR, frame=0.025, hop=0.010, margin_db=12.0, gap=0.25, min_len=0.15, pad=0.04):
    x = x / (np.abs(x).max() or 1.0)                      # normalize by channel peak
    fl, hl = int(frame * sr), int(hop * sr)
    n = 1 + max(0, (len(x) - fl) // hl)
    idx = np.arange(fl)[None, :] + hl * np.arange(n)[:, None]
    e = 10 * np.log10((x[idx] ** 2).mean(1) + 1e-10)
    thr = np.percentile(e, 10) + margin_db               # noise floor = p10, +12 dB
    segs = [(a * hop, (b - 1) * hop + frame) for a, b in _runs(e > thr)]

    merged = []                                           # close gaps <= 250 ms
    for s, e_ in segs:
        if merged and s - merged[-1][1] <= gap:
            merged[-1] = (merged[-1][0], e_)
        else:
            merged.append((s, e_))
    dur = len(x) / sr
    out = []                                              # drop < 150 ms, pad 40 ms
    for s, e_ in merged:
        if e_ - s < min_len:
            continue
        s, e_ = max(0.0, s - pad), min(dur, e_ + pad)
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], e_)
        else:
            out.append((s, e_))
    return out


def compact(x, segs, sr=SR, sil=0.5):
    gap = np.zeros(int(sil * sr), dtype=x.dtype)
    parts, tmap, t = [], [], 0.0
    for s, e in segs:
        a, b = int(s * sr), int(e * sr)
        parts += [x[a:b], gap]
        tmap.append((t, s, (b - a) / sr))
        t += (b - a) / sr + sil
    return (np.concatenate(parts) if parts else x[:0]), tmap


def to_original(t, tmap):
    i = max(0, bisect.bisect_right([r[0] for r in tmap], t) - 1)
    c, o, d = tmap[i]
    return o + min(max(t - c, 0.0), d)                    # inside a gap -> clamp to segment end


if __name__ == "__main__":
    # Fase 1 acceptance: 20 calls, VAD on channel 0 vs turns.json (±100 ms), mean compacted ≈ 40 s.
    import csv
    root = pathlib.Path(__file__).parent / "data" / "hackmty26"
    ids = [r["anon_id"] for r in csv.DictReader(open(root / "manifest.csv"))][:20]
    errs, comp_s, orig_s = [], [], []
    for cid in ids:
        x, sr = read_wav(root / "audio" / f"{cid}.wav")
        segs = vad(x[:, 0], sr)
        ref = [(t["start"], t["end"]) for t in json.load(open(root / "turns" / f"{cid}.json"))["turns"] if t["channel"] == 0]
        bounds = np.array([b for s in segs for b in s])
        errs += [np.abs(bounds - b).min() for s in ref for b in s] if len(bounds) else [np.inf] * 2 * len(ref)
        y, tmap = compact(x[:, 0], segs, sr)
        comp_s.append(len(y) / sr); orig_s.append(len(x) / sr)
        # round-trip check of the time map
        for c, o, d in tmap:
            assert abs(to_original(c + d / 2, tmap) - (o + d / 2)) < 1e-6
    errs = np.array(errs)
    print(f"calls={len(ids)}  boundaries={len(errs)}")
    print(f"boundary |err| median={np.median(errs)*1000:.0f} ms  mean={errs.mean()*1000:.0f} ms  within100ms={(errs<=0.1).mean()*100:.1f}%")
    print(f"original mean {np.mean(orig_s):.1f} s  ->  compacted mean {np.mean(comp_s):.1f} s")
    assert (errs <= 0.1).mean() > 0.9, "VAD boundaries do not match turns.json"
    print("OK")
