"""Fase 3: ElevenLabs Scribe on VAD-compacted audio, chunked at segment boundaries and sent
concurrently; word times mapped back to the original timeline; cached per call.

transcribe(x, segs, key)   -> {"words": [{"text","start","end",...}], "latency_s", "n_chunks"}
transcribe_cached(cid, ch, x, segs)
annotate(words_by_channel) -> the transcript text Gemini sees ([AGENTE t] / [CLIENTE t] / [silencio Xs])
"""
import io, json, os, pathlib, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import numpy as np
from vad import compact, to_original, write_wav, SR

import config  # noqa: F401  (.env loader)

URL = "https://api.elevenlabs.io/v1/speech-to-text"
MODEL = os.environ.get("SCRIBE_MODEL", "scribe_v1")   # model.pkl records the model it was trained with; changing it requires `asr.py all` + `rubric.py all` + `model.py`
CACHE = pathlib.Path(__file__).parent / "cache" / ("asr" if MODEL == "scribe_v1" else f"asr_{MODEL}")   # one transcript cache per ASR model
CACHE.mkdir(parents=True, exist_ok=True)
CHUNK_S = 20.0   # measured: 20 s chunks give the lowest latency for both channels
HEDGE_S = 1.1    # Scribe p50 ≈ 1.0 s with a random tail to 2.5 s: duplicate any request still pending after this
_client = None
_ex = ThreadPoolExecutor(max_workers=64)   # one executor per process for all Scribe requests (hedges included)
MAX_INFLIGHT = int(os.environ.get("SCRIBE_MAX_INFLIGHT", 9))   # measured: this subscription allows ~20 concurrent Scribe requests (429 concurrent_limit_exceeded above); excess requests queue here instead of failing. Per process: with `--workers N` keep N x this under ~20
_sem = threading.BoundedSemaphore(MAX_INFLIGHT)


def chunks(segs, max_s=None):
    """Group VAD segments into runs of <= max_s voice. Cuts only on segment boundaries, never inside a word."""
    max_s = CHUNK_S if max_s is None else max_s
    out, cur, acc = [], [], 0.0
    for s, e in segs:
        if cur and acc + (e - s) > max_s:
            out.append(cur); cur, acc = [], 0.0
        cur.append((s, e)); acc += e - s
    return out + [cur] if cur else out


def wav_bytes(x, sr=SR):
    buf = io.BytesIO(); write_wav(buf, x, sr); return buf.getvalue()


class Contended(RuntimeError):
    """Raised by post_lowprio when the in-flight cap is saturated: the agent channel yields to caller channels."""


def _post(wav, wait_s=None):
    global _client
    if _client is None:                                 # one keep-alive session per process
        import httpx
        _client = httpx.Client(timeout=3.0, headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"]},
                               limits=httpx.Limits(max_connections=200, max_keepalive_connections=50))
    for attempt in (0, 1):
        if not _sem.acquire(timeout=wait_s):
            raise Contended("scribe in-flight cap saturated")
        try:
            r = _client.post(URL, data={"model_id": MODEL, "language_code": "es", "timestamps_granularity": "word",
                                        "diarize": "false", "tag_audio_events": "false"},
                             files={"file": ("a.wav", wav, "audio/wav")})
        finally:
            _sem.release()
        if r.status_code == 429 and attempt == 0:
            time.sleep(float(r.headers.get("retry-after", 0.5)))   # one retry; the hedge and the deadline bound the total
            continue
        r.raise_for_status()
        return r.json()


def post_lowprio(wav):
    """For the agent channel: never queues behind other calls' caller chunks; under load the rubric is skipped (-> f4)."""
    return _post(wav, wait_s=0.6)


def _first_ok(futs):
    futs, last = set(futs), None
    while futs:
        done, futs = wait(futs, return_when=FIRST_COMPLETED)
        for f in done:
            if f.exception() is None:
                return f.result()
            last = f.exception()
    raise last                                           # the real upstream error (HTTPStatusError 429, ReadTimeout...) reaches the abstain reason


def transcribe(x, segs, sr=SR, post=_post, hedge_s=None):
    """Tail-latency hedging: requests still pending after hedge_s get a duplicate; first success wins."""
    hedge_s = HEDGE_S if hedge_s is None else hedge_s
    t0 = time.perf_counter()
    parts = [compact(x, c, sr) for c in chunks(segs)]
    wavs = [wav_bytes(p[0], sr) for p in parts]
    futs = [[_ex.submit(post, w)] for w in wavs]
    _, pending = wait([f[0] for f in futs], timeout=hedge_s)
    if len(pending) <= max(2, len(wavs) // 2):          # a few stragglers: hedge them. Most of many pending: upstream is slow, don't double its load
        for i, w in enumerate(wavs):
            if futs[i][0] in pending:
                futs[i].append(_ex.submit(post, w))
    raws = [_first_ok(f) for f in futs]                  # losers of a hedge finish on their own (bounded by the HTTP timeout)
    words = []
    for (y, tmap), raw in zip(parts, raws):
        for w in raw.get("words", []):
            if w.get("type", "word") != "word":
                continue
            words.append({**w, "start": to_original(w["start"], tmap), "end": to_original(w["end"], tmap)})
    words.sort(key=lambda w: w["start"])
    return {"words": words, "text": " ".join(w["text"] for w in words),
            "latency_s": time.perf_counter() - t0, "n_chunks": len(parts), "n_hedged": sum(len(f) > 1 for f in futs), "raw": raws}


def transcribe_cached(cid, channel, x, segs, sr=SR):
    p = CACHE / f"{cid}_ch{channel}.json"
    if p.exists():
        return json.load(open(p))
    out = transcribe(x, segs, sr)
    json.dump(out, open(p, "w"), ensure_ascii=False)
    return out


def annotated_cached(cid):
    """Annotated transcript from the cache, or None if either channel is missing."""
    ps = [CACHE / f"{cid}_ch{ch}.json" for ch in (0, 1)]
    if not all(p.exists() for p in ps):
        return None
    return annotate({ch: json.load(open(p))["words"] for ch, p in enumerate(ps)})


def _fmt(t):
    return f"{int(t // 60)}:{t % 60:05.2f}"


def annotate(words_by_channel, names=("CLIENTE", "AGENTE"), line_gap=1.0, silence_gap=2.0):
    """Interleave both channels' words in time; new line on speaker change or a pause > line_gap;
    mark pauses > silence_gap. Same shape as the organizers' sample transcripts."""
    ws = sorted(((w["start"], w["end"], ch, w["text"]) for ch, wl in words_by_channel.items() for w in wl))
    lines, cur, last_end = [], None, None
    for s, e, ch, txt in ws:
        if cur is None or ch != cur[0] or s - cur[2] > line_gap:
            if cur is not None:
                lines.append(f"[{names[cur[0]]:7s} {_fmt(cur[1])}] {' '.join(cur[3])}")
            if last_end is not None and s - last_end > silence_gap:
                lines.append(f"{'':17s}[silencio {s - last_end:.1f}s]")
            cur = [ch, s, e, [txt]]
        else:
            cur[2] = e; cur[3].append(txt)
        last_end = max(last_end or 0.0, e)
    if cur is not None:
        lines.append(f"[{names[cur[0]]:7s} {_fmt(cur[1])}] {' '.join(cur[3])}")
    return "\n".join(lines)


if __name__ == "__main__" and sys.argv[1:] == ["all"]:
    # Fase 3 batch: transcribe both channels of the 353 calls (cached), report latency and repetition loops.
    import csv
    from concurrent.futures import ThreadPoolExecutor as _Pool
    from vad import read_wav, vad
    root = pathlib.Path(__file__).parent / "data" / "hackmty26"
    ids = [r["anon_id"] for r in csv.DictReader(open(root / "manifest.csv"))]

    def one(cid):
        x, sr = read_wav(root / "audio" / f"{cid}.wav")
        return [transcribe_cached(cid, ch, x[:, ch], vad(x[:, ch], sr), sr) for ch in (0, 1)]

    with _Pool(int(os.environ.get('BATCH_POOL', 4))) as ex:   # 4 calls x ~3 chunks per channel stays under the ~20 concurrent the key allows
        res = list(ex.map(one, ids))
    lat = np.array([r[0]["latency_s"] + r[1]["latency_s"] for r in res])
    loops = 0
    for r in res:
        toks = r[0]["text"].lower().split()
        grams = [" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)]
        loops += any(grams.count(g) >= 4 for g in set(grams))
    print(f"calls={len(res)}  scribe latency (sum of 2 channels, sequential) mean={lat.mean():.2f}s p95={np.percentile(lat, 95):.2f}s")
    print(f"caller words mean={np.mean([len(r[0]['words']) for r in res]):.0f}  calls with a 4-gram repeated >=4x: {loops}")
    sys.exit(0)

if __name__ == "__main__":
    # self-check with a fake ASR: chunking respects boundaries, word times round-trip to the original timeline
    segs = [(1.0, 4.0), (5.0, 9.0), (10.0, 14.0), (20.0, 22.0), (30.0, 31.0)]
    CHUNK_S = 10.0
    cs = chunks(segs, 10.0)
    assert cs == [[(1.0, 4.0), (5.0, 9.0)], [(10.0, 14.0), (20.0, 22.0), (30.0, 31.0)]], cs
    x = np.zeros(int(40 * SR), dtype=np.float32)

    def fake_post(wav):   # one word per second of compacted audio
        n = (len(wav) - 44) // 2 // SR
        return {"words": [{"text": f"w{i}", "start": i + 0.1, "end": i + 0.9, "type": "word"} for i in range(n)]}

    out = transcribe(x, segs, post=fake_post)
    assert out["n_chunks"] == 2 and out["words"][0]["start"] == 1.1, out["words"][:2]
    assert all(any(s <= w["start"] <= e for s, e in segs) for w in out["words"]), "word outside a voice segment"
    print(annotate({0: out["words"][:3], 1: [{"text": "hola", "start": 25.0, "end": 25.4}]}))

    seen = set()
    def slow_first(wav):   # first request per payload stalls 3 s; the hedge must win in well under that
        if wav not in seen:
            seen.add(wav); time.sleep(3.0)
        return fake_post(wav)
    t0 = time.perf_counter(); out = transcribe(x, segs, post=slow_first, hedge_s=0.2); dt = time.perf_counter() - t0
    assert out["n_hedged"] == 2 and dt < 1.5 and len(out["words"]) == 16, (out["n_hedged"], dt, len(out["words"]))

    def all_slow(wav):   # every chunk slow: the hedge cap must NOT duplicate (would double load on a struggling upstream)
        time.sleep(0.4); return fake_post(wav)
    segs4 = [(0, 8), (10, 18), (20, 28), (30, 38)]   # 4 chunks of 8 s at CHUNK_S = 10
    out = transcribe(x, segs4, post=all_slow, hedge_s=0.1)
    assert out["n_chunks"] == 4 and out["n_hedged"] == 0, (out["n_chunks"], out["n_hedged"])
    print("OK")
