"""
HOW EARLY, not just what - the detection-latency timeline of a call, and the whole "scene" payload the
web visualiser plays back.

Altur judges latency as "how fast the system can determine an outcome with reasonable confidence", so
"synthetic, 91 %" is only half an answer. The other half is *when*: after how much of the caller's voice,
how far into the call, and how long before the call would have ended anyway.

WHAT MAKES THIS REAL RATHER THAN A REPLAY ANIMATION. The acoustic layer cuts caller speech into 4 s
chunks, scores each one independently, and aggregates the chunk log-odds by mean - so the aggregate of
the first k chunks IS exactly what the deployed acoustic model would have answered having heard only
those k. The behaviour layer is simply re-run on the call truncated at each chunk boundary (it costs
60-500 ms and abstains until it has two interaction events, exactly as it would live). The two are then
mixed by the deployed combiner with the deployed weights - so the running verdict on the page is what the
50 / 50 fusion would have said at that moment, not an approximation of it.

WHAT IT IS NOT. The model is not streaming: it scores a finished recording. This is a prefix evaluation -
"what would it have concluded by second t" - the standard way to measure detection latency for a
non-streaming model, and what the behaviour module's own scripts/evaluate_prefixes.py does. It is stated
on the page rather than implied.

    progression(...)      one step per chunk: the ACOUSTIC running verdict
    fusion_progression()  one step per chunk: the FUSION running verdict (acoustic + behaviour prefixes)
    metrics_from_steps()  first prediction, first confident detection, lead time, early-vs-final
    scene(...)            everything the visualiser needs for one call, both channels included
"""
import io
import time
import wave

import numpy as np

import _bootstrap  # noqa: F401
import config
from metrics import aggregate_chunk_scores


def _sigmoid(z):
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -40, 40))))


def calibrated(score, calibration):
    a = (calibration or {}).get("a", 1.0)
    b = (calibration or {}).get("b", 0.0)
    return _sigmoid(a * float(score) + b)


def is_confident(p, hi=None, lo=None):
    hi = config.CONFIDENT_SYNTHETIC_THRESHOLD if hi is None else hi
    lo = config.CONFIDENT_HUMAN_THRESHOLD if lo is None else lo
    return p >= hi or p <= lo


# ============================================================================= the acoustic prefix walk


def progression(chunk_scores, chunk_spans, calibration=None, aggregation=None, hi=None, lo=None):
    """
    One step per chunk: what the acoustic model would have said having heard only the chunks up to it.

    `chunk_spans` are (start, end) in seconds of the ORIGINAL call, so call_time_s is where the step sits
    on the conversation's timeline, while caller_speech_s is how much actual caller voice the model had
    consumed by then. Those two are very different numbers on a call with long agent turns, and the
    difference is the point: the model needs seconds of SPEECH, not seconds of call.
    """
    aggregation = aggregation or config.CHUNK_AGGREGATION
    scores = [float(s) for s in (chunk_scores or [])]
    spans = [(float(a), float(b)) for a, b in (chunk_spans or [])]
    if not scores or len(spans) != len(scores):
        return []
    steps, speech = [], 0.0
    for k, (score, (start, end)) in enumerate(zip(scores, spans), start=1):
        speech += end - start
        p = calibrated(aggregate_chunk_scores(scores[:k], aggregation), calibration)
        steps.append({
            "chunk": k,
            "call_time_s": round(end, 2),
            "chunk_start_s": round(start, 2),
            "caller_speech_s": round(speech, 2),
            "chunk_score": round(score, 4),
            "chunk_probability": round(calibrated(score, calibration), 4),   # this chunk ALONE
            "probability": round(p, 4),                                      # the running verdict
            "confident": is_confident(p, hi, lo),
        })
    return steps


# ============================================================================= the metrics


def _first(steps, predicate):
    return next((s for s in steps if predicate(s)), None)


def metrics_from_steps(steps, duration_s, threshold=None, hi=None, lo=None, inference_ms=None):
    """
    The latency story for one sequence of running verdicts. Works for the acoustic walk and the fusion
    walk alike - it only needs `probability`, `call_time_s`, `caller_speech_s` and `confident` per step.

    Every field is None when the call does not support it - no chunks at all, or a call that never
    becomes confident - rather than a zero that would read as "detected instantly".
    """
    threshold = config.DECISION_THRESHOLD if threshold is None else threshold
    hi = config.CONFIDENT_SYNTHETIC_THRESHOLD if hi is None else hi
    lo = config.CONFIDENT_HUMAN_THRESHOLD if lo is None else lo
    duration = float(duration_s or 0.0)
    final = steps[-1] if steps else None
    first = steps[0] if steps else None
    confident = _first(steps, lambda s: s["confident"])

    def verdict(p):
        return None if p is None else ("synthetic" if p >= threshold else "human")

    m = {
        "call_duration_sec": round(duration, 2) or None,
        "n_chunks": len(steps),
        "caller_speech_sec": final["caller_speech_s"] if final else 0.0,
        "first_prediction_at_call_sec": first["call_time_s"] if first else None,
        "first_prediction_after_caller_speech_sec": first["caller_speech_s"] if first else None,
        "first_prediction_probability": first["probability"] if first else None,
        "confident_detection_at_call_sec": confident["call_time_s"] if confident else None,
        "confident_detection_after_caller_speech_sec": confident["caller_speech_s"] if confident else None,
        "confident_detection_probability": confident["probability"] if confident else None,
        "confident_detection_chunk": confident["chunk"] if confident else None,
        "inference_ms": round(inference_ms) if inference_ms is not None else None,
    }
    if confident and duration:
        m["seconds_before_call_end"] = round(max(duration - confident["call_time_s"], 0.0), 2)
        m["percentage_of_call_elapsed_at_detection"] = round(
            100.0 * min(confident["call_time_s"] / duration, 1.0), 1)
    else:
        m["seconds_before_call_end"] = None
        m["percentage_of_call_elapsed_at_detection"] = None

    early_p = confident["probability"] if confident else None
    final_p = final["probability"] if final else None
    m["early"] = None if confident is None else {
        "verdict": verdict(early_p), "probability": early_p,
        "at_call_sec": confident["call_time_s"], "after_caller_speech_sec": confident["caller_speech_s"]}
    m["final"] = None if final is None else {
        "verdict": verdict(final_p), "probability": final_p,
        "after_caller_speech_sec": final["caller_speech_s"]}
    # Shown, never hidden: if the early commitment disagreed with the finished call, that is the single
    # most useful thing on the page for anyone deciding whether to trust an early verdict.
    m["early_agrees_with_final"] = (None if (confident is None or final is None)
                                    else verdict(early_p) == verdict(final_p))
    m["never_confident"] = bool(steps) and confident is None
    m["thresholds"] = {"decision": threshold, "confident_synthetic": hi, "confident_human": lo}
    return m


def timeline(chunk_scores, chunk_spans, duration_s, calibration=None, aggregation=None,
             threshold=None, hi=None, lo=None, inference_ms=None):
    """The acoustic walk plus its metrics - what /demo/benchmark aggregates over the split."""
    steps = progression(chunk_scores, chunk_spans, calibration, aggregation, hi, lo)
    return {"steps": steps, "metrics": metrics_from_steps(steps, duration_s, threshold, hi, lo, inference_ms)}


# ============================================================================= the fusion prefix walk


def _prefix_wav(stereo, sr, seconds):
    """The call truncated at `seconds`, as the stereo 8 kHz 16-bit WAV every layer expects."""
    n = max(int(seconds * sr), 1)
    pcm = (np.clip(stereo[:n], -1.0, 1.0) * 32767.0).astype("<i2")
    if pcm.ndim == 1:
        pcm = np.stack([pcm, np.zeros_like(pcm)], axis=1)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def fusion_progression(fusion_det, acoustic_steps, stereo, sr, hi=None, lo=None):
    """
    What the 50 / 50 FUSION would have said at each chunk boundary.

    The acoustic prefix is exact (see progression). The behaviour layer is re-run on the call truncated at
    that boundary - it abstains until it has two interaction events, exactly as it would live, and the
    combiner hands its weight to the acoustic layer until then. The verifier is never consulted on a
    prefix: it costs a paid API call, and the page is honest that it is only asked at the end. The mixing
    is combine() with the deployed config, so these numbers are the deployed decision, not a model of it.
    """
    import fusion as fusion_mod
    cfg = fusion_det.config
    layers = {l.key: l for l in fusion_det.layers}
    beh = layers.get("behaviour")
    if beh is not None:
        try:
            beh.ensure_loaded()
        except Exception:
            beh = None
    steps = []
    for s in acoustic_steps:
        t = s["call_time_s"]
        results = [fusion_mod.LayerResult("acoustic", "Acoustic", s["probability"], 1.0, False, "")]
        beh_p, beh_state = None, "absent"
        if beh is not None:
            t0 = time.perf_counter()
            r = beh.score(_prefix_wav(stereo, sr, t))
            r.latency_ms = (time.perf_counter() - t0) * 1000
            results.append(r)
            beh_p, beh_state = (None, "abstained") if r.abstained else (round(r.probability, 4), "voted")
        if "semantic" in layers:
            results.append(fusion_mod.LayerResult("semantic", "Semantic", 0.5, 0.0, False,
                                                  "not consulted on a prefix", scored=False))
        out = fusion_mod.combine(results, cfg)
        p = out["synthetic_probability"]
        steps.append({
            "chunk": s["chunk"], "call_time_s": t, "chunk_start_s": s["chunk_start_s"],
            "caller_speech_s": s["caller_speech_s"],
            "acoustic": s["probability"], "behaviour": beh_p, "behaviour_state": beh_state,
            "probability": round(p, 4), "confident": is_confident(p, hi, lo) and out["decisive"],
            "n_layers_used": out["n_layers_used"],
        })
    return steps


# ============================================================================= the scene


def _envelope(samples, buckets=1200):
    """A peak envelope of one channel - what the page draws, not what any model sees."""
    n = len(samples)
    if n == 0:
        return []
    edges = np.linspace(0, n, min(buckets, n) + 1).astype(int)
    peaks = [float(np.abs(samples[a:b]).max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    top = max(peaks) or 1.0
    return [round(v / top, 4) for v in peaks]


def _speech_regions(wave_1d, sr, method):
    import audio as audio_mod
    try:
        return [[round(float(a), 2), round(float(b), 2)] for a, b in audio_mod.detect_speech(wave_1d, sr, method=method)]
    except Exception:
        return []


def scene(fusion_det, source, hi=None, lo=None, run_fusion_prefixes=True):
    """
    Everything the visualiser needs for one call, from the deployed fusion:

      envelope / agent_envelope            both channels, for the waveforms and the speaking rings
      speech_regions / agent_speech_regions  VAD on each channel: when each party is talking
      chunk_spans + steps                  the ACOUSTIC running verdict per chunk
      fusion_steps + fusion_metrics        the FUSION running verdict per chunk (the 50 / 50 story)
      final                                the real end-of-call verdict, verifier gate included
    """
    import audio as audio_mod
    wav_bytes = source if isinstance(source, (bytes, bytearray)) else open(source, "rb").read()
    stereo, sr = audio_mod.read_wav(bytes(wav_bytes))
    caller = audio_mod.get_channel(stereo, config.CALLER_CHANNEL)
    agent = audio_mod.get_channel(stereo, config.AGENT_CHANNEL) if stereo.shape[1] > 1 else np.zeros_like(caller)
    duration = len(stereo) / sr

    import fusion as fusion_mod
    t0 = time.perf_counter()
    results = fusion_det.score_layers(bytes(wav_bytes))
    final = fusion_mod.combine(results, fusion_det.config)
    inference_ms = (time.perf_counter() - t0) * 1000

    ac = next((r for r in results if r.key == "acoustic"), None)
    ac_layer = next((l for l in fusion_det.layers if l.key == "acoustic"), None)
    det = getattr(ac_layer, "detector", None)
    d = (ac.details if ac else {}) or {}
    chunk_scores, chunk_spans = d.get("chunk_scores", []), d.get("chunk_spans", [])
    calibration = det.calibration if det else None
    aggregation = det.aggregation if det else None
    vad = det.vad if det else config.VAD_METHOD

    ac_steps = progression(chunk_scores, chunk_spans, calibration, aggregation, hi, lo)
    ac_metrics = metrics_from_steps(ac_steps, duration, None, hi, lo, inference_ms)
    if run_fusion_prefixes and ac_steps:
        fu_steps = fusion_progression(fusion_det, ac_steps, stereo, sr, hi, lo)
    else:
        fu_steps = []
    fu_metrics = metrics_from_steps(fu_steps, duration, fusion_det.config.threshold, hi, lo, inference_ms)

    return {
        "acoustic_model": det.backbone if det else None,
        "acoustic_display": det.display if det else None,
        "vad": vad,
        "weights": fusion_det.config.weight_map(),
        "roles": fusion_det.config.role_map(),
        "duration_s": round(duration, 2), "sample_rate": int(sr), "channels": int(stereo.shape[1]),
        "envelope": _envelope(caller),
        "agent_envelope": _envelope(agent),
        "speech_regions": _speech_regions(caller, sr, vad),
        "agent_speech_regions": _speech_regions(agent, sr, vad) if stereo.shape[1] > 1 else [],
        "chunk_spans": [[round(float(a), 2), round(float(b), 2)] for a, b in chunk_spans],
        "steps": ac_steps,
        "metrics": ac_metrics,
        "fusion_steps": fu_steps,
        "fusion_metrics": fu_metrics,
        "final": {
            "is_synthetic": bool(final["is_synthetic"]),
            "confidence": float(final["confidence"]),
            "probability": float(final["synthetic_probability"]),
            "decisive": bool(final["decisive"]),
            "note": final.get("note", ""),
            "verifiers_consulted": bool(final.get("verifiers_consulted", False)),
            "layers": [{"key": c["key"], "display": c["display"], "probability": c["probability"],
                        "abstained": c["abstained"], "scored": c.get("scored", True),
                        "consulted": c.get("consulted", True), "role": c.get("role"),
                        "share": c["share"], "reason": c.get("reason", ""),
                        "latency_ms": c.get("latency_ms", 0)} for c in final["layers"]],
        },
    }


# ============================================================================= across calls


def benchmark(timelines, labels=None):
    """
    Detection latency across many calls, from measured timelines only. Medians, not means: one 200 s
    call should not drag the figure everyone quotes. With labels, the synthetic calls are reported on
    their own, because "how fast do we catch an attack" is the question being asked.
    """
    rows = []
    for i, t in enumerate(timelines):
        m = t["metrics"] if "metrics" in t else t
        rows.append({"label": (labels[i] if labels else None), **m})

    def summarise(subset):
        got = [r for r in subset if r.get("confident_detection_at_call_sec") is not None]
        if not got:
            return {"n": len(subset), "n_detected": 0}
        speech = sorted(r["confident_detection_after_caller_speech_sec"] for r in got)
        call_t = sorted(r["confident_detection_at_call_sec"] for r in got)
        lead = sorted(r["seconds_before_call_end"] for r in got if r["seconds_before_call_end"] is not None)
        pct = [r["percentage_of_call_elapsed_at_detection"] for r in got
               if r["percentage_of_call_elapsed_at_detection"] is not None]
        agree = [r["early_agrees_with_final"] for r in got if r["early_agrees_with_final"] is not None]
        med = lambda xs: round(float(np.median(xs)), 2) if xs else None  # noqa: E731
        return {
            "n": len(subset), "n_detected": len(got),
            "n_never_confident": sum(1 for r in subset if r.get("never_confident")),
            "median_caller_speech_sec": med(speech),
            "median_call_time_sec": med(call_t),
            "median_lead_time_sec": med(lead),
            "median_percent_of_call_at_detection": med(pct),
            "detected_before_halfway_pct": (round(100.0 * sum(1 for p in pct if p <= 50) / len(pct), 1)
                                            if pct else None),
            "early_agrees_with_final_pct": (round(100.0 * sum(agree) / len(agree), 1) if agree else None),
        }

    out = {"all": summarise(rows)}
    if labels:
        out["synthetic"] = summarise([r for r in rows if r["label"] == "synthetic"])
        out["human"] = summarise([r for r in rows if r["label"] == "human"])
    return out
