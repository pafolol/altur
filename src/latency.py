"""
HOW EARLY, not just what - the detection-latency timeline of a single call.

Altur judges latency as "how fast the system can determine an outcome with reasonable confidence", so
"synthetic, 91 %" is only half an answer. The other half is *when*: after how much of the caller's voice,
how far into the call, and how long before the call would have ended anyway.

WHAT MAKES THIS REAL RATHER THAN A REPLAY ANIMATION. The acoustic layer cuts caller speech into 4 s
chunks, scores each one independently, and aggregates the chunk log-odds into the call score. The
aggregation is a mean, so the aggregate of the first k chunks IS exactly what the deployed model would
have answered having heard only those k chunks - same aggregate_chunk_scores(), same Platt calibration,
no re-running and no approximation. This module walks k = 1..n and reports that sequence.

WHAT IT IS NOT. The model is not streaming: it scores a finished recording. This is a prefix evaluation -
"what would it have concluded by second t" - which is the standard way to measure detection latency for a
non-streaming model, and it is what the behaviour module's own scripts/evaluate_prefixes.py does too. It
is stated on the page rather than implied, because a judge asking "is this live?" deserves a straight
answer: the numbers are real, the timeline is reconstructed.

    progression(...)  ->  one step per chunk: cumulative probability, call time, caller speech consumed
    timeline(...)     ->  progression + the metrics + the early-vs-final comparison
"""
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


def progression(chunk_scores, chunk_spans, calibration=None, aggregation=None, hi=None, lo=None):
    """
    One step per chunk: what the model would have said having heard only the chunks up to that point.

    `chunk_spans` are (start, end) in seconds of the ORIGINAL call, so call_time_s is where the step sits
    on a timeline of the whole conversation, while caller_speech_s is how much actual caller voice the
    model had consumed by then. Those two are very different numbers on a call with long agent turns, and
    the difference is the point: the model needs seconds of SPEECH, not seconds of call.
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
            "call_time_s": round(end, 2),                       # where this step lands in the call
            "chunk_start_s": round(start, 2),
            "caller_speech_s": round(speech, 2),                # how much caller voice it had heard
            "chunk_score": round(score, 4),
            "chunk_probability": round(calibrated(score, calibration), 4),   # this chunk ALONE
            "probability": round(p, 4),                          # the running verdict
            "confident": is_confident(p, hi, lo),
        })
    return steps


def _first(steps, predicate):
    return next((s for s in steps if predicate(s)), None)


def timeline(chunk_scores, chunk_spans, duration_s, calibration=None, aggregation=None,
             threshold=None, hi=None, lo=None, inference_ms=None):
    """
    The whole latency story for one call: the progression, the metrics, and early verdict vs final.

    Every field is None when the call does not support it - no chunks at all, or a call that never
    becomes confident - rather than a zero that would read as "detected instantly".
    """
    threshold = config.DECISION_THRESHOLD if threshold is None else threshold
    hi = config.CONFIDENT_SYNTHETIC_THRESHOLD if hi is None else hi
    lo = config.CONFIDENT_HUMAN_THRESHOLD if lo is None else lo
    steps = progression(chunk_scores, chunk_spans, calibration, aggregation, hi, lo)
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
        # the first moment the model had anything to say at all
        "first_prediction_at_call_sec": first["call_time_s"] if first else None,
        "first_prediction_after_caller_speech_sec": first["caller_speech_s"] if first else None,
        "first_prediction_probability": first["probability"] if first else None,
        # the first moment it was far enough from the fence to act on
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
    return {"steps": steps, "metrics": m}


def benchmark(timelines, labels=None):
    """
    Detection latency across many calls, from measured timelines only.

    `labels` (per call, "synthetic"/"human") is optional; when given, the summary is reported for the
    synthetic calls separately, because "how fast do we catch an attack" is the question being asked and
    averaging it with the human calls answers a different one. Medians, not means: one 200 s call should
    not drag the figure everyone quotes.
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
