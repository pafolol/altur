"""Explicit feature allowlists: no automatic inclusion of metadata columns."""

import math
import numpy as np

from .config import EventConfig
from .events import detect_events
from .turns import normalize_turns, speaking_states

GLOBAL = ["duration_s", "caller_speech_s", "agent_speech_s", "silence_s", "overlap_s",
          "caller_proportion", "leading_silence_s", "trailing_silence_s", "caller_turn_count",
          "agent_turn_count"]
TEMPORAL = ["caller_turn_mean", "caller_turn_median", "caller_turn_std", "caller_turn_iqr",
            "agent_turn_mean", "agent_turn_median", "caller_pause_median",
            "response_latency_count", "response_latency_mean", "response_latency_median"]
REACTION = ["interruption_count", "interruption_rate", "interruption_stop_latency_mean",
            "interruption_stop_latency_median", "interruption_yield_fraction",
            "interruption_immediate_yield_fraction", "recovery_latency_count",
            "recovery_latency_mean", "recovery_gap_mean", "immediate_retake_fraction",
            "barge_in_count", "barge_in_rate", "barge_overlap_mean", "barge_continues_fraction",
            "extended_gap_count", "extended_gap_caller_fraction"]
CONSISTENCY = [f"{prefix}_{stat}" for prefix in
               ("response_latency", "interruption_stop_latency", "recovery_latency", "barge_overlap")
               for stat in ("std", "iqr", "cv")] + [
                   "reaction_entropy", "short_context_response_std", "long_context_response_std"]
FEATURE_SETS = {
    "dummy": ["duration_s"],
    "shortcut": GLOBAL,
    "temporal": TEMPORAL,
    "reaction": TEMPORAL + REACTION,
    "consistency": TEMPORAL + REACTION + CONSISTENCY,
    "reaction_only": REACTION + CONSISTENCY,
    "no_counts": [f for f in TEMPORAL + REACTION + CONSISTENCY if not f.endswith("_count")],
    "no_agent": [f for f in TEMPORAL + REACTION + CONSISTENCY if not f.startswith("agent_turn_")],
    "with_global": GLOBAL + [f for f in TEMPORAL + REACTION + CONSISTENCY if f not in GLOBAL],
}


def summarize(values, config=EventConfig()):
    x = np.asarray([v for v in values if v is not None and math.isfinite(v)], dtype=float)
    n = len(x)
    out = {"count": n, "mean": np.nan, "median": np.nan, "std": np.nan, "iqr": np.nan, "cv": np.nan}
    if n:
        out.update(mean=float(x.mean()), median=float(np.median(x)))
    # A single measurement has NO empirical consistency evidence.
    if n >= config.min_variability_events:
        out.update(std=float(x.std(ddof=1)), iqr=float(np.quantile(x, .75) - np.quantile(x, .25)))
        if x.mean() >= config.cv_min_mean_s and np.all(x >= 0):
            out["cv"] = out["std"] / out["mean"]
    return out


def fraction(values):
    known = [v for v in values if v is not None]
    return float(np.mean(known)) if known else np.nan


def extract_features(turns, duration, config=EventConfig()):
    turns = normalize_turns(turns, duration, config.merge_gap_s)
    caller = [t for t in turns if t.channel == 0]
    agent = [t for t in turns if t.channel == 1]
    states = speaking_states(turns, duration)
    ev = detect_events(turns, duration, config)
    f = {"duration_s": duration,
         "caller_speech_s": sum(t.duration for t in caller),
         "agent_speech_s": sum(t.duration for t in agent),
         "silence_s": sum(b - a for a, b, s in states if s == 0),
         "overlap_s": sum(b - a for a, b, s in states if s == 3),
         "caller_turn_count": len(caller), "agent_turn_count": len(agent),
         "leading_silence_s": min((t.start for t in turns), default=duration),
         "trailing_silence_s": duration - max((t.end for t in turns), default=0)}
    f["caller_proportion"] = f["caller_speech_s"] / duration
    distributions = {
        "caller_turn": [t.duration for t in caller if t.end < duration - 1e-9],
        "agent_turn": [t.duration for t in agent if t.end < duration - 1e-9],
        "caller_pause": [b.start - a.end for a, b in zip(caller, caller[1:])],
        "response_latency": [e["latency"] for e in ev.responses],
        "interruption_stop_latency": [e["stop_latency"] for e in ev.interruptions],
        "recovery_latency": [e["recovery"] for e in ev.interruptions],
        "recovery_gap": [e["recovery_gap"] for e in ev.interruptions],
        "barge_overlap": [e["overlap"] for e in ev.barge_ins],
        "short_context_response": [e["latency"] for e in ev.responses if e["agent_duration"] < config.short_agent_turn_s],
        "long_context_response": [e["latency"] for e in ev.responses if e["agent_duration"] >= config.short_agent_turn_s],
    }
    for prefix, values in distributions.items():
        f.update({f"{prefix}_{stat}": v for stat, v in summarize(values, config).items()
                  if not (prefix in ("caller_turn", "agent_turn") and stat == "count")})
    f.update(
        interruption_count=len(ev.interruptions),
        interruption_rate=len(ev.interruptions) / len(agent) if agent else np.nan,
        interruption_yield_fraction=fraction([e["yielded"] for e in ev.interruptions]),
        interruption_immediate_yield_fraction=fraction([e["immediate_yield"] for e in ev.interruptions]),
        immediate_retake_fraction=fraction([e["retake"] for e in ev.interruptions]),
        barge_in_count=len(ev.barge_ins),
        barge_in_rate=len(ev.barge_ins) / len(caller) if caller else np.nan,
        barge_continues_fraction=fraction([e["continues"] for e in ev.barge_ins]),
        extended_gap_count=len(ev.silences),
        extended_gap_caller_fraction=fraction([e["caller_next"] for e in ev.silences]),
    )
    reactions = [0 if e["immediate_yield"] else 1 if e["yielded"] else 2
                 for e in ev.interruptions if e["yielded"] is not None]
    f["reaction_entropy"] = np.nan
    if len(reactions) >= config.min_variability_events:
        p = np.bincount(reactions, minlength=3) / len(reactions)
        f["reaction_entropy"] = float(-np.sum(p[p > 0] * np.log2(p[p > 0])))
    return f, ev


def evidence_quality(features, vad_stability=1.0):
    """A transparent heuristic for evidence sufficiency, NOT probability of correctness."""
    response = features["response_latency_count"]
    interruptions = features["interruption_count"]
    barge = features["barge_in_count"]
    count = int(response + interruptions + barge)
    both_speech = min(1., features["caller_speech_s"] / 10., features["agent_speech_s"] / 10.)
    temporal = min(1., response / 8.)
    perturbation = min(1., (interruptions + barge) / 5.)
    repeated = min(1., max(0, interruptions - 2) / 4.)
    quality = both_speech * min(1., features["duration_s"] / 30.) * (
        .45 * temporal + .35 * perturbation + .20 * repeated) * np.clip(vad_stability, 0, 1)
    return float(np.clip(quality, 0, 1)), count
