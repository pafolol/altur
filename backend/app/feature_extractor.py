import math
import time
from typing import Dict

import librosa
import numpy as np

from .feature_validation import validate_finite_features

# This is the canonical training/production order. The model trainer must reuse it.
FEATURE_NAMES = ["duration", "caller_speech_seconds", "agent_speech_seconds", "caller_speech_ratio", "agent_speech_ratio", "caller_turns", "agent_turns", "caller_turn_duration_mean", "caller_turn_duration_std", "caller_turn_duration_min", "caller_turn_duration_max", "caller_turn_duration_median", "agent_turn_duration_mean", "agent_turn_duration_std", "agent_turn_duration_min", "agent_turn_duration_max", "agent_turn_duration_median", "caller_pause_mean", "caller_pause_std", "caller_pause_min", "caller_pause_max", "caller_pause_median", "pause_cv", "response_count", "response_latency_mean", "response_latency_std", "response_latency_min", "response_latency_max", "response_latency_median", "response_latency_cv", "quick_response_ratio", "interruptions", "interruption_rate", "overlap_seconds", "overlap_ratio", "rms_mean", "rms_std", "rms_min", "rms_max", "zcr_mean", "zcr_std", "spectral_centroid_mean", "spectral_centroid_std", "spectral_bandwidth_mean", "spectral_bandwidth_std", "spectral_rolloff_mean", "spectral_rolloff_std", "spectral_flatness_mean", "spectral_flatness_std"]
FEATURE_SCHEMA_VERSION = "1.0"
for prefix in ("mfcc", "mfcc_delta", "mfcc_delta2"):
    FEATURE_NAMES.extend(f"{prefix}_{i}_{stat}" for i in range(13) for stat in ("mean", "std"))
FEATURE_NAMES.extend(["pitch_mean", "pitch_std", "voiced_ratio"])


def _stats(values, prefix, output):
    values = np.asarray(values, dtype=np.float64)
    output[f"{prefix}_mean"] = float(np.mean(values)) if values.size else 0.0
    output[f"{prefix}_std"] = float(np.std(values)) if values.size else 0.0


def _segments(mask, sr):
    padded = np.pad(mask.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends) if e > s], [(e - s) / sr for s, e in zip(starts, ends) if e > s]


def _fast_vad(signal, sample_rate, frame_length=256, hop_length=128):
    """Energy VAD for 8 kHz telephony; returns one mask and intervals."""
    if signal.size < frame_length:
        frames = signal.reshape(1, -1)
    else:
        frames = np.lib.stride_tricks.sliding_window_view(signal, frame_length)[::hop_length]
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float32), axis=1))
    peak = float(np.max(rms)) if rms.size else 0.0
    threshold = max(min(float(np.percentile(rms, 25)) * 1.8, peak * 0.5), 1e-4) if rms.size else 1e-4
    mask = rms > threshold
    padded = np.pad(mask.astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    intervals = [(int(start), int(end)) for start, end in zip(starts, ends) if end > start]
    return rms, mask, intervals


def extract_features(caller, agent, sample_rate, timing=None, enable_pitch=True) -> Dict[str, float]:
    extraction_started = time.perf_counter()
    timing = timing if timing is not None else {}
    caller = np.asarray(caller, dtype=np.float32).ravel(); agent = np.asarray(agent, dtype=np.float32).ravel()
    n = min(caller.size, agent.size)
    if n == 0 or sample_rate <= 0: raise ValueError("Audio vacío o sample rate inválido")
    caller, agent = caller[:n], agent[:n]
    duration = n / float(sample_rate)
    vad_frame_length, vad_hop = 256, 128
    vad_started = time.perf_counter()
    caller_rms, cm, caller_intervals = _fast_vad(caller, sample_rate, vad_frame_length, vad_hop)
    agent_rms, am, agent_intervals = _fast_vad(agent, sample_rate, vad_frame_length, vad_hop)
    timing["vad_ms"] = (time.perf_counter() - vad_started) * 1000
    behavior_started = time.perf_counter()
    cs, cturn = caller_intervals, [(e - s) * vad_hop / sample_rate for s, e in caller_intervals]
    as_, aturn = agent_intervals, [(e - s) * vad_hop / sample_rate for s, e in agent_intervals]
    def durations_stats(values, prefix):
        out = {}; vals = np.asarray(values, dtype=float)
        for suffix, fn in (("mean", np.mean), ("std", np.std), ("min", np.min), ("max", np.max), ("median", np.median)):
            out[f"{prefix}_{suffix}"] = float(fn(vals)) if vals.size else 0.0
        return out
    out = {"duration": float(duration), "caller_speech_seconds": float(cm.sum() * vad_hop / sample_rate), "agent_speech_seconds": float(am.sum() * vad_hop / sample_rate)}
    out.update({"caller_speech_ratio": out["caller_speech_seconds"] / duration, "agent_speech_ratio": out["agent_speech_seconds"] / duration, "caller_turns": len(cturn), "agent_turns": len(aturn)})
    out.update(durations_stats(cturn, "caller_turn_duration")); out.update(durations_stats(aturn, "agent_turn_duration"))
    pauses = [max(0.0, cs[i + 1][0] - cs[i][1]) * vad_hop / sample_rate for i in range(len(cs) - 1)]
    out.update(durations_stats(pauses, "caller_pause")); out["pause_cv"] = out["caller_pause_std"] / out["caller_pause_mean"] if out["caller_pause_mean"] > 0 else 0.0
    latencies = [max(0.0, (cs[i][0] - as_[j][1]) * vad_hop / sample_rate) for i in range(len(cs)) for j in range(len(as_)) if as_[j][1] <= cs[i][0] and not any(as_[k][1] > as_[j][1] and as_[k][1] <= cs[i][0] for k in range(len(as_)))]
    out["response_count"] = len(latencies); out.update(durations_stats(latencies, "response_latency")); out["response_latency_cv"] = out["response_latency_std"] / out["response_latency_mean"] if out["response_latency_mean"] > 0 else 0.0; out["quick_response_ratio"] = float(sum(x <= 0.5 for x in latencies) / len(latencies)) if latencies else 0.0
    overlap = float(np.sum(cm & am) * vad_hop / sample_rate); out.update({"interruptions": sum(1 for s, _ in cs if any(a <= s < b for a, b in as_)), "overlap_seconds": overlap, "overlap_ratio": overlap / duration})
    out["interruption_rate"] = out["interruptions"] / max(1, len(cturn))
    timing["behavior_features_ms"] = (time.perf_counter() - behavior_started) * 1000
    acoustic_started = time.perf_counter()
    n_fft, hop = 512, 256
    spectral_started = time.perf_counter()
    S = np.abs(librosa.stft(caller, n_fft=n_fft, hop_length=hop, center=False))
    rms = librosa.feature.rms(S=S, frame_length=n_fft)[0]
    _stats(rms, "rms", out)
    out["rms_min"] = float(np.min(rms)) if rms.size else 0.0
    out["rms_max"] = float(np.max(rms)) if rms.size else 0.0
    for name, values in (("zcr", librosa.feature.zero_crossing_rate(caller, frame_length=n_fft, hop_length=hop, center=False)[0]), ("spectral_centroid", librosa.feature.spectral_centroid(S=S, sr=sample_rate)[0]), ("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=S, sr=sample_rate)[0]), ("spectral_rolloff", librosa.feature.spectral_rolloff(S=S, sr=sample_rate)[0]), ("spectral_flatness", librosa.feature.spectral_flatness(S=S)[0])): _stats(values, name, out)
    timing["spectral_features_ms"] = (time.perf_counter() - spectral_started) * 1000
    mfcc_started = time.perf_counter()
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(S ** 2), sr=sample_rate, n_mfcc=13)
    timing["mfcc_ms"] = (time.perf_counter() - mfcc_started) * 1000
    def delta(values, order=1):
        # Very short valid recordings may not have enough frames for librosa's default window.
        if values.shape[1] < 3:
            return np.zeros_like(values)
        width = min(9, values.shape[1] if values.shape[1] % 2 else values.shape[1] - 1)
        return librosa.feature.delta(values, order=order, width=max(3, width))
    delta_started = time.perf_counter()
    mfcc_delta, mfcc_delta2 = delta(mfcc), delta(mfcc, 2)
    timing["mfcc_delta_ms"] = (time.perf_counter() - delta_started) * 1000
    for prefix, values in (("mfcc", mfcc), ("mfcc_delta", mfcc_delta), ("mfcc_delta2", mfcc_delta2)):
        for i, row in enumerate(values): _stats(row, f"{prefix}_{i}", out)
    pitch_started = time.perf_counter()
    if enable_pitch and n >= n_fft * 2 and caller_intervals:
        speech_only = np.concatenate([caller[start * vad_hop:min(n, end * vad_hop)] for start, end in caller_intervals])
        pitches = librosa.yin(speech_only, fmin=50, fmax=400, sr=sample_rate, frame_length=n_fft, hop_length=hop)
        voiced = np.isfinite(pitches) & (pitches >= 50) & (pitches <= 400)
    else:
        pitches, voiced = np.array([]), np.array([])
    timing["pitch_ms"] = (time.perf_counter() - pitch_started) * 1000
    valid = pitches[voiced] if pitches.size else np.array([]); out.update({"pitch_mean": float(np.mean(valid)) if valid.size else 0.0, "pitch_std": float(np.std(valid)) if valid.size else 0.0, "voiced_ratio": float(np.mean(voiced)) if voiced.size else 0.0})
    result = {name: float(value) if isinstance(value, (float, np.floating)) else int(value) for name, value in out.items()}
    result = validate_finite_features(result)
    timing["acoustic_features_ms"] = (time.perf_counter() - acoustic_started) * 1000
    timing["total_feature_extraction_ms"] = (time.perf_counter() - extraction_started) * 1000
    return result


def features_to_vector(features):
    missing = [name for name in FEATURE_NAMES if name not in features]
    if missing: raise ValueError(f"Faltan features requeridas: {', '.join(missing)}")
    validate_finite_features(features)
    return np.asarray([features[name] for name in FEATURE_NAMES], dtype=np.float64).reshape(1, -1)


def extract_behavior_features(caller, agent, sample_rate):
    from .behavioral_features import BEHAVIOR_FEATURE_NAMES
    values = extract_features(caller, agent, sample_rate)
    return {name: values[name] for name in BEHAVIOR_FEATURE_NAMES}


def extract_acoustic_features(caller, sample_rate):
    from .acoustic_features import ACOUSTIC_FEATURE_NAMES
    values = extract_features(caller, np.zeros_like(caller), sample_rate)
    return {name: values[name] for name in ACOUSTIC_FEATURE_NAMES}
