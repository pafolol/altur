"""Stable WAV-bytes interface. No dataset paths, labels, JSON turns or network required."""

import json
import math
import os
from pathlib import Path
from time import perf_counter
from .audio import load_wav
from .config import ROOT, VADConfig, feature_config
from .features import extract_features, evidence_quality
from .decision import evidence_state
from .portable import linear_score
from .vad import SileroVAD, MODEL_PATH


def score_features(artifact, features, stability=1., debug=False):
    quality, count = evidence_quality(features, stability)
    insufficient = count < 2 or min(features["caller_speech_s"], features["agent_speech_s"]) < 1.
    if insufficient:
        probability, quality = .5, 0.
    else:
        probability = float(linear_score(artifact, [[features[n] for n in artifact["feature_names"]]])[0])
    result = {"synthetic_probability": probability, "quality_score": quality,
              "event_count": count, "behavior_confidence": quality * abs(2 * probability - 1)}
    if debug:
        result["features"] = {k: float(v) if math.isfinite(v) else None for k, v in features.items()}
        result["diagnostics"] = {
            "insufficient_evidence": insufficient,
            "evidence_state": evidence_state(result),
            "vad_stability": stability,
            "interruption_count": features["interruption_count"],
            "barge_in_count": features["barge_in_count"],
            "response_count": features["response_latency_count"],
            "interruption_consistency_available": features["interruption_stop_latency_count"] >= 3,
            "calibration": artifact["calibration"]["method"],
            "confidence_definition": "quality * abs(2*p-1); heuristic, not probability of correctness",
        }
    return result


class BehaviorDetector:
    """Load once and reuse. ONNX recurrent state is isolated per prediction.

    predict(wav_bytes) -> four JSON-compatible scalars. Optional debug output
    contains temporal diagnostics but never waveform, transcript, filename or label.
    """

    def __init__(self, model_path=None, vad_model_path=None, max_duration_s=600.):
        start = perf_counter()
        path = Path(model_path or os.getenv("BEHAVIOR_MODEL_PATH", ROOT / "artifacts/models/behavior.json"))
        self.artifact = json.loads(path.read_text())
        if self.artifact.get("schema_version") != 1 or self.artifact.get("feature_config") != feature_config():
            raise ValueError("Model/feature configuration mismatch; retrain using this code version")
        self.threshold = self.artifact["threshold"]
        self.max_duration_s = max_duration_s
        self.vad = SileroVAD(vad_model_path or os.getenv("BEHAVIOR_VAD_PATH", str(MODEL_PATH)),
                             VADConfig(**self.artifact["metadata"]["vad_config"]))
        self.model_load_s = perf_counter() - start

    def predict(self, wav_bytes: bytes, debug: bool = False):
        start = perf_counter()
        audio = load_wav(wav_bytes, self.max_duration_s)
        loaded = perf_counter()
        turns, diagnostics = self.vad.extract(audio)
        segmented = perf_counter()
        features, _ = extract_features(turns, audio.duration)
        featured = perf_counter()
        result = score_features(self.artifact, features, diagnostics["stability"], debug)
        scored = perf_counter()
        if debug:
            result["diagnostics"].update(diagnostics)
            result["diagnostics"]["timing_ms"] = {
                "decode": (loaded - start) * 1000, "vad": (segmented - loaded) * 1000,
                "features": (featured - segmented) * 1000, "classifier": (scored - featured) * 1000,
                "total": (scored - start) * 1000,
            }
        return result
