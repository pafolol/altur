"""
The seam between this backend and the real detector (`src/fusion.py` at the repository root).

This backend was written as a serving shell around a detector that did not exist yet: `DETECTOR_MODE=mock`
answers 0.5 to everything, and `app/fusion.py` implements a weighted mean over modality scores nobody was
producing. The detector exists now, so `DETECTOR_MODE=fusion` points this shell at it.

Everything this backend already does is kept and still runs first - the request size limit, the duration
bounds, the minimum caller speech, the channel checks, the feature warm-up, the typed error contract.
Only the inference step changes: instead of `Detector.predict_synthetic_probability(features)` on this
backend's own features, the audio goes to the two-stage fusion (acoustic + behaviour deciding 50 / 50,
semantic consulted only when they do not settle it).

Why the WAV is re-encoded rather than the arrays passed through: the behaviour layer requires stereo
16-bit PCM at 8 kHz and validates it strictly, and this backend has already resampled and split the audio
by the time we get here. Re-encoding the channels it produced means the fusion sees exactly the audio this
backend validated, not the bytes that arrived.

Import failures are not fatal. If the repository's src/ is not importable, or the models are missing, the
bridge reports `available = False`, the backend logs it once and keeps its previous behaviour. A serving
shell that will not start because an optional detector is absent is worse than one that says so.
"""
import io
import logging
import sys
import wave
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
TARGET_SAMPLE_RATE = 8000


class FusionUnavailable(RuntimeError):
    pass


def encode_stereo_wav(caller, agent, sample_rate=TARGET_SAMPLE_RATE):
    """The two channels this backend split, back to the stereo 16-bit PCM WAV every layer expects."""
    n = min(len(caller), len(agent))
    stereo = np.stack([np.asarray(caller[:n], dtype=np.float32),
                       np.asarray(agent[:n], dtype=np.float32)], axis=1)
    pcm = np.clip(stereo, -1.0, 1.0) * 32767.0
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.astype("<i2").tobytes())
    return buf.getvalue()


class FusionBridge:
    """Loads the repository's fusion once and scores calls through it."""

    def __init__(self, enabled=False, acoustic_model="wav2vec2_spanish", semantic_url=None):
        self.enabled = bool(enabled)
        self.detector = None
        self.error = None
        if not self.enabled:
            return
        try:
            if str(SRC) not in sys.path:
                sys.path.insert(0, str(SRC))
            import fusion  # noqa: PLC0415 - imported lazily so the backend starts without it
            self._fusion = fusion
            self.detector = fusion.FusionDetector(
                layers=fusion.build_layers(acoustic_model, include_unavailable=True,
                                           semantic_url=semantic_url))
            if not any(l.available() for l in self.detector.layers):
                raise FusionUnavailable("no detection layer is available; check models/ and behaviour/artifacts/")
            logger.info("Fusion bridge ready: %s", ", ".join(
                f"{l.key} w={self.detector.config.weight_of(l):.2f} ({self.detector.config.role_of(l)})"
                + ("" if l.available() else " [not answering]") for l in self.detector.layers))
        except Exception as exc:
            self.detector = None
            self.error = f"{type(exc).__name__}: {exc}"
            logger.error("Fusion bridge unavailable: %s", self.error)

    @property
    def available(self):
        return self.detector is not None

    def warm_up(self):
        """Load every layer before traffic arrives, so the first real request is not the slow one."""
        if self.available:
            self.detector.preload()

    def describe(self):
        if not self.available:
            return {"available": False, "error": self.error}
        cfg = self.detector.config
        return {"available": True, "layers": self.detector.describe(),
                "weights": cfg.weight_map(), "roles": cfg.role_map(),
                "verify_threshold": cfg.verify_threshold}

    def predict(self, caller, agent, sample_rate=TARGET_SAMPLE_RATE):
        """
        (caller channel, agent channel) -> the fusion's own verdict.

        `is_synthetic` comes from the fusion rather than being recomputed from the probability: when no
        layer could vote it answers an explicit non-flag at 0.5, and `0.5 >= 0.5` would otherwise accuse
        a caller on no evidence at all.
        """
        if not self.available:
            raise FusionUnavailable(self.error or "fusion bridge not loaded")
        wav = encode_stereo_wav(caller, agent, sample_rate)
        result = self.detector.score(wav)
        layers = result.get("layers", [])
        return {
            "probability": float(result["synthetic_probability"]),
            "is_synthetic": bool(result["is_synthetic"]),
            "confidence": float(result["confidence"]),
            "decisive": bool(result.get("decisive", True)),
            "note": result.get("note", ""),
            "verifiers_consulted": bool(result.get("verifiers_consulted", False)),
            "primary_probability": result.get("primary_probability"),
            "layers": [{"key": c["key"], "probability": c["probability"], "role": c.get("role"),
                        "abstained": c.get("abstained"), "consulted": c.get("consulted"),
                        "share": c.get("share"), "reason": c.get("reason", "")} for c in layers],
        }
