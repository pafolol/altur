"""
DETECTOR_MODE=fusion: this backend's /detect answering with the repository's real detector.

Added alongside the backend's own suite, which is unchanged and still covers the default `mock` mode.
The point of these is that wiring the fusion in did not cost any of the guarantees the shell was written
for - the size limit, the duration bounds, the channel checks, the typed errors - and that the fusion's
own verdict survives the trip rather than being re-derived from the probability.

Most of them use a stub bridge, so they need no models and no network.
"""
import base64
import io
import sys
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

BACKEND = Path(__file__).resolve().parents[1]
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app import main as main_module            # noqa: E402
from app.config import Settings                # noqa: E402
from app.detector import Detector              # noqa: E402
from app.fusion_bridge import FusionBridge, encode_stereo_wav   # noqa: E402
from app.main import create_app                # noqa: E402


def wav_base64(channels=2, seconds=1.0, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(2); w.setframerate(rate)
        t = np.arange(int(rate * seconds)) / rate
        tone = (0.2 * np.sin(2 * np.pi * 200 * t) * 32767).astype("<i2")
        w.writeframes(np.stack([tone] * channels, axis=1).tobytes())
    return base64.b64encode(buf.getvalue()).decode()


class StubBridge:
    """A bridge with a fixed answer, so the wiring can be tested without the models."""

    def __init__(self, **answer):
        self.available = True
        self.error = None
        self.calls = 0
        self.answer = {"probability": 0.9, "is_synthetic": True, "confidence": 0.9, "decisive": True,
                       "note": "", "verifiers_consulted": False, "primary_probability": 0.9,
                       "layers": [{"key": "acoustic", "probability": 0.9, "role": "primary",
                                   "abstained": False, "consulted": True, "share": 1.0, "reason": ""}]} | answer

    def warm_up(self):
        pass

    def describe(self):
        return {"available": True, "layers": [], "weights": {}, "roles": {}, "verify_threshold": 0.8}

    def predict(self, caller, agent, sample_rate=8000):
        self.calls += 1
        self.last = (caller, agent, sample_rate)
        return self.answer


def client(bridge, monkeypatch, **kw):
    monkeypatch.setattr(main_module, "FusionBridge", lambda *a, **k: bridge)
    return TestClient(create_app(Settings(detector_mode="fusion", enable_feature_warmup=False, **kw)))


# --------------------------------------------------------------------------- the wiring


def test_detect_answers_with_the_fusions_verdict(monkeypatch):
    bridge = StubBridge(probability=0.83, is_synthetic=True, confidence=0.83)
    r = client(bridge, monkeypatch).post("/detect", json={"audio": wav_base64()})
    assert r.status_code == 200
    assert r.json() == {"is_synthetic": True, "confidence": pytest.approx(0.83)}
    assert bridge.calls == 1


def test_the_fusions_own_is_synthetic_is_used_not_a_rederivation(monkeypatch):
    """
    The non-flag case. No layer could vote, so the fusion answers 0.5 and says "not synthetic".
    Recomputing `probability >= threshold` here would give 0.5 >= 0.5 -> True and accuse a caller on no
    evidence at all, which is exactly the bug this asserts against.
    """
    bridge = StubBridge(probability=0.5, is_synthetic=False, confidence=0.5, decisive=False)
    body = client(bridge, monkeypatch, model_threshold=0.5).post("/detect", json={"audio": wav_base64()}).json()
    assert body["is_synthetic"] is False and body["confidence"] == pytest.approx(0.5)


def test_the_mock_path_is_untouched_when_the_mode_is_not_fusion():
    """
    DETECTOR_MODE=mock still behaves exactly as it did; nothing was taken away.

    And what it does is worth writing down, because it is not what "returns 0.5" sounds like: the mock
    probability is 0.5, `is_synthetic` is `probability >= threshold`, and 0.5 >= 0.5 is True - so mock
    mode flags EVERY caller as synthetic at 50 % confidence. This is pre-existing and left alone (it is
    this backend's own mode and its README says the mock is not a real detection), but it is the reason
    DETECTOR_MODE=fusion matters before anyone points judges at this service.
    """
    c = TestClient(create_app(Settings(detector_mode="mock", enable_feature_warmup=False)))
    body = c.post("/detect", json={"audio": wav_base64()}).json()
    assert body == {"is_synthetic": True, "confidence": 0.5}


def test_a_broken_bridge_does_not_stop_the_backend_from_starting(monkeypatch):
    """An optional detector that will not load must not take the serving shell down with it."""
    class Dead:
        available = False
        error = "boom"
        def warm_up(self): pass
        def describe(self): return {"available": False, "error": self.error}
    c = client(Dead(), monkeypatch)
    assert c.get("/health").json() == {"status": "ok"}
    assert c.get("/ready").status_code == 503          # honest: not ready, rather than silently wrong
    assert c.get("/fusion").json()["available"] is False


# --------------------------------------------------------------------------- the guards still run


def test_every_input_guard_still_applies_in_fusion_mode(monkeypatch):
    bridge = StubBridge()
    c = client(bridge, monkeypatch, min_audio_duration_seconds=1.0, max_audio_duration_seconds=2.0)
    assert c.post("/detect", json={}).status_code == 422                                  # no audio
    assert c.post("/detect", json={"audio": "not base64!"}).status_code == 400
    assert c.post("/detect", json={"audio": base64.b64encode(b"bad wav").decode()}).status_code == 400
    assert c.post("/detect", json={"audio": wav_base64(channels=1)}).status_code == 400   # mono
    assert c.post("/detect", json={"audio": wav_base64(seconds=0.5)}).status_code == 400  # too short
    assert c.post("/detect", json={"audio": wav_base64(seconds=3.0)}).status_code == 400  # too long
    assert bridge.calls == 0          # not one of those reached the detector


def test_the_request_id_and_debug_breakdown_survive(monkeypatch):
    bridge = StubBridge(note="the primaries settled it", verifiers_consulted=False)
    c = client(bridge, monkeypatch, enable_debug_endpoint=True)
    r = c.post("/detect", json={"audio": wav_base64()})
    assert r.headers.get("X-Request-ID")
    d = c.post("/debug/analyze", json={"audio": wav_base64()}).json()
    assert "timing" in d and "parameters" in d           # this backend's own diagnostics, still there
    assert d["fusion"]["note"] == "the primaries settled it"
    assert d["fusion"]["layers"][0]["key"] == "acoustic"


def test_the_audio_handed_to_the_fusion_is_the_audio_this_backend_validated(monkeypatch):
    """It must see the resampled, split channels - not the bytes that happened to arrive."""
    bridge = StubBridge()
    client(bridge, monkeypatch).post("/detect", json={"audio": wav_base64(seconds=1.0, rate=16000)})
    caller, agent, sr = bridge.last
    assert sr == 8000 and len(caller) == len(agent) == 8000     # resampled by the backend, then passed on


# --------------------------------------------------------------------------- the encoder


def test_encode_stereo_wav_round_trips_to_what_every_layer_expects():
    caller = np.linspace(-0.5, 0.5, 800, dtype=np.float32)
    agent = np.zeros(800, dtype=np.float32)
    raw = encode_stereo_wav(caller, agent, 8000)
    with wave.open(io.BytesIO(raw)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (2, 2, 8000)
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)
    assert pcm.shape == (800, 2)
    assert np.allclose(pcm[:, 0] / 32767.0, caller, atol=1e-3)
    assert not pcm[:, 1].any()


def test_encode_stereo_wav_clips_instead_of_wrapping():
    """A sample above 1.0 must saturate, not wrap around to a loud negative."""
    raw = encode_stereo_wav(np.array([2.0, -2.0], dtype=np.float32), np.zeros(2, dtype=np.float32))
    with wave.open(io.BytesIO(raw)) as w:
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)
    assert pcm[0, 0] == 32767 and pcm[1, 0] == -32767


def test_encode_stereo_wav_truncates_to_the_shorter_channel():
    raw = encode_stereo_wav(np.zeros(100, dtype=np.float32), np.zeros(60, dtype=np.float32))
    with wave.open(io.BytesIO(raw)) as w:
        assert w.getnframes() == 60


# --------------------------------------------------------------------------- the detector mode itself


def test_fusion_mode_is_a_valid_detector_mode_and_refuses_feature_inference():
    d = Detector("fusion")
    assert d.available is True and d.mode == "fusion"
    with pytest.raises(Exception):           # the verdict comes from the whole call, not a feature vector
        d.predict_synthetic_probability({})


def test_a_disabled_bridge_reports_itself_unavailable_without_importing_anything():
    b = FusionBridge(enabled=False)
    assert b.available is False and b.describe()["available"] is False
