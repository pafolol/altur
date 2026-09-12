import base64
from concurrent.futures import ThreadPoolExecutor
import io
import json
import wave
import numpy as np
import pytest
from fastapi.testclient import TestClient
from behavior.audio import load_wav, encode_wav, AudioError
from behavior.config import ROOT
from behavior.inference import BehaviorDetector
from behavior.service import create_app


def test_strict_wav_input_and_stereo_identity():
    a = np.zeros((8000, 2), dtype=np.float32)
    a[100, 0] = .5
    decoded = load_wav(encode_wav(a))
    assert decoded.samples[100, 0] == .5 and decoded.samples[:, 1].sum() == 0
    with pytest.raises(AudioError):
        load_wav(b"not a wav")
    with pytest.raises(AudioError):
        load_wav(encode_wav(a)[:-10])
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as f:
        f.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        f.writeframes(b"\0" * 16000)
    with pytest.raises(AudioError):
        load_wav(buffer.getvalue())


@pytest.fixture(scope="module")
def detector():
    if not (ROOT / "artifacts/models/behavior.json").exists():
        pytest.skip("Train/download local model artifacts for integration tests")
    return BehaviorDetector()


@pytest.mark.integration
def test_silent_wav_returns_neutral_evidence(detector):
    result = detector.predict(encode_wav(np.zeros((16000, 2), dtype=np.float32)), debug=True)
    assert result["synthetic_probability"] == .5
    assert result["quality_score"] == result["behavior_confidence"] == 0
    assert result["event_count"] == 0
    assert result["diagnostics"]["insufficient_evidence"]
    json.dumps(result, allow_nan=False)


@pytest.mark.integration
def test_real_wav_end_to_end_and_no_organizer_dependency(detector, monkeypatch):
    from behavior.dataset import read_manifest, locate_audio
    from behavior import turns
    row = read_manifest().iloc[0]
    payload = (locate_audio() / f"{row.anon_id}.wav").read_bytes()
    monkeypatch.setattr(turns, "load_turns", lambda *a: pytest.fail("Organizer turns accessed during inference"))
    result = detector.predict(payload)
    assert set(result) == {"synthetic_probability", "quality_score", "event_count", "behavior_confidence"}
    assert 0 <= result["synthetic_probability"] <= 1
    assert result["event_count"] > 0
    # State isolation: two concurrent calls give the same score as isolated inference.
    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(detector.predict, [payload, payload]))
    assert all(r["synthetic_probability"] == pytest.approx(result["synthetic_probability"], abs=1e-10) for r in outputs)


@pytest.mark.integration
def test_http_contract_validation_and_no_body_echo(detector):
    app = create_app(lambda: detector)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ready"
        wav = encode_wav(np.zeros((8000, 2), dtype=np.float32))
        body = {"audio_base64": base64.b64encode(wav).decode()}
        response = client.post("/detect", json=body)
        assert response.status_code == 200
        assert set(response.json()) == {"is_synthetic", "confidence"}
        assert response.json() == {"is_synthetic": False, "confidence": .5}
        assert response.headers["X-Behavior-Evidence"] == "insufficient_evidence"
        assert client.post("/behavior", json=body).json()["quality_score"] == 0
        bad = client.post("/detect", json={"audio_base64": "CONFIDENTIAL_INVALID_DATA"})
        assert bad.status_code == 422 and "CONFIDENTIAL" not in bad.text
        assert client.post("/detect", content="x", headers={"content-type": "text/plain"}).status_code == 415
        assert client.post("/detect", json={"audio_base64": 123}).status_code == 422


@pytest.mark.integration
def test_http_body_limit_enforced_before_parsing(detector, monkeypatch):
    monkeypatch.setenv("BEHAVIOR_MAX_DURATION_S", "1")
    with TestClient(create_app(lambda: detector)) as client:
        response = client.post("/detect", content=b"x" * 200_000,
                               headers={"Content-Type": "application/json"})
        assert response.status_code == 413
