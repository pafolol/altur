"""Targeted blocker regressions from ASTRA2; no validation/model tuning."""

import base64
import struct

import numpy as np
import pytest
from fastapi.testclient import TestClient

from behavior.audio import AudioError, encode_wav, load_wav
from behavior.decision import challenge_response, evidence_state
from behavior.features import extract_features
from behavior.inference import score_features
from behavior.service import create_app
from behavior.turns import Turn


@pytest.mark.parametrize("turns,duration", [
    ([], 10.),
    ([Turn(1., 1.2, 1), Turn(2., 2.2, 0), Turn(3., 3.2, 1), Turn(4., 4.2, 0)], 10.),
    ([Turn(1., 4., 0), Turn(6., 9., 0)], 10.),
    ([Turn(1., 4., 1), Turn(6., 9., 1)], 10.),
    ([Turn(1., 3., 1), Turn(3.5, 6., 0)], 10.),
], ids=["silence", "extremely_sparse_both_channels", "caller_only", "agent_only", "one_usable_event"])
def test_low_evidence_is_explicit_abstention_not_synthetic(turns, duration):
    features, _ = extract_features(turns, duration)
    # Empty artifact proves the classifier is not invoked on insufficient evidence.
    result = score_features({"calibration": {"method": "not_used"}}, features, debug=True)
    assert result["synthetic_probability"] == .5
    assert result["quality_score"] == result["behavior_confidence"] == 0
    assert result["diagnostics"]["evidence_state"] == "insufficient_evidence"
    assert challenge_response(result, .5) == {"is_synthetic": False, "confidence": .5}


@pytest.mark.parametrize("p,positive", [(.2, False), (.5, True), (.8, True)])
def test_available_evidence_retains_frozen_threshold(p, positive):
    result = {"synthetic_probability": p, "quality_score": .1, "event_count": 3}
    assert evidence_state(result) == "available"
    assert challenge_response(result, .5) == {
        "is_synthetic": positive, "confidence": p if positive else 1 - p}


class SparseDetector:
    """Deterministic timeline to exercise HTTP fallback separately from VAD quality."""
    threshold = .5
    model_load_s = 0.

    def predict(self, payload):
        load_wav(payload)
        return {"synthetic_probability": .5, "quality_score": 0.,
                "event_count": 1, "behavior_confidence": 0.}


def test_http_insufficient_evidence_nonflag_and_original_behavior_contract():
    payload = {"audio_base64": base64.b64encode(encode_wav(np.zeros((8000, 2)))).decode()}
    with TestClient(create_app(SparseDetector)) as client:
        raw = client.post("/behavior", json=payload)
        assert raw.json() == SparseDetector().predict(base64.b64decode(payload["audio_base64"]))
        verdict = client.post("/detect", json=payload)
        assert verdict.json() == {"is_synthetic": False, "confidence": .5}
        assert verdict.headers["X-Behavior-Evidence"] == "insufficient_evidence"


def test_malformed_riff_chunk_is_clean_audio_error():
    # Oversized unknown chunk makes the stdlib WAV parser seek beyond the RIFF boundary.
    payload = b"RIFF" + struct.pack("<I", 12) + b"WAVEJUNK" + struct.pack("<I", 0x7FFFFFFF)
    with pytest.raises(AudioError):
        load_wav(payload)
    with TestClient(create_app(SparseDetector)) as client:
        response = client.post("/detect", json={"audio_base64": base64.b64encode(payload).decode()})
        assert response.status_code == 422
        assert response.json() == {"detail": "Invalid WAV container"}


@pytest.mark.parametrize("rate,width", [(16000, 2), (8000, 1)])
def test_wrong_rate_or_encoding_is_clean_audio_error(rate, width):
    import io
    import wave
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setparams((2, width, rate, 0, "NONE", "not compressed"))
        wav.writeframes(bytes(2 * width * rate))
    with pytest.raises(AudioError):
        load_wav(stream.getvalue())
