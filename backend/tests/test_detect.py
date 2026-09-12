import base64
import io

import numpy as np
import soundfile as sf
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.feature_extractor import FEATURE_NAMES, extract_features
from app.feature_validation import validate_finite_features
from app.fusion import Fusion, FusionError
from app.detector import Detector, DetectorIncompatible


def wav_base64(channels=2, seconds=1.0, sr=8000):
    data = np.zeros((int(sr * seconds), channels), dtype=np.float32)
    if channels >= 1: data[:, 0] = 0.1 * np.sin(2 * np.pi * 220 * np.arange(len(data)) / sr)
    if channels >= 2: data[:, 1] = 0.1 * np.sin(2 * np.pi * 180 * np.arange(len(data)) / sr)
    stream = io.BytesIO(); sf.write(stream, data, sr, format="WAV")
    return base64.b64encode(stream.getvalue()).decode()


def client(**kwargs):
    config = Settings(**kwargs)
    return TestClient(create_app(config))


def test_health():
    assert client().get("/health").json() == {"status": "ok"}


def test_missing_audio():
    assert client().post("/detect", json={}).status_code == 422


def test_invalid_base64_and_wav():
    c = client()
    assert c.post("/detect", json={"audio": "not base64!"}).status_code == 400
    assert c.post("/detect", json={"audio": base64.b64encode(b"bad wav").decode()}).status_code == 400


def test_mono_and_empty_audio():
    c = client()
    assert c.post("/detect", json={"audio": wav_base64(1)}).status_code == 400
    empty = io.BytesIO(); sf.write(empty, np.empty((0, 2), dtype=np.float32), 8000, format="WAV")
    assert c.post("/detect", json={"audio": base64.b64encode(empty.getvalue()).decode()}).status_code == 400


def test_stereo_mock_response_and_data_uri():
    response = client().post("/detect", json={"audio_base64": "data:audio/wav;base64," + wav_base64()})
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result["is_synthetic"], bool)
    assert 0 <= result["confidence"] <= 1


def test_too_long_audio():
    c = client(max_audio_duration_seconds=0.1)
    assert c.post("/detect", json={"audio": wav_base64(seconds=1)}).status_code == 400


def test_feature_extraction_has_no_non_finite_values():
    raw = np.zeros(8000, dtype=np.float32)
    features = extract_features(raw, raw, 8000)
    assert set(FEATURE_NAMES) == set(features)
    assert all(np.isfinite(list(features.values())))


def test_model_missing_returns_503(tmp_path):
    c = client(detector_mode="model", model_path=str(tmp_path / "missing.joblib"))
    assert c.post("/detect", json={"audio": wav_base64()}).status_code == 503


def test_ready_in_mock_mode():
    response = client().get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready"}


def test_ready_model_missing_is_503(tmp_path):
    response = client(detector_mode="model", model_path=str(tmp_path / "none.joblib")).get("/ready")
    assert response.status_code == 503


def test_feature_warmup_runs():
    with TestClient(create_app(Settings())) as test_client:
        assert test_client.get("/ready").status_code == 200


def test_feature_warmup_can_be_disabled():
    with TestClient(create_app(Settings(enable_feature_warmup=False))) as test_client:
        assert test_client.get("/ready").status_code == 200


def test_feature_warmup_failure_marks_backend_not_ready(monkeypatch):
    import app.main as main_module
    monkeypatch.setattr(main_module, "run_feature_warmup", lambda: (_ for _ in ()).throw(RuntimeError("warmup failed")))
    with TestClient(create_app(Settings())) as test_client:
        assert test_client.get("/ready").status_code == 503


def test_finite_validation_rejects_nan_and_infinity():
    with pytest.raises(ValueError): validate_finite_features({"response_latency_cv": np.nan})
    with pytest.raises(ValueError): validate_finite_features({"pitch_mean": np.inf})


def test_short_audio_is_rejected():
    assert client(min_audio_duration_seconds=1.0).post("/detect", json={"audio": wav_base64(seconds=0.5)}).status_code == 400


def test_silent_caller_is_rejected():
    data = np.zeros((8000, 2), dtype=np.float32)
    stream = io.BytesIO(); sf.write(stream, data, 8000, format="WAV")
    encoded = base64.b64encode(stream.getvalue()).decode()
    assert client().post("/detect", json={"audio": encoded}).status_code == 400


def test_single_model_fusion():
    assert Fusion("single_model").predict(single_model=0.73) == pytest.approx(0.73)


def test_weighted_fusion_renormalizes_missing_semantic():
    fusion = Fusion("weighted", {"behavior": 2, "acoustic": 1, "semantic": 100})
    assert fusion.predict({"behavior": 0.2, "acoustic": 0.8, "semantic": None}) == pytest.approx(0.4)


def test_invalid_fusion_weights_and_range():
    with pytest.raises(FusionError): Fusion("weighted", {"behavior": -1})
    with pytest.raises(FusionError): Fusion("single_model").predict(single_model=1.1)


def test_incompatible_feature_schema(tmp_path):
    import joblib
    path = tmp_path / "bad.joblib"
    joblib.dump({"model": object(), "feature_names": FEATURE_NAMES[:-1]}, path)
    with pytest.raises(DetectorIncompatible): Detector("model", str(path))


def test_incompatible_schema_version(tmp_path):
    import joblib
    path = tmp_path / "bad-version.joblib"
    joblib.dump({"model": object(), "feature_names": FEATURE_NAMES, "feature_schema_version": "9.0"}, path)
    with pytest.raises(DetectorIncompatible): Detector("model", str(path))
