"""
Unit tests of the audio layer, metrics and (when a trained model exists) the endpoint contract.
Run:  .venv/Scripts/python.exe -m pytest tests -q
"""
import base64
import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import audio  # noqa: E402
import config  # noqa: E402
import metrics  # noqa: E402


def synthetic_call(sr=8000, seconds=12.0, seed=0):
    """Stereo 8 kHz test signal: caller (ch 0) speaks at 2-5 s and 7-9 s, agent (ch 1) at 0-1.5 s and 5.5-6.5 s."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    caller = 0.0005 * rng.standard_normal(n)
    agent = 0.0005 * rng.standard_normal(n)
    for s, e in ((2, 5), (7, 9)):
        m = (t >= s) & (t < e)
        caller[m] += (0.2 * np.sin(2 * np.pi * 420 * t[m]) + 0.1 * np.sin(2 * np.pi * 1260 * t[m])) * (1 + 0.5 * np.sin(2 * np.pi * 3 * t[m]))
    for s, e in ((0, 1.5), (5.5, 6.5)):
        m = (t >= s) & (t < e)
        agent[m] += 0.3 * np.sin(2 * np.pi * 220 * t[m])
    return np.stack([caller, agent], axis=1).astype(np.float32), sr


def wav_bytes(stereo, sr):
    buf = io.BytesIO()
    sf.write(buf, stereo, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def test_base64_roundtrip_and_channels():
    stereo, sr = synthetic_call()
    b64 = base64.b64encode(wav_bytes(stereo, sr)).decode()
    decoded, sr2 = audio.decode_base64_wav(b64)
    assert sr2 == sr and decoded.shape == stereo.shape
    assert np.abs(audio.get_channel(decoded, 0) - stereo[:, 0]).max() < 1e-3
    assert np.abs(audio.get_channel(decoded, 1) - stereo[:, 1]).max() < 1e-3


def test_resample_doubles_length_and_keeps_content():
    stereo, sr = synthetic_call()
    caller = audio.get_channel(stereo, 0)
    up = audio.resample(caller, sr, 16000)
    assert len(up) == 2 * len(caller)
    assert abs(audio.rms_db(up) - audio.rms_db(caller)) < 0.5


def test_level_normalize():
    x = np.random.default_rng(1).standard_normal(16000).astype(np.float32) * 0.01
    y = audio.level_normalize(x, -26.0)
    assert abs(audio.rms_db(y) - (-26.0)) < 0.01


def test_vad_finds_caller_speech_not_agent():
    stereo, sr = synthetic_call()
    regions = audio.detect_speech(audio.get_channel(stereo, 0), sr)
    assert len(regions) == 2
    (s1, e1), (s2, e2) = regions
    assert abs(s1 - 2) < 0.2 and abs(e1 - 5) < 0.2 and abs(s2 - 7) < 0.2 and abs(e2 - 9) < 0.2
    # the agent's speech at 0-1.5 s must not appear in the caller regions
    assert s1 > 1.5


def test_chunking_rules():
    chunks = audio.chunk_regions([(0.0, 10.5), (20.0, 20.5), (30.0, 33.0)], chunk_s=4.0, min_s=1.0)
    assert chunks == [(0.0, 4.0), (4.0, 8.0), (8.0, 10.5), (30.0, 33.0)]  # 0.5 s region dropped, 2.5 s leftover kept
    assert audio.chunk_regions([(0.0, 8.5)], 4.0, 1.0) == [(0.0, 4.0), (4.0, 8.0)]  # 0.5 s leftover dropped
    assert all(e - s <= 4.0 + 1e-9 for s, e in chunks)


def test_caller_chunks_shapes():
    stereo, sr = synthetic_call()
    chunks, spans = audio.caller_chunks(stereo, sr)
    assert len(chunks) == len(spans) >= 2
    for c, (s, e) in zip(chunks, spans):
        assert c.dtype == np.float32
        assert abs(len(c) / config.MODEL_SAMPLE_RATE - (e - s)) < 0.01
        assert abs(audio.rms_db(c) - config.TARGET_RMS_DB) < 0.1


def test_metrics_perfect_and_random():
    y = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([-3, -2, -1, 1, 2, 3.0])
    m = metrics.evaluate_scores(y, perfect)
    assert m["accuracy"] == 1.0 and m["auc"] == 1.0 and m["eer"] == 0.0 and m["f1"] == 1.0
    worst = -perfect
    assert metrics.evaluate_scores(y, worst)["accuracy"] == 0.0
    assert metrics.aggregate_chunk_scores([1, 2, 3, 10], "mean") == 4.0
    assert metrics.aggregate_chunk_scores([1, 2, 3, 10], "median") == 2.5
    assert metrics.aggregate_chunk_scores([1, 2, 3, 10], "max") == 10.0
    assert metrics.aggregate_chunk_scores([1, 2, 3, 10], "top25_mean") == 10.0
    cal = metrics.calibration_metrics(y, np.array([0.01, 0.02, 0.03, 0.97, 0.98, 0.99]))
    assert cal["brier"] < 0.001 and cal["ece"] < 0.05


@pytest.mark.skipif(not any((config.MODELS_DIR / k / "mlp.pt").exists() for k in config.BACKBONE_ORDER),
                    reason="no trained model yet")
def test_endpoint_contract():
    from fastapi.testclient import TestClient
    import server
    server.app.state.primary, server.app.state.classifier = "specialist", "mlp"
    client = TestClient(server.app)
    stereo, sr = synthetic_call()
    b64 = base64.b64encode(wav_bytes(stereo, sr)).decode()
    # Exact judge contract: one complete stereo 8 kHz WAV plus its metadata.
    r = client.post("/detect", json={"call_id": "contract_test", "audio_base64": b64,
                                     "sample_rate": 8000, "channels": 2})
    assert r.status_code == 200
    body = r.json()
    assert isinstance(body["is_synthetic"], bool)
    assert 0.0 <= body["confidence"] <= 1.0
    # raw base64 body and multipart upload are accepted too
    r2 = client.post("/detect", content=b64, headers={"Content-Type": "text/plain"})
    assert r2.status_code == 200 and "is_synthetic" in r2.json()
    r3 = client.post("/detect", files={"file": ("call.wav", wav_bytes(stereo, sr), "audio/wav")})
    assert r3.status_code == 200 and "is_synthetic" in r3.json()
    # a real validation call, if the dataset is present
    if config.MANIFEST.exists():
        import dataset
        row = dataset.load_split("val").iloc[0]
        r4 = client.post("/detect", json={"audio": audio.encode_wav_base64(row.path)})
        assert r4.status_code == 200 and isinstance(r4.json()["is_synthetic"], bool)


@pytest.mark.skipif(not any((config.MODELS_DIR / k / "mlp.pt").exists() for k in config.BACKBONE_ORDER),
                    reason="no trained model yet")
def test_truncated_backbone_matches_full_model():
    """predict.py drops the transformer layers above the selected one; the embedding must be identical."""
    import torch
    from transformers import AutoModel
    from predict import AcousticDetector
    det = AcousticDetector(verbose=False)
    full = AutoModel.from_pretrained(det.hf_name).to(det.device).eval()
    stereo, sr = synthetic_call()
    chunks, _ = audio.caller_chunks(stereo, sr)
    x = torch.from_numpy(audio.normalize_for_model(chunks[0], det.do_normalize))[None].to(det.device)
    with torch.inference_mode():
        h_full = full(x, output_hidden_states=True).hidden_states[det.layer].float().mean(1)
        h_trunc = det.model(x, output_hidden_states=True).hidden_states[det.layer].float().mean(1)
    assert torch.allclose(h_full, h_trunc, atol=1e-4)


@pytest.mark.skipif(not any((config.MODELS_DIR / k / "mlp.pt").exists() for k in config.BACKBONE_ORDER),
                    reason="no trained model yet")
def test_demo_page_and_details():
    """GET / serves the one-file demo page; /detect returns the fields the page explains the verdict with."""
    from fastapi.testclient import TestClient
    import server
    server.app.state.primary, server.app.state.classifier = "specialist", "mlp"
    client = TestClient(server.app)
    r = client.get("/")
    assert r.status_code == 200 and "Line" in r.text and "Preguntar al detector" in r.text
    stereo, sr = synthetic_call()
    body = client.post("/detect", json={"audio": base64.b64encode(wav_bytes(stereo, sr)).decode()}).json()
    for key in ("synthetic_probability", "chunk_scores", "chunk_spans", "model_display", "calibration", "threshold", "n_speech_regions"):
        assert key in body["details"], key
    assert len(body["details"]["chunk_scores"]) == body["details"]["n_chunks"]


@pytest.mark.skipif(not config.MANIFEST.exists(), reason="dataset not present")
def test_validation_deck_endpoints():
    """The demo page's test deck: the held-out calls are listed (with labels for the reveal) and streamable as WAV."""
    from fastapi.testclient import TestClient
    import server
    client = TestClient(server.app)
    calls = client.get("/validation").json()
    assert len(calls) == 71 and {c["label"] for c in calls} == {"human", "synthetic"}
    assert all("path" not in c for c in calls)
    r = client.get(f"/validation/{calls[0]['anon_id']}/audio")
    assert r.status_code == 200 and r.headers["content-type"].startswith("audio/wav") and r.content[:4] == b"RIFF"
    assert client.get("/validation/call_doesnotexist/audio").status_code == 404
