"""Server contract and degradation ladder with Scribe and Gemini faked. No network. Run: .venv/bin/pytest -q"""
import base64, io, struct, sys, pathlib, threading, time, wave
import numpy as np
import pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import server, asr, rubric
from fastapi.testclient import TestClient

client = TestClient(server.app)


def wav_b64(seconds=12.0, channels=2, rate=8000, width=2, tone=True):
    n = int(seconds * rate); t = np.arange(n) / rate
    sig = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 1.5 * t) > 0)) if tone else np.zeros(n)   # bursts of tone = "speech" for the energy VAD
    pcm = (sig * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels); w.setsampwidth(width); w.setframerate(rate)
        frames = np.zeros((n, channels), dtype="<i2"); frames[:, 0] = pcm          # caller channel has "speech", agent channel is silent
        w.writeframes(frames.tobytes() if width == 2 else b"\x00" * n * channels)
    return base64.b64encode(buf.getvalue()).decode()


def words(n=30, delay=0.0):
    time.sleep(delay)
    return {"words": [{"text": f"w{i}", "start": i * 0.3, "end": i * 0.3 + 0.2, "logprob": -0.01} for i in range(n)], "n_chunks": 1, "n_hedged": 0}


def set_asr(monkeypatch, caller, agent):
    def t(x, segs, sr=8000, **k):
        return caller() if np.abs(x).max() > 0 else agent()    # the test WAV has signal only on the caller channel
    monkeypatch.setattr(asr, "transcribe", t)


def set_gemini(monkeypatch, fn):
    monkeypatch.setattr(rubric, "score", fn)


def good_rubric(transcript):
    return {"scores": {"sobrecompletitud": {"score": 1.0}, "registro_formal": {"score": 1.0}}}


def test_health():
    r = client.get("/health"); j = r.json()
    assert r.status_code == 200 and j["ok"] and j["config"] == server.CONFIG and set(j["keys_present"]) == {"ELEVENLABS_API_KEY", "GEMINI_API_KEY"}


def test_full_path_contract(monkeypatch):
    set_asr(monkeypatch, words, words); set_gemini(monkeypatch, good_rubric)
    j = client.post("/detect", json={"audio": wav_b64()}).json()
    assert j["used"] == "f4+f5" and j["abstain"] is False and j["reason"] == ""
    assert set(j) == {"is_synthetic", "confidence", "score", "abstain", "reason", "used", "ms"}
    assert j["confidence"] == j["score"] and j["is_synthetic"] == (j["score"] >= 0.5) and 0 <= j["score"] <= 1


def test_slow_gemini_degrades_to_f4(monkeypatch):
    monkeypatch.setattr(server, "BUDGET_S", 0.6); monkeypatch.setattr(server, "HARD_S", 0.9)
    set_asr(monkeypatch, words, words); set_gemini(monkeypatch, lambda t: (time.sleep(2.0), good_rubric(t))[1])
    t0 = time.perf_counter(); j = client.post("/detect", json={"audio": wav_b64()}).json()
    assert j["used"] == "f4" and j["reason"].startswith("degraded") and time.perf_counter() - t0 < 1.2


def test_slow_agent_channel_degrades_to_f4(monkeypatch):
    monkeypatch.setattr(server, "BUDGET_S", 0.6); monkeypatch.setattr(server, "HARD_S", 0.9)
    set_asr(monkeypatch, words, lambda: words(delay=2.0)); set_gemini(monkeypatch, good_rubric)
    j = client.post("/detect", json={"audio": wav_b64()}).json()
    assert j["used"] == "f4"


def test_broken_scribe_abstains_with_reason(monkeypatch):
    def boom(): raise RuntimeError("all requests failed")
    set_asr(monkeypatch, boom, boom)
    j = client.post("/detect", json={"audio": wav_b64()}).json()
    assert j["abstain"] is True and j["score"] == 0.5 and j["reason"] == "asr_error:RuntimeError" and j["is_synthetic"] is False


def test_silent_caller_abstains_no_speech(monkeypatch):
    set_asr(monkeypatch, lambda: words(n=2), words); set_gemini(monkeypatch, good_rubric)
    j = client.post("/detect", json={"audio": wav_b64()}).json()
    assert j["used"] == "abstain" and j["reason"] == "no_speech"


def test_asr_without_logprob_abstains(monkeypatch):
    def nolp():
        w = words(); [x.pop("logprob") for x in w["words"]]; return w
    set_asr(monkeypatch, nolp, words); set_gemini(monkeypatch, good_rubric)
    j = client.post("/detect", json={"audio": wav_b64()}).json()
    assert j["abstain"] is True and j["reason"].startswith("internal:KeyError")


@pytest.mark.parametrize("payload,code,frag", [
    ("not base64!!", 400, "not base64"),
    (base64.b64encode(b"RIFFjunk").decode(), 400, "not base64-encoded WAV"),
    (wav_b64(channels=1), 400, "got 1 channel"),
    (wav_b64(rate=16000), 400, "16000 Hz"),
    (wav_b64(seconds=0.0), 400, "zero frames"),
])
def test_invalid_input_is_400(payload, code, frag):
    r = client.post("/detect", json={"audio": payload})
    assert r.status_code == code and frag in r.json()["detail"]


def test_oversized_body_is_413(monkeypatch):
    monkeypatch.setattr(server, "MAX_B64", 1000)
    r = client.post("/detect", json={"audio": "A" * 2000})
    assert r.status_code == 413


def test_missing_field_is_422():
    assert client.post("/detect", json={"wav": "x"}).status_code == 422


def test_sixteen_concurrent_requests_do_not_starve(monkeypatch):
    monkeypatch.setattr(server, "BUDGET_S", 2.5); monkeypatch.setattr(server, "HARD_S", 3.0)
    set_asr(monkeypatch, lambda: words(delay=1.0), lambda: words(delay=1.0)); set_gemini(monkeypatch, lambda t: (time.sleep(0.5), good_rubric(t))[1])
    body = {"audio": wav_b64()}; out = []
    def one(): out.append(client.post("/detect", json=body).json())
    th = [threading.Thread(target=one) for _ in range(16)]
    t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; dt = time.perf_counter() - t0
    assert len(out) == 16 and sum(j["abstain"] for j in out) == 0, [j["used"] for j in out]
    assert sum(j["used"] == "f4+f5" for j in out) >= 12 and dt < 6.0, (dt, [j["used"] for j in out])


def test_pkl_guard_refuses_other_scribe_model(monkeypatch):
    import importlib
    monkeypatch.setattr(asr, "MODEL", "scribe_v2")
    with pytest.raises(SystemExit):
        importlib.reload(server)
    monkeypatch.undo(); importlib.reload(server)


def test_index_and_calls():
    c = TestClient(server.app)
    assert c.get("/").status_code == 200 and "<html" in c.get("/").text.lower()
    rows = c.get("/calls").json()
    assert all(r["label"] in ("human", "synthetic") and 0 <= r["score"] <= 1 for r in rows)
    assert c.get("/audio/not_a_call").status_code == 404


def test_check_converts_mono_44k_and_scores(monkeypatch):
    # a 44.1 kHz mono WAV: afconvert -> 8 kHz, caller in ch0, agent silent -> f4+f5 like /detect (fake ASR keyed by channel signal)
    set_asr(monkeypatch, words, lambda: words(n=2)); set_gemini(monkeypatch, good_rubric)
    sr = 44100; x = (np.sin(np.arange(sr * 3) * 2 * np.pi * 440 / sr) * 8000).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes(x.tobytes())
    r = TestClient(server.app).post("/check", content=buf.getvalue())
    assert r.status_code == 200, r.text
    d = r.json(); assert d["used"] in ("f4+f5", "f4") and abs(d["duration_s"] - 3.0) < 0.1
    assert TestClient(server.app).post("/check", content=b"not audio at all").status_code == 400
