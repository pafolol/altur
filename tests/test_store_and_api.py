"""
Tests for the call log (src/store.py) and the admin-panel API it feeds (src/server.py).

The panel in web/ is written against a TypeScript interface, so the thing worth testing is that the JSON
these endpoints produce still has the fields and the shapes that interface promises. Everything runs
against a temporary database; no model and no network.
"""
import base64
import io
import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import store  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    """A fresh database per test - store keeps one module-level connection, so reset both."""
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "calls.db")
    monkeypatch.setattr(store, "_conn", None)
    monkeypatch.delenv("TELEMETRY_URL", raising=False)   # the project .env may set it; tests opt in below
    yield store
    if store._conn is not None:
        store._conn.close()
    store._conn = None


def verdict(p=0.9, is_synthetic=True, decisive=True, latency=120, layers=None, duration=42.0):
    layers = layers if layers is not None else [
        {"key": "acoustic", "display": "A", "probability": p, "quality": 1.0, "abstained": False,
         "scored": True, "role": "primary", "consulted": True, "share": 0.5, "reason": "",
         "details": {"chunk_scores": [2.0, 2.0], "chunk_spans": [[0.0, 4.0], [4.0, 8.0]]}},
        {"key": "behaviour", "display": "B", "probability": p, "quality": 0.8, "abstained": False,
         "scored": True, "role": "primary", "consulted": True, "share": 0.5, "reason": ""},
    ]
    return {"is_synthetic": is_synthetic, "confidence": max(p, 1 - p), "decisive": decisive,
            "layers": layers,
            "details": {"synthetic_probability": p, "latency_ms": latency, "duration_s": duration,
                        "layers": layers}}


# --------------------------------------------------------------------------- the log


def test_a_recorded_call_comes_back(db):
    cid = db.record(verdict(), duration_s=42.0, channels=2, queue="cards")
    rows = db.recent(24)
    assert len(rows) == 1
    r = rows[0]
    assert r["id"] == cid and r["verdict"] == "synthetic" and r["queue"] == "cards"
    assert r["probability"] == pytest.approx(0.9) and r["latency_ms"] == 120
    assert {l["key"] for l in r["layers"]} == {"acoustic", "behaviour"}


def test_the_log_keeps_the_role_and_whether_a_verifier_was_consulted(db):
    """The point of the two-stage design is visible per call, or the log is not worth keeping."""
    layers = verdict()["layers"] + [
        {"key": "semantic", "display": "S", "probability": 0.5, "quality": 0.0, "abstained": False,
         "scored": False, "role": "verifier", "consulted": False, "share": 0.0,
         "reason": "not consulted: the primary layers settled this call on their own"}]
    db.record(verdict(layers=layers))
    sem = next(l for l in db.recent(24)[0]["layers"] if l["key"] == "semantic")
    assert sem["role"] == "verifier" and sem["consulted"] is False and sem["scored"] is False
    assert "not consulted" in sem["reason"]


def test_an_undecided_call_is_logged_as_abstained(db):
    db.record(verdict(p=0.5, is_synthetic=False, decisive=False))
    assert db.recent(24)[0]["verdict"] == "abstained"


def test_a_failed_request_is_logged_but_kept_out_of_the_call_list(db):
    db.record(verdict(), ok=True)
    db.record(verdict(), ok=False)
    assert len(db.recent(24)) == 1                 # the panel lists answers, not crashes
    assert db.health_stats(24) == {"n": 2, "ok": 1, "uptime_24h": 50.0}


def test_recording_never_raises_on_a_malformed_verdict(db):
    """A log row is not worth failing a detection over."""
    assert db.record({}) is not None or True       # returns None on failure, but must not raise
    assert db.record({"details": {"layers": "not a list"}}) is None or True
    assert db.record(None) is None


def test_a_verdict_is_mirrored_to_the_telemetry_service_when_configured(db, monkeypatch):
    import http.server
    import threading
    got, arrived = [], threading.Event()

    class Sink(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            got.append((self.path, json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
            self.send_response(201)
            self.end_headers()
            arrived.set()

        def log_message(self, *_):
            pass

    sink = http.server.HTTPServer(("127.0.0.1", 0), Sink)
    threading.Thread(target=sink.serve_forever, daemon=True).start()
    monkeypatch.setenv("TELEMETRY_URL", f"http://127.0.0.1:{sink.server_port}/")
    try:
        cid = db.record(verdict(p=0.9), duration_s=42.0, channels=2)
        assert arrived.wait(3), "the telemetry POST never arrived"
        assert got[0] == ("/telemetry", {
            "call_id": cid, "is_synthetic": True, "confidence": 0.9,
            "acoustic_score": 0.9, "behavioral_score": 0.9, "semantic_score": None,   # never asked -> null
            "final_score": 0.9, "latency_ms": 120})
        arrived.clear()
        db.record({"is_synthetic": False, "confidence": 0.5, "decisive": False, "details": {"latency_ms": 5}}, ok=False)
        assert not arrived.wait(0.5), "a failed request is uptime, not a detection: it must not be mirrored"
    finally:
        sink.shutdown()


def test_latency_series_is_the_shape_the_sparkline_wants(db):
    for ms in (100, 200, 300):
        db.record(verdict(latency=ms))
    series = db.latency_series(24, buckets=48)
    assert len(series) == 48
    assert series[-1] == 200                        # median of the three, in the newest bucket
    assert all(isinstance(v, (int, float)) for v in series)


def test_latency_series_of_an_empty_log_is_zeros_not_a_crash(db):
    assert db.latency_series(24, buckets=48) == [0] * 48


# --------------------------------------------------------------------------- "decided at"


def test_decided_at_is_the_end_of_the_chunk_after_which_the_verdict_stops_changing():
    layers = [{"key": "acoustic", "abstained": False, "scored": True,
               "details": {"chunk_scores": [-1.0, 3.0, 2.0], "chunk_spans": [[0, 4], [4, 8], [8, 12]]}}]
    # running means: -1.0 (human), +1.0 (synthetic), +1.33 (synthetic) -> settles on chunk 2, ending at 8 s
    assert store.decided_at(layers) == 8.0


def test_decided_at_is_none_when_the_layer_had_nothing_to_say():
    assert store.decided_at([]) is None
    assert store.decided_at([{"key": "acoustic", "abstained": True, "details": {}}]) is None
    assert store.decided_at([{"key": "acoustic", "scored": False, "details": {}}]) is None
    assert store.decided_at([{"key": "acoustic", "details": {"chunk_scores": [], "chunk_spans": []}}]) is None


def test_decided_at_ignores_spans_that_do_not_match_the_scores():
    layers = [{"key": "acoustic", "details": {"chunk_scores": [1.0, 2.0], "chunk_spans": [[0, 4]]}}]
    assert store.decided_at(layers) is None


# --------------------------------------------------------------------------- the panel's API


@pytest.fixture
def client(db, monkeypatch):
    from fastapi.testclient import TestClient
    import server
    monkeypatch.setattr(server, "store", db)
    return TestClient(server.app)


def silence_wav(seconds=1.0, rate=8000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(np.zeros((int(rate * seconds), 2), dtype="<i2").tobytes())
    return buf.getvalue()


def test_api_calls_matches_the_typescript_Call_shape(client, db):
    db.record(verdict(), duration_s=42.4, channels=2, queue="cards")
    rows = client.get("/api/calls?range=24h").json()
    assert len(rows) == 1
    c = rows[0]
    assert set(c) == {"id", "at", "duration_s", "channels", "verdict", "confidence", "evidence",
                      "latency_ms", "trap", "queue"}
    assert set(c["evidence"]) == {"acoustic", "conversational", "semantic", "decided_at_s"}
    assert c["verdict"] in {"human", "synthetic", "abstained"}
    assert c["trap"] in {"refused", "answered", "none"}
    assert c["channels"] in (1, 2)
    # the panel renders m:ss from this, and its own mock always produced whole seconds
    assert isinstance(c["duration_s"], int)


def test_the_behaviour_layer_is_reported_under_the_panels_name(client, db):
    db.record(verdict())
    e = client.get("/api/calls").json()[0]["evidence"]
    assert e["conversational"] == pytest.approx(0.9)     # "behaviour" here, "conversational" there
    assert e["semantic"] == 0.0                          # absent layer reads 0, not null


def test_a_layer_that_abstained_or_was_never_asked_reads_zero(client, db):
    layers = [{"key": "acoustic", "probability": 0.8, "abstained": False, "scored": True},
              {"key": "behaviour", "probability": 0.7, "abstained": True, "scored": True},
              {"key": "semantic", "probability": 0.99, "abstained": False, "scored": False}]
    db.record(verdict(layers=layers))
    e = client.get("/api/calls").json()[0]["evidence"]
    assert e == {"acoustic": pytest.approx(0.8), "conversational": 0.0, "semantic": 0.0,
                 "decided_at_s": None}


def test_api_stats_matches_the_Stats_shape(client, db):
    db.record(verdict(latency=250))
    s = client.get("/api/stats?range=24h").json()
    assert set(s) == {"latency_series"} and len(s["latency_series"]) == 48


def test_api_health_matches_the_Health_shape_and_does_not_invent_fields(client):
    h = client.get("/api/health").json()
    assert {"endpoint", "model", "calibrated_on", "streaming", "uptime_24h", "queue_depth",
            "checked_at"} <= set(h)
    assert h["endpoint"] in {"up", "degraded", "down"}
    # this detector scores a finished call and serves synchronously; it says so rather than pretending
    assert h["streaming"] is False and h["queue_depth"] == 0
    assert isinstance(h["uptime_24h"], (int, float))


@pytest.mark.skipif(not any((Path(__file__).resolve().parents[1] / "models" / k / "mlp.pt").exists()
                            for k in ("wav2vec2_spanish", "robust_v2")), reason="no trained model")
def test_detect_writes_a_row_and_returns_its_id(client, db):
    body = {"audio": base64.b64encode(silence_wav()).decode(), "queue": "onboarding"}
    r = client.post("/detect", json=body).json()
    assert isinstance(r["is_synthetic"], bool)
    cid = r["details"]["call_id"]
    rows = db.recent(24)
    assert cid and [x["id"] for x in rows] == [cid]
    assert rows[0]["queue"] == "onboarding"          # the caller's own label, carried through


@pytest.mark.skipif(not any((Path(__file__).resolve().parents[1] / "models" / k / "mlp.pt").exists()
                            for k in ("wav2vec2_spanish", "robust_v2")), reason="no trained model")
def test_the_queue_label_is_optional_and_bounded(client, db):
    client.post("/detect", json={"audio": base64.b64encode(silence_wav()).decode()})
    assert db.recent(24)[0]["queue"] == "api"
    client.post("/detect", json={"audio": base64.b64encode(silence_wav()).decode(), "queue": "x" * 200})
    assert len(db.recent(24)[0]["queue"]) <= 40
