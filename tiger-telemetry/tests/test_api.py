from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app import database
from app.main import app


client = TestClient(app)
EVENT = {
    "timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc).isoformat(),
    "call_id": "abc123",
    "is_synthetic": True,
    "confidence": 0.91,
    "acoustic_score": 0.84,
    "behavioral_score": 0.95,
    "semantic_score": 0.72,
    "final_score": 0.91,
    "latency_ms": 240,
}


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_successful_post(monkeypatch) -> None:
    monkeypatch.setattr(database, "insert_detection", lambda telemetry: EVENT)

    response = client.post(
        "/telemetry",
        json={key: value for key, value in EVENT.items() if key != "timestamp"},
    )

    assert response.status_code == 201
    assert response.json()["call_id"] == EVENT["call_id"]
    assert response.json()["timestamp"].startswith("2026-01-01T00:00:00")


def test_validation_rejects_out_of_range_score_and_unknown_pii() -> None:
    response = client.post(
        "/telemetry",
        json={
            "call_id": "abc123",
            "is_synthetic": False,
            "confidence": 1.1,
            "phone_number": "+15551234567",
        },
    )
    assert response.status_code == 422


def test_database_unavailable_returns_safe_503(monkeypatch) -> None:
    def unavailable(_telemetry):
        raise database.DatabaseUnavailable("private connection details")

    monkeypatch.setattr(database, "insert_detection", unavailable)
    response = client.post(
        "/telemetry",
        json={"call_id": "abc123", "is_synthetic": True, "confidence": 0.5},
    )
    assert response.status_code == 503
    assert response.json() == {"detail": "Tiger Data is unavailable"}
    assert "private" not in response.text


def test_detections(monkeypatch) -> None:
    monkeypatch.setattr(database, "fetch_detections", lambda: [EVENT])
    response = client.get("/detections")
    assert response.status_code == 200
    assert response.json()[0]["call_id"] == EVENT["call_id"]
    assert response.json()[0]["latency_ms"] == EVENT["latency_ms"]


def test_stats(monkeypatch) -> None:
    monkeypatch.setattr(
        database,
        "fetch_stats",
        lambda: {
            "total_detections": 4,
            "synthetic_detections": 3,
            "human_detections": 1,
            "average_confidence": 0.8,
            "average_latency_ms": 210.5,
        },
    )
    response = client.get("/stats")
    assert response.status_code == 200
    assert response.json()["total_detections"] == 4
    assert response.json()["average_latency_ms"] == 210.5
