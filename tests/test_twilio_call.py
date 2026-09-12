from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
import app.twilio_routes as twilio_routes


def test_outbound_call_missing_configuration(monkeypatch):
    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "TWILIO_TO_NUMBER"):
        monkeypatch.delenv(name, raising=False)
    response = TestClient(create_app(Settings(enable_feature_warmup=False))).post("/twilio/call")
    assert response.status_code == 503


def test_outbound_call_success_is_mocked(monkeypatch):
    calls = []

    class FakeCalls:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(sid="CA_TEST", status="queued")

    class FakeClient:
        def __init__(self, account_sid, auth_token):
            assert account_sid == "AC_TEST"
            assert auth_token == "token-test"
            self.calls = FakeCalls()

    for name, value in {
        "TWILIO_ACCOUNT_SID": "AC_TEST",
        "TWILIO_AUTH_TOKEN": "token-test",
        "TWILIO_PHONE_NUMBER": "+10000000001",
        "TWILIO_TO_NUMBER": "+10000000002",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(twilio_routes, "Client", FakeClient)
    response = TestClient(create_app(Settings(enable_feature_warmup=False))).post("/twilio/call")
    assert response.status_code == 200
    assert response.json() == {"status": "call_started", "call_sid": "CA_TEST", "call_status": "queued"}
    assert calls[0]["from_"] == "+10000000001"
    assert calls[0]["to"] == "+10000000002"
    assert "Hola Esteban" in calls[0]["twiml"]


def test_outbound_call_twilio_failure(monkeypatch):
    class FakeClient:
        def __init__(self, *_):
            self.calls = self

        def create(self, **kwargs):
            raise twilio_routes.TwilioRestException(
                400,
                "https://api.twilio.com/2010-04-01/Accounts/AC_SECRET/Calls.json",
                "secret error message",
                21219,
            )

    for name in ("TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_PHONE_NUMBER", "TWILIO_TO_NUMBER"):
        monkeypatch.setenv(name, "configured")
    monkeypatch.setattr(twilio_routes, "Client", FakeClient)
    response = TestClient(create_app(Settings(enable_feature_warmup=False))).post("/twilio/call")
    assert response.status_code == 502
    assert response.json() == {"detail": "Twilio rejected the outbound call (code=21219, status=400)"}
    assert "secret" not in response.text
    assert "AC_SECRET" not in response.text
