import xml.etree.ElementTree as ET

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def test_twilio_voice_returns_record_twiml():
    response = TestClient(create_app(Settings(enable_feature_warmup=False))).post("/twilio/voice")
    assert response.status_code == 200
    assert "xml" in response.headers["content-type"]
    root = ET.fromstring(response.text)
    record = root.find("Record")
    assert record is not None
    assert record.attrib["action"] == "/twilio/playback"


def test_twilio_playback_accepts_recording_url():
    recording_url = "https://api.twilio.com/recordings/RE123"
    response = TestClient(create_app(Settings(enable_feature_warmup=False))).post(
        "/twilio/playback", data={"RecordingUrl": recording_url}
    )
    assert response.status_code == 200
    root = ET.fromstring(response.text)
    assert root.find("Play").text == recording_url
    assert "Esta es tu grabación." in response.text
    assert "Prueba terminada." in response.text
