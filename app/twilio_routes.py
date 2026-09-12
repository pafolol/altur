import xml.etree.ElementTree as ET

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response

router = APIRouter(prefix="/twilio", tags=["Twilio Voice"])


def _twiml(response: ET.Element) -> Response:
    # ElementTree escapes user-provided values such as RecordingUrl safely.
    content = ET.tostring(response, encoding="unicode", short_empty_elements=True)
    return Response(content=f'<?xml version="1.0" encoding="UTF-8"?>{content}', media_type="application/xml")


@router.post("/voice")
async def twilio_voice():
    response = ET.Element("Response")
    say = ET.SubElement(response, "Say", {"language": "es-MX", "voice": "alice"})
    say.text = "Hola. Esta es una prueba de voz. Habla después del tono y presiona gato cuando termines."
    ET.SubElement(response, "Record", {
        "action": "/twilio/playback",
        "method": "POST",
        "finishOnKey": "#",
        "playBeep": "true",
    })
    return _twiml(response)


@router.post("/playback")
async def twilio_playback(recording_url: str | None = Form(default=None, alias="RecordingUrl")):
    if not recording_url:
        raise HTTPException(status_code=400, detail="RecordingUrl is required")
    response = ET.Element("Response")
    say_start = ET.SubElement(response, "Say", {"language": "es-MX", "voice": "alice"})
    say_start.text = "Esta es tu grabación."
    ET.SubElement(response, "Play").text = recording_url
    say_end = ET.SubElement(response, "Say", {"language": "es-MX", "voice": "alice"})
    say_end.text = "Prueba terminada."
    return _twiml(response)
