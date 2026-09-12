import xml.etree.ElementTree as ET
import os

from fastapi import APIRouter, Form, HTTPException
from fastapi.responses import Response
from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

router = APIRouter(prefix="/twilio", tags=["Twilio Voice"])


def _twiml(response: ET.Element) -> Response:
    # ElementTree escapes user-provided values such as RecordingUrl safely.
    content = ET.tostring(response, encoding="unicode", short_empty_elements=True)
    return Response(content=f'<?xml version="1.0" encoding="UTF-8"?>{content}', media_type="application/xml")


def _twilio_config():
    values = {
        "account_sid": os.getenv("TWILIO_ACCOUNT_SID", "").strip(),
        "auth_token": os.getenv("TWILIO_AUTH_TOKEN", "").strip(),
        "from_number": os.getenv("TWILIO_PHONE_NUMBER", "").strip(),
        "to_number": os.getenv("TWILIO_TO_NUMBER", "").strip(),
    }
    if not all(values.values()):
        raise HTTPException(status_code=503, detail="Twilio outbound configuration is incomplete")
    return values


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


@router.post("/call")
async def twilio_call():
    config = _twilio_config()
    twiml = (
        '<Response><Say language="es-MX" voice="alice">'
        "Hola Esteban. Esta llamada fue iniciada desde tu backend de Altur. La prueba funcionó."
        "</Say></Response>"
    )
    try:
        call = Client(config["account_sid"], config["auth_token"]).calls.create(
            twiml=twiml,
            from_=config["from_number"],
            to=config["to_number"],
        )
    except TwilioRestException as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Twilio rejected the outbound call (code={exc.code}, status={exc.status})",
        ) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Unable to start outbound call") from exc
    return {"status": "call_started", "call_sid": call.sid, "call_status": call.status}
