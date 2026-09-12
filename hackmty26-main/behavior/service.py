"""CPU HTTP adapter, retaining no request audio and logging no request bodies."""

import asyncio
import base64
import binascii
from contextlib import asynccontextmanager
import json
import os
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from .audio import AudioError
from .decision import challenge_response, evidence_state
from .inference import BehaviorDetector
from .schemas import BehaviorResponse, DetectRequest, DetectResponse


def create_app(detector_factory=None):
    max_seconds = float(os.getenv("BEHAVIOR_MAX_DURATION_S", "600"))
    max_body = int((max_seconds * 8000 * 4 + 65536) * 4 / 3) + 4096

    @asynccontextmanager
    async def lifespan(app):
        app.state.detector = (detector_factory or (lambda: BehaviorDetector(max_duration_s=max_seconds)))()
        app.state.gate = asyncio.Semaphore(2)
        yield

    app = FastAPI(title="Altur Conversational Behaviour", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health(request: Request):
        return {"status": "ready", "module": "behavior", "version": "1.0.0",
                "model_load_s": request.app.state.detector.model_load_s}

    async def predict_request(request):
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(415, "Use application/json with audio_base64")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > max_body:
                raise HTTPException(413, "Request exceeds WAV size limit")
            body.extend(chunk)
        try:
            payload = DetectRequest.model_validate(json.loads(body))
            wav = base64.b64decode(payload.audio_base64, validate=True)
        except (ValidationError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error, ValueError):
            # Never echo the submitted base64 in validation responses.
            raise HTTPException(422, "Invalid JSON/base64; expected audio_base64") from None
        try:
            async with request.app.state.gate:
                return await run_in_threadpool(request.app.state.detector.predict, wav)
        except AudioError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/behavior", response_model=BehaviorResponse)
    async def behavior(request: Request):
        return await predict_request(request)

    @app.post("/detect", response_model=DetectResponse)
    async def detect(request: Request, response: Response):
        """Behaviour-only demo adapter. The team fusion layer should own final /detect."""
        r = await predict_request(request)
        response.headers["X-Behavior-Evidence"] = evidence_state(r)
        return challenge_response(r, request.app.state.detector.threshold)

    return app


app = create_app()
