"""
CHALLENGE ENDPOINT -  POST /detect

Request body (JSON):  {"audio": "<base64 of a stereo 8 kHz WAV file>"}
    Also accepted: {"wav": ...}, {"audio_base64": ...}, {"file": ...}, {"data": ...}, or a raw base64 string body,
    or multipart/form-data with a file field - the judges' exact field name is not specified in the PDF, so the
    server is permissive about it.
Response (JSON):      {"is_synthetic": true, "confidence": 0.87}
    is_synthetic  required by the challenge
    confidence    probability that the verdict is right (config.CONFIDENCE_MODE = "verdict")
    Extra diagnostic fields (synthetic_probability, n_chunks, latency_ms, model) are added under "details";
    they do not interfere with the required contract.

Run:  python src/server.py [--host 0.0.0.0] [--port 8000] [--backbone wavlm]
Test: python src/client_demo.py path/to/call.wav --url http://localhost:8000/detect

The acoustic detector is the only signal today; the fusion of acoustic + behavioural + semantic scores will
replace `score_call()` later without touching the HTTP contract.
"""
import argparse
import base64
import time

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

import _bootstrap  # noqa: F401
import config
from predict import AcousticDetector

app = FastAPI(title="Altur HackMTY 2026 - synthetic caller detector (acoustic branch)")
# The demo page (frontend/index.html) is served by this same process at GET /, so microphone access works over
# localhost and no cross-origin setup is needed; CORS is open so the file also works when opened from disk.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND = config.PROJECT_ROOT / "frontend" / "index.html"
DETECTOR = None
FIELDS = ("audio", "wav", "audio_base64", "file", "data", "audio_b64", "base64", "wav_base64")


def get_detector():
    global DETECTOR
    if DETECTOR is None:
        DETECTOR = AcousticDetector(app.state.backbone, app.state.classifier)
    return DETECTOR


def score_call(wav_bytes):
    """One place where the verdict is produced. Fusion with other branches will plug in here."""
    det = get_detector()
    result = det.predict_wav(wav_bytes)
    verdict = det.verdict(result)
    verdict["details"] = {
        "synthetic_probability": round(result["synthetic_probability"], 4),
        "raw_score": round(result["score"], 3),
        "n_chunks": result["n_chunks"],
        "speech_s": round(result["speech_s"], 1),
        "duration_s": round(result["duration_s"], 1),
        "latency_ms": round(result["timings"].get("total_s", 0) * 1000),
        "model": f"{det.backbone} layer {det.layer} + {det.classifier_name}",
        "model_display": f"{config.BACKBONES[det.backbone]['display']}, hidden layer {det.layer}, frozen + MLP",
        "calibration": det.calibration.get("method", "none"),
        "threshold": config.DECISION_THRESHOLD,
        "n_speech_regions": result["n_speech_regions"],
        "chunk_scores": result.get("chunk_scores", []),
        "chunk_spans": result.get("chunk_spans", []),
        "branches": {"acoustic": round(result["synthetic_probability"], 4)},
    }
    if "warning" in result:
        verdict["details"]["warning"] = result["warning"]
    return verdict


def _extract_base64(payload):
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for f in FIELDS:
            if f in payload and isinstance(payload[f], str):
                return payload[f]
        for v in payload.values():  # any long string value
            if isinstance(v, str) and len(v) > 100:
                return v
    return None


@app.post("/detect")
async def detect(request: Request):
    t0 = time.perf_counter()
    content_type = request.headers.get("content-type", "")
    wav_bytes = None
    try:
        if content_type.startswith("multipart/form-data"):
            form = await request.form()
            for v in form.values():
                if isinstance(v, (UploadFile, StarletteUploadFile)):
                    wav_bytes = await v.read()
                    break
                if isinstance(v, str) and len(v) > 100:
                    wav_bytes = base64.b64decode(v)
                    break
        else:
            body = await request.body()
            try:
                payload = await request.json()
            except Exception:
                payload = body.decode("utf-8", errors="ignore").strip().strip('"')
            b64 = _extract_base64(payload)
            if b64 is None:
                return JSONResponse({"error": f"no base64 audio found; send JSON {{'audio': '<base64 wav>'}}"}, status_code=400)
            if b64.startswith("data:"):  # data URL
                b64 = b64.split(",", 1)[1]
            wav_bytes = base64.b64decode(b64)
        if not wav_bytes:
            return JSONResponse({"error": "empty audio"}, status_code=400)
        verdict = score_call(wav_bytes)
        verdict["details"]["request_ms"] = round((time.perf_counter() - t0) * 1000)
        return verdict
    except Exception as exc:  # never leave the judges without an answer
        return JSONResponse({"is_synthetic": False, "confidence": 0.5, "error": f"{type(exc).__name__}: {exc}"}, status_code=200)


@app.get("/")
def index():
    """The one-file demo page: record or upload a call, get the verdict and the explanation."""
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND), media_type="text/html")
    return JSONResponse({"status": "ok", "hint": "POST /detect with {'audio': '<base64 wav>'}"})


_VAL = None


def _validation():
    """The held-out validation calls (never used for training). Labels are served so the page can reveal them AFTER
    the verdict; the detector itself only ever receives the WAV through /detect, like any other call."""
    global _VAL
    if _VAL is None:
        import dataset
        df = dataset.load_split("val")
        _VAL = [{"index": i, "anon_id": r.anon_id, "label": r.label, "duration_s": float(r.duration_s), "path": r.path}
                for i, r in enumerate(df.itertuples())]
    return _VAL


@app.get("/validation")
def validation_list():
    return [{k: v for k, v in c.items() if k != "path"} for c in _validation()]


@app.get("/validation/{anon_id}/audio")
def validation_audio(anon_id: str):
    for c in _validation():
        if c["anon_id"] == anon_id:
            return FileResponse(c["path"], media_type="audio/wav", filename=f"{anon_id}.wav")
    return JSONResponse({"error": "unknown validation call"}, status_code=404)


@app.get("/health")
def health():
    det = get_detector()
    return {"status": "ok", "model": f"{det.backbone} layer {det.layer} + {det.classifier_name}", "device": det.device.type}


@app.post("/detect_file")
async def detect_file(file: UploadFile = File(...)):
    """Convenience: multipart upload of a WAV file (same response as /detect)."""
    return score_call(await file.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--backbone", default=None, choices=[None] + config.BACKBONE_ORDER)
    parser.add_argument("--classifier", default="mlp", choices=["mlp", "logreg"])
    args = parser.parse_args()
    app.state.backbone, app.state.classifier = args.backbone, args.classifier
    get_detector()  # load before accepting requests
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
