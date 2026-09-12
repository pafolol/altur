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
FIELDS = ("audio", "wav", "audio_base64", "file", "data", "audio_b64", "base64", "wav_base64")

# Two models are served side by side so the demo page can show what a telephone line does to each of them:
#   specialist  the Altur specialist, 100 % on the clean validation split, fitted on the recording pipeline
#   robust_v2   the same frozen backbone with a classifier that has also seen telephone-channel audio
# /detect (the challenge contract) answers with ONE of them - app.state.primary. /detect_all answers with both.
DETECTORS = {}
CANDIDATES = {"specialist": "wav2vec2_spanish", "robust_v2": "robust_v2"}


# Defaults so that importing this module (from a test, a notebook) gives a working app without calling main().
app.state.classifier = "mlp"
app.state.primary = None


def available_models():
    return [n for n, d in CANDIDATES.items() if (config.MODELS_DIR / d / "meta.json").exists()]


def default_model():
    """The same pointer predict.py reads, so the endpoint and the fusion interface never disagree."""
    import json
    names = available_models()
    if not names:
        raise RuntimeError("no trained model found under models/")
    pointer = config.REPORTS_DIR / "deployed_model.json"
    if pointer.exists():
        key = json.loads(pointer.read_text()).get("model")
        if key in names:
            return key
    return "robust_v2" if "robust_v2" in names else names[0]


def get_detector(name=None):
    name = name or getattr(app.state, "primary", None) or default_model()
    if name not in CANDIDATES:
        raise KeyError(f"unknown model '{name}'; available: {available_models()}")
    if name not in DETECTORS:
        DETECTORS[name] = AcousticDetector(CANDIDATES[name], getattr(app.state, "classifier", "mlp"))
    return DETECTORS[name]


def score_call(wav_bytes, model=None):
    """One place where the verdict is produced. Fusion with other branches will plug in here."""
    det = get_detector(model)
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
        "model_key": next((k for k, v in CANDIDATES.items() if v == det.backbone), det.backbone),
        "vad": det.vad,
        "model_display": f"{det.display}, hidden layer {det.layer}, frozen + MLP",
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


@app.post("/detect_all")
async def detect_all(request: Request):
    """
    The demo endpoint: the same audio scored by every loaded model, so the page can put them side by side.
    Not part of the challenge contract - /detect is, and its shape is unchanged.
    """
    body = await request.body()
    try:
        payload = await request.json()
    except Exception:
        payload = body.decode("utf-8", errors="ignore").strip().strip('"')
    b64 = _extract_base64(payload)
    if b64 is None:
        return JSONResponse({"error": "no base64 audio found"}, status_code=400)
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]
    wav_bytes = base64.b64decode(b64)
    out = {}
    for name in available_models():
        t0 = time.perf_counter()
        try:
            verdict = score_call(wav_bytes, name)
            verdict["details"]["request_ms"] = round((time.perf_counter() - t0) * 1000)
            out[name] = verdict
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"models": out, "primary": getattr(app.state, "primary", None) or default_model()}


@app.get("/models")
def models():
    """What the page can ask for, and which one /detect answers with."""
    rows = []
    for name in available_models():
        det = get_detector(name)
        rows.append({"key": name, "display": det.display, "layer": det.layer, "vad": det.vad,
                     "calibration": det.calibration.get("method", "none"),
                     "trained_on": det.meta.get("train_sets", ["altur_original"]),
                     "is_primary": name == (getattr(app.state, "primary", None) or default_model())})
    return rows


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
    return {"status": "ok", "model": f"{det.backbone} layer {det.layer} + {det.classifier_name}",
            "primary": getattr(app.state, "primary", None) or default_model(),
            "available": available_models(), "device": det.device.type}


@app.post("/detect_file")
async def detect_file(file: UploadFile = File(...)):
    """Convenience: multipart upload of a WAV file (same response as /detect)."""
    return score_call(await file.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=None, choices=[None, *CANDIDATES],
                        help="which model answers /detect (default: robust_v2 if it exists)")
    parser.add_argument("--classifier", default="mlp", choices=["mlp", "logreg"])
    parser.add_argument("--preload-all", action="store_true", help="load every model at startup (demo mode)")
    args = parser.parse_args()
    app.state.classifier = args.classifier
    names = available_models()
    if not names:
        raise SystemExit("no trained model found under models/")
    app.state.primary = args.model or default_model()
    for name in (names if args.preload_all else [app.state.primary]):
        get_detector(name)  # load before accepting requests
    print(f"[server] models available: {names}; /detect answers with '{app.state.primary}'")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
