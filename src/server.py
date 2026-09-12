"""
CHALLENGE ENDPOINT -  POST /detect

Request body (JSON):  {"audio": "<base64 of a stereo 8 kHz WAV file>"}
    Also accepted: {"wav": ...}, {"audio_base64": ...}, {"file": ...}, {"data": ...}, or a raw base64 string body,
    or multipart/form-data with a file field - the judges' exact field name is not specified in the PDF, so the
    server is permissive about it.
Response (JSON):      {"is_synthetic": true, "confidence": 0.87}
    is_synthetic  required by the challenge
    confidence    probability that the verdict is right (config.CONFIDENCE_MODE = "verdict")
    Extra diagnostic fields (the per-layer scores, the weights that produced the verdict, latency) are added
    under "details"; they do not interfere with the required contract.

The verdict is now the FUSION of every available detection layer (src/fusion.py): the acoustic model on the
caller's voice and the behaviour model on the conversation timing, weighted 50 / 50 by default. The weights
are live-tunable (POST /fusion/config) and a third layer needs no change here - the endpoints iterate over
the registry.

Run:  python src/server.py [--host 0.0.0.0] [--port 8000] [--acoustic-model robust_v2] [--weight acoustic=0.7]
Test: python src/client_demo.py path/to/call.wav --url http://localhost:8000/detect

Pages:  GET /         the single-call inspector (frontend/index.html)
        GET /fusion   the fusion console: weights, the 71 held-out calls, every layer's score side by side
"""
import argparse
import base64
import json
import time

import uvicorn
from fastapi import FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.datastructures import UploadFile as StarletteUploadFile

import _bootstrap  # noqa: F401
import config
import fusion
from predict import AcousticDetector

app = FastAPI(title="Altur HackMTY 2026 - synthetic caller detector (acoustic + behaviour fusion)")
# The demo pages are served by this same process at GET / and GET /fusion, so microphone access works over
# localhost and no cross-origin setup is needed; CORS is open so the files also work when opened from disk.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND_DIR = config.PROJECT_ROOT / "frontend"
FRONTEND = FRONTEND_DIR / "index.html"
FUSION_PAGE = FRONTEND_DIR / "fusion.html"
FIELDS = ("audio", "wav", "audio_base64", "file", "data", "audio_b64", "base64", "wav_base64")

# The acoustic models that can sit in the acoustic slot of the fusion. /detect_all and the A/B panel of the
# inspector page score BOTH of them; the fusion uses ONE - app.state.acoustic_model, V1 by default.
DETECTORS = {}
CANDIDATES = {"specialist": "wav2vec2_spanish", "robust_v2": "robust_v2"}

# Defaults so that importing this module (from a test, a notebook) gives a working app without calling main().
app.state.classifier = "mlp"
app.state.primary = None
app.state.acoustic_model = "wav2vec2_spanish"   # V1, the specialist
app.state.fusion = None


# ============================================================================= the fusion


def get_fusion():
    """The single FusionDetector this process serves. Built on first use, layers loaded lazily."""
    if app.state.fusion is None:
        app.state.fusion = fusion.FusionDetector(acoustic_model=app.state.acoustic_model)
    return app.state.fusion


def fusion_config():
    return get_fusion().config


def score_call(wav_bytes, cfg=None):
    """One place where the verdict is produced. Every layer runs; the config decides how they are mixed."""
    det = get_fusion()
    t0 = time.perf_counter()
    results = det.score_layers(wav_bytes)
    verdict = fusion.combine(results, cfg or det.config)
    layers = [r.to_dict() for r in results]
    acoustic = next((r for r in layers if r["key"] == "acoustic"), None)
    verdict["details"] = {
        "synthetic_probability": verdict["synthetic_probability"],
        "layers": layers,
        "branches": {r["key"]: r["probability"] for r in layers},
        "weights": verdict["config"]["weights"],
        "mode": verdict["config"]["mode"],
        "on_abstain": verdict["config"]["on_abstain"],
        "threshold": verdict["config"]["threshold"],
        "n_layers_used": verdict["n_layers_used"],
        "latency_ms": round((time.perf_counter() - t0) * 1000),
        "acoustic_model": app.state.acoustic_model,
    }
    # Fields the original single-model contract exposed, kept so the inspector page - which explains a
    # verdict with the acoustic model's chunk scores - keeps working now that /detect answers with a fusion.
    used = [c for c in verdict["layers"] if c["share"] > 0]
    verdict["details"]["model_display"] = (
        "fusion — " + " + ".join(f"{c['display']} {c['share']:.0%}" for c in used) if used
        else "fusion — no layer is voting")
    if acoustic:
        d = acoustic["details"]
        verdict["details"] |= {"raw_score": d.get("raw_score"), "n_chunks": d.get("n_chunks"),
                               "speech_s": d.get("speech_s"), "duration_s": d.get("duration_s"),
                               "n_speech_regions": d.get("n_speech_regions"), "vad": d.get("vad"),
                               "model": d.get("model"), "calibration": d.get("calibration", "none"),
                               "chunk_scores": d.get("chunk_scores", []),
                               "chunk_spans": d.get("chunk_spans", [])}
    if verdict["note"]:
        verdict["details"]["warning"] = verdict["note"]
    return verdict


# ============================================================================= single-model access (A/B panel)


def available_models():
    return [n for n, d in CANDIDATES.items() if (config.MODELS_DIR / d / "meta.json").exists()]


def default_model():
    """The same pointer predict.py reads, so the single-model endpoints and the report never disagree."""
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


def score_acoustic_only(wav_bytes, model=None):
    """One acoustic model, no fusion: what /detect_all and the A/B panel compare."""
    det = get_detector(model)
    result = det.predict_wav(wav_bytes)
    verdict = det.verdict(result)
    verdict["details"] = {
        "synthetic_probability": round(result["synthetic_probability"], 4),
        "raw_score": round(result["score"], 3), "n_chunks": result["n_chunks"],
        "speech_s": round(result["speech_s"], 1), "duration_s": round(result["duration_s"], 1),
        "latency_ms": round(result["timings"].get("total_s", 0) * 1000),
        "model": f"{det.backbone} layer {det.layer} + {det.classifier_name}",
        "model_key": next((k for k, v in CANDIDATES.items() if v == det.backbone), det.backbone),
        "vad": det.vad, "model_display": f"{det.display}, hidden layer {det.layer}, frozen + MLP",
        "calibration": det.calibration.get("method", "none"), "threshold": config.DECISION_THRESHOLD,
        "n_speech_regions": result["n_speech_regions"], "chunk_scores": result.get("chunk_scores", []),
        "chunk_spans": result.get("chunk_spans", []),
        "branches": {"acoustic": round(result["synthetic_probability"], 4)},
    }
    if "warning" in result:
        verdict["details"]["warning"] = result["warning"]
    return verdict


# ============================================================================= request helpers


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


async def _payload(request):
    body = await request.body()
    try:
        return await request.json()
    except Exception:
        return body.decode("utf-8", errors="ignore").strip().strip('"')


async def _wav_from(request):
    b64 = _extract_base64(await _payload(request))
    if b64 is None:
        return None
    if b64.startswith("data:"):  # data URL
        b64 = b64.split(",", 1)[1]
    return base64.b64decode(b64)


# ============================================================================= the contract


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
            wav_bytes = await _wav_from(request)
            if wav_bytes is None:
                return JSONResponse({"error": "no base64 audio found; send JSON {'audio': '<base64 wav>'}"}, status_code=400)
        if not wav_bytes:
            return JSONResponse({"error": "empty audio"}, status_code=400)
        verdict = score_call(wav_bytes)
        verdict["details"]["request_ms"] = round((time.perf_counter() - t0) * 1000)
        return verdict
    except Exception as exc:  # never leave the judges without an answer
        return JSONResponse({"is_synthetic": False, "confidence": 0.5, "error": f"{type(exc).__name__}: {exc}"}, status_code=200)


@app.post("/detect_layers")
async def detect_layers(request: Request):
    """
    Every layer's own score plus the fused verdict, for the fusion console.

    Optional body keys alongside the audio: "config" (a partial FusionConfig) scores this one call with
    different weights without changing the server's own configuration.
    """
    payload = await _payload(request)
    wav_bytes = await _wav_from(request)
    if wav_bytes is None:
        return JSONResponse({"error": "no base64 audio found"}, status_code=400)
    cfg = fusion_config()
    if isinstance(payload, dict) and payload.get("config"):
        try:
            cfg = fusion.FusionConfig.from_dict(payload["config"], cfg)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=400)
    t0 = time.perf_counter()
    det = get_fusion()
    results = det.score_layers(wav_bytes)
    out = fusion.combine(results, cfg)
    out["layer_details"] = [r.to_dict() for r in results]
    out["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return out


@app.post("/fusion/recombine")
async def recombine(request: Request):
    """
    Re-decide from scores that already exist. No model runs.

    Body: {"layers": [ <LayerResult dicts, or {key, probability, quality, abstained}> ], "config": {...}}
       or {"calls": [{"id": "...", "layers": [...]}, ...], "config": {...}}  for a whole batch at once.

    This is what the weight sliders call: the page scores the 71 calls once, then re-tunes as often as it
    likes. One combiner, one source of truth - the page never re-implements the arithmetic.
    """
    payload = await _payload(request)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "send a JSON object"}, status_code=400)
    try:
        cfg = fusion.FusionConfig.from_dict(payload.get("config"), fusion_config())
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    if "calls" in payload:
        return {"config": cfg.to_dict(),
                "calls": [{"id": c.get("id"), **fusion.combine(c.get("layers") or [], cfg)}
                          for c in payload["calls"]]}
    return fusion.combine(payload.get("layers") or [], cfg)


# ============================================================================= configuration


@app.get("/layers")
def layers():
    """The registry: what detection systems exist, whether they loaded, and their current weight."""
    det = get_fusion()
    return {"layers": det.describe(), "config": det.config.to_dict(),
            "acoustic_model": app.state.acoustic_model,
            "modes": list(fusion.COMBINE_MODES), "abstain_policies": list(fusion.ABSTAIN_POLICIES)}


@app.get("/fusion/config")
def get_config():
    return fusion_config().to_dict()


@app.post("/fusion/config")
async def set_config(request: Request):
    """Live-tune the weights / combine mode / abstention policy. Takes a partial config."""
    payload = await _payload(request)
    if not isinstance(payload, dict):
        return JSONResponse({"error": "send a JSON object"}, status_code=400)
    det = get_fusion()
    try:
        det.config = fusion.FusionConfig.from_dict(payload, det.config)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return det.config.to_dict()


@app.post("/detect_all")
async def detect_all(request: Request):
    """
    The A/B endpoint: the same audio scored by every ACOUSTIC model, so the page can put them side by side.
    Not part of the challenge contract - /detect is, and its shape is unchanged.
    """
    wav_bytes = await _wav_from(request)
    if wav_bytes is None:
        return JSONResponse({"error": "no base64 audio found"}, status_code=400)
    out = {}
    for name in available_models():
        t0 = time.perf_counter()
        try:
            verdict = score_acoustic_only(wav_bytes, name)
            verdict["details"]["request_ms"] = round((time.perf_counter() - t0) * 1000)
            out[name] = verdict
        except Exception as exc:
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
    return {"models": out, "primary": getattr(app.state, "primary", None) or default_model()}


@app.get("/models")
def models():
    """What the A/B panel can ask for, and which acoustic model sits in the fusion."""
    rows = []
    for name in available_models():
        det = get_detector(name)
        rows.append({"key": name, "display": det.display, "layer": det.layer, "vad": det.vad,
                     "calibration": det.calibration.get("method", "none"),
                     "trained_on": det.meta.get("train_sets", ["altur_original"]),
                     "in_fusion": CANDIDATES[name] == app.state.acoustic_model,
                     "is_primary": name == (getattr(app.state, "primary", None) or default_model())})
    return rows


# ============================================================================= the held-out calls


@app.get("/")
def index():
    """The single-call inspector: record or upload a call, get the verdict and the explanation."""
    if FRONTEND.exists():
        return FileResponse(str(FRONTEND), media_type="text/html")
    return JSONResponse({"status": "ok", "hint": "POST /detect with {'audio': '<base64 wav>'}"})


@app.get("/fusion")
def fusion_page():
    """The fusion console: tune the weights, score all 71 held-out calls, see every layer's own score."""
    if FUSION_PAGE.exists():
        return FileResponse(str(FUSION_PAGE), media_type="text/html")
    return JSONResponse({"error": "frontend/fusion.html is missing"}, status_code=404)


_VAL = None
SCORE_CACHE_FILE = config.OUTPUTS_DIR / "fusion" / "validation_layer_scores.json"
_SCORE_CACHE = None


def _validation():
    """The held-out validation calls (never used for training). Labels are served so the page can reveal them AFTER
    the verdict; the detectors themselves only ever receive the WAV, like any other call."""
    global _VAL
    if _VAL is None:
        import dataset
        df = dataset.load_split("val")
        _VAL = [{"index": i, "anon_id": r.anon_id, "label": r.label, "duration_s": float(r.duration_s), "path": r.path}
                for i, r in enumerate(df.itertuples())]
    return _VAL


def _cache():
    """Per-layer scores of the held-out calls, kept on disk so a restart does not re-score 71 calls."""
    global _SCORE_CACHE
    if _SCORE_CACHE is None:
        _SCORE_CACHE = {}
        if SCORE_CACHE_FILE.exists():
            try:
                _SCORE_CACHE = json.loads(SCORE_CACHE_FILE.read_text())
            except Exception:
                _SCORE_CACHE = {}
    return _SCORE_CACHE


def _cache_key(anon_id):
    return f"{app.state.acoustic_model}:{anon_id}"


def _cache_save():
    SCORE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    SCORE_CACHE_FILE.write_text(json.dumps(_cache(), indent=1))


@app.get("/validation")
def validation_list():
    cached = _cache()
    return [{k: v for k, v in c.items() if k != "path"} | {"scored": _cache_key(c["anon_id"]) in cached}
            for c in _validation()]


@app.get("/validation/{anon_id}/audio")
def validation_audio(anon_id: str):
    for c in _validation():
        if c["anon_id"] == anon_id:
            return FileResponse(c["path"], media_type="audio/wav", filename=f"{anon_id}.wav")
    return JSONResponse({"error": "unknown validation call"}, status_code=404)


@app.get("/validation/{anon_id}/score")
def validation_score(anon_id: str, refresh: bool = False):
    """
    Score ONE held-out call through every layer, reading the WAV from disk instead of asking the page to
    upload it back. The per-layer scores are cached (they do not depend on the weights), so re-tuning the
    fusion afterwards costs nothing: that is what /fusion/recombine is for.
    """
    call = next((c for c in _validation() if c["anon_id"] == anon_id), None)
    if call is None:
        return JSONResponse({"error": "unknown validation call"}, status_code=404)
    cache, key = _cache(), _cache_key(anon_id)
    if refresh or key not in cache:
        t0 = time.perf_counter()
        results = get_fusion().score_layers(open(call["path"], "rb").read())
        cache[key] = {"layers": [r.to_dict() for r in results],
                      "scored_ms": round((time.perf_counter() - t0) * 1000),
                      "acoustic_model": app.state.acoustic_model}
        _cache_save()
    entry = cache[key]
    out = fusion.combine(entry["layers"], fusion_config())
    return {"anon_id": anon_id, "index": call["index"], "label": call["label"],
            "duration_s": call["duration_s"], "layers": entry["layers"],
            "scored_ms": entry["scored_ms"], "cached": not (refresh or False), **out}


@app.get("/validation/scores")
def validation_scores():
    """Everything already scored, in one response - so a reloaded page comes back instantly."""
    cache = _cache()
    out = []
    for c in _validation():
        entry = cache.get(_cache_key(c["anon_id"]))
        if entry:
            out.append({"anon_id": c["anon_id"], "index": c["index"], "label": c["label"],
                        "duration_s": c["duration_s"], "layers": entry["layers"],
                        "scored_ms": entry["scored_ms"]})
    return {"acoustic_model": app.state.acoustic_model, "n_total": len(_validation()),
            "n_scored": len(out), "calls": out}


@app.delete("/validation/scores")
def clear_scores():
    cache = _cache()
    for k in [k for k in cache if k.startswith(f"{app.state.acoustic_model}:")]:
        del cache[k]
    _cache_save()
    return {"cleared": True, "n_scored": 0}


@app.get("/health")
def health():
    det = get_fusion()
    return {"status": "ok", "mode": "fusion",
            "layers": [{"key": l.key, "display": l.display, "loaded": l._loaded} for l in det.layers],
            "weights": det.config.weight_map(), "acoustic_model": app.state.acoustic_model,
            "model": f"fusion of {len(det.layers)} layers",
            "primary": getattr(app.state, "primary", None) or default_model(),
            "available": available_models(),
            "device": config.get_device(verbose=False).type}


@app.post("/detect_file")
async def detect_file(file: UploadFile = File(...)):
    """Convenience: multipart upload of a WAV file (same response as /detect)."""
    return score_call(await file.read())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--acoustic-model", default="wav2vec2_spanish",
                        help="which acoustic model sits in the fusion: wav2vec2_spanish (V1, default) or robust_v2")
    parser.add_argument("--weight", action="append", default=[], metavar="KEY=W",
                        help="starting weight of one layer, e.g. --weight acoustic=0.7 (tunable live afterwards)")
    parser.add_argument("--mode", default="weighted_mean", choices=fusion.COMBINE_MODES)
    parser.add_argument("--model", default=None, choices=[None, *CANDIDATES],
                        help="which single acoustic model the A/B endpoints call primary")
    parser.add_argument("--classifier", default="mlp", choices=["mlp", "logreg"])
    parser.add_argument("--no-preload", action="store_true", help="load the layers on the first request instead")
    args = parser.parse_args()
    app.state.classifier = args.classifier
    app.state.acoustic_model = args.acoustic_model
    if not available_models():
        raise SystemExit("no trained model found under models/")
    app.state.primary = args.model or default_model()

    det = get_fusion()
    if not det.layers:
        raise SystemExit("no detection layer is available: check models/ and behaviour/artifacts/")
    overrides = {k: float(v) for k, v in (w.split("=", 1) for w in args.weight)}
    det.config = fusion.FusionConfig.from_dict({"weights": overrides, "mode": args.mode}, det.config)
    if not args.no_preload:
        det.preload()
    print(f"[server] fusion layers: " +
          ", ".join(f"{l.display} (w={det.config.weight_of(l):.2f})" for l in det.layers))
    print(f"[server] acoustic slot: {app.state.acoustic_model}; A/B models available: {available_models()}")
    # 127.0.0.1, not localhost: uvicorn binds IPv4 only, and on Windows "localhost" resolves to ::1
    # first, which costs a two-second connection timeout on every single request.
    print(f"[server] pages: http://127.0.0.1:{args.port}/  (inspector)   "
          f"http://127.0.0.1:{args.port}/fusion  (fusion console)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
