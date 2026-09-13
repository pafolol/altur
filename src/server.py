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
import latency
import store
import twilio_demo
from predict import AcousticDetector

app = FastAPI(title="Altur HackMTY 2026 - synthetic caller detector (acoustic + behaviour fusion)")
# The demo pages are served by this same process at GET / and GET /fusion, so microphone access works over
# localhost and no cross-origin setup is needed; CORS is open so the files also work when opened from disk.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
FRONTEND_DIR = config.PROJECT_ROOT / "frontend"
FRONTEND = FRONTEND_DIR / "index.html"
FUSION_PAGE = FRONTEND_DIR / "fusion.html"
DEMO_PAGE = FRONTEND_DIR / "demo.html"
FIELDS = ("audio", "wav", "audio_base64", "file", "data", "audio_b64", "base64", "wav_base64")

# The acoustic models that can sit in the acoustic slot of the fusion. /detect_all and the A/B panel of the
# inspector page score BOTH of them; the fusion uses ONE - app.state.acoustic_model, V1 by default.
DETECTORS = {}
CANDIDATES = {"specialist": "wav2vec2_spanish", "robust_v2": "robust_v2"}

# Defaults so that importing this module (from a test, a notebook) gives a working app without calling main().
app.state.classifier = "mlp"
app.state.primary = None
app.state.acoustic_model = "wav2vec2_spanish"   # V1, the specialist
app.state.semantic_url = None
app.state.fusion = None


# ============================================================================= the fusion


def get_fusion():
    """
    The single FusionDetector this process serves. Built on first use, layers loaded lazily.

    Every registered layer is included even when it is not answering right now, so the console always
    shows the full registry - a semantic channel whose service is down is a greyed-out fader with its
    share reserved, not a layer that silently vanished. Such a layer abstains at score time and
    combine() hands its weight to the others.
    """
    if app.state.fusion is None:
        app.state.fusion = fusion.FusionDetector(
            layers=fusion.build_layers(app.state.acoustic_model, include_unavailable=True,
                                       semantic_url=app.state.semantic_url))
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
                               "channels": d.get("channels", 2),
                               "chunk_scores": d.get("chunk_scores", []),
                               "chunk_spans": d.get("chunk_spans", [])}
    if verdict["note"]:
        verdict["details"]["warning"] = verdict["note"]
    return verdict


# ============================================================================= detection latency
#
# "Synthetic, 91 %" is half an answer; the other half is WHEN. The acoustic layer scores 4 s chunks of
# caller speech and aggregates their log-odds by mean, so the aggregate of the first k chunks is exactly
# what the deployed model would have answered having heard only those k - same aggregation, same
# calibration. src/latency.py walks that sequence; these endpoints serve it with the audio needed to draw
# it. It is a PREFIX evaluation of a non-streaming model, not a live stream, and the page says so.

DEMO_CACHE = {}
BENCH_FILE = config.OUTPUTS_DIR / "fusion" / "latency_benchmark.json"


def _envelope(samples, buckets=900):
    """A peak envelope for the waveform - what the page draws, not what the model sees."""
    import numpy as np
    n = len(samples)
    if n == 0:
        return []
    edges = np.linspace(0, n, min(buckets, n) + 1).astype(int)
    peaks = [float(np.abs(samples[a:b]).max()) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:])]
    top = max(peaks) or 1.0
    return [round(v / top, 4) for v in peaks]


def _demo_detector(model):
    if model not in DETECTORS:
        DETECTORS[model] = AcousticDetector(CANDIDATES[model], getattr(app.state, "classifier", "mlp"))
    return DETECTORS[model]


def analyse_for_demo(path, model="specialist", hi=None, lo=None):
    """One call, everything the visualiser needs: audio shape, speech regions, chunks, timeline."""
    import audio as audio_mod
    det = _demo_detector(model)
    stereo, sr = audio_mod.read_wav(path)
    caller = audio_mod.get_channel(stereo, config.CALLER_CHANNEL)
    _, spans, regions = audio_mod.caller_chunks(stereo, sr, return_regions=True, vad_method=det.vad)
    t0 = time.perf_counter()
    result = det.predict_wav(path)
    inference_ms = (time.perf_counter() - t0) * 1000
    tl = latency.timeline(result["chunk_scores"], result["chunk_spans"], result["duration_s"],
                          det.calibration, det.aggregation, hi=hi, lo=lo, inference_ms=inference_ms)
    return {
        "model": model, "model_dir": det.backbone, "display": det.display, "vad": det.vad,
        "duration_s": round(result["duration_s"], 2), "sample_rate": sr,
        "envelope": _envelope(caller),
        "speech_regions": [[round(a, 2), round(b, 2)] for a, b in regions],
        "chunk_spans": [[round(a, 2), round(b, 2)] for a, b in result["chunk_spans"]],
        "final": {"probability": round(result["synthetic_probability"], 4),
                  "is_synthetic": bool(result["synthetic_probability"] >= config.DECISION_THRESHOLD),
                  "raw_score": round(result["score"], 3), "speech_s": round(result["speech_s"], 2)},
        **tl,
    }


@app.get("/demo")
def demo_page():
    """The latency visualiser: how early the verdict was reachable, not just what it was."""
    if DEMO_PAGE.exists():
        return FileResponse(str(DEMO_PAGE), media_type="text/html")
    return JSONResponse({"error": "frontend/demo.html is missing"}, status_code=404)


@app.get("/demo/calls")
def demo_calls():
    return [{k: v for k, v in c.items() if k != "path"} for c in _validation()]


@app.get("/demo/call/{anon_id}")
def demo_call(anon_id: str, model: str = "specialist", confident_synthetic: float = None,
              confident_human: float = None):
    call = next((c for c in _validation() if c["anon_id"] == anon_id), None)
    if call is None:
        return JSONResponse({"error": "unknown call"}, status_code=404)
    if model not in CANDIDATES:
        return JSONResponse({"error": f"unknown model; try {list(CANDIDATES)}"}, status_code=400)
    key = (anon_id, model, confident_synthetic, confident_human)
    if key not in DEMO_CACHE:
        DEMO_CACHE[key] = analyse_for_demo(call["path"], model, confident_synthetic, confident_human)
    # The label rides along so the page can show whether the call was really synthetic - AFTER the
    # verdict, the same way the inspector's deck does. The model never receives it.
    return DEMO_CACHE[key] | {"anon_id": anon_id, "label": call["label"], "index": call["index"]}


@app.get("/demo/benchmark")
def demo_benchmark(model: str = "specialist", refresh: bool = False):
    """
    Detection latency across the whole held-out split, measured rather than asserted.

    ~45 s the first time (it scores 71 calls); cached on disk afterwards.
    """
    if model not in CANDIDATES:
        return JSONResponse({"error": f"unknown model; try {list(CANDIDATES)}"}, status_code=400)
    cache = {}
    if BENCH_FILE.exists() and not refresh:
        try:
            cache = json.loads(BENCH_FILE.read_text())
        except Exception:
            cache = {}
    if model in cache and not refresh:
        return cache[model]
    det = _demo_detector(model)
    timelines, labels, notable = [], [], []
    for c in _validation():
        r = det.predict_wav(c["path"])
        tl = latency.timeline(r["chunk_scores"], r["chunk_spans"], r["duration_s"],
                              det.calibration, det.aggregation)
        timelines.append(tl)
        labels.append(c["label"])
        ps = [s["probability"] for s in tl["steps"]]
        m = tl["metrics"]
        notable.append({
            "anon_id": c["anon_id"], "label": c["label"], "n_chunks": len(ps),
            # how far the running verdict travelled: 0 = certain from the first chunk
            "spread": round(max(ps) - min(ps), 4) if ps else 0.0,
            "first": round(ps[0], 4) if ps else None, "final": round(ps[-1], 4) if ps else None,
            "early_agrees_with_final": m["early_agrees_with_final"],
            "confident_after_caller_speech_sec": m["confident_detection_after_caller_speech_sec"],
        })
    # Which calls are worth looking at, decided by the numbers rather than by hand: the ones where the
    # running verdict actually moved, and the ones where the early commitment did not survive the call.
    gradual = sorted([n for n in notable if n["n_chunks"] >= 3], key=lambda n: -n["spread"])[:6]
    disagreed = [n for n in notable if n["early_agrees_with_final"] is False]
    flat = sum(1 for n in notable if n["spread"] < 0.02)
    out = latency.benchmark(timelines, labels) | {
        "notable": {"most_gradual": gradual, "early_disagreed_with_final": disagreed,
                    "n_flat": flat, "n_total": len(notable)},
        "model": model, "model_dir": det.backbone, "display": det.display,
        "thresholds": {"decision": config.DECISION_THRESHOLD,
                       "confident_synthetic": config.CONFIDENT_SYNTHETIC_THRESHOLD,
                       "confident_human": config.CONFIDENT_HUMAN_THRESHOLD},
        "generated": time.strftime("%Y-%m-%d %H:%M:%S")}
    cache[model] = out
    BENCH_FILE.parent.mkdir(parents=True, exist_ok=True)
    BENCH_FILE.write_text(json.dumps(cache, indent=1))
    return out


# ============================================================================= the live-call demo
#
# Dial a real number, record what the person says, score it. The only path in this repository where the
# audio has actually been through a telephone network, so it uses ROBUST V2 rather than V1 - see
# src/twilio_demo.py. Unconfigured, /twilio/config says exactly which variables are missing and the
# button on the console explains itself; nothing else in the server is affected.


def get_twilio():
    return twilio_demo.get_demo(app.state.semantic_url)


@app.get("/twilio/config")
def twilio_config():
    return get_twilio().describe()


@app.post("/twilio/call")
async def twilio_call(request: Request):
    """Body: {"to": "+52...", "seconds": 30}. Returns immediately; poll /twilio/status/{call_sid}."""
    payload = await _payload(request)
    payload = payload if isinstance(payload, dict) else {}
    try:
        job = get_twilio().start(payload.get("to"), max_seconds=int(payload.get("seconds", 30)))
    except (RuntimeError, ValueError) as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        return JSONResponse({"error": f"{type(exc).__name__}: {exc}"}, status_code=502)
    return job.to_dict()


@app.get("/twilio/status/{call_sid}")
def twilio_status(call_sid: str):
    job = get_twilio().status(call_sid)
    if job is None:
        return JSONResponse({"error": "unknown call"}, status_code=404)
    # Log it once, the moment it is decided, so a demo call lands in the same call log as everything else
    # and shows up in the admin panel beside the uploads. `logged` lives on the job, so polling the status
    # repeatedly - which the page does - cannot write the same call twice.
    demo = get_twilio()
    live = demo.jobs.get(call_sid)
    if live is not None and live.state == "done" and live.verdict and not live.logged:
        v = live.verdict
        store.record({"is_synthetic": v["is_synthetic"], "confidence": v["confidence"], "decisive": True,
                      "layers": v["layers"],
                      "details": {"synthetic_probability": v["synthetic_probability"],
                                  "latency_ms": 0, "duration_s": live.duration_s,
                                  "layers": v["layers"]}},
                     duration_s=live.duration_s, channels=1, queue="twilio",
                     call_id="tw" + call_sid[-10:])
        live.logged = True
        job = live.to_dict()
    return job


# ============================================================================= the admin panel's API
#
# web/ (the landing page + admin panel) defines an `IsisiApi` interface in src/admin/api.ts and ships a
# seeded mock behind it. These four endpoints are the real implementation of that interface, and the
# field names below are ITS names, not this server's: `conversational` is the behaviour layer, and
# `evidence` is the three modality scores. Where this system genuinely does not have something the panel
# asks for, it says so rather than inventing a number - see /api/health.


def _evidence(layers):
    """The panel's three-modality shape. A layer that abstained or was never asked reads 0."""
    by = {l["key"]: l for l in layers}
    def score(key):
        l = by.get(key)
        if not l or l.get("abstained") or l.get("scored") is False:
            return 0.0
        return round(float(l["probability"]), 4)
    return {"acoustic": score("acoustic"), "conversational": score("behaviour"),
            "semantic": score("semantic")}


def _call_row(r):
    # duration_s is whole seconds on purpose: the panel's formatter renders it as m:ss and the mock it
    # was written against always produced integers, so a raw float shows up as "2:22.80000000000001".
    return {"id": r["id"], "at": r["at"], "duration_s": round(r.get("duration_s") or 0),
            "channels": r.get("channels") or 2, "verdict": r["verdict"],
            "confidence": round(float(r["confidence"]), 4),
            "evidence": _evidence(r["layers"]) | {"decided_at_s": r.get("decided_at_s")},
            "latency_ms": round(float(r["latency_ms"])), "trap": r.get("trap") or "none",
            "queue": r.get("queue") or "api"}


@app.get("/api/calls")
def api_calls(range: str = "24h", limit: int = 500):
    hours = {"24h": 24, "7d": 168}.get(range, 24)
    return [_call_row(r) for r in store.recent(hours, limit)]


@app.get("/api/stats")
def api_stats(range: str = "24h"):
    hours = {"24h": 24, "7d": 168}.get(range, 24)
    return {"latency_series": store.latency_series(hours)}


@app.get("/api/health")
def api_health():
    """
    What the panel's health card shows. Two of its fields this system does not have, and they are
    answered honestly rather than dressed up:

      streaming    false - the detector scores a complete call, it does not decide as audio arrives.
      queue_depth  0 - requests are handled synchronously; there is no queue to be deep.

    `calibrated_on` is the real mtime of the deployed acoustic model, not a marketing date.
    """
    import datetime
    det = get_fusion()
    layers = det.describe()
    up = all(l["available"] for l in layers if l["role"] == "primary")
    degraded = up and not all(l["available"] for l in layers)
    h = store.health_stats(24)
    model_dir = config.MODELS_DIR / app.state.acoustic_model
    meta = model_dir / "meta.json"
    calibrated = (datetime.datetime.fromtimestamp(meta.stat().st_mtime, datetime.timezone.utc).date().isoformat()
                  if meta.exists() else "unknown")
    return {
        "endpoint": "up" if up and not degraded else "degraded" if up else "down",
        "model": " + ".join(f"{l['key']} {det.config.weight_of_key(l['key']):.2f}" for l in layers
                            if l["available"]) or "no layer available",
        "calibrated_on": calibrated,
        "streaming": False,
        "uptime_24h": h["uptime_24h"],
        "queue_depth": 0,
        "checked_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "layers": layers,
        "calls_24h": h["n"],
    }


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
    wav_bytes, queue_label = None, "api"
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
            payload = await _payload(request)
            if isinstance(payload, dict) and isinstance(payload.get("queue"), str):
                queue_label = payload["queue"][:40]
            wav_bytes = await _wav_from(request)
            if wav_bytes is None:
                return JSONResponse({"error": "no base64 audio found; send JSON {'audio': '<base64 wav>'}"}, status_code=400)
        if not wav_bytes:
            return JSONResponse({"error": "empty audio"}, status_code=400)
        verdict = score_call(wav_bytes)
        verdict["details"]["request_ms"] = round((time.perf_counter() - t0) * 1000)
        # Logged AFTER the verdict exists, so a store failure can never cost a detection. `queue` is
        # the caller's own label for where the call came from - the admin panel groups by it - and it
        # is optional: anything that just posts audio lands in "api".
        verdict["details"]["call_id"] = store.record(
            verdict,
            duration_s=verdict["details"].get("duration_s"),
            channels=verdict["details"].get("channels", 2),
            queue=queue_label)
        return verdict
    except Exception as exc:  # never leave the judges without an answer
        store.record({"is_synthetic": False, "confidence": 0.5, "decisive": False,
                      "details": {"latency_ms": (time.perf_counter() - t0) * 1000}}, ok=False)
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
    """
    The cache holds PER-LAYER scores, so it is only valid for the layer set that produced it. Adding a
    semantic layer (or swapping the acoustic model) changes the key, the old entries are ignored, and
    the console shows those calls as not scored rather than silently missing a column.
    """
    layers = "+".join(l.key for l in get_fusion().layers)
    return f"{app.state.acoustic_model}|{layers}:{anon_id}"


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
        # anon_id is passed on purpose: a layer whose deployed model was refitted on this call answers
        # from its own held-out scores rather than being asked about audio it has already seen.
        results = get_fusion().score_layers(open(call["path"], "rb").read(), anon_id=anon_id)
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
    prefix = _cache_key("").rsplit(":", 1)[0] + ":"   # everything scored by the CURRENT layer set
    for k in [k for k in cache if k.startswith(prefix)]:
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
    parser.add_argument("--semantic-url", default=None,
                        help="the semantic service's /detect (default: $SEMANTIC_URL or http://127.0.0.1:8100/detect)")
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
    app.state.semantic_url = args.semantic_url
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
    print("[server] fusion layers: " +
          ", ".join(f"{l.display} w={det.config.weight_of(l):.2f}"
                    f"{'' if l.available() else ' [NOT ANSWERING - abstains, its share goes to the others]'}"
                    for l in det.layers))
    print(f"[server] acoustic slot: {app.state.acoustic_model}; A/B models available: {available_models()}")
    # 127.0.0.1, not localhost: uvicorn binds IPv4 only, and on Windows "localhost" resolves to ::1
    # first, which costs a two-second connection timeout on every single request.
    print(f"[server] pages: http://127.0.0.1:{args.port}/  (inspector)   "
          f"http://127.0.0.1:{args.port}/fusion  (fusion console)")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
