import logging
import time
import uuid
from contextlib import asynccontextmanager

import numpy as np

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .audio_processor import AudioValidationError, decode_base64_audio, load_wav_from_memory, resample_if_needed, split_channels
from .config import Settings, settings
from .detector import Detector, DetectorUnavailable, DetectorIncompatible
from .feature_extractor import extract_features
from .feature_validation import validate_finite_features
from .fusion import Fusion, FusionError
from .fusion_bridge import FusionBridge
from .schemas import DetectRequest, DetectResponse

logging.basicConfig(level=getattr(logging, settings.log_level, logging.INFO), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_feature_warmup():
    sample_rate = 8000
    samples = np.arange(sample_rate, dtype=np.float32) / sample_rate
    caller = (0.15 * np.sin(2 * np.pi * 220 * samples)).astype(np.float32)
    agent = (0.15 * np.sin(2 * np.pi * 180 * samples)).astype(np.float32)
    extract_features(caller, agent, sample_rate, enable_pitch=True)


def create_app(config: Settings = settings) -> FastAPI:
    # ASGI servers run lifespan before serving traffic. The initial value also
    # preserves direct TestClient usage that does not enter the lifespan context.
    feature_ready = True
    try:
        detector = Detector(config.detector_mode, config.model_path, config.model_threshold)
        detector_error = None
    except (DetectorUnavailable, DetectorIncompatible, ValueError) as exc:
        detector = None
        detector_error = str(exc)
        logger.error("Detector unavailable: %s", detector_error)
    try:
        fusion = Fusion(config.fusion_mode, config.fusion_weights, config.fusion_model_path)
        fusion_error = None
    except FusionError as exc:
        fusion = None
        fusion_error = str(exc)
        logger.error("Fusion unavailable: %s", fusion_error)
    # DETECTOR_MODE=fusion: the repository's own detector answers instead of this backend's placeholder.
    # Every guard above and below still runs; only the inference step changes.
    bridge = FusionBridge(config.detector_mode == "fusion", config.acoustic_model,
                          config.semantic_url or None)
    if config.detector_mode == "fusion" and not bridge.available:
        detector_error = bridge.error or "fusion bridge unavailable"
        detector = None

    @asynccontextmanager
    async def lifespan(application):
        nonlocal feature_ready
        if config.enable_feature_warmup:
            feature_ready = False
            logger.info("Starting feature warm-up...")
            warmup_started = time.perf_counter()
            try:
                run_feature_warmup()
                logger.info("Feature warm-up completed in %.0f ms", (time.perf_counter() - warmup_started) * 1000)
            except Exception:
                logger.exception("Feature warm-up failed")
                yield
                return
        try:
            if detector is not None:
                detector.warm_up()
            if bridge.available:
                bridge.warm_up()
            if detector is None or fusion is None:
                logger.error("Backend not ready: detector or fusion unavailable")
            else:
                feature_ready = True
                logger.info("Backend ready")
            yield
        finally:
            pass

    application = FastAPI(title="Altur Voice Deepfake Detector", lifespan=lifespan)

    @application.middleware("http")
    async def request_size_limit(request: Request, call_next):
        request.state.request_id = str(uuid.uuid4())
        logger.info("request received request_id=%s path=%s", request.state.request_id, request.url.path)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > config.max_request_bytes:
                    return JSONResponse(status_code=413, content={"detail": "Request demasiado grande"})
            except ValueError:
                return JSONResponse(status_code=400, content={"detail": "Content-Length inválido"})
        body = await request.body()
        if len(body) > config.max_request_bytes:
            return JSONResponse(status_code=413, content={"detail": "Request demasiado grande"})
        request._body = body
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        logger.info("response request_id=%s status=%s", request.state.request_id, response.status_code)
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(status_code=422, content={"detail": "Request inválido"})

    @application.exception_handler(AudioValidationError)
    async def audio_error(request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @application.get("/health")
    async def health():
        return {"status": "ok"}

    @application.get("/ready")
    async def ready():
        if not feature_ready or detector is None or fusion is None:
            return JSONResponse(status_code=503, content={"status": "not_ready"})
        return {"status": "ready"}

    @application.get("/fusion")
    async def fusion_status():
        """Which detector is actually answering /detect, and what it is made of."""
        return {"detector_mode": config.detector_mode, **bridge.describe()}

    async def analyze(payload: DetectRequest, include_probability=False, request_id="-"):
        started = time.perf_counter()
        if detector is None or fusion is None:
            raise DetectorUnavailable(detector_error or fusion_error or "Detector no disponible")
        encoded = payload.audio if payload.audio else payload.audio_base64
        timing = {}
        decoded_start = time.perf_counter(); raw = decode_base64_audio(encoded); timing["base64_decode_ms"] = (time.perf_counter() - decoded_start) * 1000
        logger.info("audio decoded request_id=%s bytes=%s", request_id, len(raw))
        wav_started = time.perf_counter(); audio, sr = load_wav_from_memory(raw); timing["wav_load_ms"] = (time.perf_counter() - wav_started) * 1000
        resample_started = time.perf_counter(); audio, sr = resample_if_needed(audio, sr); timing["resample_ms"] = (time.perf_counter() - resample_started) * 1000; validate_duration = audio.shape[0] / sr
        if validate_duration > config.max_audio_duration_seconds: raise AudioValidationError("El audio supera la duración máxima configurada")
        caller, agent = split_channels(audio)
        if validate_duration < config.min_audio_duration_seconds: raise AudioValidationError("El audio es demasiado corto")
        features_start = time.perf_counter(); features = extract_features(caller, agent, sr, timing, config.enable_pitch_features); validate_finite_features(features); timing["feature_extraction_ms"] = (time.perf_counter() - features_start) * 1000
        if features["caller_speech_seconds"] < config.min_caller_speech_seconds: raise AudioValidationError("El caller no contiene suficiente voz para clasificar")
        inference_start = time.perf_counter()
        fused = None
        if bridge.available:
            # The fusion decides on the whole call. `is_synthetic` is ITS verdict, not a re-derivation
            # from the probability: when no layer could vote it answers an explicit non-flag at 0.5, and
            # `0.5 >= threshold` would otherwise accuse a caller on no evidence at all.
            fused = bridge.predict(caller, agent, sr)
            probability, is_synthetic, confidence = fused["probability"], fused["is_synthetic"], fused["confidence"]
        else:
            model_probability = detector.predict_synthetic_probability(features) if config.fusion_mode == "single_model" else None
            probability = fusion.predict(single_model=model_probability)
            is_synthetic = bool(probability >= detector.threshold)
            confidence = probability if is_synthetic else 1.0 - probability
        timing["inference_ms"] = (time.perf_counter() - inference_start) * 1000
        logger.info("inference request_id=%s synthetic_probability=%.4f%s", request_id, probability,
                    "" if fused is None else f" layers={ {l['key']: round(l['probability'], 3) for l in fused['layers']} }")
        timing["total_processing_ms"] = (time.perf_counter() - started) * 1000
        logger.info("timing request_id=%s %s", request_id, ", ".join(f"{name}={value:.2f}ms" for name, value in timing.items()))
        result = {"is_synthetic": is_synthetic, "confidence": float(confidence)}
        if include_probability:
            result["probability_synthetic"] = float(probability); result["parameters"] = features; result["timing"] = timing
            if fused is not None: result["fusion"] = fused
        return result

    @application.post("/detect", response_model=DetectResponse)
    async def detect(payload: DetectRequest, request: Request):
        try: return await analyze(payload, request_id=request.state.request_id)
        except DetectorUnavailable as exc: raise RuntimeError(str(exc))

    if config.enable_debug_endpoint:
        @application.post("/debug/analyze")
        async def debug_analyze(payload: DetectRequest, request: Request):
            return await analyze(payload, True, request.state.request_id)

    @application.exception_handler(RuntimeError)
    async def runtime_error(request, exc):
        return JSONResponse(status_code=503 if "Detector" in str(exc) or "modelo" in str(exc).lower() else 500, content={"detail": str(exc)})

    @application.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.exception("Unexpected request failure")
        return JSONResponse(status_code=500, content={"detail": "Error interno inesperado"})

    return application


app = create_app()
