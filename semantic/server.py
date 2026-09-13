"""Fase 7: POST /detect {"audio": "<base64 stereo 8 kHz WAV>"}
  -> {"is_synthetic", "confidence", "score", "abstain", "reason", "used", "ms"}
score = confidence = P(caller is synthetic), calibrated. Always answers within HARD_S, degrading in order:
"f4+f5" (Scribe + Gemini rubric) -> "f4" (Scribe text features only) -> "abstain" (0.5, with a reason).
Invalid input is a client error (400 / 413), never an abstention.

  request --[413 if > MAX_B64]--> decode (400 if not stereo 16-bit 8 kHz) --> VAD --> score_call   [cpu_pool]
                                                                                 |--> Scribe ch0, ch1, Gemini   [io_pool]
                                                                                 |    (pending futures cancelled at the deadline)
                                       f4 (< MIN_WORDS -> abstain no_speech) --> model.pkl --> JSON
  GET /health -> config, model provenance, keys_present
  GET /        -> demo page (static/index.html): the 71 val calls with audio, truth and out-of-fold score, plus an upload box
  GET /calls   -> those 71 rows;  GET /audio/<id> -> their WAV;  POST /check (raw audio bytes, any format afconvert reads) -> same as /detect
Run: .venv/bin/uvicorn server:app --workers 4
"""
import asyncio, base64, csv, io, json, logging, os, pathlib, pickle, subprocess, tempfile, time, wave
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
try:
    from . import config, asr, features, rubric
    from .model import vector, CONFIG, provenance
    from .vad import vad, SR
except ImportError:  # standalone `uvicorn server:app` run from semantic/
    import config, asr, features, rubric
    from model import vector, CONFIG, provenance
    from vad import vad, SR

BUDGET_S = float(os.environ.get("BUDGET_S", 2.8))      # deadline for the network calls (Scribe, Gemini)
HARD_S = float(os.environ.get("HARD_S", 3.0))          # backstop: the pipeline must have degraded by BUDGET_S; this only catches bugs
GEMINI_MIN_S = 0.5                                     # don't start the rubric call with less than this left
MAX_B64 = 20_000_000                                   # 20 MB base64 ≈ 15 MB WAV ≈ 470 s of stereo 8 kHz; the longest call is 11.7 MB
MAX_S = 400                                            # seconds of audio accepted
MIN_WORDS = 5                                          # fewer caller words than this: no evidence, abstain

log = logging.getLogger("semantic")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)   # one JSON line per request, not one per upstream call
config.require_keys()
_pkl = pathlib.Path(__file__).parent / "model.pkl"


class _ModelUnpickler(pickle.Unpickler):
    """Read NumPy 2 model artifacts in the NumPy 1 runtime still used by some demo machines."""

    def find_class(self, module, name):
        return super().find_class(module.replace("numpy._core", "numpy.core"), name)


with open(_pkl, "rb") as _model_file:
    MODELS = _ModelUnpickler(_model_file).load()
_prov = provenance()
_mismatch = {k: (MODELS.get(k), _prov[k]) for k in ("rubric_model", "prompt_id", "scribe_model", "chunk_s", "f4", "agent", "dims") if MODELS.get(k) != _prov[k]}
if _mismatch:
    raise SystemExit(f"model.pkl was trained with {_mismatch} (trained, runtime): retrain (rubric.py all / asr.py all, model.py) or match the env")
if MODELS.get("sklearn") != sklearn.__version__:
    log.warning(json.dumps({"warn": "sklearn version differs from model.pkl", "trained": MODELS.get("sklearn"), "runtime": sklearn.__version__}))
if CONFIG not in MODELS["models"]:
    raise SystemExit(f"SEMANTIC_CONFIG={CONFIG} not in model.pkl ({list(MODELS['models'])})")

app = FastAPI()
ROOT = pathlib.Path(__file__).parent
DATA = ROOT / "data/hackmty26"
# demo page rows: the val split with its out-of-fold score from scores_semantico.csv (scored by a model that never saw the call;
# the live model.pkl is fit on all calls, so scoring these through /detect would not be a held-out check)
VAL = []
if (DATA / "manifest.csv").exists() and (ROOT / "scores_semantico.csv").exists():
    _oof = {r["anon_id"]: float(r["score"]) for r in csv.DictReader(open(ROOT / "scores_semantico.csv"))}
    VAL = [{"id": r["anon_id"], "label": r["label"], "duration_s": int(r["duration_s"]), "score": _oof.get(r["anon_id"])}
           for r in csv.DictReader(open(DATA / "manifest.csv")) if r["split"] == "val"]
cpu_pool = ThreadPoolExecutor(max_workers=32)          # decode + score_call (mostly blocked waiting on io_pool): a queued request burns its own budget, so keep this wide
io_pool = ThreadPoolExecutor(max_workers=128)          # Scribe per channel + Gemini; never shares a pool with its waiter


class Req(BaseModel):
    audio: str


class BadInput(ValueError):
    pass


def decode(b64):
    try:
        raw = base64.b64decode(b64, validate=True)
        w = wave.open(io.BytesIO(raw))
    except Exception as e:
        raise BadInput(f"audio is not base64-encoded WAV: {type(e).__name__}") from e
    with w:
        fmt = (w.getnchannels(), w.getsampwidth(), w.getframerate())
        if fmt != (2, 2, SR):
            raise BadInput(f"expected stereo 16-bit {SR} Hz WAV, got {fmt[0]} channel(s), {8 * fmt[1]}-bit, {fmt[2]} Hz")
        if w.getnframes() > MAX_S * SR:
            raise BadInput(f"audio longer than {MAX_S} s")
        if w.getnframes() == 0:
            raise BadInput("audio has zero frames")
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").reshape(-1, 2)
    return x.astype(np.float32) / 32768.0


def _predict(name, f4, scores, agent=None):
    cols, m = MODELS["models"][name]
    return float(m.predict_proba(np.array(vector(f4, scores, agent))[None, cols])[0, 1])


def score_call(x, deadline, stats):
    """Returns (p, used, reason). Degrades on its own deadlines; cancels what it no longer needs."""
    segs = {ch: vad(x[:, ch], SR) for ch in (0, 1)}
    use_f5 = CONFIG != "F4"
    futs = {0: io_pool.submit(asr.transcribe, x[:, 0], segs[0], SR)}                       # caller: waits for a Scribe slot
    if use_f5:
        futs[1] = io_pool.submit(asr.transcribe, x[:, 1], segs[1], SR, post=asr.post_lowprio)   # agent: yields under load -> f4
    try:
        try:
            r0 = futs[0].result(timeout=deadline - time.perf_counter())
        except Exception as e:
            return 0.5, "abstain", f"asr_error:{type(e).__name__}"
        stats.update(n_chunks=r0["n_chunks"], n_hedged=r0["n_hedged"], n_words=len(r0["words"]))
        if len(r0["words"]) < MIN_WORDS:
            return 0.5, "abstain", "no_speech"
        f4 = features.f4(r0["words"], segs[0])
        p4 = _predict("F4", f4, None)
        if not use_f5:
            return p4, "f4", ""
        try:
            agent = futs[1].result(timeout=deadline - time.perf_counter() - GEMINI_MIN_S)["words"]
            rub = io_pool.submit(rubric.score, asr.annotate({0: r0["words"], 1: agent}))
            futs["rubric"] = rub
            sc = rub.result(timeout=deadline - time.perf_counter())["scores"]
        except Exception as e:
            return p4, "f4", f"degraded:{type(e).__name__}"
        return _predict(CONFIG, f4, sc, features.f_agent(r0["words"], agent)), "f4+f5", ""
    finally:
        for f in futs.values():
            f.cancel()                                   # queued-but-not-started work is dropped; running requests end at their HTTP timeout


@app.get("/health")
def health():
    return {"ok": True, "config": CONFIG, **{k: MODELS.get(k) for k in ("rubric_model", "prompt_id", "scribe_model", "chunk_s", "sklearn")},
            "runtime_sklearn": sklearn.__version__, "keys_present": config.keys_present(), "budget_s": BUDGET_S, "hard_s": HARD_S}


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/calls")
def calls():
    return VAL


@app.get("/audio/{cid}")
def audio(cid: str):
    if not any(r["id"] == cid for r in VAL):            # whitelist: only the val calls, never an arbitrary path
        raise HTTPException(404)
    return FileResponse(DATA / "audio" / f"{cid}.wav", media_type="audio/wav")


def to_stereo_8k(raw):
    """Any audio afconvert reads -> stereo 16-bit 8 kHz WAV bytes. Mono goes to the caller channel (ch0), agent channel silent."""
    with tempfile.TemporaryDirectory() as d:
        src, dst = pathlib.Path(d) / "in", pathlib.Path(d) / "out.wav"
        src.write_bytes(raw)
        r = subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@8000", str(src), str(dst)], capture_output=True, text=True)
        if r.returncode:
            raise BadInput("afconvert could not read the file: " + (r.stderr.strip().splitlines() or ["?"])[-1])
        with wave.open(str(dst)) as w:
            nch, frames = w.getnchannels(), w.readframes(w.getnframes())
    x = np.frombuffer(frames, dtype="<i2").reshape(-1, nch)
    y = np.zeros((len(x), 2), dtype="<i2"); y[:, :min(nch, 2)] = x[:, :2]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes(y.tobytes())
    return buf.getvalue()


@app.post("/check")
async def check(request: Request):
    raw = await request.body()
    if len(raw) > MAX_B64:
        raise HTTPException(413, "file too large")
    try:
        wav = await asyncio.get_running_loop().run_in_executor(cpu_pool, to_stereo_8k, raw)
    except BadInput as e:
        raise HTTPException(400, str(e))
    out = await detect(Req(audio=base64.b64encode(wav).decode()))
    return {**out, "duration_s": round((len(wav) - 44) / (4 * SR), 1)}


@app.post("/detect")
async def detect(req: Req):
    t0 = time.perf_counter(); deadline = t0 + BUDGET_S
    if len(req.audio) > MAX_B64:
        raise HTTPException(413, f"audio exceeds {MAX_B64 // 1_000_000} MB of base64")
    loop = asyncio.get_running_loop()
    stats = {}
    try:
        x = await loop.run_in_executor(cpu_pool, decode, req.audio)
    except BadInput as e:
        raise HTTPException(400, str(e))
    try:
        p, used, reason = await asyncio.wait_for(loop.run_in_executor(cpu_pool, score_call, x, deadline, stats), t0 + HARD_S - time.perf_counter())
    except Exception as e:                               # ponytail: last-resort net so the contract "always answers" holds; everything above already degrades
        log.exception("detect failed")
        p, used, reason = 0.5, "abstain", f"internal:{type(e).__name__}"
    p = float(min(max(p, 0.0), 1.0)); ms = int((time.perf_counter() - t0) * 1000)
    log.info(json.dumps({"used": used, "reason": reason, "ms": ms, "score": round(p, 3), "dur_s": round(len(x) / SR, 1), **stats}))
    abstain = used == "abstain"
    return {"is_synthetic": (not abstain) and p >= 0.5, "confidence": p, "score": p, "abstain": abstain, "reason": reason, "used": used, "ms": ms}
