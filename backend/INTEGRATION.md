# The backend, and how it relates to the fusion

Merged from the `backend-esteban` branch. **Nothing in `app/` has been edited** — this note only records
how the two systems line up, because right now the repository contains *two* FastAPI services that both
serve `POST /detect`, and only one of them can be the answer the judges get.

## What this backend is

An operational shell, and a good one. It has things `src/server.py` does not:

| | `backend/app` | `src/server.py` |
|---|---|---|
| request size limit (413) | yes, `MAX_REQUEST_SIZE_MB`, checked on header *and* body | no |
| request ids / structured logs | yes, `X-Request-ID` on every response | uvicorn's default |
| readiness | `GET /ready` → 503 until warm-up finishes | `/health` only |
| feature warm-up before traffic | yes, `ENABLE_FEATURE_WARMUP` | models loaded at startup |
| input validation | min/max duration, minimum caller speech → 400 | decode errors only |
| typed error contract | 400 / 413 / 422 / 503 / 500, all JSON | 400 + a 200 fallback |

## What it does not have

Its detector and its feature extractors are **placeholders**. `DETECTOR_MODE=mock` — the default — returns
`0.5` for every call and says so in the logs. `app/semantic_features.py` returns `None` and is documented as
"reserved for a future semantic model". `app/fusion.py` implements `single_model`, `weighted` and
`meta_model`, but `weighted` is a plain weighted mean with no abstention handling, no evidence quality and
no verifier stage.

So: **this branch is the serving layer, and `src/fusion.py` is the detector it was written to wrap.** They
were built to meet.

## The seam

`Fusion.predict()` already takes exactly the shape the fusion produces:

```python
scores = {"behavior": 0.97, "acoustic": 1.00, "semantic": None}
```

and `src/fusion.py` produces precisely that, per layer, plus the things this backend's version cannot
express: which layers abstained, how much evidence each had, and whether the verifier was consulted at all.

Wiring them is a small, contained change — replace the body of `analyze()`'s inference step with a call to
`fusion.FusionDetector.score()` and let `combine()` do the mixing, keeping every one of this backend's
guards, limits and error codes. It has **not** been done here, because `app/` is someone else's work and
which service answers `/detect` is a team decision, not a merge decision.

## ⚠ Before submitting: only one `/detect` can be live

| service | port | what it answers today |
|---|---|---|
| `python src/server.py` | 8000 | the real three-layer verdict |
| `uvicorn app.main:app` (this one) | 8000 | **`0.5` for every call**, in `mock` mode |

Both default to port 8000. If the judges are pointed at this one as it stands, every answer is a coin flip
at 50 % confidence. Decide which one is the endpoint before the deadline:

- **keep `src/server.py`** — it is the one with the models — and optionally port this backend's request
  limits, `/ready` and request ids into it; or
- **keep this backend** and wire its `analyze()` to `src/fusion.py` as described above.

Either is a short job. Doing neither is the risk.

## No database here either

Worth stating plainly, since this branch was merged while looking for one: it has no persistence of any
kind. `app/schemas.py` is Pydantic request/response models, not tables; `.env.example` has no connection
string; no migrations, no ORM, nothing in `requirements.txt`. Nothing in this repository writes a verdict
anywhere except the console's per-session cache in `outputs/fusion/`.
