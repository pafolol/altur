# The backend, wired to the real detector

Merged from the `backend-esteban` branch. It was written as a serving shell around a detector that did
not exist yet; the detector exists now, and **`DETECTOR_MODE=fusion` points this shell at it**.

```
cd backend
DETECTOR_MODE=fusion SEMANTIC_URL=http://127.0.0.1:8100/detect uvicorn app.main:app --port 8000
```

`GET /fusion` reports which detector is actually answering and what it is made of. `GET /ready` returns
503 until the layers are loaded, so a container never takes traffic before it can answer.

## What changed, and what deliberately did not

Everything this shell was written for still runs, and runs **first**: the request size limit on header and
body (413), the base64 and WAV validation (400), the duration bounds, the minimum caller speech, the
channel checks, the feature warm-up, the request id on every response, and the typed error contract.

Only the inference step is different. Instead of

```python
detector.predict_synthetic_probability(features)   # this backend's own feature vector
```

the validated audio goes to the two-stage fusion: **acoustic and behaviour decide 50 / 50, and the
semantic verifier is consulted only when they do not settle the call**. `app/fusion_bridge.py` is the
whole seam, about a hundred lines, and `app/` is otherwise untouched apart from the branch in `analyze()`
and a third value for `DETECTOR_MODE`.

Two details worth knowing:

- **The WAV is re-encoded, not passed through.** By the time inference happens this backend has already
  decoded, resampled and split the channels, and the behaviour layer requires stereo 16-bit PCM at 8 kHz
  and validates it strictly. Re-encoding the channels this backend produced means the fusion scores
  exactly the audio this backend validated, not the bytes that happened to arrive.
- **`is_synthetic` comes from the fusion, not from `probability >= threshold`.** When no layer can vote,
  the fusion answers an explicit non-flag at 0.5 — and `0.5 >= 0.5` would otherwise accuse a caller on no
  evidence at all. There is a test for exactly that.

This backend's own feature extraction still runs in fusion mode (~100 ms of a ~900 ms request). That is on
purpose: `MIN_CALLER_SPEECH_SECONDS` is enforced from those features, and `/debug/analyze` reports them.

## ⚠ DETECTOR_MODE=mock is worse than it sounds

The mock probability is 0.5, `is_synthetic` is `probability >= threshold`, and `0.5 >= 0.5` is **True**.
So the default mode does not answer "don't know" — it **flags every caller as synthetic** at 50 %
confidence. It is left exactly as it was (it is this backend's own mode, and its README already says the
mock is not a real detection), but it is why `.env.example` now ships `DETECTOR_MODE=fusion`.

## Which service should judges hit?

Either, now — they answer the same verdict from the same `src/fusion.py`:

| | `python src/server.py` | `uvicorn app.main:app` with `DETECTOR_MODE=fusion` |
|---|---|---|
| verdict | the real fusion | the same real fusion |
| request size limit, `/ready`, request ids, typed errors | no | **yes** |
| the fusion console and the inspector pages | **yes** | no |
| the admin panel's `/api/*` and the call log | **yes** | no |

If the judges get one URL and nothing else, this backend is the better-behaved front door. If the demo
matters, `src/server.py` is the one with the pages. Running both is fine — they are the same detector.

## Still no database here

`app/schemas.py` is Pydantic models, not tables. The call log that backs the admin panel lives in
`src/store.py` and is written by `src/server.py`; requests served by this backend are not recorded in it.
