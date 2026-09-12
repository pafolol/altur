# TODOS

## Semantic module

### Spike: local ASR fallback (faster-whisper on the first 60 s of the caller)

**What:** Measure faster-whisper (small/medium, int8, CPU) latency and logprob availability on 60 s of compacted 8 kHz Spanish caller audio on the i9; if it fits under ~1.5 s, wire it as the fallback when Scribe fails.

**Why:** Today a Scribe outage or 429 storm means every call abstains. The module has no network-free path.

**Context:** `asr.transcribe(post=...)` takes any callable; a local backend would replace `_post`. The F4 features depend on per-word log-probabilities, so the local ASR must expose them (whisper does via segment/word probs, on a different scale: F4 would need retraining on local transcripts to avoid train/inference mismatch). Deferred by /autoplan on 2026-09-12 as outside the served scope; decide with a measurement, not an argument.

**Effort:** M (human) → S with CC
**Priority:** P2
**Depends on:** None

### 60 s prefix path as a budget degradation

**What:** When the deadline is tight, transcribe only the first 60 s of the caller (F4 only).

**Why:** AUDIT §3 measured AUC 0.915 (60 s) and 0.937 (30 s) vs 0.912 full-call for F4 alone; one API instead of two.

**Context:** Rejected as the main path at the /autoplan gate (the rubric needs the whole call; the difference is within noise on 71 calls). Would be a third rung between `f4` and `abstain` in `server.score_call`.

**Effort:** S
**Priority:** P3
**Depends on:** None

### Dockerfile and deployment recipe

**What:** Container with pinned requirements, `.env` injected, `uvicorn --workers 4`, `/health` probe.

**Why:** The backend team has not said where the modules run; when they do, this is the first thing they ask for.

**Context:** `requirements.txt` is pinned and `model.pkl` records the sklearn version. Payload of the longest call is 11.7 MB base64: any proxy in front needs `client_max_body_size 20m`.

**Effort:** S
**Priority:** P2
**Depends on:** backend infra decision

### ~~Try scribe_v2~~ (done 2026-09-12, rejected)

Re-transcribed the 353 calls (`cache/asr_scribe_v2`), re-ran the rubric, compared under repeated CV (FINDINGS.md):
F4 −0.001, F4+F5 −0.009, and 40 % slower (3.04 s vs 2.19 s for the two channels). `scribe_v1` stays.

### Endpoint auth

**What:** Shared-secret header on `/detect`.

**Why:** Anyone on the network can burn Scribe/Gemini credit.

**Context:** Acceptable risk inside the hackathon network; add before any public exposure.

**Effort:** S
**Priority:** P3
**Depends on:** deployment topology
