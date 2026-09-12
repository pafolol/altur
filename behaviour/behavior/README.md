# Conversational Behaviour — Altur HackMTY 2026

Offline CPU module estimating `P(synthetic caller | temporal interaction)` from
**stereo 8 kHz PCM16 WAV bytes**. Caller is channel 0; bank agent is channel 1.
It supplies a score and evidence quality for team fusion.

**Frozen integration version:** see [FINAL_BEHAVIOR_FREEZE.md](../reports/FINAL_BEHAVIOR_FREEZE.md)
for the exact artifact/runtime hashes, and [FINAL_DELTA_REVIEW.md](../reports/FINAL_DELTA_REVIEW.md)
for ASTRA2 findings. Use the saved model; no further validation-driven model selection is authorized.

```text
WAV → separated-channel Silero VAD → turns → temporal events
    → 24 timing features → train-fitted preprocessing → logistic → sigmoid calibration
    → synthetic_probability + quality_score + event_count
```

**Official validation:** 69/71 correct, ROC-AUC 0.9913, Brier 0.0406;
confusion `[[36,1],[1,33]]` with rows/columns human, synthetic. Validation informed
development; these are not hidden-test results. Timing consistency is a tested
feature hypothesis, not proof that a caller is human.

## Setup

Run commands from **`hackmty26-main/`**, not from `behavior/` or the empty
`Documents/Default Project`. On the prepared Mac:

```bash
source .venv/bin/activate
python -m scripts.evaluate_behavior
```

For a clean local installation, the same pinned bootstrap used during development:

```bash
python3 -m venv .bootstrap
.bootstrap/bin/python -m pip install uv==0.8.22
UV_PYTHON_INSTALL_DIR="$PWD/.python" .bootstrap/bin/uv venv --python 3.11.13 .venv
.bootstrap/bin/uv pip sync requirements.lock --python .venv/bin/python
source .venv/bin/activate
```

No CUDA, PyTorch, STT, LLM or API keys are needed. Dependencies and public model
weights download during setup; inference is offline. `requirements-runtime.lock`
is the smaller inference-only environment (NumPy, ONNX Runtime, FastAPI/Uvicorn).

## Required files

Training needs `manifest.csv`, `turns/`, and `audio/` either inside the repository
or as its sibling (`../audio`, the current layout). `dataset.locate_audio` detects
exactly one location; `scripts.extract_features --audio-dir PATH` supports an
explicit directory. Never merge/re-split the official train and validation data.

**Inference only needs code and these two generated artifacts:**

- `artifacts/models/behavior.json` — complete fitted model/preprocessing/calibration/config.
- `artifacts/vad/silero_vad.onnx` — pinned, checksum-verified Silero v6.2.1 (MIT).

Both are already present in this local workspace. If VAD weights are missing:

```bash
python -m scripts.download_vad
```

## Train and reproduce

The commands below preserve the **historical development procedure**, which included
validation-informed family/count-removal and raw-versus-calibrated selection. They
overwrite generated artifacts and do not establish independent validation. Do not
run them for integration or judging; use the frozen artifacts and inference commands.
No complete train-only selection winner can be reconstructed from existing results.

```bash
# Full staged experiment/report pipeline; existing caches are reused safely.
python -m scripts.reproduce

# Classifier-only retraining after valid VAD feature extraction:
python -m scripts.train_behavior --source vad
python -m scripts.finalize_model --exclude-counts

# Frozen model evaluation and critical tests:
python -m scripts.evaluate_behavior
python -m pytest -q
```

On first setup use `python -m scripts.reproduce --download-vad` to explicitly
permit downloading the public VAD artifact. Reproduction measures a small VAD
smoke test and stops if projected full-dataset VAD exceeds 20 minutes. The measured
complete VAD job took about 6.7 minutes on this CPU. Reports/robustness/prefix tests
add several minutes; classifier fitting itself takes seconds or less.

All experimental models are retained under `artifacts/models/organizer/` and
`artifacts/models/vad/`. Per-call caches and diagnostics are confidential local
derivatives, ignored by Git. Never commit or upload the dataset or these caches.

## Inference and Python integration

```bash
python -m scripts.predict /absolute/path/to/call.wav
# Optional local timing diagnostics:
python -m scripts.predict /absolute/path/to/call.wav --debug
```

```python
from behavior import BehaviorDetector

detector = BehaviorDetector()             # initialize once at backend startup
behavior = detector.predict(wav_bytes)   # no labels or organizer turns required
behavior["synthetic_probability"]        # calibrated P(synthetic), positive=1
behavior["quality_score"]                # evidence sufficiency heuristic [0,1]
behavior["event_count"]                  # observed normal/interrupt/barge events
behavior["behavior_confidence"]          # quality*abs(2*p-1), NOT correctness probability

# Let the team's trained fusion consume acoustic/behavior/semantic scores + quality.
# No fixed fusion weights are supplied by this module.
```

`AudioError` reports invalid format/size/truncation. Sparse/no usable interaction
returns neutral `synthetic_probability=0.5` with quality zero. Debug fields are
optional and JSON-safe; `diagnostics.evidence_state` explicitly reports
`insufficient_evidence` or `available`. They never include audio or transcript content.
The four-field `/behavior` contract is unchanged. Fusion must give zero-quality
Behaviour no positive synthetic vote, and may downweight other low-quality evidence.

## HTTP and deployment

```bash
uvicorn behavior.service:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

- `GET /health` — readiness after model load.
- `POST /behavior` — internal four-field score/quality contract.
- `POST /detect` — **Behaviour-only demo**, response `{"is_synthetic":bool,"confidence":float}`.
- JSON request: `{"audio_base64":"<base64-encoded stereo WAV>"}`.
- `/detect` confidence is probability of the chosen class (`p` or `1-p`), using
  threshold 0.5 **when evidence is available**. For zero quality or fewer than two
  events, the Boolean-only demo returns `{"is_synthetic":false,"confidence":0.5}`
  and `X-Behavior-Evidence: insufficient_evidence`. This is a non-flag fallback,
  not evidence of a human or authorization. Other responses set the header to `available`.
  The team fusion layer should consume `/behavior`, not infer evidence from this fallback Boolean.
- Wrong inputs return sanitized 422/415/413 errors. Requests are not persisted.

No secrets are needed. `.env.example` contains model paths, duration limit and
CPU thread settings. Export variables or pass an env file to your process manager;
the module does not automatically load `.env`. Default maximum WAV duration is 600 s.

```bash
python -m scripts.smoke_service
docker build -t altur-behavior:local .
docker run --rm --cpus=2 --memory=1g -p 127.0.0.1:8000:8000 altur-behavior:local
```

Docker is not installed on the development Mac, so image build/run is unverified.
The Docker context is allowlisted to exclude all source/derived dataset files.
Provisioning a Vultr/other cloud VM, obtaining credentials, or transferring
confidential data requires explicit user permission. A CPU VM should suffice
for this module; combined-team resource requirements need separate measurement.

## Results and limitations

- Separate 20-call CPU benchmark: about 0.86 s p50 / 1.19 s p95; ~125 MiB peak RSS.
- Global timing is already predictive; workflow and indirect VAD shortcuts may remain.
- One fixed audit diagnostic found agent-only marginal timing AUC 0.8776 and
  missingness-only AUC 0.6292. Count-free does not mean count/protocol-independent:
  15 missingness indicators, event rates and context-dependent availability remain.
- Very few repeated interruptions; no supported repeated recovery variability.
- Short prefixes are poor: do not treat 15/30 s full-call-model output as reliable.
- Historical probability-threshold prefix BA was 51.23% / 62.16% / 82.31% at
  15/30/60 s. Corrected Boolean fallback BA is 63.43% / 58.74% / 82.31%, with
  32/10/0 abstentions; this policy arithmetic does not establish early detection.
- VAD threshold changes can flip decisions; automatic organizer turns are not ground truth.
- Calibration is dataset-specific. Advanced callers can imitate timing.
- This dataset targets autonomous AI callers. Human-controlled real-time voice
  conversion can have synthetic acoustics and human timing; it is not validated here.
- Do not independently use Behaviour as identity proof, a transaction authorizer,
  or assume independence from Acoustic/Semantic scores.

Read [the full report](../reports/BEHAVIOUR_REPORT.md),
[the student handoff](../reports/WHAT_I_NEED_TO_KNOW.md),
and [the experiment log](../reports/experiments.csv).
