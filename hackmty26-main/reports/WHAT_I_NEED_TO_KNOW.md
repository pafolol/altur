# What I need to know to own and present the Behaviour module

This is your handoff, not a script for pretending to understand. You should be
able to trace one feature from a synthetic timeline to the score, explain one
mistake, and describe what the evaluation does **not** establish.

**Integration is frozen:** use [FINAL_BEHAVIOR_FREEZE.md](FINAL_BEHAVIOR_FREEZE.md)
and read [FINAL_DELTA_REVIEW.md](FINAL_DELTA_REVIEW.md). The trained 24-feature
artifact is unchanged after ASTRA2; the insufficient-evidence Boolean fallback and
malformed RIFF error handling were corrected. Retraining commands below document
the historical development workflow, not permission to optimize the 71 validation calls.

## 1. The shortest correct explanation

We look at **when the caller interacts with the bank agent**, not at the sound of
their voice or the words they say. An offline VAD converts both audio channels
into speech segments. We detect temporal responses, talk-over and long gaps,
summarize them with 24 numbers, and feed those numbers to a small calibrated
logistic model. A separate quality score tells fusion how much evidence existed.

**Final validation:** 69/71 correct, one human false positive and one synthetic
false negative, AUC 0.9913, Brier 0.0406. This is official speaker-disjoint
validation, **not hidden judging**. The development process used validation for
finite model choices, so the result is not an unbiased final-test guarantee.

## 2. Architecture you should draw from memory

```text
WAV bytes (stereo PCM16, 8kHz)
  → caller channel 0 and agent channel 1
  → frozen Silero VAD (CPU; no API)
  → merged speech intervals
  → responses / interruptions / barge-ins / interior silence events
  → 24 temporal measurements + explicit missingness
  → training-fitted preprocessing
  → L2 logistic regression
  → sigmoid calibrated on training out-of-fold logits
  → synthetic probability + quality → team fusion
```

Do not average the stereo channels. Without the agent channel, you do not know
which event the caller was reacting to. Do not attach an ASR/LLM just to make
the architecture sound advanced: the complete MVP has neither.

## 3. What each source file does

| File | Responsibility |
| --- | --- |
| `behavior/__init__.py` | Public `from behavior import BehaviorDetector` interface |
| `behavior/config.py` | Fixed event/VAD defaults, feature version, project root and seed |
| `behavior/audio.py` | Strict WAV decode, stereo preservation, size/rate checks; test encoder |
| `behavior/vad.py` | Offline ONNX speech probabilities, state isolation, timestamp postprocessing and stability |
| `behavior/turns.py` | `Turn`, clipping/merging, interval intersections, four speaking states |
| `behavior/events.py` | Precise directional onsets, normal responses, gap events, censored recoveries |
| `behavior/features.py` | Numeric statistics, explicit feature families, missing values and evidence quality |
| `behavior/dataset.py` | Local manifest/audio lookup and cache/official-split checks; never runtime identification |
| `behavior/model.py` | Training-only sklearn pipeline, support selector and JSON exporter |
| `behavior/portable.py` | Small NumPy implementation of the exported classifier and preprocessing |
| `behavior/calibrate.py` | Fold-fitted pipelines and train-only sigmoid calibration |
| `behavior/metrics.py` | Synthetic-positive metrics, confusion matrices, calibration bins and paired bootstrap |
| `behavior/inference.py` | `BehaviorDetector.predict(wav_bytes)`; score/quality and optional diagnostics |
| `behavior/decision.py` | Explicit evidence state and non-flag, 0.5-confidence Boolean fallback for insufficient evidence |
| `behavior/schemas.py` | HTTP input/output contracts |
| `behavior/service.py` | FastAPI readiness, `/behavior`, and demo `/detect`; no request persistence |

### Experiment/maintenance scripts

| Script (`python -m scripts.NAME`) | What it does |
| --- | --- |
| `inspect_data` | Local format, duration, duplicate and acoustic/metadata shortcut audit |
| `extract_features --source organizer` | First fast feature table from provided JSON |
| `train_behavior --source organizer` | All initial dummy/shortcut/linear/tree baselines |
| `analyze_organizer` | Early ablation deltas and organizer error review |
| `download_vad` | Public, pinned Silero download only; no Altur data touched |
| `compare_turns --smoke 3` | Estimate CPU cost from a few training calls |
| `compare_turns --tune --extract-all` | Label-blind VAD setting choice and full local turn comparison |
| `extract_features --source vad` | Same features from our own WAV-derived turns |
| `train_behavior --source vad` | Repeat the ladder with production-like segmentation |
| `finalize_model --exclude-counts` | Fit final support-filtered, count-free linear model and train-only calibration |
| `evaluate_behavior` | Evaluate the frozen JSON artifact on official cached features |
| `run_ablations` | Refit family removals, duration-matched and shuffled-label controls |
| `evaluate_prefixes` | Re-run actual WAV prefixes through the same frozen detector |
| `robustness` | Fixed threshold/jitter/gain/noise probes, no retuning |
| `benchmark_latency --calls 20` | Fresh-process startup and runtime memory/latency |
| `make_figures` | Plots and all final-error timelines/contributions, locally |
| `write_report` | Render full educational report from its template and measured results |
| `predict /path/to/call.wav` | Single-call WAV inference |
| `smoke_service` | Start, test and stop a real loopback-only HTTP service |
| `reproduce` | Entire staged workflow with a measured full-VAD compute gate |
| `verify_deliverables` | Final inventory, feature/model checks, plot/link validity and local data-ignore checks |
| `delta_review_checks` | One-shot cached evidence verification and two fixed nuisance diagnostics; already completed, no tuning |
| `delta_runtime_smoke` | Data-free package, runtime-only Python, OS-denied original dataset access and network egress |
| `finalize_delta_freeze` | Record verified final suite/smoke results and hashes; no model fitting |
| `common.py` | JSON-safe reports and reproducibility metadata helper; not a CLI |

## 4. Commands you will actually need

Start in `/Users/chai/Documents/hack_mty_2026/hackmty26-main`, then:

```bash
source .venv/bin/activate
python -m pytest -q
python -m scripts.evaluate_behavior
python -m scripts.predict /absolute/path/to/call.wav
```

Historical retraining procedure (do not execute for frozen integration/judging):

```bash
python -m scripts.train_behavior --source vad
python -m scripts.finalize_model --exclude-counts
python -m scripts.evaluate_behavior
```

If a future authorized version changes temporal logic or VAD, regenerate its inputs
and independently revalidate. The old development workflow was:

```bash
python -m scripts.compare_turns --tune --extract-all
python -m scripts.extract_features --source vad
python -m scripts.train_behavior --source vad
python -m scripts.finalize_model --exclude-counts
python -m scripts.run_ablations
python -m scripts.evaluate_prefixes
python -m scripts.robustness
python -m scripts.make_figures
python -m scripts.write_report
```

**Important:** the final deployed model is the `--exclude-counts` variant. Running
`finalize_model` without that flag intentionally recreates the earlier 29-feature
candidate. The full reproduction command runs both candidates in the correct order.

Start the service:

```bash
uvicorn behavior.service:app --host 127.0.0.1 --port 8000 --workers 1 --no-access-log
```

Send `application/json` to `/behavior` or `/detect` with
`{"audio_base64":"..."}`. `/detect` is the Behaviour-only demo; the team fusion
layer must replace its final decision logic. See `behavior/README.md` for setup,
Docker commands and the permission-dependent cloud plan.

## 5. Where the important outputs live

- **Production classifier:** `artifacts/models/behavior.json` — inspect it to see
  the actual feature list, medians, standardization, weights, calibration and threshold.
- **Speech detector:** `artifacts/vad/silero_vad.onnx` — public MIT weights, checksum-verified.
- **VAD settings:** `artifacts/vad/config.json`, also frozen into the production model.
- **Baseline models:** `artifacts/models/organizer/`, `artifacts/models/vad/`.
- **Earlier supported candidate:** `artifacts/models/supported_with_counts.json`.
- **Feature tables:** `artifacts/features/organizer.csv`, `artifacts/features/vad.csv`.
- **Final results:** `reports/metrics/final_model.json`, `frozen_evaluation.json`.
- **Experiments:** `reports/experiments.csv` and detailed model JSON files in `reports/metrics/`.
- **Errors/per-call diagnostics:** `reports/private/` (confidential local derivatives).
- **Plots:** `reports/figures/`; explanation: `reports/BEHAVIOUR_REPORT.md`.

The original manifest, organizer turns and WAV files were preserved. Git ignores
the dataset, caches, artifacts, environments and `.env`. That is not permission
to upload a derived report or model arbitrarily: keep project confidentiality and
ask before external transfer. Never push the entire home directory repository.

## 6. Understand one event and one feature

Draw caller `[1,4)` and agent `[3.5,5)`. The agent starts over an already active
caller at 3.5. Overlap is `4−3.5=0.5 s`; caller stop latency is also 0.5 s. The
caller yields because it ends before the agent. If the caller resumes at 5.3,
the recovery is 1.3 s from caller stop and the gap is 0.3 s from agent end.

Now draw the agent ending at 10.0 and caller beginning at 10.35, without intervening
speech: response latency is 0.35 s. Three latencies `[0.2,0.4,0.6]` have mean 0.4 s,
sample standard deviation 0.2 s, IQR 0.2 s, CV 0.5. A single latency can estimate a
mean but **not** our required repeated-event consistency. No latency is NaN, not 0.

At a prefix boundary, a still-active caller has no observed stop. Do not substitute
the cutoff time as a voluntary yield. Read `tests/test_events.py` and change the
synthetic timeline by hand to predict what should happen before running the test.

## 7. How to read the scores and metrics

```python
from behavior import BehaviorDetector
d = BehaviorDetector()
r = d.predict(wav_bytes)
```

- `synthetic_probability`: calibrated model score, synthetic=positive.
- `quality_score`: heuristic sufficiency of observed interaction; not accuracy.
- `event_count`: normal responses + agent interruptions + caller barge-ins.
- `behavior_confidence`: `quality * abs(2*p−1)`, heuristic decisiveness, not calibrated correctness.

Never say “quality 0.6 means 60% accurate.” No usable interaction returns p=0.5,
quality=0. In the corrected `/detect` demo, insufficient evidence returns
`is_synthetic=false, confidence=0.5`, with `X-Behavior-Evidence: insufficient_evidence`.
This means **no Behaviour flag**, not evidence of humanity. Fusion should consume
the four-field `/behavior` result, give zero-quality Behaviour no positive vote,
and separately resolve the final decision. Available evidence keeps the frozen
`p>=0.5` rule. Debug output exposes the explicit `evidence_state`.

The final validation confusion matrix is:

| True ↓ / predicted → | Human | Synthetic |
| --- | --- | --- |
| Human | 36 | 1 false positive |
| Synthetic | 1 false negative | 33 |

A false positive inconveniences a human; a false negative misses synthetic evidence.
Balanced accuracy averages both class recalls. ROC-AUC measures ranking, not the
quality of a chosen threshold. Brier evaluates probability accuracy. Calibration
can improve Brier without changing AUC. EER is an interpolated tradeoff summary,
not today's deployed error threshold. The report defines every metric.

## 8. What the evidence supports — and what it does not

You can defend **temporal predictive information in this dataset**, direct WAV
inference, preserved official splits, a shortcut audit, and a CPU working pipeline.

You cannot defend “all humans recover messily and all machines consistently.”
Synthetic response variability was often higher. Only 13 training calls supported
interruption variability; none supported repeated recovery variability. Those
unsupported features were excluded. Removing interruption features barely changes
AUC; response and long-gap timing are more useful here.

The global shortcut model reaches about 0.909 AUC. Final Behaviour improves over
it, including a duration-matched subset, but some workflow/missingness/VAD artifacts
may remain. VAD itself is acoustic, even though our classifier does not use raw
acoustic embeddings. Never call the three modules statistically independent.

The ASTRA2 delta diagnostics add detail: agent-only marginal timing gives 0.8776
validation AUC; missingness-only gives 0.6292. The 15 missingness indicators account
for about 19% of absolute standardized logit contributions on validation, so they
matter but are not the whole signal. This is descriptive arithmetic, not causal proof.
Caller VAD/reference IoU differs by class (validation human 0.771 vs synthetic 0.866).

Historical 15/30/60 s balanced accuracies were 51.23%/62.16%/82.31%. Correcting
the non-flag fallback changes Boolean accuracies to 63.43%/58.74%/82.31%, with
32/10/0 abstentions. No probabilities or thresholds were retrained. These remain
weak early-decision results, not evidence of real-time early detection.
An autonomous-caller benchmark does not establish detection of human-controlled
voice conversion or every synthetic-voice attack.

## 9. Common failures and how to debug them

| Symptom | Check / fix |
| --- | --- |
| `ModuleNotFoundError` | Activate `.venv`, run from repository root, sync the correct dependency lock |
| `Offline VAD model missing` | Run `python -m scripts.download_vad` during setup; this downloads public weights only |
| Model JSON missing | Extract VAD features and run `finalize_model --exclude-counts` |
| VAD checksum mismatch | Do not disable the check; restore the pinned public model and its correct path |
| Feature configuration mismatch | Code and artifact disagree; regenerate features/retrain after bumping feature version |
| Stale cache or split mismatch | Regenerate from original manifest/WAVs; never edit split/label columns to silence the error |
| 422 HTTP response | Field must be `audio_base64`; use valid base64 of an actual stereo 8k PCM16 WAV |
| 415 response | Send `Content-Type: application/json` |
| 413/over-duration error | Request exceeds configured bound; adjust only after measuring memory and confirming contract |
| p=0.5, quality=0 | Check both channels contain detected speech and at least two usable interaction events |
| Strange early synthetic score | Full-call model is poorly calibrated on 15/30 s prefixes; do not retune threshold blindly |
| High score with unexpected mistakes | Read timing features and plot/error reports; VAD, workflow or domain shift may be responsible |
| Slow service | Load the detector once; keep CPU threads/worker counts bounded; measure audio duration and VAD timing |
| Container cannot start | Docker build is unverified locally; verify artifacts are in allowlisted context and install the runtime lock |

Use `--debug` locally to inspect features and `diagnostics.timing_ms`. Compare
`reports/figures/vad_agreement.png` and local error timelines. Do not print base64,
raw audio or transcripts into shared logs. No transcription is needed to debug
most interval mistakes.

## 10. Parameters you may change safely versus revalidation boundaries

**Usually operational:** file paths to the same artifacts, HTTP port, log level
(without request bodies), plot size, Markdown wording. Paths can change without
changing numerical inference. CPU worker/concurrency settings require new latency
and memory checks even if model accuracy is unaffected.

**Require feature regeneration and revalidation:** VAD threshold, hysteresis,
speech/silence duration, padding, merge gap, onset tolerance, overlap cutoff,
event rules, context cutoff, missingness rules, variability minimum and feature
families. Updating `VADConfig` alone does not modify a detector loaded from an
already trained artifact: its saved configuration is authoritative.

**Require model/calibration revalidation:** regularization C, support threshold,
feature selection, calibrator method, class weighting, deployment threshold, quality
formula or sparse-evidence policy. A quality change affects fusion even if the raw
classifier output is unchanged. Do not tune these during the judging window.

**Never change for convenience:** channel order, label orientation, official split,
training-only preprocessing discipline, model checksum validation, or permission
boundaries. Do not merge train and validation to obtain a nicer number.

## 11. Cloud and privacy facts for the presentation

The sponsor guide lists Vultr; no VM, cloud account, credits, payment or API key
was used. CPU measurements do not justify a GPU for this module. Docker support
is supplied, but Docker is not installed on the Mac, so image build/run is not
verified. Actual local Python inference, tests and loopback HTTP are verified.

No confidential audio, transcript or dataset-record uploads occurred. Public
package/Python/model downloads are different from uploading the dataset. External
STT/LLM annotation, cloud provisioning/transfer, credentials, remote Git changes
or data deletion require explicit user permission.

## 12. Five exercises before you present

1. Explain the `[1,4)` caller / `[3.5,5)` agent test without looking at the code.
2. Point to the exact allowlist that prevents filenames and labels becoming inputs.
3. Explain why a scaler fitted on train+val leaks, even without labels.
4. Open the ablation graph and describe a negative result honestly.
5. Read one anonymized error example and explain its arithmetic without claiming
   you know the caller's intent. Then explain why 30-second inference is a limitation.

If you can do those five things, you can explain, defend and begin modifying the
module rather than merely presenting generated code.
