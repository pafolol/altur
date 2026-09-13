# Project file map — Altur HackMTY 2026 synthetic-caller detector

Annotated tree of `D:\altur\acoustic`. Every entry says what the file is for, which lifecycle stage it
belongs to, and what depends on it. Companion to `PROJECT_TECHNICAL_REPORT.md`.

**Role tags**

| Tag | Meaning |
|---|---|
| `PRODUCTION` | on the path a real `/detect` request takes, or serves the product UI |
| `TRAINING` | produced a model checkpoint that exists today |
| `EVALUATION` | measures something; writes to `outputs/` or `reports/` |
| `EXPERIMENTAL` | an experiment that was run and recorded; its result informed a decision but its output is not deployed |
| `LEGACY` | superseded, kept for lineage; nothing runs it |
| `UTILITY` | glue, configuration, helpers |

Sizes are the real byte counts of the working tree.

---

## Top level

```text
D:/altur/
├── acoustic/                    THE PROJECT (everything below is relative to here)
├── hackmty26/                   the official challenge repo: manifest.csv + turns/  (DATA, not ours)
├── altur-challenge-audio/audio/ the 353 unzipped stereo 8 kHz WAVs               (DATA, not ours)
├── channel_dataset/             4.7 GB of generated telephone-channel audio       (GENERATED DATA)
└── hackmty26-altur-challenge.pdf  the challenge brief
```

The dataset never moves. `config.py` points at it by absolute path; nothing in the repository copies or
redistributes the recordings.

---

## `acoustic/` — the repository root

```text
acoustic/
├── config.py                10.0 KB   UTILITY/PRODUCTION  every setting in the system
├── requirements.txt          1.5 KB   UTILITY             pinned dependency set
├── .env / .env.example       0.7/2.4  UTILITY             the one configuration file, git-ignored
├── README.md                37.8 KB   documentation
├── src/                               all Python: pipeline, training, evaluation, server
├── models/                            every trained checkpoint
├── outputs/                           every experiment result and figure
├── reports/                           the written reports and their figures
├── tests/                             the pytest suite (219 tests)
├── frontend/                          three server-rendered HTML pages
├── web/                               the Vite + React product app
├── behaviour/                         the frozen behaviour layer (merged, unedited)
├── semantic/                          the semantic layer (merged, imported in-process)
├── backend/                 LEGACY    a second FastAPI shell, superseded
└── report_assets/                     figures and build scripts for this report
```

### `config.py` — `UTILITY` / `PRODUCTION`

The single source of truth. Every script reads it; nothing hard-codes a threshold.

| Block | What it fixes |
|---|---|
| paths | `PROJECT_ROOT`, `CHALLENGE_DIR`, `AUDIO_DIR`, `MODELS_DIR`, `OUTPUTS_DIR`, `REPORTS_DIR` |
| `load_env()` | dependency-free `.env` reader; a real environment variable always wins (`os.environ.setdefault`) |
| data facts | `SEED=42`, `CALLER_CHANNEL=0`, `AGENT_CHANNEL=1`, `SOURCE_SAMPLE_RATE=8000`, `MODEL_SAMPLE_RATE=16000` |
| segmentation | `VAD_METHOD="energy"`, `VAD_FRAME_S=0.02`, `VAD_HOP_S=0.01`, `VAD_THRESHOLD_DB=15.0`, `VAD_MIN_ABS_DB=-55.0`, `VAD_MIN_SPEECH_S=0.30`, `VAD_MIN_GAP_S=0.30`, `VAD_PAD_S=0.10`, `VAD_CROSSTALK_MARGIN_DB=None` |
| chunking | `CHUNK_SECONDS=4.0`, `MIN_CHUNK_SECONDS=1.0`, `MAX_CHUNKS_PER_CALL=60`, `TARGET_RMS_DB=-26.0` |
| backbones | the three `BACKBONES` entries and `BACKBONE_ORDER` |
| classifier | `LOGREG_C=1.0`, `MLP_HIDDEN=256`, `MLP_DROPOUT=0.3`, `MLP_EPOCHS=60`, `MLP_BATCH_SIZE=64`, `MLP_LR=1e-3`, `MLP_WEIGHT_DECAY=1e-4`, `MLP_PATIENCE=15`, `CHUNK_AGGREGATION="mean"` |
| contract | `CONFIDENCE_MODE="verdict"`, `DECISION_THRESHOLD=0.5` |
| latency policy | `CONFIDENT_SYNTHETIC_THRESHOLD=0.85`, `CONFIDENT_HUMAN_THRESHOLD=0.15` |
| fine-tuning | the `FT_*` block |

---

## `src/` — 40 Python modules

### The production inference path

| File | Size | Role | What it does | Depended on by |
|---|---|---|---|---|
| `_bootstrap.py` | 334 B | `UTILITY` | puts the repo root and `src/` on `sys.path` so `python src/x.py` works | every `src/` script |
| `audio.py` | 12.4 KB | `PRODUCTION` | decode, channel split, VAD, chunking, resampling, loudness normalisation | everything that touches audio |
| `dataset.py` | 3.7 KB | `PRODUCTION`/`TRAINING` | manifest, splits, turn files, turn-derived statistics | training, evaluation, server |
| `predict.py` | 13.3 KB | `PRODUCTION` | `AcousticDetector`: loads a backbone + classifier + calibration, scores one call | `fusion.py`, `server.py` |
| `metrics.py` | 7.5 KB | `PRODUCTION`/`EVALUATION` | `aggregate_chunk_scores`, `evaluate_scores`, calibration metrics | `predict.py`, `latency.py`, every evaluator |
| `fusion.py` | 46.9 KB | `PRODUCTION` | **the fusion layer**: `Layer`, `LayerResult`, `FusionConfig`, `combine()`, the registry, the two-stage gate | `server.py`, `latency.py`, `twilio_demo.py` |
| `server.py` | 41.7 KB | `PRODUCTION` | **the only backend**: `/detect`, the fusion console API, the demo API, Twilio, the admin API, the pages | the product |
| `store.py` | 10.2 KB | `PRODUCTION` | the SQLite call log; scores and timings only, never audio | `server.py` |
| `latency.py` | 17.7 KB | `PRODUCTION`/`EVALUATION` | the prefix walk: what the model would have said after each chunk; the `scene` payload | `server.py`, `twilio_demo.py` |
| `twilio_demo.py` | 17.3 KB | `PRODUCTION` | the live outbound call: dial, record, convert, score with Robust V2 | `server.py` |
| `client_demo.py` | 1.7 KB | `UTILITY` | a one-file client that posts a WAV to `/detect` | manual testing |

### Stage 1–4: the acoustic model V1

| File | Size | Role | What it does |
|---|---|---|---|
| `inspect_dataset.py` | 20.2 KB | `EVALUATION` | the exploratory analysis: formats, hashes, levels, turn statistics, VAD agreement, spectra, the 23-feature shortcut scan, the id-leakage check. Writes `outputs/dataset_inspection/` and six figures |
| `extract_embeddings.py` | 9.6 KB | `TRAINING` | runs the frozen backbone over every chunk, caches mean+std pooled hidden states for **all** layers as float16 `.npy` |
| `train_classifier.py` | 20.4 KB | `TRAINING` | the layer probe, the logistic-regression probe and the `MLP` class; trains and checkpoints |
| `calibrate.py` | 9.3 KB | `TRAINING` | Platt and temperature scaling, chosen by out-of-fold log-loss |
| `benchmark.py` | 20.1 KB | `EVALUATION` | the controlled three-backbone comparison; writes `reports/benchmark.csv` and `best_model.json` |
| `finetune.py` | 18.8 KB | `EXPERIMENTAL` | unfreezes the last 4 transformer layers. **Result: did not beat frozen; not deployed** |
| `augment_train.py` | 10.2 KB | `EXPERIMENTAL` | waveform augmentation at training time. **Result: did not beat baseline; not deployed** |

### The shortcut and sanity investigation

| File | Size | Role | What it does |
|---|---|---|---|
| `sanity_checks.py` | 16.8 KB | `EVALUATION` | label permutation, few-chunk test, voice clustering, leave-one-cluster-out, nearest neighbour, provenance, tiny-train, reversed split |
| `train_shortcut.py` | 6.7 KB | `EVALUATION` | trains logistic regression and gradient boosting on 24 **trivial call statistics** — no audio content. The key shortcut evidence |
| `voice_identity.py` | 12.8 KB | `EVALUATION` | uses a VoxCeleb speaker-verification model as an instrument to ask whether voices repeat across the split |
| `signal_processing_demo.py` | 17.8 KB | `EVALUATION` | generates the teaching figures: waveform, Nyquist, FFT, spectrogram, mel, MFCC, pitch, turn structure |

### The telephone channel and Robust V2

| File | Size | Role | What it does |
|---|---|---|---|
| `phone_channel.py` | 24.8 KB | `TRAINING`/`EVALUATION` | **the telephone channel**: exact G.711 in numpy, 11 ffmpeg codecs, band-limiting, spectral tilt, AGC, Gilbert-Elliott packet loss, clock drift, DTX comfort noise, line noise, cross-correlation alignment |
| `build_channel_dataset.py` | 13.4 KB | `TRAINING` | pushes every call through a randomly sampled channel; writes `D:/altur/channel_dataset` (706 samples, 29.0 h, 4.7 GB) |
| `channel_pilot.py` | 13.5 KB | `EXPERIMENTAL` | the pilot that measured what a channel does before the full dataset was built |
| `channel_figures.py` | 13.7 KB | `EVALUATION` | the channel figures, including the digital-silence / dynamic-range analysis |
| `extract_channel_embeddings.py` | 8.5 KB | `TRAINING` | the same embedding extraction over the channel dataset, with Silero VAD |
| `train_robust.py` | 23.7 KB | `TRAINING` | trains the four robust variants and selects one. **Produced the deployed model** |
| `evaluate_models.py` | 22.7 KB | `EVALUATION` | the 4-model × 6-domain evaluation matrix; writes `reports/deployed_model.json` |
| `stress_test.py` | 18.6 KB | `EVALUATION` | 11 signal perturbations applied to the validation split |
| `external_check.py` | 13.5 KB | `EVALUATION` | 108 out-of-dataset clips: edge-tts, gTTS, Windows SAPI, FLEURS, ASVspoof bonafide |

### The fusion

| File | Size | Role | What it does |
|---|---|---|---|
| `evaluate_fusion.py` | 13.3 KB | `EVALUATION` | all 71 held-out calls through every layer, the per-layer table, the gate statistics |

### Report builders

| File | Size | Role | What it does |
|---|---|---|---|
| `build_report.py` | 31.7 KB | `EVALUATION` | assembles `reports/ACOUSTIC_LEARNING_SUMMARY.{md,pdf}` |
| `build_report_results.py` | 48.9 KB | `EVALUATION` | the results half of that report, and most `outputs/figures/bench_*` plots |
| `report_robust.py` | 40.3 KB | `EVALUATION` | assembles `reports/TELEPHONE_ROBUSTNESS_REPORT.{md,pdf}` |
| `report_sanity.py` | 10.6 KB | `EVALUATION` | the "is 100% believable" section |
| `report_stress.py` | 5.6 KB | `EVALUATION` | the stress-test section |
| `report_external.py` | 5.6 KB | `EVALUATION` | the external / out-of-distribution section |
| `report_augment.py` | 3.8 KB | `EVALUATION` | the augmentation-experiment section |
| `report_utils.py` | 17.2 KB | `UTILITY` | shared PDF/markdown helpers (ReportLab) |

---

## `models/` — every checkpoint

```text
models/
├── wav2vec2_spanish/        THE V1 SPECIALIST  (never overwritten)
│   ├── meta.json            backbone, layer 5, pooling, aggregation, vad, seed, metrics
│   ├── mlp.pt               the deployed V1 head
│   ├── mlp_augmented.pt     EXPERIMENTAL — the augmentation run's head, not selected
│   ├── logreg.joblib        the linear probe
│   ├── calibration_mlp.json    Platt a=0.9025, b=0.8366, fit on 781 call-conditions
│   └── calibration_logreg.json temperature, a=3.0 (slope capped), fit on 71 clean calls
├── robust_v2/               THE DEPLOYED MODEL
│   ├── meta.json            layer 5, vad=silero, variant=balanced, train_sets=[original, channel]
│   ├── mlp.pt               the deployed V2 head (= the "balanced" variant)
│   ├── mlp_balanced.pt      the four variants, kept side by side
│   ├── mlp_channel_only.pt
│   ├── mlp_mixed.pt
│   ├── mlp_orig_only.pt
│   └── calibration_mlp.json
├── robust_balanced/ robust_channel_only/ robust_mixed/ robust_orig_only/
│                            EXPERIMENTAL — per-variant meta + head, for the comparison table
├── wavlm/                   EXPERIMENTAL — the English backbone arm of the benchmark
├── xlsr/                    EXPERIMENTAL — the 300M multilingual arm
└── wav2vec2_spanish_finetuned/  EXPERIMENTAL — the fine-tuning run. Lost to frozen; not deployed
```

`reports/deployed_model.json` is the **single pointer** that decides which of these answers. Both
`predict.py` and `server.py` read it, so the report and the endpoint can never disagree.

---

## `outputs/` — 428 artifacts

```text
outputs/
├── dataset_inspection/   summary.json, call_stats.csv (353 rows × 40 cols), shortcut_features.csv,
│                         vad_sweep.csv, vad_silero_vs_energy.csv, ltas.npy
├── embeddings/           per backbone: {train,val}_{mean,std}.npy, *_index.csv, extraction_stats.json
│   ├── wav2vec2_spanish/ train [4102, 13, 768], val [977, 13, 768]
│   ├── wavlm/            same shapes
│   ├── xlsr/             train [4102, 25, 1024]
│   └── robust/           the channel-domain embeddings + five index CSVs
├── classifier/           per backbone: layer_probe.json, mlp_history.json, results.json, 3 figures
├── calibration/          per backbone: the fitted calibration and its out-of-fold metrics
├── sanity/               wav2vec2_spanish/results.json — all eight checks
├── shortcut/             results.json — the 14 feature-set × classifier rows
├── voice_identity/       xvectors.npy, calls.csv, results.json
├── stress/               per backbone: mlp_stress.json, probe_stress.json
├── robust/               results.json, evaluation_matrix.{csv,json}, dynamic_range.{csv,json}
├── external/             results.json — the 108 out-of-dataset clips, two variants
├── augmentation/         wav2vec2_spanish/results.json
├── finetune/             wav2vec2_spanish/{history,results}.json
├── signal_processing/    examples.json, pitch_stats.csv, examples/
├── fusion/               calls.db (the live call log), val_layer_scores.csv,
│                         val_weight_sweep.json, latency_benchmark.json, validation_layer_scores.json
├── predictions/          cached per-call scores
└── figures/              53 PNGs
```

---

## `reports/`

| File | What it is |
|---|---|
| `EXPERIMENT_LOG.md` | 41 KB — the running record of every stage, in order, with numbers |
| `ACOUSTIC_LEARNING_SUMMARY.{md,pdf}` | 80 KB / 6.4 MB — the three-backbone comparison write-up |
| `TELEPHONE_ROBUSTNESS_REPORT.{md,pdf}` | 29 KB / 3.4 MB — the domain-shift and Robust V2 write-up |
| `ISISI_PRESENTER_GUIDE.{md,pdf}` | 49 KB / 3.6 MB — the presentation narrative with figures |
| `RUNPOD_USAGE.md` | how the GPU work was run off the laptop |
| `benchmark.csv` | the three-backbone results table |
| `model_comparison.csv` | every backbone × classifier × split |
| `shortcut_check.csv` | the 14 shortcut rows |
| `best_model.json` | winner of the backbone benchmark (V1 specialist, layer 5) |
| `deployed_model.json` | **the deployed pointer** — currently `robust_v2` |
| `guide_figures/` | 7 PNGs made for the presenter guide |

---

## `tests/` — 219 tests, all passing

| File | Tests | Covers |
|---|---|---|
| `test_pipeline.py` | 11 | audio round-trip, resampling, normalisation, VAD, chunking, metrics, the `/detect` contract, truncated-backbone equivalence |
| `test_fusion.py` | 65 | the combiner arithmetic, abstention policies, the verifier gate, the registry, latency carry-through |
| `test_latency.py` | 19 | the prefix walk equals `aggregate_chunk_scores`, call-time vs speech-time, never-confident handling |
| `test_phone_channel.py` | — | **G.711 bit-exactness against `audioop` over all 65 536 int16 values**, seen/unseen disjointness |
| `test_channel_dataset.py` | — | no unseen codec family ever reaches the training side |
| `test_stress.py` | — | the perturbations do what they claim |
| `test_store_and_api.py` | 17 | the call-log schema, `decided_at()`, the admin API shapes |
| `test_twilio_demo.py` | 29 | mono→stereo, the TwiML, the state machine, number masking, the warm-up |

---

## `frontend/` — three server-rendered pages (`PRODUCTION`)

| File | Route | What it is |
|---|---|---|
| `index.html` | `GET /` | the Line Inspector: record or upload one call, verdict plus explanation, A/B the two acoustic models, play the 71 held-out calls |
| `fusion.html` | `GET /fusion` | the Fusion Console: weight faders, combine mode, the gate, batch-score all 71, CSV export, live call |
| `demo.html` | `GET /demo` | the original latency visualiser (superseded for demos by `web/demo/`) |

## `web/` — the Vite + React product app (`PRODUCTION`)

```text
web/
├── vite.config.ts        3 entry points; the MPA trailing-slash redirect; envDir: '..'
├── index.html            /         the landing page
├── admin/index.html      /admin/   the operations dashboard
├── demo/index.html       /demo/    THE PRODUCT DEMO
└── src/
    ├── main.tsx, App.tsx, scenes/, sections/    the landing narrative
    ├── admin/  Admin.tsx, api.ts (mock), api.live.ts (real), charts.tsx
    └── demo/   Demo.tsx, api.ts, demo.css       live mic · upload · "Try being the caller"
```

## `behaviour/` — the frozen second layer (`PRODUCTION`, merged unedited)

L2 logistic regression, C=0.1, on 24 conversation-timing features, Silero VAD at 8 kHz, sigmoid
calibration. Val accuracy 0.972, AUC 0.991. Needs two git-ignored artifacts:
`behaviour/artifacts/models/behavior.json` and `behaviour/artifacts/vad/silero_vad.onnx`.

## `semantic/` — the verifier layer (`PRODUCTION`, imported in-process)

ElevenLabs Scribe ASR → 12 text features + 1 grounding feature + 2 Gemini rubric dimensions → calibrated
logistic regression. Runs inside `src/server.py` via `LocalSemanticLayer`. Held-out accuracy 0.915,
AUC 0.975.

## `backend/` — `LEGACY`

A second FastAPI shell from the `backend-esteban` branch. Marked superseded at the top of
`backend/INTEGRATION.md`. Nothing starts it. Its only surviving advantage is serving hygiene (request
size limit, `/ready`, request ids, typed errors) that has not yet been ported into `src/server.py`.

## `report_assets/` — this report

| File | What it is |
|---|---|
| `build_report.py` | Markdown → HTML → PDF via Chromium; the report stylesheet lives here |
| `make_report_figures.py` | the three figures generated for this report from measured artifacts |
| `figures/` | 48 PNGs: 45 copied from the project's own output, 3 generated here |
