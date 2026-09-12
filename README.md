# Altur HackMTY 2026 - synthetic-caller detector (acoustic + behaviour + semantic)

"Defend the Bank Against Voice Deepfakes": given a stereo 8 kHz call (channel 0 = caller, channel 1 = bank
agent), decide whether the caller is a human or a synthetic voice.

Independent detectors answer that question from **different evidence**, and `POST /detect` returns their
verdict in **two stages**: the acoustic and behaviour layers decide every call **50 / 50**, and the semantic
layer is a **verifier** consulted only when those two do not settle it. Every weight, role and gate is tunable
at runtime. This repository is the merge of the `modelo`, `behaviour`, `fusion` and `backend-esteban` branches.

```
stereo 8 kHz WAV (base64 at the API)
  |
  +-- ACOUSTIC LAYER  - how the caller SOUNDS                                       PRIMARY    w 0.50
  |     channel 0 -> VAD -> caller speech chunks (<= 4 s), RMS-normalised -> resample 16 kHz
  |     -> FROZEN speech backbone, hidden layer 5 -> mean over time -> MLP -> calibrated P(synthetic)
  |
  +-- BEHAVIOUR LAYER - how the caller BEHAVES  (behaviour/, frozen, unedited)      PRIMARY    w 0.50
  |     both channels -> Silero VAD at 8 kHz -> turns -> interruption / barge-in / response timing
  |     -> 24 features -> logistic + sigmoid calibration -> calibrated P(synthetic) + evidence quality
  |
  +-> the two of them, 50 / 50                                                  -> p_primary
  |
  |     confident?  |p - 0.5| puts it at >= 80 % -> THAT IS THE VERDICT. Stop. Nothing else is asked.
  |     unsure?     they disagree, or neither is committed -> escalate:
  |
  +-- SEMANTIC LAYER - what the caller SAYS  (separate service, HTTP, PAID, ~2.6 s)   VERIFIER   w 0.15
  |     agent channel -> trap bank;  caller channel -> VAD -> ElevenLabs Scribe -> words + logprobs
  |     -> text features + a 7-dimension Gemini rubric -> logistic, Platt-calibrated -> P(synthetic)
  |
  +-> FUSION (src/fusion.py)                          -> {"is_synthetic": bool, "confidence": float}
```

The layers fail in different places, which is the point of running more than one: the acoustic layer needs
*audible* caller speech, the behaviour layer needs *interaction* (a caller who never gets interrupted leaves
it nothing to time), and the semantic layer needs a *transcript* and two third-party APIs. Each one says when
it has nothing to go on instead of guessing, and its share of the vote goes to the others.

**Why the semantic layer is a verifier rather than a third vote.** It is the only one that leaves the machine:
a paid ASR call, a paid LLM call and about 2.6 s of network per call, against ~500 ms locally for the other
two. Asking it about a call the acoustic and behaviour layers already agree on at 95 % confidence buys nothing
and costs money. So it is asked only when they do not settle it. On the 71 held-out calls that is **5 of
71** — the primaries settle **66** on their own, and a confident call answers in **~1.0 s instead of
~3.9 s** because the service is never contacted.

### The acoustic layer's two models

Both stay on disk; **V1, the specialist, is the one in the fusion** (`--acoustic-model robust_v2` swaps it):

| | trained on | segmentation | Altur validation | down a telephone line |
|---|---|---|---|---|
| **V1 — Specialist** (Model A) — *in the fusion* | the Altur recordings | energy VAD | 100.0% | 77.5% |
| **V2 — Robust** (Model B) | + the same calls through telephone channels | Silero VAD | 100.0% | 100.0% |

Model A is the original work and is untouched. Model B exists because Model A degrades badly on audio that has
actually been through a phone call — the full story, with every number and figure, is in
`reports/TELEPHONE_ROBUSTNESS_REPORT.pdf`. The three-backbone comparison that produced Model A is in
`reports/ACOUSTIC_LEARNING_SUMMARY.pdf`. Every experiment is recorded in `reports/EXPERIMENT_LOG.md`.

## Setup

```
cd D:\altur\acoustic
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dataset stays where the challenge put it: `D:\altur\hackmty26` (manifest + turns) and
`D:\altur\altur-challenge-audio\audio` (unzipped WAVs); paths are in `config.py`. The telephone-channel
dataset is generated into `D:\altur\channel_dataset` and the official one is never written to.
`ffmpeg` must be on PATH (the real codecs run through it).

**The behaviour layer** needs two frozen artifacts that are deliberately NOT in Git (`behaviour/.gitignore`
keeps `artifacts/` out, because the same folder also holds per-call caches derived from the dataset):

```
behaviour/artifacts/models/behavior.json     sha256 631dc6d7...  the fitted model  (52 KB)
behaviour/artifacts/vad/silero_vad.onnx      sha256 1a153a22...  Silero v6.2.1, MIT (2.3 MB)
```

Copy the `artifacts/` folder of the behaviour deliverable to `behaviour/artifacts/`. Both hashes are checked
at load time — the ONNX by the module itself, the model against
`behaviour/reports/FINAL_BEHAVIOR_MANIFEST.json`. Without them the behaviour layer reports itself unavailable,
abstains on every call, and its 35 % is shared out among the layers that did answer.

To run the behaviour module's *own* test suite it also wants the dataset beside it (its 41 tests read real
calls); only the last line copies any bytes:

```
mklink /J  D:/altur/acoustic/behaviour/audio  D:/altur/altur-challenge-audio/audio
copy       D:/altur/hackmty26/manifest.csv    D:/altur/acoustic/behaviour/
xcopy /E /I D:/altur/hackmty26/turns          D:/altur/acoustic/behaviour/turns
```

All three are git-ignored. The fusion itself never needs them - it is given WAV bytes like any caller.

**The semantic layer** lives in `semantic/`, merged from the `fusion` branch, and runs as its **own
service on its own interpreter**. That is not incidental: it needs an ElevenLabs key and a Gemini key, it is
the only layer that leaves the machine, its budget is a network deadline rather than compute — and it
requires **Python 3.12+** (it uses PEP 701 f-strings), which the 3.11 venv here cannot parse. The HTTP
boundary is what lets the two live in one repository without one dictating the other's runtime.

```
cd semantic
set UV_PYTHON_INSTALL_DIR=%CD%\.python
uv venv --python 3.13 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
copy .env.example .env                     :: ELEVENLABS_API_KEY, GEMINI_API_KEY - it will not start without them
.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8100
```

Then point the fusion at it — `python src/server.py --semantic-url http://127.0.0.1:8100/detect`, or set
`SEMANTIC_URL`. Port 8100 rather than its default 8000 so it does not collide with this server.

It answers `GET /health` and `POST /detect {"audio": "<base64 wav>"}` with
`{"is_synthetic", "confidence", "score", "abstain", "reason", "used", "ms"}`. `score` is the calibrated
P(synthetic) and is what the fusion reads; this service sets `confidence` equal to it, but a service
following the challenge's own reading of "confidence" would return P(the verdict is right), and averaging
that would be wrong. `used` names the path that answered — `"f4+f5"` (Scribe + rubric), `"f4"` (text
features only, recorded at half evidence quality) or `"abstain"`.

**If nothing answers, the layer abstains on every call and its 15 % goes to the other two** — the console
greys the channel out with the reason, and no verdict is blocked or delayed waiting for it.

It also needs the dataset where its code looks for it, which is one junction and no copying:

```
mklink /J D:/altur/acoustic/semantic/data/hackmty26  D:/altur/hackmty26
mklink /J D:/altur/hackmty26/audio                   D:/altur/altur-challenge-audio/audio
```

(the second is the layout the organizers' own README describes, and `audio/` is gitignored there)

## Pipeline (in order)

### Part 1 - the specialist

| Step | Command | What it does |
| --- | --- | --- |
| 1 | `python src/inspect_dataset.py` | counts, formats, per-call statistics, shortcut/leakage checks, VAD agreement |
| 2 | `python src/train_shortcut.py` | how far trivial statistics get without listening to the voice |
| 3 | `python src/signal_processing_demo.py` | waveform / FFT / spectrogram / mel / MFCC / pitch figures from real calls |
| 4 | `python src/extract_embeddings.py --backbone all` | frozen backbones -> cached embeddings of every layer |
| 5 | `python src/train_classifier.py --backbone all` | layer probe, logistic regression, MLP, aggregation diagnostics |
| 5b | `python src/stress_test.py --stage extract` then `--stage probe` | validation calls under 10 channel perturbations; robust layer per backbone |
| 5c | `python src/train_classifier.py --backbone <k> --layer <robust layer>` then `python src/stress_test.py --stage mlp` | MLP on the robust layer, scored under stress |
| 6 | `python src/calibrate.py` | Platt / temperature scaling of the call score (cross-validated on val) |
| 7 | `python src/benchmark.py --cpu` | side-by-side tables, figures, latency, winner -> `reports/best_model.json` |
| 8 | `python src/finetune.py` | optional: conservative partial fine-tuning of the winner |
| 8b | `python src/augment_train.py` | optional: augmentation experiment (seen vs unseen perturbation families) |
| 8c | `python src/sanity_checks.py` and `python src/voice_identity.py` | is 100% believable? permutation test, one-chunk test, speaker x-vectors, leave-one-voice-group-out |
| 8d | `python src/external_check.py` | new TTS engines + new speakers through a simulated phone line |
| 9 | `python src/build_report.py` | `reports/ACOUSTIC_LEARNING_SUMMARY.{md,pdf}` |

### Part 2 - surviving a real telephone line

| Step | Command | What it does |
| --- | --- | --- |
| 10 | `python src/build_channel_dataset.py --split all --family all` | the Altur caller channels through real narrowband codecs -> `D:\altur\channel_dataset` (~6 min) |
| 11 | `python src/channel_pilot.py` | the 10-call gate: does the channel actually break the deployed detector? GO / STOP |
| 12 | `python src/channel_figures.py` | why the detector goes blind, what the factory drew, the architecture diagram |
| 13 | `python src/extract_channel_embeddings.py --sets all` | frozen backbone -> embeddings of the original and channel audio (~35 min, Silero VAD on CPU is the bottleneck) |
| 14 | `python src/train_robust.py` | Model B plus the three controls that let the report attribute the gain |
| 15 | `python src/evaluate_models.py` | four models x six domains -> `reports/deployed_model.json` (~35 min) |
| 16 | `python src/report_robust.py` | `reports/TELEPHONE_ROBUSTNESS_REPORT.{md,pdf}` |

### Part 3 - the fusion

| Step | Command | What it does |
| --- | --- | --- |
| 17 | `python src/evaluate_fusion.py` | scores the 71 held-out calls through **every** layer, sweeps the weights, writes the cache the console reads (~60 s) |
| 18 | `python src/server.py --port 8000` | `/detect` answers with the fused verdict; `/fusion` is the console |
| 18b | `python src/server.py --semantic-url http://127.0.0.1:8100/detect` | ... with the semantic service attached (start it first, see Setup) |
| 18c | `cd semantic && .venv/Scripts/python holdout_val.py` | train-only fit applied to val -> `scores_val_trainfit.csv` (caches only, no API calls) |

Smoke tests: add `--fraction 0.05` to steps 4 and 8, `--limit 10` to step 10, `--limit 6 --skip-stress` to
step 15. Step 15 can rebuild its tables and figures without re-scoring anything with `--from-cache`.
Unit tests: `python -m pytest tests -q` (114 tests, 27 of them the fusion combiner) and, for the frozen
behaviour module, `cd behaviour && python -m pytest tests -q` (41 tests).

## Using the detector

```
python src/server.py --port 8000
```

Use **`http://127.0.0.1:8000`**, not `localhost` - uvicorn binds IPv4 only, and on Windows `localhost`
resolves to `::1` first, which costs a two-second timeout on every request.

Two pages, one process:

| page | what it is for |
| --- | --- |
| `GET /` | **the inspector** - record from the microphone or upload one call, get the verdict and the explanation, and put the two acoustic models side by side |
| `GET /fusion` | **the fusion console** - set what each layer's vote is worth, run all 71 held-out calls, and read every system's own score next to the decision they add up to |

The console scores each call once per layer and caches the result, so moving a fader re-decides all 71 calls
in about a millisecond without a model running. Every column, fader and card is generated from the layer
registry, so a third layer appears in all of them on its own.

```
python src/fusion.py call.wav                             # every layer's score + the fused verdict
python src/fusion.py call.wav --weight acoustic=0.7 --weight behaviour=0.3
python src/fusion.py call.wav --mode logit_mean           # average the log-odds instead
python src/predict.py path\to\call.wav                    # the acoustic layer on its own
python src/evaluate_fusion.py                             # the 71 held-out calls + the weight sweep
python src/server.py --acoustic-model robust_v2           # put V2 in the acoustic slot instead of V1
python src/server.py --weight acoustic=0.7                # start somewhere other than 50 / 35 / 15
python src/server.py --semantic-url http://127.0.0.1:8100/detect   # attach the semantic service
python src/client_demo.py call.wav --url http://127.0.0.1:8000/detect
```

| endpoint | contract |
| --- | --- |
| `POST /detect` | `{"audio": "<base64 wav>"}` -> `{"is_synthetic", "confidence"}` - the challenge contract, unchanged; the verdict is now the fused one and every layer's score is under `details` |
| `POST /detect_layers` | the same body -> each layer's own score, quality and latency, plus the fused verdict. An optional `"config"` scores that one call under different weights without changing the server's |
| `POST /fusion/recombine` | `{"calls": [{"id", "layers"}], "config"}` -> the decisions again from scores that already exist. **No model runs** - this is what the faders call |
| `GET` / `POST /fusion/config` | read / live-tune the weights, the combine mode, the abstention policy and the threshold |
| `GET /layers` | the registry: what detection systems exist, whether they are answering, why not if not, and what each one is looking at |
| `GET /validation`, `/validation/{id}/score`, `/validation/scores` | the 71 held-out calls, scored server-side from disk and cached |
| `POST /detect_all`, `GET /models` | unchanged: the two ACOUSTIC models side by side, for the inspector's A/B panel |

From Python:

```python
from fusion import FusionDetector, FusionConfig, build_layers
det = FusionDetector()                       # whatever is answering, at 0.50 / 0.35 / 0.15
det.score(open("call.wav", "rb").read())     # fused verdict + what every layer contributed

# every registered layer, including ones not answering (what the server serves, so the console
# shows a greyed-out channel with its share reserved instead of silently dropping it)
det = FusionDetector(layers=build_layers(include_unavailable=True))

det.config = FusionConfig.from_dict({"weights": {"semantic": 0.3}}, det.config)   # partial update
det.config = det.equal_config()                                                   # assume nothing

from predict import predict_acoustic         # one layer on its own, unchanged
```

### Adding a third layer

`src/fusion.py` is the only file that has to change. Subclass `Layer` with `available()`, `_load()` and
`_score()` returning a `LayerResult`, give it a `default_weight`, then add it to `build_layers()`. The
endpoints, the console, the batch evaluation and the CSV export all iterate over the registry, so the new
layer arrives in each of them with its own column, its own fader, its own preset button and its own share of
the vote. `SemanticLayer` is the worked example: about sixty lines, and it reaches a service over HTTP.

A layer that raises is caught and abstains, so it can never take the verdict down with it - there is a test
for exactly that. A layer that fails to load is not retried for 30 s, so a batch of 71 calls does not pay a
connection timeout each. And a layer that is registered but not answering still appears, greyed out, with its
share reserved - it does not silently vanish from the table.

### How the votes are combined

`combine()` is the single implementation, in `src/fusion.py`, and both `/detect` and the console's faders go
through it. Only the *ratio* of the weights matters, so a 0..1 fader per layer behaves the way you expect and
any number of layers works.

| knob | options |
| --- | --- |
| weights | any non-negative number per layer. Shipped: acoustic 0.50 / behaviour 0.50 as primaries, semantic 0.15 when it is asked |
| roles | `primary` votes on every call; `verifier` only on an escalation. Shipped: acoustic and behaviour primary, semantic verifier |
| verify_threshold | the joint confidence the primaries must reach for the verifier to be skipped (0.80) |
| use_verifiers | `false` turns the gate off, and every layer becomes a plain weighted vote again |
| mode | `weighted_mean` (average the probabilities - the literal reading of "50 / 50") or `logit_mean` (average the log-odds; two layers that agree reinforce each other) |
| on_abstain | `renormalise` (a layer with no evidence is dropped and the rest share its weight) or `neutral` (it stays in at 0.5) |
| use_quality | multiply each weight by that layer's evidence quality on that call. Off by default, so the split means exactly what it says on every call |
| threshold | where the fused probability becomes a verdict |

If nothing is left voting - every layer abstained, or every weight is zero - the answer is an explicit
non-flag, `is_synthetic: false` at confidence 0.5, and `decisive: false` says so. That is **not** a vote for
"human": a probability of 0.5 would otherwise clear a threshold of 0.5 and accuse a caller on no evidence at
all.

Which acoustic model sits in the fusion is a flag (`--acoustic-model`, V1 by default). The separate
single-model endpoints still follow `reports/deployed_model.json`, written by `evaluate_models.py`, so they
and the report cannot drift apart.


## Layout

```
config.py                       every setting (paths, segmentation, backbones, classifier, endpoint contract)
src/audio.py                    decode, caller channel, VAD, chunking, resampling, loudness normalisation
src/dataset.py                  manifest, splits, turn files
src/inspect_dataset.py          stage 1: dataset analysis + shortcut checks
src/extract_embeddings.py       stage 2: frozen backbones -> cached embeddings
src/train_classifier.py         stage 3: layer probe + classifiers (identical for every backbone)
src/calibrate.py                probability calibration
src/benchmark.py                stage 4: the controlled comparison
src/predict.py                  acoustic inference (AcousticDetector, predict_acoustic)
src/server.py                   the HTTP surface: /detect, /detect_layers, /fusion/*, /validation/*, both pages
src/build_report.py             the specialist's report

src/phone_channel.py            THE TELEPHONE CHANNEL: exact G.711, ffmpeg codecs, RTP loss, drift, AGC, alignment
src/build_channel_dataset.py    the dataset factory (resumable, idempotent, split-preserving)
src/channel_pilot.py            the 10-call GO/STOP gate
src/channel_figures.py          the channel figures
src/extract_channel_embeddings.py  frozen backbone -> embeddings of original + channel audio
src/train_robust.py             Model B + the orig_only / channel_only / mixed controls
src/evaluate_models.py          four models x six domains, AUC-vs-threshold analysis
src/report_robust.py            the telephone-robustness report

src/fusion.py                   THE FUSION LAYER: Layer, LayerResult, FusionConfig, combine(), the registry
                                AcousticLayer (0.50) + BehaviourLayer (0.35) + SemanticLayer (0.15, over HTTP)
src/evaluate_fusion.py          all 71 held-out calls through every layer + the weight sweep
frontend/index.html             the single-call inspector (GET /)
frontend/fusion.html            the fusion console (GET /fusion)

behaviour/                      THE BEHAVIOUR LAYER, merged in frozen from the behaviour branch and not edited
behaviour/behavior/             the module itself (inference.py, vad.py, features.py, events.py, ...)
behaviour/artifacts/            its two frozen artifacts - git-ignored, see Setup
behaviour/reports/              its report, its freeze record and its metrics

backend/                        THE SERVING SHELL, merged from the backend-esteban branch and not edited
backend/INTEGRATION.md          how it relates to src/server.py - READ BEFORE SUBMITTING: two /detect exist
backend/app/                    FastAPI with request limits, /ready, warm-up, request ids; detector is a stub

semantic/                       THE SEMANTIC LAYER, merged from the fusion branch and not edited
semantic/server.py              its own FastAPI service - runs on its own Python 3.13 venv, see Setup
semantic/model.pkl              its fitted logistic + Platt calibration
semantic/cache/                 Scribe transcripts + the Gemini rubric for every call (no API calls to re-run)
semantic/scores_semantico.csv   its own out-of-fold scores, a random 5-fold over all 353 calls
semantic/scores_val_trainfit.csv  train-only fit applied to val, what the fusion reads for the 71
semantic/holdout_val.py         added by this integration to write that file; nothing else is modified
semantic/AUDIT.md               its shortcut audit, ablation ladder and error analysis
outputs/fusion/                 the cached per-layer scores, the per-call CSV and the weight sweep

models/wav2vec2_spanish/        Model A (specialist) = V1, in the fusion - never overwritten
models/robust_v2/               Model B = V2 (phone-hardened), one flag away
models/robust_{orig_only,channel_only,mixed,balanced}/   the controls, loadable for evaluation
D:\altur\channel_dataset\       the transformed audio, manifests, logs, pilot, figures
reports/                        ACOUSTIC_LEARNING_SUMMARY.{pdf,md}, TELEPHONE_ROBUSTNESS_REPORT.{pdf,md},
                                EXPERIMENT_LOG.md, deployed_model.json, best_model.json
```

## Results

### The fusion, on the 71 held-out calls

Speaker-disjoint from everything any layer trained on. `python src/evaluate_fusion.py`:

| system | accuracy | ROC AUC | Brier | reads |
|---|---|---|---|---|
| Acoustic V1 alone | 100.0% | 1.000 | 0.000 | how the caller sounds |
| Behaviour alone | 97.2% | 0.991 | 0.041 | when the caller speaks, yields, interrupts |
| Semantic alone | 91.5% | 0.975 | 0.070 | what the caller actually says |
| **Fused, 50 / 50 + verifier** | **100.0%** | **1.000** | **0.010** | the two primaries, plus the verifier on 5 calls |
| Flat three-way vote, no gate | 100.0% | 1.000 | 0.009 | all three, on every call |

Confusion of the fused verdict: `[[37, 0], [0, 34]]` (rows and columns human, synthetic). No layer
abstained on this split.

**The gate costs nothing and saves almost everything.** The primaries settle **66 of 71** calls on
their own; the verifier is asked on **5** — 7% of the paid API calls a flat three-way vote
would have made — and the accuracy is identical either way. These are the five:

| call | truth | acoustic | behaviour | primary | conf. | semantic | fused |
|---|---|---|---|---|---|---|---|
| `0847d7417bb1` | synthetic | 1.000 | 0.435 | 0.717 | 72 % | 0.952 | 0.748 -> synthetic |
| `569ffb0869eb` | human | 0.000 | 0.964 | 0.482 | 52 % | 0.046 | 0.425 -> human |
| `678ee1dd2242` | human | 0.001 | 0.476 | 0.239 | 76 % | 0.178 | 0.231 -> human |
| `95ccefc6e4ce` | human | 0.013 | 0.428 | 0.221 | 78 % | 0.280 | 0.228 -> human |
| `b658b1155216` | human | 0.000 | 0.434 | 0.217 | 78 % | 0.390 | 0.240 -> human |

Every escalation is a call where the behaviour layer is sitting near the fence and drags the pair's confidence
under the gate. `569ffb0869eb` is the interesting one: the behaviour layer is not merely unsure there, it is
confidently **wrong** (0.964 for a human caller), and the verifier independently agrees with the acoustic
layer at 0.046. That is the case the second stage exists for.

**The semantic column is not scored through the live service, on purpose.** Its shipped `model.pkl` is
refitted on all 353 calls, so asking it about a validation call would be asking a model about audio it has
already seen. `Layer.held_out_score()` exists for exactly this: an evaluation path passes the call's
`anon_id`, and a layer that was fitted on it answers from its own held-out scores instead. Live `/detect`
never has an `anon_id`, so it always runs every layer it actually needs, for real. The acoustic and behaviour
layers are fitted on the train split alone, so for them the validation WAV is already a held-out input and
they return `None`. A held-out lookup is free, so the evaluation path fills the semantic column on all 71 even
where the live path would have skipped it - which is what lets the console re-tune the gate after the fact.

The semantic numbers come from `semantic/scores_val_trainfit.csv` - the served configuration fitted on the 282
train calls and applied to the 71 val ones, written by `semantic/holdout_val.py` from the caches with no API
calls. The module's own `scores_semantico.csv` gives 0.973 instead of 0.975, because its rows come from a
random 5-fold over all 353 calls, which ignores the speaker-disjoint official split; its own code calls that
"slightly optimistic". The two agreeing to 0.002 is reassuring rather than alarming.

> One inconsistency worth flagging to whoever owns that module: `AUDIT.md` labels a **10-feature** row
> ("F4 limpio + F5 2 dims", val AUC 0.921) as *"lo que se sirve ahora"*, but `model.py` actually serves
> **15** features - F4 with `SEMANTIC_EXTRA` on (12), plus `overlap_agent`, plus the 2 rubric dimensions.
> The audit table and the served configuration have drifted apart; 0.975 is the served one measured the
> same way.

Read the whole table honestly: the split is **saturated for the acoustic layer**, which is already at 100 %
alone for the reason the next section explains. No weighting and no gate can be justified from these 71 calls,
and a sweep maximum here describes them rather than transferring. Every layer was also developed while looking
at this split. The case for the arrangement is not the accuracy column - it is that the layers fail in
different places, and that the expensive one is spent only where the cheap ones ran out of confidence.

### The specialist, on the official split

Validation (71 calls from speakers never seen in training), frozen backbone + identical MLP:

| Backbone | Params | Layer | Accuracy | Bal. acc. | F1 | AUC | EER | Latency/call (GPU) |
|---|---|---|---|---|---|---|---|---|
| WavLM Base+ | 94M | 9 | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% | 255 ms |
| Wav2Vec2 Spanish (VoxPopuli) | 94M | 5 | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% | 119 ms |
| XLS-R 300M | 315M | 23 | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% | 310 ms |

The clean split is **saturated**: a classifier on trivial statistics alone (loudness, band edge, noise floor —
no voice modelling) also reaches AUC 1.000 on it. It cannot rank models, and it cannot tell "detects synthetic
speech" from "detects the recording pipeline".

### What a telephone line does to it

The caller channel of the Altur recordings is *digitally silent* between turns — across 40 validation calls the
10th-percentile frame level is −72 dB and so is the median, i.e. more than half of every call is a run of zero
samples. A real line never delivers that: gain control lifts the quiet passages and comfort noise fills the
gaps, so the noise floor rises to −38 dB and the dynamic range falls from 52 dB to 16 dB. The deployed
segmenter needs 15 dB of range to fire, so **it returns no speech at all** and the endpoint answers an
undecided 0.5 before the model is ever reached.

Accuracy at the deployed threshold / ROC AUC, four models on six domains:

| domain | Specialist | + VAD swapped | Retrained, orig. only | **Robust V2** |
|---|---|---|---|---|
| Altur validation | 100.0% / 1.000 | 100.0% / 1.000 | 100.0% / 1.000 | **100.0% / 1.000** |
| through a SEEN-family line | 77.5% / 0.908 | 91.5% / 1.000 | 98.6% / 1.000 | **100.0% / 1.000** |
| through an UNSEEN-family line | 81.7% / 0.929 | 74.6% / 1.000 | 93.0% / 1.000 | **100.0% / 1.000** |
| out-of-dataset clips, through a line | 62.0% / 0.794 | 64.8% / 0.894 | 76.9% / 0.891 | **81.5% / 0.891** |
| 11 channel perturbations | 97.4% / 0.997 | 99.0% / 1.000 | 98.2% / 1.000 | 98.5% / 0.999 |
| out-of-dataset clips, *clean* | **68.5% / 0.899** | 60.2% / 0.849 | 55.6% / 0.886 | 54.6% / 0.867 |
| **worst telephone domain** | 62.0% | 64.8% | 76.9% | **81.5%** |

The **UNSEEN** family (GSM, AMR-NB, iLBC, Speex, Opus, G.722) never appears in any training set — those files
are not generated for TRAIN at all, so there is nothing on disk that could be trained on by accident. Across
all six domains the deployed segmenter found no caller speech in **47 files**; for Robust V2 that number is 2.

Two honest caveats. On **clean, non-telephone audio** the specialist is better (68.5% vs 54.6%) — but at the
best available threshold the two are 85.2% and 82.4% with AUCs of 0.899 and 0.867, so most of that gap is a
threshold nobody has fitted for that kind of audio, and Robust V2 is the more miscalibrated there precisely
because it stopped keying on loudness and band edge. And the channel is built from *real codecs* but a
*modelled* network: no carrier was involved, which the report states wherever it matters.

**Deployed: Robust V2** — better on every domain that is a phone call, including codecs and engines nothing
was trained on. The specialist is kept intact and is one flag away: `python src/server.py --model specialist`.
