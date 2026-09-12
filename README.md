# Altur HackMTY 2026 - acoustic synthetic-caller detector

Acoustic branch of the "Defend the Bank Against Voice Deepfakes" challenge: given a stereo 8 kHz call
(channel 0 = caller, channel 1 = bank agent), decide from the **caller's voice alone** whether the caller is
a human or a synthetic voice.

```
stereo 8 kHz WAV (base64 at the API)
  -> channel 0 (caller)  -> voice activity detection -> caller speech chunks (<= 4 s), RMS-normalised
  -> resample 16 kHz     -> FROZEN pretrained speech backbone -> hidden layer L, mean over time
  -> small MLP           -> chunk log-odds -> mean over chunks -> calibrated synthetic probability
```

There are **two models**, and both stay on disk:

| | trained on | segmentation | Altur validation | down a telephone line |
|---|---|---|---|---|
| **Specialist** (Model A) | the Altur recordings | energy VAD | 100.0% | 77.5% |
| **Robust V2** (Model B) — *deployed* | + the same calls through telephone channels | Silero VAD | 100.0% | 100.0% |

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

Smoke tests: add `--fraction 0.05` to steps 4 and 8, `--limit 10` to step 10, `--limit 6 --skip-stress` to
step 15. Step 15 can rebuild its tables and figures without re-scoring anything with `--from-cache`.
Unit tests: `python -m pytest tests -q` (87 tests).

## Using the model

Demo page — record from the microphone or upload a call, get the verdict and the explanation, and put the two
models side by side on the same audio:

```
python src/server.py --port 8000 --preload-all      # then open http://localhost:8000/
```

The page never tells the model where the audio came from: both paths produce the same 8 kHz stereo WAV and
POST it to `/detect`. A "simulate telephone line" option is applied identically to recordings and to uploads
that are not already 8 kHz phone audio, because the detector was trained on phone calls.

```
python src/predict.py path\to\call.wav              # verdict, probability, chunk scores, latency
python src/predict.py --demo                        # six validation calls with known labels
python src/server.py --model specialist             # answer /detect with Model A instead
python src/client_demo.py call.wav --url http://localhost:8000/detect
```

| endpoint | contract |
| --- | --- |
| `POST /detect` | `{"audio": "<base64 wav>"}` -> `{"is_synthetic", "confidence"}` — the challenge contract, unchanged |
| `POST /detect_all` | the same body -> every loaded model's verdict, for the comparison panel |
| `GET /models` | what is loaded, which layer, which VAD, which one answers `/detect` |

From Python (what the fusion system will call):

```python
from predict import predict_acoustic
p = predict_acoustic("call.wav")      # path, bytes or base64 string -> float, 0 = human ... 1 = synthetic
```

Which model answers is decided by `reports/deployed_model.json`, written by `evaluate_models.py` from the
measured cross-domain matrix. `predict.py` and `server.py` both read it, so they cannot drift apart.

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
src/predict.py                  inference (AcousticDetector, predict_acoustic)
src/server.py                   POST /detect, POST /detect_all, GET /models
src/build_report.py             the specialist's report

src/phone_channel.py            THE TELEPHONE CHANNEL: exact G.711, ffmpeg codecs, RTP loss, drift, AGC, alignment
src/build_channel_dataset.py    the dataset factory (resumable, idempotent, split-preserving)
src/channel_pilot.py            the 10-call GO/STOP gate
src/channel_figures.py          the channel figures
src/extract_channel_embeddings.py  frozen backbone -> embeddings of original + channel audio
src/train_robust.py             Model B + the orig_only / channel_only / mixed controls
src/evaluate_models.py          four models x six domains, AUC-vs-threshold analysis
src/report_robust.py            the telephone-robustness report

models/wav2vec2_spanish/        Model A (specialist) - never overwritten
models/robust_v2/               Model B (deployed)
models/robust_{orig_only,channel_only,mixed,balanced}/   the controls, loadable for evaluation
D:\altur\channel_dataset\       the transformed audio, manifests, logs, pilot, figures
reports/                        ACOUSTIC_LEARNING_SUMMARY.{pdf,md}, TELEPHONE_ROBUSTNESS_REPORT.{pdf,md},
                                EXPERIMENT_LOG.md, deployed_model.json, best_model.json
```

## Results

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
