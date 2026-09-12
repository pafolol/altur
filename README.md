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

Three backbones were benchmarked under identical conditions (same chunks, classifier, seed, metrics):
`microsoft/wavlm-base-plus`, `facebook/wav2vec2-base-es-voxpopuli-v2`, `facebook/wav2vec2-xls-r-300m`.
The full story, with every number and figure, is in `reports/ACOUSTIC_LEARNING_SUMMARY.pdf`
(source: `reports/ACOUSTIC_LEARNING_SUMMARY.md`); every experiment is recorded in `reports/EXPERIMENT_LOG.md`.

## Setup

```
cd D:\altur\acoustic
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

The dataset stays where the challenge put it: `D:\altur\hackmty26` (manifest + turns) and
`D:\altur\altur-challenge-audio\audio` (unzipped WAVs); paths are in `config.py`.

## Pipeline (in order)

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
| 8b | `python src/augment_train.py` | optional: augmentation experiment on the winner (seen vs unseen perturbation families) |
| 8c | `python src/sanity_checks.py` and `python src/voice_identity.py` | is 100% believable? permutation test, one-chunk test, speaker x-vectors, leave-one-voice-group-out |
| 8d | `python src/external_check.py` | the honest number: new TTS engines + new speakers through a simulated phone line |
| 9 | `python src/build_report.py` | the PDF + Markdown report |

Smoke tests: add `--fraction 0.05` to steps 4 and 8. Unit tests: `python -m pytest tests -q`.

## Using the model

Demo page (record from the microphone or upload a call, get the verdict + explanation, keep score of what it got right):

```
python src/server.py --port 8000        # then open http://localhost:8000/  (frontend/index.html, one file)
```

The page never tells the model where the audio came from: both paths produce the same 8 kHz stereo WAV and POST it to
`/detect`. A "simulate telephone line" option (300-3400 Hz + mu-law) is applied identically to recordings and to uploads
that are not already 8 kHz phone audio, because the detector was trained on phone calls (see report, Part 8c).

```
python src/predict.py path\to\call.wav              # verdict, probability, chunk scores, latency
python src/predict.py --demo                        # six validation calls with known labels
python src/server.py --port 8000                    # POST /detect  {"audio": "<base64 wav>"} -> {"is_synthetic", "confidence"}
python src/client_demo.py call.wav --url http://localhost:8000/detect
```

From Python (what the fusion system will call):

```python
from predict import predict_acoustic
p = predict_acoustic("call.wav")      # path, bytes or base64 string -> float, 0 = human ... 1 = synthetic
```

## Layout

```
config.py                 every setting (paths, segmentation, backbones, classifier, endpoint contract)
src/audio.py              decode, caller channel, VAD, chunking, resampling, loudness normalisation
src/dataset.py            manifest, splits, turn files
src/inspect_dataset.py    stage 1: dataset analysis + shortcut checks
src/extract_embeddings.py stage 2: frozen backbones -> cached embeddings
src/train_classifier.py   stage 3: layer probe + classifiers (identical for every backbone)
src/calibrate.py          probability calibration
src/benchmark.py          stage 4: the controlled comparison
src/finetune.py           optional partial fine-tuning of the winner
src/predict.py            inference (AcousticDetector, predict_acoustic)
src/server.py             POST /detect
src/build_report.py       the report
models/<backbone>/        logreg.joblib, mlp.pt, meta.json, calibration_*.json
outputs/embeddings/       cached embeddings (float16, all layers)
outputs/predictions/      per-call validation/train scores of every model
outputs/figures/          every figure of the report
reports/                  ACOUSTIC_LEARNING_SUMMARY.{pdf,md}, benchmark.csv, model_comparison.csv, EXPERIMENT_LOG.md
```

## Results

Validation split (71 calls from speakers never seen in training), frozen backbone + identical MLP:

| Backbone | Params | Layer | Accuracy (primary) | Bal. acc. | F1 | AUC | EER | Latency/call (GPU) |
|---|---|---|---|---|---|---|---|---|
| WavLM Base+ | 94M | 9 | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% | 255 ms |
| Wav2Vec2 Spanish (VoxPopuli) **(deployed)** | 94M | 5 | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% | 119 ms |
| XLS-R 300M | 315M | 23 | 100.0% | 100.0% | 1.000 | 1.000 | 0.0% | 310 ms |

Shortcut check (trivial statistics, no voice): best all_trivial/logreg = AUC 1.000, accuracy 98.6%.
Fine-tuning the winner: mean stressed validation AUC 1.000 vs frozen 1.000 (clean validation: both 100%) -> frozen model deployed.

Out-of-dataset check (new TTS engines + new speakers through a simulated PSTN line, 108 clips): accuracy 88.0%, AUC 0.945, EER 12.9%. This, not the 100%, is the number to expect on unseen engines.

Winner: **Wav2Vec2 Spanish (VoxPopuli) layer 5 + MLP** (highest mean validation AUC under the 10 stress conditions (0.9998; stressed accuracy 0.972, worst condition white_snr10 AUC 0.998); clean validation accuracy 1.000 / AUC 1.0000 - the clean split is saturated for every backbone and cannot rank them; 3 model(s) within 0.005 -> tie-break by stressed accuracy, EER, latency).
