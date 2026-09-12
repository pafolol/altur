# Runpod usage

## Why a cloud GPU was used

The frozen benchmark (dataset inspection, embedding extraction of three backbones, layer probes, stress test,
classifiers, calibration, latency benchmark) ran on the local RTX 4080 Laptop GPU (12 GB): each of those stages is
minutes of bursty load. The two **sustained-load** stages - partial fine-tuning of the winner and the augmentation
experiment (five extra passes over the training audio) - were first launched locally and the laptop shut itself
down mid fine-tuning (the previous ASVspoof project had documented the same thermal/power protection). They were
therefore moved to a Runpod pod, per the brief's instruction to use Runpod when local compute is insufficient.

## Compute estimate before renting

| Quantity | Value |
| --- | --- |
| training chunks / validation chunks | 4,102 / 977 (<= 4 s each, ~3.7 h of caller speech) |
| frozen embedding extraction of the winner (clean + 10 stress conditions) | ~15 min on an L4 |
| fine-tuning (6 kept layers, 4 unfrozen, <= 11 epochs, batch 8) | ~20-30 min on an L4 |
| augmentation experiment (5 perturbed passes over the training chunks + MLP training) | ~10 min on an L4 |
| GPU memory needed | < 8 GB -> any 24 GB card is sufficient; an H100/A100 would be waste |

GPU choice: RTX 4090 (secure $0.74/hr, community $0.34/hr) was requested first but had no available instances
at the time; RTX A6000 and A40 were also unavailable; the pod was created on the next cheapest 24 GB card with stock.

## Record

| Field | Value |
| --- | --- |
| Pod id | `s1mv90hscq6vls` (name `altur-acoustic`) |
| GPU | NVIDIA L4, 24 GB, secure cloud, data center US-MO-2 |
| Template | `runpod-torch-v280` (torch 2.8.0+cu128, Python 3.12, Ubuntu 24.04) |
| Price | USD 0.49 / hour (+ container/volume disk, 30 + 30 GB) |
| Created | 2026-09-12 06:37 UTC |
| Terminated | 2026-09-12 07:32 UTC (deleted right after `results.tgz` was downloaded; `pod list` confirmed empty) |
| Hours | 0.92 h (55 min; the job itself ran 06:38-06:56, the rest was queueing behind a paused local session) |
| Approximate cost | ~USD 0.45 GPU + ~USD 0.01 disk = **~USD 0.46** |
| Cost guard | local 3 h deletion watchdog (not needed); manual delete at 07:32 UTC |

## Experiments executed on the pod

1. Frozen embedding extraction of the winner (`facebook/wav2vec2-base-es-voxpopuli-v2`), clean train/val and the 10
   stress conditions of validation (`src/extract_embeddings.py`, `src/stress_test.py --stage extract`).
2. Conservative partial fine-tuning of the winner (`src/finetune.py`): backbone truncated at layer 5, last 4 kept
   layers unfrozen, head warm-started from the frozen MLP, early stopping on validation, evaluated under the stress
   conditions.
3. Augmentation experiment (`src/augment_train.py`): the same MLP trained on clean + perturbed training chunks (seen
   families: gain, white/pink noise, mu-law), evaluated on seen and unseen (band-limit, reverb, tilt) families.

## Artifacts generated on the pod (downloaded as `results.tgz`, then unpacked into the project)

- `outputs/finetune/wav2vec2_spanish/{results.json, history.json}`, `models/wav2vec2_spanish_finetuned/`,
  `outputs/figures/finetune_wav2vec2_spanish.png`, `outputs/predictions/wav2vec2_spanish_finetuned_val.csv`
- `outputs/augmentation/wav2vec2_spanish/results.json`, `models/wav2vec2_spanish/mlp_augmented.pt`,
  `outputs/figures/augmentation_wav2vec2_spanish.png`
- the pod's `reports/EXPERIMENT_LOG.md` entries (appended to the local log, marked "Runpod")

## How it was driven

`runpodctl` 2.14.0 with `RUNPOD_API_KEY` (key kept in `.cache/runpod_key.txt`, git-ignored): `ssh add-key`,
`pod create --template-id runpod-torch-v280 --gpu-id "NVIDIA L4" --ssh --wait`, `ssh info`, `scp` of a 1 MB code
bundle, the job launched detached (`setsid ... > /workspace/job.log`) and polled over SSH, results fetched with
`scp`, `pod delete`. The Runpod MCP server was registered for structured pod management in future sessions.

## Measured on the pod

| Stage | Wall time on the L4 |
| --- | --- |
| dataset download + unzip + dependencies | ~2 min |
| frozen embedding extraction of the winner (clean train + val) | 63 s (348x realtime) |
| stress extraction (10 conditions of validation) | ~4 min |
| fine-tuning (3 head epochs + 8 unfrozen epochs, early stop at epoch 4 of the unfrozen phase) | 323 s |
| augmentation experiment (5 perturbed passes + 2 MLP trainings + stress evaluation) | 316 s |
| **job total** | **17.5 min** |
