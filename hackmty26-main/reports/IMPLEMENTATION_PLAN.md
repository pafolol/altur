# Implementation record

## Initial inspection (2026-09-12)

Actual project: `Documents/hack_mty_2026/hackmty26-main`; audio is a sibling.
Only organizer README, manifest and turns existed; no Acoustic/Semantic/backend code,
dependency environment or project-level Git history exists. Parent Git repository is
the user's home directory; no commits or remote changes will be made.

Read all three supplied PDFs. The ASVspoof learning report is background, NOT an
Altur result. Its frozen-feature advantage motivates small models; its silence-only
success motivates an explicit shortcut baseline. Sponsor guide pages 4/13 list
Vultr CPU hosting and conditional GPU offers; no service is required for this MVP.

Dataset inspection: 282 train / 71 val; no speaker/session IDs. Only aggregate
inspection results are returned to the coding assistant; raw records/audio stay local.
Manifest duration is not precise enough: use WAV frame count. Audit turn endpoints.

## Fixed experiment plan (before seeing model results)

1. Unit-tested half-open timelines, directional onset events, non-causal interpretation,
   censoring, missing statistics below three comparable observations.
2. Dummy; 10 global timing shortcuts; temporal; temporal+reaction;
   temporal+reaction+consistency. L2 logistic C=0.1 first; C=1 as capacity comparison;
   shallow ExtraTrees as a bounded nonlinear comparison. No broad validation search.
3. Offline Silero ONNX CPU. Verify tagged official model source and license; native
   8 kHz, no diarization/transcription. Time a small training-only sample first.
4. Compare three predeclared segmentation settings on 24 training calls without labels;
   select maximum mean channel speech-mask IoU, then freeze. Organizer annotations
   are automatic references, not manual ground truth.
5. Evaluate organizer-trained models on VAD inputs, then train the same model ladder
   on VAD-derived training features. Select an interpretable model considering both
   validation performance and shortcut exposure. Keep all baseline artifacts.
6. Train-only five-fold out-of-fold sigmoid calibration; official validation remains
   unfitted. Internal folds are call-stratified, NOT demonstrably speaker-disjoint.
7. Fixed threshold 0.5 for comparison; report optional operating points separately.
   Same final full-call model and frozen threshold on 15/30/60/full prefixes.
8. Family ablations, metadata audit, duration strata, label-permutation control,
   light gain/noise and segmentation sensitivity, error timelines, calibration,
   CPU latency/memory. Export stable inference and HTTP adapter, no fusion weights.
9. Reproducible scripts, plots, comprehensive educational report and short student guide.

## Early engineering corrections

- An initial one-off inspection used `Path(".").parent` and looked in the wrong audio
  directory. Corrected to resolved project root; production locator checks ambiguity.
- Guessed upstream tag `v6.2.0` returned 404. Enumerated official tags before pinning
  a real release. This is why model URLs/versions must be verified.
- System Python 3.14 has no ML packages. Created isolated Python 3.11.13 using
  local uv 0.8.22; no system Python packages changed.
