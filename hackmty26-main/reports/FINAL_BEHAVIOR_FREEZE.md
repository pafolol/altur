# Final Behaviour integration freeze

**READY FOR INTEGRATION: YES** — engineering-ready for team integration under the
documented development-validation evidence. **Not proven robust on Altur's hidden
set or production banking traffic.** No further model optimization is authorized.

- Freeze timestamp (UTC): `2026-09-12T14:07:53.269274+00:00`.
- Trained artifact: `artifacts/models/behavior.json` (**unchanged** during delta review).
- Artifact SHA-256: `631dc6d7504f44c2d1f3533438738b2327258d4494143bd638c255e43fa6bacc`.
- Artifact training timestamp: `2026-09-12T09:15:55.724311+00:00`.
- VAD: `artifacts/vad/silero_vad.onnx`; SHA-256 `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`.
- Runtime policy: `1.1-low-evidence-abstention`.
- Runtime revision SHA-256: `85e87fb5919a2ec57cba9d8ce36da61e9d1d70c755010826dd034b100174e4bf`.
- Full source/test revision SHA-256: `33c7a1e17d3adb539b69757b6bad2081583aa7e43cce19747ef4a2a038afc2b4`.
- Git commit: unavailable (project has no own `.git`; no commits/pushes made).
- Reference baseline: `reports/reference_baseline/20260912T133418.398155Z/REFERENCE_BASELINE.json` (read-only, 102 files).

## Exact model and metrics

- **24 primary features +15 missing indicators**; 40 base linear parameters including
  intercept, plus two sigmoid parameters. Full feature list and configuration:
  [FINAL_BEHAVIOR_MANIFEST.json](FINAL_BEHAVIOR_MANIFEST.json), `feature_names`,
  `missingness_indicators`, `feature_configuration`, `model_configuration`.
- L2 logistic C=0.1; train-only preprocessing; count-free primary feature candidate.
- Calibration: `train_5fold_oof_sigmoid`, seed 2026;
  `sigmoid(1.719003805525593 * logit + 0.09308606398592335)`.
  Parameters fitted using training OOF logits/labels; the calibration option and
  count-free variant were selected with official validation involvement.
- Official train/val: **282 /71**. Accuracy **0.9718309859 (69/71)**;
  balanced accuracy **0.9717806041**; ROC-AUC **0.9912559618**;
  FPR **1/37=0.0270270270**; FNR **1/34=0.0294117647**;
  Brier **0.0406352173**; confusion **[[36,1],[1,33]]** (human, synthetic).
  Same artifact/hash and cached actual-WAV predictions; zero full-call fallback cases.

## Decision and fusion contract

Use `BehaviorDetector.predict(wav_bytes)` or `/behavior` and consume all four fields:
`synthetic_probability`, `quality_score`, `event_count`, `behavior_confidence`.
No manifest, labels, organizer JSON or training caches are required by inference.

For available evidence the demo `/detect` uses **p>=0.5**, confidence p or 1−p.
For q<=0 or fewer than two events it abstains internally and returns the Boolean-only
non-flag **false, confidence=0.5**, with `X-Behavior-Evidence: insufficient_evidence`.
This is **not evidence of humanity or access approval**. Fusion must give absent
Behaviour no positive synthetic vote; other low quality can be downweighted. No
fusion weights were invented. `behavior_confidence=q*abs(2p−1)` is a heuristic,
not probability of correctness. Debug diagnostics expose `evidence_state`.

## Verification

- Final complete suite run **once** in this delta review: **41 passed,
  0 skipped, 0 failures, 0 errors**.
  Receipt: [final_tests.json](metrics/delta_review/final_tests.json) and
  [JUnit output](metrics/delta_review/final_pytest.xml). Earlier targeted regression
  exposed/fixed malformed RIFF handling, then all 12 targeted tests passed.
- Real local WAV in isolated data-free package: **PASS**, Python 3.11.13,
  existing `.venv-runtime`; original project/data reads **OS-denied** and all network
  egress **OS-denied**. No sklearn/pandas/torch. Silence, sparse burst and one-channel
  audio abstain; repeated different calls preserve VAD state isolation.
- Actual loopback HTTP `/health`, `/behavior`, `/detect`, silence fallback/header
  and sanitized invalid input: **PASS**. Receipts: `metrics/delta_review/isolated_runtime.json`
  and `metrics/delta_review/http_smoke.json`.
- CPU benchmark retained (20 full calls): p50 **860.446 ms**, p95 **1193.124 ms**,
  peak **125.441 MiB**. Excludes HTTP/network and is not a new post-fix benchmark.
- Runtime dependencies are pinned in `requirements-runtime.lock`; installed package
  versions are recorded in the JSON manifest. No installation or cloud resource was needed.
- **Docker build/run: NOT VERIFIED LOCALLY**. Static Dockerfile/build-context review only.

## Remaining limitations and changes

The delta review fixed only the insufficient-evidence adapter and malformed RIFF
exception path, plus debug evidence-state/docs/reproducibility receipts. **No trained
artifact, VAD settings, selected features, calibration parameter or threshold changed.**

Prior validation selection cannot be undone; 71 calls do not establish banking error
rates. Group/engine/session provenance, causal interaction mechanisms and manual
speech truth remain unverified. Rates/missingness/protocol signal and acoustic VAD
dependence remain. Early prefixes are weak, robustness probes limited, calibration
domain-specific. Advanced agents can imitate timing; human-controlled voice conversion
is outside demonstrated autonomous-caller coverage. Docker/cloud reachability unverified.

See [FINAL_DELTA_REVIEW.md](FINAL_DELTA_REVIEW.md) for all ASTRA2 issue statuses,
exact evidence, diagnostic results, corrected cached-prefix policy tables and changes.

Manifest SHA-256: `523d956429f9081a5553d482afdca774abc0352a37a47822a4d90894a06bda17`. Verify the model/runtime hashes before integration.
Freeze complete. **Stop here; no further model optimization.**
