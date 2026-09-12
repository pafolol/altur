# ASTRA2 final delta review — freeze decision

Scope: compare the completed module against `reports/ASTRA2_ADVERSARIAL_REVIEW.md`.
No architecture redesign, web research, feature search, deployed-model refit,
threshold optimization, external data transfer or cloud action was performed.

Reference state was preserved **before** this review under
`reports/reference_baseline/20260912T133418.398155Z/`. Its read-only
`REFERENCE_BASELINE.json` records the model, feature/configuration lists, calibration,
metrics, the previously observed 29-test pass, and hashes of 102 preserved files.

Deployed artifact: `artifacts/models/behavior.json`, SHA-256
`631dc6d7504f44c2d1f3533438738b2327258d4494143bd638c255e43fa6bacc`.
**The artifact is unchanged.** Runtime changes are the explicit low-evidence adapter
policy, debug evidence state, and clean malformed-RIFF error handling.

## FINAL DELTA REVIEW TABLE

Statuses describe the actual concern, not just whether its documentation was edited.
`PARTIALLY FIXED` means safeguards or narrowed claims exist but the methodological
risk remains. `NOT VERIFIABLE` means provided evidence cannot establish the claim.
All paths below are relative to `hackmty26-main/`; `delta/` in evidence denotes
`reports/metrics/delta_review/`.

| Issue | Status | Evidence | Action |
| --- | --- | --- | --- |
| Headline: mixing model-family and deployed metrics | FIXED | `scripts/delta_review_checks.py::main`, `delta/checks.json`, baseline artifact hash, `reports/metrics/final_model.json` and `frozen_evaluation.json`. Same immutable 24-feature artifact gives 69/71, AUC 0.9912559618, Brier 0.0406352173, CM `[[36,1],[1,33]]`; matches cached actual WAV predictions. The older 29-feature candidate remains separately archived. | NONE |
| A1. Validation-selection contamination | PARTIALLY FIXED | `scripts/train_behavior.py::LADDER` and `scripts/finalize_model.py` lines 25–40, 53–61 disclose train fits but **validation-Brier calibration selection**. Validation also informed the supported/count-free revision. `BEHAVIOUR_REPORT.md` §4 records each selection stage; `reports/experiments.csv` retains both candidates and ablations. Historical bias cannot be removed. No further model choice made in this review. | DOCUMENT LIMITATION |
| A2. Early-call failure | PARTIALLY FIXED | `reports/metrics/prefixes.json`, `scripts/evaluate_prefixes.py`, report §15. Historical frozen-score BA 15/30/60/full = 0.512321/0.621622/0.823132/0.971781. Corrected Boolean policy applied to cached scores is in `delta/decision_policy_impact.json`; weak early evidence remains. | DOCUMENT LIMITATION |
| A3. No evidence silently becomes synthetic | FIXED | `behavior/decision.py::evidence_state/challenge_response`; `behavior/service.py::detect`; `behavior/inference.py` debug state. Zero quality or <2 events yields non-flag `false`, confidence 0.5 and `X-Behavior-Evidence: insufficient_evidence`. `tests/test_delta_review.py::test_low_evidence_is_explicit_abstention_not_synthetic` covers silence, sparse speech, caller-only, agent-only and one usable event; HTTP regression and final isolated WAV smoke cover the adapter/runtime. Full-call validation has zero fallback cases and is unaffected. | NONE |
| A4. Caller/source/engine/session disjointness | NOT VERIFIABLE | Organizer README; `behavior/dataset.py`; `reports/metrics/dataset_audit.json`: no speaker/session/engine IDs, no byte-identical duplicates. `behavior/calibrate.py` uses call-stratified folds. Neither hashes nor preserved official membership establish derivative/source/engine/session independence. | DOCUMENT LIMITATION |
| A5. Reaction consistency as the mechanism | PARTIALLY FIXED | `behavior/features.py::summarize`, deployed feature list and `delta/counts_missingness.json`: interruption-stop variability supported by 3 human +10 synthetic train calls; repeated recovery 0+0, both excluded from final variability inputs. Response variability supported by 113+163, barge overlap by 31+54. `final_family_ablations.json`: without consistency AUC 0.978537 vs final 0.991256; without interruption family 0.989666. No universal or causal mechanism established. | DOCUMENT LIMITATION |
| B1. Duration/silence/speech/count shortcuts | PARTIALLY FIXED | Final artifact and `delta/counts_missingness.json` exclude five raw event-count inputs, duration, edge silence, totals and direct agent-duration summaries. Rates, outcome fractions, event-minimum availability and call-phase proxies remain. `shortcut_controls.json`: duration-matched 23 pairs, final AUC 0.998110 vs global 0.875236; controls only a duration explanation. | DOCUMENT LIMITATION |
| B2. Agent-channel protocol signal | PARTIALLY FIXED | `features.py::FEATURE_SETS['no_agent']` only removes direct agent-duration summaries. One fixed `scripts/delta_review_checks.py` agent-only diagnostic using four marginal agent timings: validation AUC 0.877583, Brier 0.170917, CM `[[24,13],[2,32]]` (`delta/nuisance_diagnostics.json`). Agent context is intentional; protocol-independent caller adaptation is unproven. | DOCUMENT LIMITATION |
| B3. Recording/loudness differences | PARTIALLY FIXED | `reports/metrics/shortcut_audit.csv` caller train RMS medians approximately 0.038 human/0.078 synthetic; RMS is not an input. Existing `robustness.json` gain/noise probes and `delta/class_specific_vad.json` expose rather than eliminate confounding. No recording-only classifier or causal/manual-boundary control was added. | DOCUMENT LIMITATION |
| B4. Missingness as a classifier | PARTIALLY FIXED | 24 primary values +15 missing flags =39 weights +1 intercept, plus 2 sigmoid parameters. One fixed missingness-only diagnostic: validation AUC 0.629173, Brier 0.238340, CM `[[21,16],[11,23]]`. Flags account for mean 19.11% of absolute logit terms on validation (`delta/counts_missingness.json`), a noncausal arithmetic measure. They matter, but do not alone reproduce full discrimination; retained for correct missing-event handling. | DOCUMENT LIMITATION |
| B5. Automatic reference is not manual truth | NOT VERIFIABLE | `scripts/compare_turns.py`, `vad_selection.json`: three fixed settings, max mean channel IoU on 24 train calls without class labels. `vad_comparison.json`: caller/agent val IoU 0.816/0.951. `events.py` defines operational proxies; no manual event/intent annotations exist. | DOCUMENT LIMITATION |
| B6. Selective normal-response measurements | PARTIALLY FIXED | `behavior/events.py` lines 61–73 measures agent-only → optional silence → caller-only; excludes overlap exits. `tests/test_events.py::test_normal_response` and `test_no_multiple_response_assignment_or_within_agent_pause` verify that definition. Barge-in features represent other events, but do not demonstrate class-neutral response sampling. Report §7 now explicitly states conditional-population bias. | DOCUMENT LIMITATION |
| B7. Censoring and non-independent events | PARTIALLY FIXED | `events.py` unknown stop/overlap/recovery outcomes remain `None`; `features.py` requires 3 observations for variability. `test_barge_in_and_censoring`, `test_new_agent_turn_prevents_recovery_attribution`, `test_no_events_not_fake_latency_zero` pass. One feature vector per call avoids treating each event as a person; call-bootstrap intervals still omit unidentified speaker clustering and selection bias. | DOCUMENT LIMITATION |
| B8. Limited robustness panel | PARTIALLY FIXED | `scripts/robustness.py`, `reports/metrics/robustness.json`: fixed 24-call gain/noise panel had no original errors and zero flips; 71-call threshold/jitter probes had 1–3 flips depending on probe. No codec/packet-loss/echo/unseen-engine guarantee. Existing evidence reused, not expanded. | DOCUMENT LIMITATION |
| VAD acoustic side channel | PARTIALLY FIXED | `behavior/vad.py` is an acoustic boundary estimator. `organizer_to_vad_transfer.json`: E_consistency same-weight AUC 0.985692 vs organizer 0.994436. `delta/class_specific_vad.json`: caller val IoU human 0.770682/synthetic 0.865602; barge-count deltas −5.14/−0.71. Report §§5,11 explicitly rejects full acoustic independence. | DOCUMENT LIMITATION |
| Calibration fitting versus confidence claims | PARTIALLY FIXED | `calibrate.py` lines 10–22 generates whole-pipeline OOF logits from train and fits train labels only; `finalize_model.py` line 39 selects via validation. Raw/calibrated final Brier 0.054215/0.040635, 8-bin ECE 0.080048, reliability-bin counts in `final_model.json`. No independent nested/grouped calibration comparison. `behavior_confidence=q*abs(2p-1)` remains explicitly heuristic, distinct from probability and quality. | DOCUMENT LIMITATION |
| WAV-only isolated runtime | FIXED | `behavior/inference.py`, `portable.py`, `vad.py`; `scripts/delta_runtime_smoke.py` and `delta/isolated_runtime.json` check a package containing only runtime code +two model artifacts, WAV via stdin, original project reads denied, all network egress denied, no sklearn/pandas/torch. | NONE |
| Final HTTP/malformed WAV behaviour | FIXED | `service.py`, `audio.py`, `tests/test_inference.py`, `tests/test_delta_review.py`. An oversized RIFF chunk initially failed a targeted regression with stdlib `RuntimeError`; now translated into sanitized `AudioError`/HTTP 422. Wrong stereo/rate/encoding/base64/body size rejected; live real-WAV routes and silence checked in `delta/http_smoke.json`. | NONE |
| Docker and remote availability | NOT VERIFIABLE | Static `Dockerfile`/`.dockerignore` review: pinned Python, runtime lock, explicit artifact copies, nonroot user, healthcheck, one worker/bounded concurrency. Docker absent. Actual build/run **NOT VERIFIED LOCALLY**; no public service or cloud provisioned. | DOCUMENT LIMITATION |
| Privacy/secrets/current Git tracking | FIXED | `delta/checks.json`: zero project-scoped tracked files, zero tracked audio or `.env`, confidential paths ignored. `.env.example` only local configuration. `vad.py` disables ORT telemetry; no dataset uploads, credentials or remote Git actions in this agent's recorded sessions, and no network action during delta diagnostics. This is not a claim to audit unrelated historical activity. | NONE |
| All synthetic-voice attacks versus autonomous interaction | PARTIALLY FIXED | Organizer README describes autonomous AI callers. Report §20, module README and student handoff now explicitly exclude demonstrated coverage of human-controlled voice conversion. An advanced agent may imitate timing; no unseen-engine evidence is claimed. | DOCUMENT LIMITATION |

## TRUE BLOCKERS BEFORE JUDGING

**None remain for engineering integration after the two correctness fixes.**
The no-evidence-to-synthetic adapter bug and uncaught malformed-RIFF exception were
corrected with targeted regressions. No classifier parameter or probability threshold
changed. A reachable team-owned final `/detect` service and actual container/cloud
deployment still need environment-specific verification; this is not a claim of
judging-network readiness or a production banking deployment.

## Focused review of completed components

| Final component | Evidence and conclusion |
| --- | --- |
| 1. Saved model | Immutable SHA above; exact 24-feature values/15 flags/calibration linked in `FINAL_BEHAVIOR_MANIFEST.json`. |
| 2. Loader | `inference.py::BehaviorDetector.__init__` validates schema/feature config; loads explicit/default local paths. Frozen hashes identify the selected file; no inference-time fitting. |
| 3. Runtime-only environment | Existing `.venv-runtime`, Python 3.11.13. Isolated smoke asserts sklearn/pandas/torch absent. |
| 4. `/behavior` | Same four JSON scalars; low evidence retains p=0.5, q=0, explicit debug state. Fusion must use quality. |
| 5. Behaviour `/detect` | Corrected non-flag fallback at confidence 0.5 plus evidence header; available evidence retains p≥0.5 rule. |
| 6. Health | `/health` becomes ready after lifespan model load; actual loopback smoke checks it. |
| 7. Malformed requests | Bounded body reading; sanitized 413/415/422; base64 and schema errors never echo content. |
| 8. Stereo/channel contract | `audio.py` requires 2 channels, 8000 Hz, 16-bit PCM. Channel identity cannot be inferred from a header; caller=0 remains caller's input contract. |
| 9. Invalid WAVs | Empty/truncated/incorrect containers and observed malformed chunk bug rejected cleanly. No replacement with silent “human” inputs. |
| 10. Concurrent requests | `test_real_wav_end_to_end_and_no_organizer_dependency` runs two concurrent predictions and compares isolated score; service semaphore=2. This is not a load test. |
| 11. VAD state | `vad.py::probabilities` allocates recurrent state/context inside each call; isolated smoke repeats valid WAV after silence/sparse/single-channel cases. |
| 12. Dockerfile | Static-only review above; no Docker installed or run. |
| 13. Dependency locks | `requirements-runtime.lock` separates 20 runtime dependencies from full training stack; exact installed versions recorded at freeze. Linux wheel/build resolution unverified. |
| 14. Artifact paths | Default paths derived from runtime package root; environment overrides supported. Sandbox package proves relocation with copied artifacts. |
| 15. Startup commands | README `uvicorn behavior.service:app ...` matches service; real loopback smoke uses same entrypoint. |
| 16. README commands | Historical retraining/reproduction now labeled as artifact-overwriting development, not freeze/integration commands. |
| 17. Fusion interface | `BehaviorDetector.predict(wav_bytes)` / `/behavior`; positive=synthetic. No hard-coded fusion weights or team-module changes. Missing Behaviour is no positive vote; final resolution belongs to team fusion. |
| 18. Reports/figures | Frozen-artifact metrics and full-call figures retained. Prefix figure is explicitly historical raw-probability threshold evaluation; corrected Boolean policy table added. |
| 19. Experiment log | Original ladder and candidate histories preserved. Two delta diagnostics are separately labeled reporting-only; no new deployment candidate selected. |
| 20. Reproducibility | Existing scripts retain historical protocol and train-only fit discipline; validation-informed selection is disclosed. Freeze record and runtime/source hashes, not re-executing a search, define integration version. |

## Disposition of ASTRA2's proposed experiments (D1–D11)

These are not all requirements for this bounded freeze review. No broad new audit
or proposed training-only nested-search program was launched.

| Proposal | Evidence reused / bounded disposition |
| --- | --- |
| D1 Source isolation | Existing byte hashes only; missing provenance remains unverified. No speaker identification or derivative-search project. |
| D2 Shortcut ladder | Existing global/count-removal results reused; exactly one fixed missingness-only and one agent-only diagnostic added. Recording-only and other proposed variants not run. |
| D3 Conditional controls | Existing duration strata and 23-pair matching only; other nuisances uncontrolled. |
| D4 Destroy coordination | Not run; protocol-versus-adaptation causal claim remains unverified. |
| D5 Manual events | Not available; no claim that temporal proxies establish intent/backchannels/comprehension. |
| D6 Acoustic invariance | Existing gain/noise/jitter probes only; not definitive. |
| D7 Independent VAD | Existing organizer-vs-Silero transfer/agreement only; both automatic. |
| D8 Consistency contribution | Existing family/feature removals and support counts, no new grouped study. |
| D9 Engine/condition holdouts | Metadata absent; not verifiable. |
| D10 Actual prefixes | Existing 15/30/60/full WAV predictions reused. Only fallback decisions remapped, no horizon-specific model/threshold chosen. |
| D11 Endpoint isolation | Executed as final smoke using existing runtime in a data-free package, with macOS file/network sandbox. |

## DISCLOSED LIMITATIONS

- 282 train calls /71 validation calls; historical validation-informed model,
  calibration-option and count-removal decisions cannot be undone.
- Calibrator **parameters** use train-only OOF logits; **selection** used validation.
  Existing raw OOF/feature-removal results cannot reconstruct a pristine train-only
  model-selection winner. No retrospective seed reset or retraining was attempted.
- Speaker disjointness is the organizer's assertion; source/session/engine grouping,
  derivative duplicates and unseen-engine independence are not verified.
- Global/protocol timing and agent-only statistics are predictive. Rates, missingness
  and workflow can remain shortcuts despite removal of direct count/global inputs.
- Response population is conditioned on non-overlap; VAD/reference agreement differs
  by class. Behaviour boundaries have acoustic dependence; no manual causal event truth.
- Reaction-consistency features contribute collectively, not in a universally
  “AI is more consistent” direction. Specific interruption/recovery evidence is weak.
- Short prefixes remain poor. CPU inference time is not time to obtain evidence.
  Corrected fallback increases or decreases Boolean accuracy through policy alone.
- Calibration is development-distribution evidence with sparse bins, not hidden-set
  confidence or bank attack-prior calibration. Quality/behavior_confidence are heuristics.
- Robustness probes cover a small, limited panel; advanced agents can imitate timing.
  Human-operated voice conversion is outside demonstrated autonomous-caller coverage.
- Docker **NOT VERIFIED LOCALLY**; no cloud/network reachability/load guarantee.

## FINAL DEPLOYMENT FACTS

- Artifact/hash: above, unchanged; complete linkage in `FINAL_BEHAVIOR_MANIFEST.json`.
- Features: **24 primary measurements +15 binary missingness indicators**; 40
  base linear parameters including intercept, plus two calibrator parameters.
- Classifier: L2 logistic C=0.1, train-only median imputation/standardization, ±6 clip;
  no direct global/edge-silence/raw-count/direct-agent-duration classifier inputs.
- Calibration: train-only five-fold call-stratified OOF sigmoid; exact slope/intercept
  copied into the freeze record. Deployment calibration option was validation-selected.
- Decision: with available evidence, synthetic iff p≥0.5. With q≤0 or <2 events,
  internal abstention /demo non-flag false with confidence 0.5; no evidence of humanity.
- Official full-call validation: **69/71**, accuracy 0.9718309859, BA 0.9717806041,
  AUC 0.9912559618, FPR 1/37=0.0270270270, FNR 1/34=0.0294117647,
  Brier 0.0406352173, CM `[[36,1],[1,33]]`. No full-call fallback cases.
- Existing offline CPU latency: p50 **860.446 ms**, p95 **1193.124 ms**, 20 calls;
  peak **125.441 MiB RSS**. Not rerun; excludes HTTP/network, unverified on Linux/cloud.
- Runtime: Python 3.11.13, NumPy 1.26.4, ONNX Runtime 1.20.1, FastAPI 0.115.6,
  Pydantic 2.10.4, Uvicorn 0.34.0 and locked transitive dependencies. No sklearn,
  pandas, torch, ASR, LLM, API key or organizer-turn dependency at inference.
- Integration: `BehaviorDetector.predict(wav_bytes)` → synthetic_probability,
  behavior_confidence, quality_score, event_count; optional debug evidence state.
  HTTP `/health`, `/behavior`, Behaviour-only `/detect`; final fusion owned by team.

Exact final full-suite counts, smoke results, timestamps and post-fix source hashes
are in [FINAL_BEHAVIOR_FREEZE.md](FINAL_BEHAVIOR_FREEZE.md) and its JSON manifest.

## DOCUMENTATION CORRECTIONS MADE

- `README.md`: links frozen integration version and this delta review.
- `behavior/README.md`: exact non-flag fallback/header, four-field fusion contract,
  historical-versus-corrected prefix metrics, nuisance diagnostics, attack scope,
  and historical retraining warning.
- `reports/BEHAVIOUR_REPORT.template.md` and generated `BEHAVIOUR_REPORT.md`:
  explicit validation-selection history; unavailable train-only counterfactual;
  corrected fallback; conditional response sampling; missingness/agent/class-VAD
  diagnostics; limited attack/grouping scope; Brier versus calibration distinction;
  unchanged historical prefix plots plus corrected cached-decision table.
- `reports/WHAT_I_NEED_TO_KNOW.md`: freeze workflow, new policy module, no-evidence
  interpretation, diagnostics, attack scope and no further validation optimization.
- `reports/figures/README.md`: provenance clarifies which existing figures are
  unchanged full-call results and which contain the old raw-threshold prefix policy.
- `reports/experiments.csv`: two clearly diagnostic-only rows; production row/hash
  link finalized without reusing diagnostic results for selection.
- This delta review, machine-readable checks and `FINAL_BEHAVIOR_FREEZE.md`:
  traceable status/evidence, limitations, exact final runtime/artifact verification.

## READY FOR INTEGRATION: YES

This means **engineering-ready for team integration under the documented
development-validation evidence**, conditional on the completed verification
record linked above. It does **not** mean proven robust on Altur's hidden set or
production banking traffic, nor a verified cloud/Docker deployment.

The freeze record is issued only after the one final complete test pass and
isolated-runtime/live-HTTP smoke succeed. No further model optimization follows.
