**Independent adversarial review — Altur Conversational Behaviour detector**

**Verdict:** The evidence supports discrimination within this dataset. It does **not yet establish engine-independent behavioural detection, reliable early detection, or calibrated confidence under hidden-set conditions**. There is a confirmed validation-selection issue, substantial unresolved shortcut risk, and an observed failure on short calls.

This review inspected the challenge documentation, dataset README, relevant ML learning material, current implementation, saved artifacts, and aggregate reports. No project files were modified, no models were trained, no audio was processed, and no confidential data was uploaded. Existing experimental results are attributed to saved reports, not independently reproduced.

**First, reconcile the headline results.**

The saved artifacts distinguish three models:

| ModelValidation ROC-AUCHumans flagged syntheticSynthetic callers missed |        |      |      |
| ----------------------------------------------------------------------- | ------ | ---- | ---- |
| Global timing baseline                                                  | 0.9086 | 8/37 | 5/34 |
| Temporal + reaction + consistency family                                | 0.9913 | 2/37 | 2/34 |
| Exported 29-feature deployment candidate                                | 0.9881 | 2/37 | 1/34 |

Do not attach the family’s 0.9913 AUC to the deployment candidate’s confusion matrix. The inspected candidate still contains event counts. A newer count-exclusion option exists in the code, but it was not reflected in the exported artifact inspected here. [Saved deployment results](app://-/Users/chai/Documents/hack_mty_2026/hackmty26-main/reports/metrics/final_model.json)

**A. Critical issues that must be addressed before judging**

**1. Official validation has already influenced deployment selection.**

This is confirmed, not hypothetical. The finalization script compares raw and calibrated validation Brier scores and deploys whichever is lower. The implementation plan also explicitly permits model selection using validation performance.

Fitting the calibrator on training predictions does not make the subsequent validation-based selection independent. Here, calibration changed the candidate from four validation errors to three. The positive-slope sigmoid leaves ROC-AUC unchanged, but changes probabilities and the effective decision boundary. [Selection code](app://-/Users/chai/Documents/hack_mty_2026/hackmty26-main/scripts/finalize_model.py:35)

The experiment ladder, family ablations, calibration comparison, and contemplated count removal create multiple opportunities to adapt to these 71 calls. A bounded search is better than an unlimited search, but still creates selection bias.

**Required:** Stop describing validation as untouched or independent of selection. Freeze further decisions using train-only procedures. Previously observed validation results cannot become pristine again through a new random seed or renamed report.

**2. Early-call behaviour already fails.**

The saved evaluation runs the frozen detector on actual cropped WAVs:

| Available audioHuman false positivesSynthetic misses |       |      |
| ---------------------------------------------------- | ----- | ---- |
| 15 seconds                                           | 37/37 | 0/34 |
| 30 seconds                                           | 35/37 | 0/34 |
| 60 seconds                                           | 12/37 | 0/34 |
| Full call                                            | 2/37  | 1/34 |

At 60 seconds, AUC is approximately **0.957**, yet **32.4% of humans are flagged**. That is a concrete example of excellent ranking coexisting with an unacceptable operating threshold. [Prefix results](app://-/Users/chai/Documents/hack_mty_2026/hackmty26-main/reports/metrics/prefixes.json)

**Required:** Either limit the claim to completed calls, or establish horizon-specific training, calibration, thresholds, and an evidence policy using training data. Report observation time separately from processing time.

**3. Insufficient evidence silently becomes a synthetic verdict.**

The module assigns `p = 0.5` when evidence is insufficient. The HTTP adapter applies `p >= 0.5`, producing `is_synthetic: true`. Silence therefore becomes a synthetic verdict through a tie-breaking rule, not detection evidence. The endpoint also ignores the module’s quality score when returning class confidence. [Evidence handling](app://-/Users/chai/Documents/hack_mty_2026/hackmty26-main/behavior/inference.py:17), [endpoint decision](app://-/Users/chai/Documents/hack_mty_2026/hackmty26-main/behavior/service.py:63)

**Required:** Specify and evaluate the fallback policy. The challenge requires a Boolean, so an internal abstention needs an explicit final-system resolution. Do not present that fallback as an evidence-backed classification.

**4. Speaker-disjoint is an organizer assertion, not an independently verified source-disjoint audit.**

The README states that callers do not cross the official splits. Speaker/session/source identifiers are unavailable. Internal calibration uses shuffled, call-stratified folds and may share speakers.

The existing audit reports no byte-identical WAV duplicates. That does **not** exclude:

- The same recording cropped, re-encoded, or padded differently.
- Repeated caller sessions or reused caller audio.
- Shared synthesis engines, voice families, generation configurations, or recording batches.
- Shared bank-agent prompts and protocol branches.

Shared agent prompts are not automatically leakage—the agent is intentionally available—but can make evaluation much easier without testing new interaction policies.

**Required:** Obtain anonymous grouping/provenance information where possible. Distinguish caller-disjoint, source-disjoint, engine-disjoint, and session-disjoint evidence. Without those identifiers, explicitly leave the latter claims unverified.

**5. “Reaction consistency” is not established as the mechanism.**

Only **13 training calls—3 human and 10 synthetic—support interruption-stop variability**. Three events make a statistic computable; they do not make it reliable.

The saved WAV comparison gives consistency an AUC increment of approximately **0.0072**, with a call-bootstrap interval whose lower endpoint is zero. That is suggestive, not decisive, and the interval does not account for model selection or unidentified speaker clusters.

Excluding interruption variability is justified. However, the current artifact retains interruption means, yield fractions, counts, and missingness indicators. It has not excluded interruption information generally.

**Required:** Withdraw universal claims that synthetic callers are more consistent. Separate response-gap variability, overlap variability, and interruption-recovery variability. Each needs its own support counts and evidence.

**B. Medium-risk concerns worth testing**

| ConcernWhy it matters here                           |                                                                                                                                                                                                                                                       |
| ---------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Duration, silence, speech totals and turn counts** | The global baseline already reaches 0.909 AUC. Removing duration itself does not remove proxies encoded by counts, missingness, rates or call phase.                                                                                                  |
| **Agent behaviour**                                  | Agent timing may reveal how the collection protocol treats different callers. The feature family called `no_agent` only removes direct agent-turn-duration features; it still uses both channels.                                                     |
| **Recording differences**                            | The existing training audit reports median caller RMS around 0.038 for humans versus 0.078 for synthetic callers. This is a substantial acoustic confound to investigate, even though RMS is not a direct production feature.                         |
| **Missingness as a classifier**                      | Median imputation adds missingness indicators. Availability of sufficient overlap or response events can encode class, duration, engine or VAD behaviour. The exported model has 45 parameters, not simply 29 feature coefficients plus an intercept. |
| **Automatic-reference dependence**                   | VAD settings were selected to resemble organizer segmentation. Agreement with another automatic system is not agreement with true speech boundaries or meaningful interruptions.                                                                      |
| **Selective response measurement**                   | Response latency includes agent-only → silence → caller-only transitions. Overlapping responses are excluded, potentially making the measured response population class-dependent.                                                                    |
| **Censoring and event dependence**                   | Recovery requires an observed yield and subsequent caller onset. Missing recovery differs from instant recovery. Multiple events within a call are not independent people or independent experiments.                                                 |
| **Limited robustness probes**                        | ±3 dB gain, 25 dB noise and small timing jitter do not cover codecs, packet loss, echo, asymmetric channels or unseen engines. Zero flips on 24 selected calls is weak protection against those failures.                                             |

Duration matching provides useful counterevidence: the saved 23-pair comparison retains approximately 0.989 behaviour AUC. But it only challenges a **duration-only explanation**. It does not control silence, caller loudness, event availability, agent policy or synthesis engine. [Shortcut controls](app://-/Users/chai/Documents/hack_mty_2026/hackmty26-main/reports/metrics/shortcut_controls.json)

**Behavioural validity needs narrower definitions.**

Response gaps, directional overlap, yielding, and subsequent resumption are **interaction proxies** because they relate the channels. They do not establish a completed question, an intentional interruption, surprise, comprehension, or causal recovery.

Caller segment length and pause variation describe speech activity but are not necessarily interactive. Duration, edge silence and speech totals are global recording/protocol properties.

The strong global baseline weakens the claim that high AUC demonstrates sophisticated behaviour. It does not prove the final model uses only shortcuts. Demonstrating an increment over that baseline is necessary, but insufficient to establish the proposed mechanism.

**VAD can covertly become the acoustic detector.**

A frozen acoustic network still responds to loudness, timbre, breath, noise and synthesis artifacts. If those properties systematically change detected boundaries, the downstream “temporal” classifier can inherit acoustic discrimination.

The inspected configuration uses 32 ms frames, 240 ms minimum speech/silence and 32 ms padding. Existing comparisons show caller boundary disagreement with organizer segments on the order of tenths of a second—large relative to some reaction thresholds. Neither segmentation is manual ground truth. VAD parameter sensitivity is also explicitly recognized in [Silero’s documentation](https://github.com/snakers4/silero-vad/wiki/FAQ).

**Most likely hidden-set collapse mechanisms, in priority order:**

1. Faster synthetic engines lose the latency signature learned from slower ASR–LLM–TTS pipelines.
2. Slow human responses caused by network delay, unfamiliar prompts or difficult call conditions resemble training bots.
3. A different agent policy changes overlaps, silence and event availability.
4. Shorter clips trigger the already observed human false-positive problem.
5. Codec, gain, echo or noise changes alter VAD boundaries and missingness.
6. Hidden engine/source groups expose memorized collection signatures.
7. Changed class prevalence or score distributions invalidate confidence and thresholds even if ranking remains good.

A human controlling real-time voice conversion is another important scope boundary: synthetic acoustics can coexist with human turn-taking. The dataset README describes autonomous AI callers; success on that population does not establish detection of every synthetic-voice attack.

**C. Things that appear methodologically sound**

These are safeguards, not proof of generalization:

- Explicit feature allowlists exclude IDs, labels, split columns and filenames from classifier inputs.
- Model fitting uses training rows; imputation, scaling and support selection live inside the fitted pipeline.
- Calibration generates training out-of-fold scores by cloning the whole pipeline.
- VAD configuration selection uses 24 training calls without class-label optimization.
- The current extractor derives duration from WAV frames, avoiding inaccurate manifest durations.
- Stereo channels remain separate; VAD recurrent state is initialized per call.
- Unobserved variability is missing rather than falsely assigned zero.
- The code acknowledges temporal events as proxies and handles several boundary-censoring cases.
- Saved evaluations include Brier, log loss, class error rates, ablations, and actual cropped-WAV inference.
- The current inference path reads WAV bytes plus local model artifacts; it does not require organizer turn JSONs.

The last point establishes **architectural WAV-only compliance**, not proof that the deployed server is reachable or correctly packaged. Existing tests check organizer-turn independence, but I did not execute them. The latency report excludes HTTP/network transfer and says Docker was not tested.

**D. Exact experiments that could validate or falsify the claims**

These are proposed experiments for the implementation team; none were run for this review. Keep development on official training data and freeze criteria before inspecting outcomes.

| ExperimentExact designEvidence against the claim |                                                                                                                                                                                                                                                                               |                                                                                                                                                                                         |
| ------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **1. Source-isolation audit**                    | Use existing hashes and anonymous provenance; check decoded-audio equivalence and suspected cropped/re-encoded duplicates locally. Keep every derivative of a source in one group.                                                                                            | Cross-split caller/source reuse undermines independent evaluation. Missing provenance leaves the claim unresolved.                                                                      |
| **2. Shortcut ladder**                           | On identical training folds compare duration-only; silence-only; counts-only; caller-speech-only; recording-only; missingness-only; agent-only; and interaction models. Fit every transform inside folds.                                                                     | Interaction fails to improve held-out loss/error rates over nuisance baselines.                                                                                                         |
| **3. Conditional shortcut control**              | Match or stratify held-out training calls on duration, silence, caller speech and recording condition, using predeclared rules. Report retained sample sizes and both classes in every stratum.                                                                               | Performance disappears when nuisance distributions overlap. Poor overlap means the claim cannot be identified from this dataset.                                                        |
| **4. Destroy channel coordination**              | From cached training intervals, circularly shift one channel by several predeclared large offsets. Control duration, speech totals and boundary artifacts. Compare frozen-model score changes; separately compare train-only models on intact versus randomized coordination. | Similar discrimination after coordination is destroyed suggests marginal/protocol information dominates. A collapse alone is not proof because perturbations create distribution shift. |
| **5. Manual event validity**                     | Select 24 balanced training calls before scoring; annotate fixed windows plus candidate events. Two reviewers mark speech, handoffs, backchannels, genuine interruptions and recovery; record disagreements.                                                                  | Large class-dependent boundary/event errors or weak agreement falsify strong interpretation of the automatic features.                                                                  |
| **6. Acoustic invariance**                       | On that small panel apply timing-preserving gain changes, telephony filtering, codec round trips, noise and controlled cross-channel echo. Compare VAD masks, feature values and scores per call. Repeat with manual intervals.                                               | Scores change strongly through VAD while manually timed features remain stable: acoustic sensitivity is driving “behaviour.”                                                            |
| **7. Independent VAD check**                     | Compare the frozen extractor with a second VAD and manual timing. Test both same-weight transfer and a separately refitted train-only pipeline.                                                                                                                               | Results depend on one segmentation system, especially through class-specific errors.                                                                                                    |
| **8. Consistency contribution**                  | Compare mean/median-only features against added variability on identical grouped folds. Match event counts, separate event types and context, and report uncertainty by call/group.                                                                                           | No repeatable incremental benefit, or benefit only from missingness/event count, fails the consistency claim.                                                                           |
| **9. Engine and condition holdouts**             | Leave out each available synthetic engine/voice family and recording condition. Include disjoint human groups in evaluation. If metadata is absent, use newly collected authorized calls.                                                                                     | Unseen-engine sensitivity or human specificity collapses at the frozen threshold.                                                                                                       |
| **10. Actual prefix evaluation**                 | Train/select horizon-specific candidates within training; test actual 15/30/60/90-second WAV prefixes and event-based stopping. Keep all prefixes from one call together.                                                                                                     | Useful accuracy arrives only near completion, or human FPR remains high despite good AUC.                                                                                               |
| **11. Endpoint isolation**                       | Run a frozen package without mounted manifests, labels, feature caches or turn JSONs, with network egress disabled. Test valid stereo WAVs, silence, sparse speech, concurrency and cold start.                                                                               | Missing files, state-dependent predictions, unacceptable fallback behaviour or timeouts invalidate deployment claims.                                                                   |

For performance comparisons, predeclare the primary metric and practical improvement needed. Use paired group-aware uncertainty where groups exist. Failure to reject a difference on a tiny sample is **inconclusive**, not equivalence.

**Train-only selection and calibration protocol**

1. Preserve the official train/validation membership. Record every prior validation-informed decision.
2. Obtain anonymous caller/source groups. Use five outer grouped folds, reducing the count if group/class support is insufficient.
3. Within each outer training fold, use inner grouped folds to select the small feature/model/VAD candidate set. Refit support selection, imputation and scaling within each fit.
4. For calibration candidates, generate inner out-of-fold logits; fit the calibrator on those logits and labels. Evaluate the complete calibrated pipeline on the untouched outer fold.
5. Select raw versus calibrated probabilities, hyperparameters and thresholds using training-only evidence. Then generate full-training OOF logits for the selected specification, fit the final calibrator, and refit the base model on all training calls.
6. Freeze artifact, feature configuration, threshold and fallback policy before the final official-validation check.

OOF training predictions are the correct input for this calibration approach. **Evaluating a calibrator on the same OOF labels used to fit it is not an independent calibration test.** An outer fold or separate calibration-evaluation split is needed. Full-training refits may also have different score distributions from smaller fold models. See [scikit-learn’s calibration guidance](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV) and [nested-selection guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

If group IDs remain unavailable, call-stratified CV is a fallback diagnostic, not speaker-disjoint evidence. Retrospective CV also cannot erase earlier human adaptation to validation.

**Metrics and confidence**

With synthetic as positive, the inspected candidate has:

- Human false-positive/rejection rate: **2/37 = 5.4%**.
- Synthetic false-negative/acceptance rate: **1/34 = 2.9%**.
- Accuracy: **68/71 = 95.8%**.

Approximate 95% Wilson intervals are **1.5–17.7%** for human FPR and **0.5–14.9%** for synthetic FNR, assuming independent calls. These intervals exclude neither serious human rejection nor substantial attack acceptance; they also omit selection bias and clustering.

Report ROC-AUC alongside fixed-threshold FPR/FNR, sensitivity, specificity, balanced accuracy, precision/PR-AUC with prevalence, Brier, log loss, reliability-bin counts, and failure/coverage rates. Include worst-group and prefix results.

EER is descriptive, not an operating guarantee: recomputing its threshold on each evaluation set can hide deployment threshold failure. The challenge material does not specify that EER or AUC determines ranking. It explicitly mentions calibration for confidence tie-breaking.

The saved Brier of approximately 0.045 is not proof of calibration: Brier also reflects discrimination, validation selected the calibration option, and reliability bins are sparse. [Calibration interpretation](https://scikit-learn.org/stable/modules/calibration.html)

Clarify whether challenge `confidence` means probability of synthetic or confidence in the chosen class. The current adapter returns the latter. The separate `quality × |2p−1|` value is a heuristic, not a probability of correctness.

For illustration, if synthetic prevalence were 1%, carrying the observed sensitivity and FPR unchanged would yield only approximately **15% precision** among flagged calls. That is not a prevalence forecast; it demonstrates why balanced validation precision is not a banking deployment guarantee.

**E. Checklist before trusting the model**

- Every reported metric maps to one immutable artifact and evaluation configuration.
- Validation-informed choices are disclosed; further selection is train-only.
- Caller/source/engine/session isolation is verified or explicitly marked unknown.
- No labels, IDs, split fields or validation-fitted transformations enter inference.
- Missingness-only, agent-only and recording-only baselines have been tested.
- Interaction adds measurable value after nuisance controls.
- Event definitions agree sufficiently with manual review, by class.
- Variability claims report event counts, comparable contexts and uncertainty.
- VAD/acoustic perturbations preserve acceptable decisions and calibration.
- Threshold and fallback policy meet a stated human-FPR/attack-FNR tradeoff.
- Short-call limitations and observation time are shown prominently.
- Calibration is evaluated outside its fitting data.
- The final endpoint works without dataset files or organizer annotations.
- Any fusion improvement uses out-of-fold module predictions and a separate evaluation.
- Unseen-engine/condition evidence exists, or robustness remains an explicit unproven claim.

**F. Fifteen hardest judge questions**

| QuestionEvidence needed to answer                                                                                      |                                                                                                     |
| ---------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------- |
| **1. Why should I believe this detects synthetic callers rather than your dataset’s bot latency?**                     | Engine holdouts, latency-matched comparisons and conditional interaction benefit.                   |
| **2. How many decisions did you change after looking at these 71 calls?**                                              | Complete experiment/decision history, including calibration, feature removal and threshold changes. |
| **3. Which exact artifact achieved the AUC and confusion matrix on this slide?**                                       | Artifact hash, frozen configuration and matching evaluation report.                                 |
| **4. Can the same caller or original recording appear in different folds?**                                            | Anonymous group mapping, derivative-aware duplicate audit and fold assignments.                     |
| **5. Can I classify your labels using only the bank agent’s channel?**                                                 | Agent-only baseline and testing across agent policies/protocol branches.                            |
| **6. Does the score survive after I destroy the relationship between channels?**                                       | Controlled coordination-randomization results and nuisance-preservation checks.                     |
| **7. How do you know these are interruptions rather than backchannels, echo or segmentation errors?**                  | Independently reviewed event annotations, agreement and class-specific error rates.                 |
| **8. You have interruption variability from only three human training calls. What population claim can that support?** | A restricted claim, broader repeated-event evidence, or removal of that claim.                      |
| **9. Why does caller loudness differ by class, and does VAD convert that into your score?**                            | Acoustic controls, manual-boundary comparison and class-specific VAD sensitivity.                   |
| **10. What happens when our next synthetic engine responds as quickly as a human?**                                    | Actual unseen fast-engine calls and frozen-threshold results.                                       |
| **11. Why does your detector flag almost every human at 30 seconds?**                                                  | Acknowledgment of the observed failure and independently evaluated prefix/fallback policy.          |
| **12. Why does silence produce** **`is_synthetic: true`****?**                                                         | Explicit decision policy and end-to-end evaluation of insufficient-evidence cases.                  |
| **13. When you return 99% confidence, what fraction of those calls are actually correct?**                             | Independent reliability evidence, adequate bin counts and documented confidence semantics.          |
| **14. Could your excellent AUC coexist with unacceptable human rejection on our hidden set?**                          | Fixed-threshold class errors, uncertainty and condition-specific stress tests—not AUC alone.        |
| **15. Can I remove every dataset file and send a new WAV to the deployed server right now?**                           | A reproducible isolated deployment, request-level results and measured cold/concurrent latency.     |

The defensible presentation claim is currently limited to **WAV-derived temporal discrimination on the available development distribution**. Claims about universal synthetic consistency, robust early detection, calibrated hidden-set confidence, or unseen-engine generalization require additional evidence.

Challenge source reviewed: hackmty26-altur-challenge.pdf. ML background reviewed: LEARNING\_SUMMARY.pdf.