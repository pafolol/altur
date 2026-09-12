# Figure provenance after the ASTRA2 delta review

These are the original completed-module figures, preserved before the delta review.
The 24-feature model bytes, fitted parameters and probability threshold are unchanged.
The authoritative artifact hash is in `../FINAL_BEHAVIOR_FREEZE.md` and
`../FINAL_BEHAVIOR_MANIFEST.json`.

- `confusion_matrix.png`, `roc_pr.png`, `calibration_scores.png`,
  `logistic_coefficients.png`, `feature_correlations.png`, and the final-error
  timelines describe the exact frozen final model. Full-call scores/decisions are unchanged.
- `ablation_ladder.png` compares distinct named organizer/VAD feature-family models;
  it is not a set of alternate metrics for the same deployed artifact.
- `final_family_ablations.png` shows training-refitted family removals, not deployment
  changes made during the delta review.
- **`prefix_performance.png` is historical raw probability-threshold evaluation**
  (`p>=0.5`, including neutral insufficient-evidence ties). It is not the corrected
  Boolean adapter policy. See `../metrics/delta_review/decision_policy_impact.json`
  and report §15 for corrected non-flag fallback matrices/coverage. The original
  probability AUC/Brier, event quality and latency traces remain valid and unchanged.
- `behavior_distributions.png`, `shortcut_distributions.png`, `vad_agreement.png`,
  and `robustness.png` show the labeled exploratory/evaluation evidence, not causal
  mechanism or comprehensive hidden-set robustness.

Do not combine a curve from one family with another candidate's confusion matrix.
No figures were cherry-picked, removed, or regenerated to seek a better validation result.
