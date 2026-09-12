"""Finite ASTRA2 delta checks. Two prespecified nuisance diagnostics; no model selection.

All computation uses existing LOCAL feature/timing caches. No audio is processed here,
no network is used, and the deployed model and original metrics are never written.
"""

from datetime import datetime, timezone
import hashlib
import json
import subprocess
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from behavior.config import ROOT, SEED
from behavior.dataset import load_feature_table
from behavior.decision import challenge_response, evidence_state, DECISION_POLICY_VERSION
from behavior.metrics import evaluate
from behavior.model import make_model
from behavior.portable import linear_score
from scripts.common import write_json

BASELINE = ROOT / "reports/reference_baseline/20260912T133418.398155Z"
OUT = ROOT / "reports/metrics/delta_review"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decision_metrics(y, decisions):
    y, decisions = np.asarray(y), np.asarray(decisions)
    tn, fp, fn, tp = skm.confusion_matrix(y, decisions, labels=[0, 1]).ravel()
    return {"n": len(y), "accuracy": skm.accuracy_score(y, decisions),
            "balanced_accuracy": skm.balanced_accuracy_score(y, decisions),
            "precision": skm.precision_score(y, decisions, zero_division=0),
            "recall": skm.recall_score(y, decisions, zero_division=0),
            "f1": skm.f1_score(y, decisions, zero_division=0),
            "fpr": fp / (tn + fp) if tn + fp else None,
            "fnr": fn / (fn + tp) if fn + tp else None,
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def main():
    if (OUT / "checks.json").exists():
        raise SystemExit("Delta diagnostics already exist; inspect them rather than rerunning/tuning.")
    artifact_path = ROOT / "artifacts/models/behavior.json"
    artifact = json.loads(artifact_path.read_text())
    baseline = json.loads((BASELINE / "REFERENCE_BASELINE.json").read_text())
    assert sha(artifact_path) == baseline["deployed_artifact_sha256"]
    table = load_feature_table("vad")
    train, val = table[table.split == "train"], table[table.split == "val"]
    names = artifact["feature_names"]
    missing_names = [names[i] for i in artifact["missing_indices"]]
    # Locked before fitting: one missingness-only and one equivalent agent-only diagnostic.
    specs = {
        "missingness_only": (train[missing_names].isna().astype(float), val[missing_names].isna().astype(float)),
        "agent_only": (train[["agent_speech_s", "agent_turn_count", "agent_turn_mean", "agent_turn_median"]],
                       val[["agent_speech_s", "agent_turn_count", "agent_turn_mean", "agent_turn_median"]]),
    }
    diagnostics = {}
    for label, (xtr, xval) in specs.items():
        start = perf_counter()
        model = make_model("logistic", c=.1)
        model.fit(xtr, train.y)
        diagnostics[label] = {"purpose": "Diagnostic only; excluded from deployment selection",
                              "input_features": xtr.columns.tolist(),
                              "input_interpretation": "binary unavailable flags" if label == "missingness_only" else "four agent-channel marginal timing summaries, no caller or interaction input",
                              "seed": SEED, "hyperparameters": model.named_steps["classifier"].get_params(),
                              "preprocessing": "median/indicators, standardization fit on official train only; fixed clip +/-6",
                              "train": evaluate(train.y, model.predict_proba(xtr)[:, 1]),
                              "val": evaluate(val.y, model.predict_proba(xval)[:, 1]),
                              "runtime_s": perf_counter() - start}
    write_json(OUT / "nuisance_diagnostics.json", diagnostics)

    contributions = {}
    for label, data in (("train", train), ("val", val)):
        x = data[names].to_numpy(dtype=float)
        missing = np.isnan(x)
        expanded = np.column_stack([np.where(missing, artifact["impute"], x), missing[:, artifact["missing_indices"]]])
        z = np.clip((expanded - artifact["mean"]) / artifact["scale"], -6, 6)
        terms = z * np.asarray(artifact["coef"])
        numerical, indicators = terms[:, :len(names)], terms[:, len(names):]
        total_absolute = np.abs(terms).sum(axis=1)
        contributions[label] = {
            "mean_missing_indicator_absolute_term_share": np.mean(np.abs(indicators).sum(axis=1) / np.maximum(total_absolute, 1e-12)),
            "fraction_calls_indicator_signed_magnitude_exceeds_numeric": np.mean(np.abs(indicators.sum(axis=1)) > np.abs(numerical.sum(axis=1))),
            "interpretation": "Descriptive standardized-logit arithmetic, not causal feature importance; correlated/redundant indicators share credit"}
    old_artifact = json.loads((ROOT / "artifacts/models/supported_with_counts.json").read_text())
    counts = {"removed_raw_counts": [n for n in old_artifact["feature_names"] if n.endswith("_count") and n not in names],
              "remaining_primary_features": names, "missing_indicator_count": len(missing_names),
              "missing_indicator_features": missing_names,
              "linear_parameters_including_intercept": len(artifact["coef"]) + 1,
              "calibration_parameters": 2, "contributions": contributions,
              "remaining_proxies": ["event count ratios (interruption_rate, barge_in_rate)",
                                    "observed-outcome fractions", "availability requires event-count minima",
                                    "caller pauses contain agent speech; temporal context uses agent duration"]}
    support = {}
    for col in ("response_latency_std", "barge_overlap_std", "interruption_stop_latency_std", "recovery_latency_std",
                "short_context_response_std", "long_context_response_std"):
        support[col] = {label: int(train.loc[train.y == y, col].notna().sum()) for y, label in ((0, "human"), (1, "synthetic"))}
    counts["train_variability_support"] = support
    write_json(OUT / "counts_missingness.json", counts)

    # One arithmetic consistency check against the exact immutable artifact, using cached features.
    probabilities = linear_score(artifact, val[names])
    measured = evaluate(val.y, probabilities, artifact["threshold"])
    final = json.loads((ROOT / "reports/metrics/final_model.json").read_text())["val"]
    for key in ("accuracy", "balanced_accuracy", "roc_auc", "brier", "fpr", "fnr"):
        np.testing.assert_allclose(measured[key], final[key], rtol=0, atol=1e-12)
    prefix_rows = pd.read_csv(ROOT / "reports/private/prefix_predictions.csv")
    full = prefix_rows[prefix_rows.prefix == "full"].set_index("anon_id").loc[val.anon_id]
    np.testing.assert_allclose(probabilities, full.synthetic_probability, rtol=0, atol=1e-12)
    impact = {}
    for prefix in ("15", "30", "60", "full"):
        group = prefix_rows[prefix_rows.prefix == prefix]
        rows = group.to_dict("records")
        states = [evidence_state(r) for r in rows]
        predictions = [challenge_response(r, artifact["threshold"])["is_synthetic"] for r in rows]
        impact[prefix] = {"probability_metrics_unchanged": evaluate(group.y, group.synthetic_probability),
                          "corrected_adapter_decision_metrics": decision_metrics(group.y, predictions),
                          "available_evidence_calls": states.count("available"),
                          "fallback_calls": states.count("insufficient_evidence"),
                          "fallback_human": int(sum(s == "insufficient_evidence" and y == 0 for s, y in zip(states, group.y))),
                          "fallback_synthetic": int(sum(s == "insufficient_evidence" and y == 1 for s, y in zip(states, group.y))),
                          "method": "Remap cached frozen probabilities/quality/events; no new VAD, training or threshold selection"}
    write_json(OUT / "decision_policy_impact.json", {"policy": DECISION_POLICY_VERSION, "prefixes": impact})
    assert impact["full"]["fallback_calls"] == 0
    assert impact["full"]["corrected_adapter_decision_metrics"]["confusion_matrix"] == [[36, 1], [1, 33]]

    agreement = pd.read_csv(ROOT / "reports/private/vad_comparison.csv")
    agreement = agreement.merge(table[["anon_id", "y"]], on="anon_id", validate="one_to_one")
    class_vad = []
    for (split, y), group in agreement.groupby(["split", "y"]):
        class_vad.append({"split": split, "class": "synthetic" if y else "human", "calls": len(group),
                          "caller_iou": group.ch0_speech_iou.mean(), "agent_iou": group.ch1_speech_iou.mean(),
                          "caller_onset_mae_s": group.ch0_onset_mae_s.mean(), "caller_offset_mae_s": group.ch0_offset_mae_s.mean(),
                          "barge_count_delta": group.barge_in_count_delta.mean()})
    write_json(OUT / "class_specific_vad.json", class_vad)

    # Check project-scoped Git state only; do not inspect unrelated home files.
    tracked = subprocess.run(["git", "ls-files", "-z", "--", str(ROOT.parent)], cwd=ROOT,
                             capture_output=True, check=True).stdout.split(b"\0")
    tracked = [p.decode() for p in tracked if p]
    dangerous = [p for p in tracked if p.lower().endswith(".wav") or p.rsplit("/", 1)[-1] == ".env"]
    assert not dangerous
    protected = ["manifest.csv", "turns/", "artifacts/", "reports/private/", ".env", "../audio/"]
    ignored = {p: subprocess.run(["git", "check-ignore", "-q", "--", p], cwd=ROOT).returncode == 0 for p in protected}
    assert all(ignored.values())
    checks = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "audit_path": "reports/ASTRA2_ADVERSARIAL_REVIEW.md", "audit_sha256": sha(ROOT / "reports/ASTRA2_ADVERSARIAL_REVIEW.md"),
              "reference_baseline": str(BASELINE.relative_to(ROOT)), "artifact_path": "artifacts/models/behavior.json",
              "artifact_sha256": sha(artifact_path), "artifact_unchanged": True,
              "feature_count": len(names), "calibration": artifact["calibration"],
              "threshold": artifact["threshold"], "validation": measured,
              "cached_wav_full_predictions_match": True, "full_validation_fallback_calls": 0,
              "two_diagnostic_specifications_only": list(diagnostics),
              "train_only_counterfactual_selection": "NOT VERIFIABLE: existing raw OOF score and leave-one-feature-out diagnostics are not a complete nested train-only comparison of model families/calibration options; cannot reconstruct an unbiased winner",
              "project_scoped_tracked_files": len(tracked), "tracked_audio_or_dotenv": len(dangerous),
              "confidential_paths_ignored": ignored,
              "network_actions_during_delta_review": "none",
              "privacy_history_scope": "No dataset uploads/API calls/credentials/pushes in this agent's recorded sessions; current tree checks cannot certify unrelated historical activity"}
    assert sha(artifact_path) == baseline["deployed_artifact_sha256"]
    write_json(OUT / "checks.json", checks)
    print(json.dumps({"artifact_unchanged": True, "artifact_sha256": checks["artifact_sha256"],
                      "diagnostics": {k: {"train_auc": v["train"]["roc_auc"], "val_auc": v["val"]["roc_auc"],
                                          "val_brier": v["val"]["brier"], "val_cm": v["val"]["confusion_matrix"]} for k, v in diagnostics.items()},
                      "missingness_contributions": contributions, "support": support,
                      "prefix_policy": {k: {"cm": v["corrected_adapter_decision_metrics"]["confusion_matrix"],
                                             "balanced_accuracy": v["corrected_adapter_decision_metrics"]["balanced_accuracy"],
                                             "fallback_calls": v["fallback_calls"]} for k, v in impact.items()},
                      "class_vad": class_vad, "tracked_project_files": len(tracked)}, indent=2, default=float))


if __name__ == "__main__":
    main()
