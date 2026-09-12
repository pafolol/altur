"""One evidence-motivated supported-feature candidate, calibration and JSON export."""

import argparse
import json
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from behavior.config import ROOT, SEED
from behavior.dataset import load_feature_table
from behavior.features import FEATURE_SETS
from behavior.metrics import evaluate, bootstrap_auc_difference
from behavior.model import make_model, export_linear, linear_score
from behavior.calibrate import fit_oof_calibrator, apply_calibrator
from behavior.vad import VAD_RELEASE, VAD_COMMIT, MODEL_SHA256
from .common import provenance, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude-counts", action="store_true", help="Audit-selected final variant: counts only inform evidence quality")
    args = parser.parse_args()
    df = load_feature_table("vad")
    train, val = df[df.split == "train"], df[df.split == "val"]
    names = [n for n in FEATURE_SETS["no_agent"] if not (args.exclude_counts and n.endswith("_count"))]
    # This support rule uses NO labels; refitted within each calibration fold.
    model = make_model(c=.1, support_min=20)
    start = perf_counter()
    model.fit(train[names], train.y)
    fit_s = perf_counter() - start
    calibration_start = perf_counter()
    calibrator, oof = fit_oof_calibrator(model, train[names], train.y)
    calibration_s = perf_counter() - calibration_start
    raw_train, raw_val = (model.predict_proba(d[names])[:, 1] for d in (train, val))
    calibrated_train, calibrated_val = (apply_calibrator(calibrator, model.decision_function(d[names])) for d in (train, val))
    raw_metrics, calibrated_metrics = evaluate(val.y, raw_val), evaluate(val.y, calibrated_val)
    use_calibration = calibrated_metrics["brier"] <= raw_metrics["brier"]
    # Two-way, explicitly validation-based calibration choice. Neither candidate fits validation.
    selected_p = calibrated_val if use_calibration else raw_val
    artifact_path = ROOT / "artifacts" / "models" / "behavior.json"
    if args.exclude_counts and artifact_path.exists():
        old = json.loads(artifact_path.read_text())
        if len(old["feature_names"]) == 29:
            write_json(ROOT / "artifacts/models/supported_with_counts.json", old)
            # Only these two results were just evaluated for the 29-feature model.
            # Do not relabel previously cached prefix/robustness results from another model.
            for filename in ("final_model.json", "final_family_ablations.json"):
                source = ROOT / "reports/metrics" / filename
                if source.exists():
                    write_json(ROOT / "reports/metrics/initial_supported29" / filename, json.loads(source.read_text()))
    metadata = {**provenance(), "source": "wav_silero", "family": "supported_no_agent_no_counts" if args.exclude_counts else "supported_no_agent",
                "vad_release": VAD_RELEASE, "vad_commit": VAD_COMMIT, "vad_model_sha256": MODEL_SHA256,
                "feature_table_metadata": json.loads((ROOT / "artifacts/features/vad.meta.json").read_text()),
                "training_calls": len(train), "validation_calls": len(val), "c": .1,
                "support_min_observed_train": 20, "class_weight": None,
                "classifier_hyperparameters": model.named_steps["classifier"].get_params(),
                "selection": "Discard direct agent-duration inputs and unsupported timing measurements; strong regularization; event counts excluded after explicit shortcut ablation" if args.exclude_counts else "Discard direct agent-duration inputs and unsupported timing measurements; strong regularization",
                "calibration_selection": "Lower validation Brier among raw and train-OOF sigmoid; no validation fitting",
                "vad_config": json.loads((ROOT / "artifacts" / "vad" / "config.json").read_text())}
    artifact = export_linear(model, names, artifact_path, metadata, calibrator if use_calibration else None)
    expected = calibrated_val if use_calibration else raw_val
    np.testing.assert_allclose(linear_score(artifact, val[artifact["feature_names"]]), expected, atol=1e-12)
    joblib.dump({"model": model, "feature_names": names, "calibrator": calibrator, "use_calibration": use_calibration},
                ROOT / "artifacts" / "models" / "final_training_bundle.joblib")
    # Critical same-weights transfer test BEFORE any VAD retraining.
    transfer = {}
    for exp in ("B_shortcut", "C_temporal", "D_reaction", "E_consistency", "E_no_agent"):
        b = joblib.load(ROOT / "artifacts" / "models" / "organizer" / f"{exp}.joblib")
        transfer[exp] = evaluate(val.y, b["model"].predict_proba(val[b["feature_names"]])[:, 1])
    write_json(ROOT / "reports" / "metrics" / "organizer_to_vad_transfer.json", transfer)
    records = df[["anon_id", "split", "y", "quality", "event_count"]].copy()
    records["raw_probability"] = model.predict_proba(df[names])[:, 1]
    records["probability"] = linear_score(artifact, df[artifact["feature_names"]])
    private = ROOT / "reports" / "private"
    private.mkdir(parents=True, exist_ok=True)
    records.to_csv(private / "final_predictions.csv", index=False)
    selected_names = artifact["feature_names"]
    # Per-feature diagnostic only, NOT a validation-driven deletion loop.
    base = make_model(c=.1)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    full_oof = cross_val_predict(base, train[selected_names], train.y, cv=cv, method="predict_proba")[:, 1]
    base_auc = evaluate(train.y, full_oof)["roc_auc"]
    loo = []
    for name in selected_names:
        subset = [f for f in selected_names if f != name]
        p = cross_val_predict(clone(base), train[subset], train.y, cv=cv, method="predict_proba")[:, 1]
        m = evaluate(train.y, p)
        loo.append({"feature": name, "train_observed": int(train[name].notna().sum()),
                    "oof_auc_without": m["roc_auc"], "oof_auc_full_minus_without": base_auc - m["roc_auc"],
                    "decision": "Diagnostic only; correlated features need joint interpretation"})
    pd.DataFrame(loo).to_csv(ROOT / "reports" / "metrics" / "feature_leave_one_out.csv", index=False)
    inference_start = perf_counter()
    linear_score(artifact, val[selected_names])
    inference_ms = (perf_counter() - inference_start) * 1000 / len(val)
    result = {"selected_feature_names": selected_names, "feature_count": len(selected_names),
              "parameters": len(artifact["coef"]) + 1, "training_s": fit_s,
              "calibration_training_s": calibration_s, "inference_ms_per_call_batched": inference_ms,
              "train": evaluate(train.y, calibrated_train if use_calibration else raw_train),
              "val": evaluate(val.y, selected_p), "raw_val": raw_metrics,
              "calibrated_val": calibrated_metrics, "calibration_selected": use_calibration,
              "calibrator": calibrator, "raw_oof_train": evaluate(train.y, 1 / (1 + np.exp(-oof))),
              "limitation": "Call-stratified internal folds may share speakers; official val is the main generalization check"}
    pred = pd.read_csv(ROOT / "artifacts" / "models" / "vad" / "predictions.csv")
    pred = pred[pred.split == "val"]
    result["versus_shortcut"] = bootstrap_auc_difference(val.y, selected_p, pred.B_shortcut)
    result["consistency_minus_reaction"] = bootstrap_auc_difference(val.y, pred.E_consistency, pred.D_reaction)
    write_json(ROOT / "reports" / "metrics" / "final_model.json", result)
    log_path = ROOT / "reports" / "experiments.csv"
    log = pd.read_csv(log_path)
    if args.exclude_counts:
        log.loc[(log.experiment == "FINAL_supported") & (log.feature_count == 29), "experiment"] = "SUPPORTED_with_counts"
    log = log[log.experiment != "FINAL_supported"]
    new = {"source": "vad", "experiment": "FINAL_supported", "hypothesis": "Drop weakly supported and agent-only measurements",
           "feature_set": metadata["family"], "model": "logistic", "c": .1,
           "feature_count": len(selected_names), "training_s": fit_s,
           "calibration_training_s": calibration_s, "inference_ms_per_call_batched": inference_ms,
           "capacity": len(artifact["coef"]) + 3, "capacity_unit": "linear_plus_calibration_coefficients",
           "conclusion": "Selected as deployment model after calibration comparison",
           "decision": "Deploy JSON artifact; preserve all baselines",
           **{f"train_{k}": result["train"][k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "brier")},
           **{f"val_{k}": result["val"][k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "f1", "fpr", "fnr", "eer", "brier")}}
    pd.concat([log, pd.DataFrame([new])], ignore_index=True).to_csv(log_path, index=False)
    print(json.dumps({"final_feature_count": len(selected_names), "calibration_selected": use_calibration,
                      "val_auc": result["val"]["roc_auc"], "val_accuracy": result["val"]["accuracy"],
                      "confusion_matrix": result["val"]["confusion_matrix"], "val_brier": result["val"]["brier"],
                      "training_s": fit_s, "calibration_training_s": calibration_s,
                      "same_weights_organizer_to_vad_auc": {k: v["roc_auc"] for k, v in transfer.items()}}, indent=2))


if __name__ == "__main__":
    main()
