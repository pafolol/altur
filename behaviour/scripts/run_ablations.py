"""Frozen-plan family removals, permutation control, and duration shortcut checks."""

import json
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
from behavior.config import ROOT, SEED
from behavior.dataset import load_feature_table
from behavior.features import CONSISTENCY
from behavior.model import make_model
from behavior.calibrate import fit_oof_calibrator, apply_calibrator
from behavior.metrics import evaluate
from .common import write_json


def main():
    artifact = json.loads((ROOT / "artifacts/models/behavior.json").read_text())
    names = artifact["feature_names"]
    df = load_feature_table("vad")
    train, val = df[df.split == "train"], df[df.split == "val"]
    groups = {
        "without_consistency": [n for n in names if n in CONSISTENCY],
        "without_caller_turns": [n for n in names if n.startswith("caller_turn_") or n == "caller_pause_median"],
        "without_responses": [n for n in names if "response" in n],
        "without_interruptions": [n for n in names if n.startswith(("interruption", "recovery", "immediate_retake"))],
        "without_barge_ins": [n for n in names if n.startswith("barge")],
        "without_extended_gaps": [n for n in names if n.startswith("extended_gap")],
        "without_counts": [n for n in names if n.endswith("_count")],
    }
    results = {}
    records = []
    for name, remove in groups.items():
        if not remove:
            continue
        selected = [n for n in names if n not in remove]
        model = make_model(c=.1)
        start = perf_counter()
        model.fit(train[selected], train.y)
        cal, _ = fit_oof_calibrator(model, train[selected], train.y)
        trained = perf_counter()
        p = apply_calibrator(cal, model.decision_function(val[selected]))
        inference_ms = (perf_counter() - trained) * 1000 / len(val)
        result = evaluate(val.y, p)
        train_result = evaluate(train.y, apply_calibrator(cal, model.decision_function(train[selected])))
        results[name] = {"removed": remove, "remaining": len(selected), "metrics": result,
                         "train": train_result, "features": selected, "calibrator": cal,
                         "hyperparameters": model.named_steps["classifier"].get_params(),
                         "training_s_including_calibration": trained - start,
                         "inference_ms_per_call_batched": inference_ms}
        model_dir = ROOT / "artifacts/models/ablations"
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "feature_names": selected, "calibrator": cal}, model_dir / f"{name}.joblib")
        records.append({"source": "vad_ablation", "experiment": name, "hypothesis": f"Contribution of {name.removeprefix('without_')}",
                        "feature_set": name, "feature_count": len(selected), "model": "logistic", "c": .1,
                        "training_s": trained - start, "inference_ms_per_call_batched": inference_ms,
                        "capacity": len(model.named_steps["classifier"].coef_[0]) + 3, "capacity_unit": "linear_plus_calibration_coefficients",
                        **{f"train_{k}": train_result[k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "brier")},
                        **{f"val_{k}": result[k] for k in ("accuracy", "balanced_accuracy", "roc_auc", "f1", "brier", "fpr", "fnr", "eer")},
                        "decision": "Diagnostic removal, no iterative validation feature tuning"})
        write_json(ROOT / "reports/metrics/final_family_ablations.json", results)
    rng = np.random.default_rng(SEED)
    control = []
    for _ in range(20):
        model = make_model(c=.1).fit(train[names], rng.permutation(train.y))
        control.append(evaluate(val.y, model.predict_proba(val[names])[:, 1])["roc_auc"])
    pred = pd.read_csv(ROOT / "reports/private/final_predictions.csv")
    val = val.merge(pred[["anon_id", "probability"]], on="anon_id", validate="one_to_one")
    shortcut = pd.read_csv(ROOT / "artifacts/models/vad/predictions.csv")
    val = val.merge(shortcut[["anon_id", "B_shortcut"]], on="anon_id", validate="one_to_one")
    cutpoints = np.quantile(train.duration_s, [.25, .5, .75])
    bins = np.digitize(val.duration_s, cutpoints)
    strata = {}
    for b in range(4):
        subset = val[bins == b]
        strata[f"duration_quartile_{b + 1}"] = {
            "human": int((subset.y == 0).sum()), "synthetic": int((subset.y == 1).sum()),
            "behavior": evaluate(subset.y, subset.probability),
            "shortcut": evaluate(subset.y, subset.B_shortcut)}
    # Matching uses only durations, without replacement; no predictions drive pairing.
    humans = val[val.y == 0].sort_values("duration_s")
    synthetic = val[val.y == 1]
    matched_ids, used = [], set()
    for idx, row in humans.iterrows():
        candidates = synthetic.loc[~synthetic.index.isin(used)]
        if len(candidates):
            j = (candidates.duration_s - row.duration_s).abs().idxmin()
            if abs(candidates.loc[j, "duration_s"] - row.duration_s) <= 5:
                matched_ids.extend([idx, j])
                used.add(j)
    matched = val.loc[matched_ids]
    audit = {"permuted_train_label_auc": {"runs": 20, "mean": float(np.mean(control)),
              "min": min(control), "max": max(control), "all_values": control},
             "duration_bin_edges_from_train_s": cutpoints.tolist(), "duration_strata": strata,
             "duration_matched_val": {"tolerance_s": 5, "pairs": len(used),
                "behavior": evaluate(matched.y, matched.probability), "shortcut": evaluate(matched.y, matched.B_shortcut)},
             "score_spearman_with_duration_val": float(val[["probability", "duration_s"]].corr(method="spearman").iloc[0, 1])}
    write_json(ROOT / "reports/metrics/shortcut_controls.json", audit)
    path = ROOT / "reports/experiments.csv"
    log = pd.read_csv(path)
    log = log[log.source != "vad_ablation"]
    pd.concat([log, pd.DataFrame(records)], ignore_index=True).to_csv(path, index=False)
    print(json.dumps({"family_ablations": {k: {m: v["metrics"][m] for m in ("roc_auc", "balanced_accuracy", "brier")}
                      for k, v in results.items()},
                      "permutation_control": {k: v for k, v in audit["permuted_train_label_auc"].items() if k != "all_values"},
                      "duration_matched": {"pairs": len(used), "behavior_auc": audit["duration_matched_val"]["behavior"]["roc_auc"],
                                           "shortcut_auc": audit["duration_matched_val"]["shortcut"]["roc_auc"]}}, indent=2))


if __name__ == "__main__":
    main()
