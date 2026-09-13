"""
Re-derive every headline claim in PROJECT_TECHNICAL_REPORT.md straight from the project's artifacts.

    .venv/Scripts/python report_assets/verify_report_claims.py

This is what makes the report an AUDIT rather than a reconstruction: if a number in the report drifts from
the file it came from, this exits non-zero. Any FAIL is a defect in the report, not in the project.
"""
import json
import sys
from pathlib import Path

import pandas as pd

R = Path("D:/altur/acoustic")
ok = fail = 0


def check(label, got, want, tol=0.0):
    global ok, fail
    good = (abs(got - want) <= tol) if isinstance(want, (int, float)) and tol else (got == want)
    print(f"{'OK  ' if good else 'FAIL'}  {label:<58} got={got!r} want={want!r}")
    ok, fail = (ok + 1, fail) if good else (ok, fail + 1)


s = json.loads((R / "outputs/dataset_inspection/summary.json").read_text())
cs = pd.read_csv(R / "outputs/dataset_inspection/call_stats.csv")
check("353 calls in the manifest", s["manifest_rows"], 353)
check("zero split overlap", s["split_overlap_ids"], 0)
check("zero duplicate md5", s["duplicate_md5"], 0)
check("train chunks", s["chunks"]["train"]["n_chunks"], 4102)
check("val chunks", s["chunks"]["val"]["n_chunks"], 977)
check("total audio hours", round(s["total_audio_hours"], 3), 14.508, 0.001)
check("caller speech hours", round(s["caller_speech_hours_official"], 3), 3.912, 0.001)

floor = cs["caller_noise_floor_db"].round(5)
digital = (floor == round(-72.24716186523438, 5)).sum()
check("calls at the digital-silence floor", int(digital), 266)
check("that as a percentage", round(digital / len(cs) * 100, 1), 75.4, 0.05)
check("human/synthetic counts", (int((cs.label == "human").sum()), int((cs.label == "synthetic").sum())),
      (150, 203))

r = json.loads((R / "outputs/classifier/wav2vec2_spanish/results.json").read_text())
check("MLP trainable parameters", r["mlp_params"], 197378)
check("epochs actually run", r["mlp_epochs_run"], 18)

cal = json.loads((R / "models/wav2vec2_spanish/calibration_mlp.json").read_text())
check("calibration method", cal["method"], "platt")
check("calibration a", round(cal["a"], 4), 0.9025, 0.0001)
check("calibration b", round(cal["b"], 4), 0.8366, 0.0001)
check("calibration fitted on n", cal["n"], 781)

m = json.loads((R / "models/robust_v2/meta.json").read_text())
check("deployed layer", m["layer"], 5)
check("deployed vad", m["vad"], "silero")
check("deployed variant", m["variant"], "balanced")

dep = json.loads((R / "reports/deployed_model.json").read_text())
check("deployed model pointer", dep["model"], "robust_v2")
check("worst telephone accuracy", round(dep["worst_telephone_accuracy"], 4), 0.8148, 0.0001)

em = pd.read_csv(R / "outputs/robust/evaluation_matrix.csv").set_index(["domain", "model"])["accuracy"]
for dom, v1, v2 in [("altur_val", 1.0, 1.0), ("channel_val_seen", 0.775, 1.0),
                    ("channel_val_unseen", 0.817, 1.0), ("external_ood_channel", 0.620, 0.815),
                    ("external_ood", 0.685, 0.546)]:
    check(f"{dom}: specialist", round(em[(dom, "specialist")], 3), v1, 0.001)
    check(f"{dom}: robust_v2", round(em[(dom, "robust_v2")], 3), v2, 0.001)

sw = json.loads((R / "outputs/fusion/val_weight_sweep.json").read_text())
fused = sw["fused"] if "fused" in sw else sw
for key in ("shipped", "fused", "default"):
    if key in sw and isinstance(sw[key], dict) and "accuracy" in sw[key]:
        fused = sw[key]
        break
check("verifier consulted on", sw["shipped_weights"]["n_verifier_consulted"], 5)
check("settled by primaries", sw["shipped_weights"]["n_settled_by_primaries"], 66)
pl = sw["per_layer_alone"]
check("behaviour alone accuracy", round(pl["behaviour"]["accuracy"], 4), 0.9718, 0.0001)
check("semantic alone accuracy", round(pl["semantic"]["accuracy"], 4), 0.9155, 0.0001)

lat = json.loads((R / "outputs/fusion/latency_benchmark.json").read_text())["specialist"]["all"]
check("median caller speech to confident", lat["median_caller_speech_sec"], 2.33, 0.001)
check("median lead time", lat["median_lead_time_sec"], 132.73, 0.001)
check("early agrees with final", lat["early_agrees_with_final_pct"], 100.0, 0.001)

sc = pd.read_csv(R / "reports/shortcut_check.csv")
row = sc[(sc.feature_set == "all_trivial") & (sc.model == "logreg")].iloc[0]
check("trivial-statistics AUC", round(row.val_auc, 4), 1.0, 0.0001)
check("trivial-statistics accuracy", round(row.val_accuracy, 4), 0.9859, 0.0001)

from scipy.stats import beta
# Clopper-Pearson for 0 errors in 71: the ONE-SIDED 95 % lower bound is 0.05**(1/71) = 95.9 %.
# The two-sided interval's lower bound is 94.9 %; the report states which it means.
check("Clopper-Pearson one-sided 95% lower bound", round(beta.ppf(0.05, 71, 1) * 100, 1), 95.9, 0.05)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
