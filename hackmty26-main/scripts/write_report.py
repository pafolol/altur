"""Render educational Markdown from local measured results, never invented numbers."""

import json
import math
import re
import numpy as np
import pandas as pd
from behavior.config import ROOT

REPORTS = ROOT / "reports"


def read_metric(name):
    return json.loads((REPORTS / "metrics" / name).read_text())


def fmt(value, decimals=3):
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return "not estimable"
    return f"{value:.{decimals}f}" if isinstance(value, (float, np.floating)) else str(value)


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    out += ["| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |" for row in rows]
    return "\n".join(out)


def feature_description(name):
    """Definitions correspond exactly to the feature registry and event code."""
    distributions = {
        "caller_turn": ("Complete merged caller segment durations: end − start. Segments still active at the audio boundary are excluded.",
                        "How the caller occupies the floor.", "VAD fragmentation, phrase length and task structure may dominate.", "[1, 2, 3]", {"mean": "2", "median": "2", "std": "1", "iqr": "1"}),
        "caller_pause": ("Gap between consecutive caller segments: next.start − previous.end; may contain agent speech.",
                         "The caller's overall pattern of yielding and returning.", "Not a pure thinking-time measure; contains agent turn duration.", "[1, 2, 3]", {"median": "2"}),
        "response_latency": ("Observed agent-only → optional silence → caller-only handoffs: caller onset − preceding agent offset. Agent continuation and overlap exits are excluded.",
                             "Normal floor-taking timing after the other participant finishes.", "A segment boundary is not necessarily the end of a semantic turn.", "[0.2, 0.4, 0.6]", {"mean": "0.4", "median": "0.4", "std": "0.2", "iqr": "0.2", "cv": "0.5"}),
        "interruption_stop_latency": ("For qualifying agent interruptions, caller segment end − agent onset, provided caller end is observed. Includes callers that outlast the agent.",
                                      "How long the caller keeps talking after an agent begins over them.", "A stop may be natural utterance completion, not caused by the interruption; events are scarce.", "[0.2, 0.4, 0.6]", {"mean": "0.4", "median": "0.4"}),
        "barge_overlap": ("For qualifying caller interruptions, min(caller end, agent end) − caller onset. Missing when neither endpoint is observed before the clip boundary.",
                          "Persistence in talk-over and how variable that persistence is.", "Padding/crosstalk can create an overlap; timing alone cannot distinguish a backchannel from deliberate interruption.", "[0.2, 0.4, 0.6]", {"mean": "0.4", "std": "0.2", "iqr": "0.2", "cv": "0.5"}),
        "short_context_response": ("Normal response latencies following a complete agent segment shorter than 3 s.",
                                   "Compare variability within a rough, fixed temporal context instead of pooling all events.", "Segment length is not a semantic question category; 3 s is a fixed design cutoff.", "[0.2, 0.4, 0.6]", {"std": "0.2"}),
        "long_context_response": ("Normal response latencies following a complete agent segment at least 3 s long.",
                                  "Response variability after longer agent contributions.", "Few qualifying events and VAD merging can change context membership.", "[0.2, 0.4, 0.6]", {"std": "0.2"}),
    }
    stat_defs = {
        "mean": ("arithmetic mean, sum(x)/n", "At least one observed event; otherwise NaN."),
        "median": ("50th percentile", "At least one observed event; otherwise NaN."),
        "std": ("sample standard deviation, sqrt(sum((x−mean)^2)/(n−1))", "At least 3 observed comparable events; otherwise NaN."),
        "iqr": ("75th percentile − 25th percentile, NumPy linear quantiles", "At least 3 observed comparable events; otherwise NaN."),
        "cv": ("sample standard deviation / arithmetic mean", "At least 3 events, all nonnegative, mean ≥0.1 s; otherwise NaN."),
    }
    ratios = {
        "interruption_rate": ("Qualifying agent-onset interruptions / all merged agent segments.", "3 of 12 agent onsets interrupt → 0.25.", "How frequently the caller was exposed to talk-over.", "Agent policy/segmentation is part of this measurement.", "No agent segments → NaN. With agent segments but no interruptions, zero is a valid observed rate."),
        "interruption_yield_fraction": ("Fraction of interruptions with observed caller end for which caller.end < agent.end − 0.064 s.", "Caller yields in 2 of 3 resolved interruptions → 0.667.", "Probability of relinquishing the floor during agent talk-over.", "Close endpoint ties and early natural utterance endings are ambiguous.", "No resolved interruptions → NaN; unresolved outcomes are excluded."),
        "interruption_immediate_yield_fraction": ("Fraction of resolved interruptions with BOTH yielding and caller stop latency ≤0.5 s.", "1 of 4 resolved interruptions produces a rapid yield → 0.25.", "Distinguishes prompt yielding from longer continued overlap.", "The 0.5 s boundary is a coarse design choice, not an experimentally proven human reflex limit.", "No resolved interruptions → NaN."),
        "barge_in_rate": ("Qualifying caller-onset interruptions / all merged caller segments.", "2 of 10 caller onsets overlap an already active agent → 0.2.", "How often the caller enters before the agent finishes.", "Agent speaking time and VAD boundaries affect the available opportunities.", "No caller segments → NaN; zero barge-ins with caller turns is a valid zero rate."),
        "barge_continues_fraction": ("Fraction of barge-ins with observed agent end where caller.end > agent.end +0.064 s.", "Caller outlasts agent in 3 of 4 barge-ins → 0.75.", "Persistence in taking the floor during talk-over.", "A short agent segment can make this high without any deliberate persistence.", "No barge-ins with observed agent end → NaN."),
        "extended_gap_caller_fraction": ("Fraction of fully observed interior joint silences ≥2 s whose next speaking state is caller-only.", "Caller ends 3 of 5 long joint gaps → 0.6.", "Who takes the floor after a long gap.", "Does not prove that silence was unexpected or intentionally caused by the agent; may encode call flow.", "No completed interior long gaps → NaN. Leading/trailing silence excluded."),
    }
    if name in ratios:
        definition, example, rationale, failure, missing = ratios[name]
        return {"definition": definition, "unit": "fraction (0–1)", "example": example,
                "rationale": rationale, "failure": failure, "missing": missing}
    for prefix in sorted(distributions, key=len, reverse=True):
        if name.startswith(prefix + "_"):
            stat = name[len(prefix) + 1:]
            definition, rationale, failure, numbers, values = distributions[prefix]
            statistic, missing = stat_defs[stat]
            return {"definition": f"{statistic} of: {definition}", "unit": "unitless" if stat == "cv" else "seconds",
                    "example": f"Observed durations {numbers} s give {values[stat]} {'(unitless)' if stat == 'cv' else 's'}.",
                    "rationale": rationale, "failure": failure, "missing": missing}
    raise ValueError(f"No educational definition for final feature {name}")


def main():
    artifact = json.loads((ROOT / "artifacts/models/behavior.json").read_text())
    final = read_metric("final_model.json")
    evaluated = read_metric("frozen_evaluation.json")
    prefixes = read_metric("prefixes.json")
    vad = read_metric("vad_comparison.json")
    families = read_metric("final_family_ablations.json")
    latency = read_metric("latency.json")
    controls = read_metric("shortcut_controls.json")
    robustness = read_metric("robustness.json")
    loo = pd.read_csv(REPORTS / "metrics/feature_leave_one_out.csv").set_index("feature")
    audit = pd.read_csv(REPORTS / "metrics/shortcut_audit.csv")
    log = pd.read_csv(REPORTS / "experiments.csv")
    # Add concrete conclusions to the machine-readable log, derived from the actual results.
    for idx, row in log.iterrows():
        source, exp = row.source, row.experiment
        if source in ("organizer", "vad") and exp.startswith(("A_", "B_", "C_", "D_", "E_", "R_", "G_")):
            source_log = log[log.source == source].set_index("experiment")
            if exp == "A_dummy":
                conclusion = "Always-synthetic majority/prior prediction; BA=0.5 is the minimum meaningful comparator."
            elif exp.startswith("B_"):
                conclusion = f"Global timing alone gives val AUC {row.val_roc_auc:.4f}; strong dataset shortcut evidence."
            else:
                baseline_id = "B_shortcut" if exp == "C_temporal" else "C_temporal" if exp == "D_reaction" else "D_reaction" if exp == "E_consistency" else "E_consistency"
                delta = row.val_roc_auc - source_log.loc[baseline_id, "val_roc_auc"]
                conclusion = f"Val AUC {row.val_roc_auc:.4f}; delta {delta:+.4f} versus {baseline_id}; BA {row.val_balanced_accuracy:.4f}. See family/calibration tradeoff in report."
            log.at[idx, "conclusion"] = conclusion
            log.at[idx, "decision"] = "Keep as reproducible comparator; final selection uses supported, count-free C=0.1 model."
        elif source == "vad_ablation":
            log.at[idx, "conclusion"] = f"Removing this family gives AUC {row.val_roc_auc:.4f}, BA {row.val_balanced_accuracy:.4f}, Brier {row.val_brier:.4f}; no further validation deletion loop."
        elif exp == "SUPPORTED_with_counts":
            log.at[idx, "conclusion"] = "Initial 29-feature supported candidate; count ablation improves validation and reduces duration-proxy risk."
            log.at[idx, "decision"] = "Archive model; final model excludes classifier count inputs."
    log.to_csv(REPORTS / "experiments.csv", index=False)
    dataset = read_metric("dataset_audit.json")
    blocks = {}
    m = final["val"]
    blocks.update(ACCURACY=f"{m['accuracy']:.2%}", AUC=f"{m['roc_auc']:.4f}", BRIER=f"{m['brier']:.4f}",
                  FEATURE_COUNT=str(final["feature_count"]), PARAMETERS=str(final["parameters"]),
                  TRAIN_DATE=artifact["metadata"]["date_utc"],
                  LATENCY_P50=f"{latency['components_ms']['total']['p50']:.0f}",
                  LATENCY_P95=f"{latency['components_ms']['total']['p95']:.0f}",
                  PEAK_MEMORY=f"{latency['peak_process_rss_mib']:.1f}",
                  RAW_BRIER=f"{final['raw_val']['brier']:.4f}",
                  ECE=f"{m['ece_8_bins']:.4f}",
                  OOF_AUC=f"{final['raw_oof_train']['roc_auc']:.4f}",
                  CALIBRATOR=f"slope={artifact['calibration']['slope']:.6f}, intercept={artifact['calibration']['intercept']:.6f}",
                  LOO_RANGE=f"{loo.oof_auc_full_minus_without.min():+.6f} to {loo.oof_auc_full_minus_without.max():+.6f}")
    http = read_metric("http_smoke.json")
    blocks["HTTP_SMOKE"] = table(["Loopback service run", "Startup to ready s", "/behavior request s", "/detect request s"],
                                  [[i + 1, fmt(r["startup_to_ready_s"]), fmt(r["single_call_http_s"]["behavior"]), fmt(r["single_call_http_s"]["detect"])] for i, r in enumerate(http.get("runs", [http]))])
    blocks["TRAINING_TIMES"] = f"Final base-model fitting took {final['training_s']:.4f} s; the five-fold calibration procedure took {final['calibration_training_s']:.4f} s. Batched exported classification averaged {final['inference_ms_per_call_batched']:.4f} ms per call (excluding VAD/features)."
    blocks["DATASET_TABLE"] = table(["Official split", "Human (0)", "Synthetic (1)", "Total"],
                                     [[r["split"], r["human"], r["synthetic"], r["human"] + r["synthetic"]] for r in dataset["split_counts"]])
    blocks["METRICS_TABLE"] = table(["Metric", "Train (resubstitution)", "Official validation"],
                                    [[k, fmt(evaluated["train"][k], 4), fmt(evaluated["val"][k], 4)] for k in
                                     ("accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc_ap", "fpr", "fnr", "eer", "brier", "log_loss", "ece_8_bins")])
    cm = m["confusion_matrix"]
    blocks["CONFUSION"] = table(["True class ↓ / predicted →", "Human", "Synthetic"],
                                [["Human", f"{cm[0][0]} (TN)", f"{cm[0][1]} (FP)"], ["Synthetic", f"{cm[1][0]} (FN)", f"{cm[1][1]} (TP)"]])
    model_rows = []
    confusion_rows = []
    for source in ("organizer", "vad"):
        model_results = read_metric(f"{source}_models.json")
        for row in log[log.source == source].itertuples():
            if row.experiment not in model_results:
                continue
            model_rows.append([source, row.experiment, int(row.feature_count), fmt(row.train_roc_auc), fmt(row.val_roc_auc), fmt(row.val_balanced_accuracy), fmt(row.val_brier)])
            confusion_rows.append([source, row.experiment, str(model_results[row.experiment]["train"]["confusion_matrix"]), str(model_results[row.experiment]["val"]["confusion_matrix"])])
    blocks["MODEL_TABLE"] = table(["Turns", "Experiment", "Inputs", "Train AUC", "Val AUC", "Val BA", "Val Brier"], model_rows)
    blocks["MODEL_CONFUSIONS"] = table(["Turns", "Experiment", "Train [[TN,FP],[FN,TP]]", "Val [[TN,FP],[FN,TP]]"], confusion_rows)
    blocks["ABLATIONS"] = table(["Removed family", "Remaining features", "Val AUC", "Val BA", "Val Brier", "Val CM"],
                                 [[k, v["remaining"], fmt(v["metrics"]["roc_auc"], 4), fmt(v["metrics"]["balanced_accuracy"], 4), fmt(v["metrics"]["brier"], 4), str(v["metrics"]["confusion_matrix"])] for k, v in families.items()])
    blocks["PREFIX_TABLE"] = table(["Prefix", "AUC", "BA", "Brier", "CM [[TN,FP],[FN,TP]]", "Mean events", "Mean quality", "p50 / p95 ms", "Insufficient evidence"],
                                    [[k, fmt(v["metrics"]["roc_auc"]), fmt(v["metrics"]["balanced_accuracy"]), fmt(v["metrics"]["brier"]), str(v["metrics"]["confusion_matrix"]), fmt(v["mean_event_count"], 2), fmt(v["mean_quality"]), f"{v['latency_p50_ms']:.0f} / {v['latency_p95_ms']:.0f}", v["insufficient_evidence_calls"]] for k, v in prefixes.items()])
    blocks["VAD_TABLE"] = table(["Split", "Channel", "Speech IoU", "Matched onset MAE (s)", "Matched offset MAE (s)", "Ref / generated / matched segments per call"],
                                 [[s, "caller" if ch == 0 else "agent", fmt(vad[s][f"ch{ch}_speech_iou"]), fmt(vad[s][f"ch{ch}_onset_mae_s"]), fmt(vad[s][f"ch{ch}_offset_mae_s"]), " / ".join(fmt(vad[s][f"ch{ch}_{key}_segments"], 2) for key in ("reference", "generated", "matched"))] for s in ("train", "val") for ch in (0, 1)])
    blocks["VAD_EVENTS"] = table(["Validation event", "Mean generated − reference", "Mean absolute difference"],
                                  [[key, fmt(vad["val"][f"{key}_delta"]), fmt(vad["val"][f"{key}_absolute_delta"])] for key in ("interruption_count", "barge_in_count", "response_latency_count")])
    transfer = read_metric("organizer_to_vad_transfer.json")
    blocks["TRANSFER"] = table(["Organizer-trained model", "Organizer-input val AUC", "WAV-derived input val AUC (same weights)"],
                                [[name, fmt(read_metric("organizer_models.json")[name]["val"]["roc_auc"], 4), fmt(value["roc_auc"], 4)] for name, value in transfer.items()])
    defs = []
    dictionary = []
    for i, name in enumerate(artifact["feature_names"], 1):
        d = feature_description(name)
        evidence = loo.loc[name]
        dictionary.append({"name": name, **d, "train_observed": int(evidence.train_observed), "oof_auc_full_minus_without": evidence.oof_auc_full_minus_without})
        defs.append(f"### 6.{i}. `{name}`\n\n"
                    f"- **Definition/equation:** {d['definition']}\n- **Unit:** {d['unit']}.\n"
                    f"- **Example:** {d['example']}\n- **Why it may help:** {d['rationale']}\n"
                    f"- **Missing/censored cases:** {d['missing']}\n- **Possible failure:** {d['failure']}\n"
                    f"- **Empirical support:** {int(evidence.train_observed)}/282 training calls observed; removing this feature changes train-internal OOF AUC by **{evidence.oof_auc_full_minus_without:+.6f}** (full minus removed). See the shared caveats below.\n")
    blocks["FEATURE_DICTIONARY"] = "\n".join(defs)
    (REPORTS / "metrics/feature_dictionary.json").write_text(json.dumps(dictionary, indent=2) + "\n")
    blocks["MISSING_INDICATORS"] = ", ".join(f"`{artifact['feature_names'][i]}__missing`" for i in artifact["missing_indices"])
    selected_audit = ["wav_duration_s", "leading_silence_s", "trailing_silence_s", "caller_turn_count", "agent_turn_count", "caller_speech_s", "agent_speech_s", "caller_rms", "agent_rms", "channel_correlation", "caller_digital_zero_fraction", "agent_digital_zero_fraction", "sample_rate", "bits_per_sample", "channels"]
    blocks["SHORTCUT_TABLE"] = table(["Measurement", "Split", "Human median", "Synthetic median", "Spearman(label)"],
                                     [[r.feature, r.split, fmt(r.human_median), fmt(r.synthetic_median), fmt(r.spearman_label)] for r in audit[audit.feature.isin(selected_audit)].itertuples()])
    delta = final["versus_shortcut"]
    blocks["SHORTCUT_DELTA"] = f"{delta['delta_auc']:+.4f}, with paired call-bootstrap 95% interval [{delta['ci95_call_bootstrap'][0]:+.4f}, {delta['ci95_call_bootstrap'][1]:+.4f}]"
    delta = final["consistency_minus_reaction"]
    blocks["CONSISTENCY_DELTA"] = f"{delta['delta_auc']:+.4f}, with paired call-bootstrap 95% interval [{delta['ci95_call_bootstrap'][0]:+.4f}, {delta['ci95_call_bootstrap'][1]:+.4f}]"
    matched = controls["duration_matched_val"]
    blocks["MATCHED"] = f"{matched['pairs']} pairs ({matched['behavior']['n']} validation calls), ±5 s matching: Behaviour AUC {matched['behavior']['roc_auc']:.4f}, global timing AUC {matched['shortcut']['roc_auc']:.4f}."
    blocks["STRATA"] = table(["Duration quartile (train boundaries)", "Val human / synthetic", "Behaviour AUC / BA", "Shortcut AUC / BA"],
                              [[k, f"{v['human']} / {v['synthetic']}", f"{fmt(v['behavior']['roc_auc'])} / {fmt(v['behavior']['balanced_accuracy'])}", f"{fmt(v['shortcut']['roc_auc'])} / {fmt(v['shortcut']['balanced_accuracy'])}"] for k, v in controls["duration_strata"].items()])
    blocks["ROBUSTNESS"] = table(["Probe", "Calls", "Original AUC", "Perturbed AUC", "Perturbed BA", "Mean absolute Δp", "Decision flips"],
                                  [[k, v["metrics"]["n"], fmt(v["original_same_subset"]["roc_auc"]), fmt(v["metrics"]["roc_auc"]), fmt(v["metrics"]["balanced_accuracy"]), fmt(v["mean_absolute_probability_change"], 4), v["decision_flips"]] for k, v in robustness.items()])
    blocks["LATENCY_COMPONENTS"] = table(["Component", "Mean ms", "p50 ms", "p95 ms"],
                                          [[k] + [fmt(v[s], 3) for s in ("mean", "p50", "p95")] for k, v in latency["components_ms"].items()])
    blocks["STARTUP"] = table(["Fresh process", "Imports s", "Model load s", "Total init s", "Process wall s"],
                               [[i + 1] + [fmt(r[k]) for k in ("import_s", "model_load_s", "total_startup_s", "process_wall_s")] for i, r in enumerate(latency["startup_fresh_processes"])])
    examples = json.loads((REPORTS / "private/error_examples.json").read_text())
    error_text = []
    for example in examples:
        r = example["measurements"]
        error_text.append(f"### {example['alias']}: {'human flagged as synthetic' if example['type'] == 'FP' else 'synthetic caller missed'}\n\n"
                          f"The frozen model assigned P(synthetic)={example['synthetic_probability']:.4f}. "
                          f"The call lasted {r['duration_s']:.2f} s, with {r['response_latency_count']:.0f} normal responses, "
                          f"{r['interruption_count']:.0f} agent interruptions and {r['barge_in_count']:.0f} caller barge-ins. "
                          f"Quality={r['quality']:.3f}. These counts are diagnostics, not classifier inputs.\n\n"
                          + table(["Feature pushing in the wrong direction", "Standardized logit contribution"],
                                  [[a["feature"], fmt(a["standardized_logit_contribution"], 4)] for a in example["largest_wrong_direction_contributions"]])
                          + f"\n\nOrganizer event counts (responses / agent interruptions / barge-ins): "
                          f"{example['organizer_event_counts']['response_latency_count']} / {example['organizer_event_counts']['interruption_count']} / {example['organizer_event_counts']['barge_in_count']}. "
                          "Differences between organizer and generated timing are a plausible contributor, not a proven explanation. "
                          "A large logit contribution explains the fitted arithmetic, not the caller's intent or the cause of the mistake.\n\n"
                          f"![{example['alias']} local temporal timeline]({example['figure']})\n")
    blocks["ERROR_EXAMPLES"] = "\n".join(error_text)
    blocks["PACKAGE_TABLE"] = table(["Runtime/training component", "Recorded version"], [[k, v] for k, v in artifact["metadata"]["packages"].items()])
    template = (REPORTS / "BEHAVIOUR_REPORT.template.md").read_text()
    for key, value in blocks.items():
        template = template.replace("{{" + key + "}}", value)
    remaining = re.findall(r"\{\{[^}]+\}\}", template)
    if remaining:
        raise ValueError(f"Unfilled report fields: {remaining}")
    (REPORTS / "BEHAVIOUR_REPORT.md").write_text(template)
    print(f"Rendered educational report: {len(template.split())} words; {len(dictionary)} fully defined final features.")


if __name__ == "__main__":
    main()
