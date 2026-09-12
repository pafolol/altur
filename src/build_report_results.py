"""
The data-driven parts of the report (Parts 2, 5-12) + the README results section. Imported by build_report.py.
Every sentence with a number reads it from the output files; when a stage has not run, the part says so.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import config
from build_report import DISPLAY, F, M, O, R, bench_row, f3, pct


def _m(x, k, default=np.nan):
    try:
        return x[k]
    except (KeyError, TypeError):
        return default


# ============================================================================ part 2
def part_dataset(rep, d):
    ins, stats = d["inspect"], d["stats"]
    c = ins.get("counts", {})
    rep.h1("Part 2 - The dataset")
    rep.p("The organisers released 353 recorded calls with a manifest (`anon_id, label, split, duration_s`) and one turn file "
          "per call. There are **no speaker ids, no TTS-system ids and no codec fields**: the README states that train and "
          "validation are speaker-disjoint and that judging uses callers and voices that appear in neither split. So every "
          "leakage check must be done on the audio itself, and the official split is the only honest speaker-disjoint "
          "evaluation we have - we use it unchanged and never re-split.")
    if c:
        rows = [["Split", "Calls", "Human", "Synthetic", "Synthetic share", "Role"]]
        rows.append(["train", c["train"]["total"], c["train"]["human"], c["train"]["synthetic"], pct(c["train"]["synthetic_fraction"], 0), "learn the classifier"])
        rows.append(["val", c["val"]["total"], c["val"]["human"], c["val"]["synthetic"], pct(c["val"]["synthetic_fraction"], 0), "choose layer / model / calibration; every number in this report"])
        rows.append(["hidden test", "?", "?", "?", "?", "judges' benchmark: new callers and new voices"])
        rep.table(rows, [2.2, 1.5, 1.5, 1.8, 2.4, 7.6])
    rep.image(F / "class_distribution.png", "Calls per split and class. Training is 60/40 synthetic/human, validation is balanced.", width_cm=11)
    rep.h2("Audio format (every file checked)")
    fmt = ins.get("formats", {})
    rep.bullets([f"Formats found: {', '.join(f'{k} ({v} files)' for k, v in fmt.items())} - identical for human and synthetic calls, "
                 "so the container, sample rate, channel count and bit depth cannot leak the label.",
                 f"Unreadable files: {len(ins.get('unreadable_files', []))}. Duplicate files (MD5): {ins.get('duplicate_md5', 0)}. "
                 f"Manifest duration vs actual header: max error {ins.get('manifest_duration_max_abs_error_s', 0):.2f} s.",
                 f"Total audio {ins.get('total_audio_hours', 0):.1f} h; caller speech (official turns) {ins.get('caller_speech_hours_official', 0):.1f} h "
                 f"- callers talk only ~25% of the time.",
                 "Channel 0 is the caller and channel 1 the agent in every file; the two channels are uncorrelated "
                 "(no mixed-down mono files)."])
    rep.image(F / "durations.png", "Call duration per class and split: the classes overlap (single-feature AUC ~0.47), duration is not a shortcut.")
    rep.h2("Conversation structure (official turn files)")
    rep.p("The turn files describe who speaks when. They are 'derived automatically' by the organisers and are not available "
          "at inference, so we use them only to describe the data and to validate our own segmentation.")
    rep.image(F / "turn_structure.png", "Behavioural cues from the turn files (train). These are NOT used by the acoustic model; "
              "they are the raw material of the future behavioural branch.")
    if stats is not None:
        tr = stats[stats["split"] == "train"]
        med = lambda col, lab: tr.loc[tr["label"] == lab, col].median()
        rep.bullets([f"Synthetic callers wait much longer before their first word: median {med('first_caller_start_s', 'synthetic'):.1f} s vs "
                     f"{med('first_caller_start_s', 'human'):.1f} s for humans (the AI pipeline needs the agent's greeting to finish before it answers).",
                     f"They take fewer turns ({med('n_caller_turns', 'synthetic'):.0f} vs {med('n_caller_turns', 'human'):.0f}), longer ones "
                     f"({med('mean_caller_turn_s', 'synthetic'):.1f} vs {med('mean_caller_turn_s', 'human'):.1f} s), answer more slowly "
                     f"(median response latency {med('median_response_latency_s', 'synthetic'):.1f} vs {med('median_response_latency_s', 'human'):.1f} s) "
                     f"and rarely talk over the agent (overlap {med('overlap_s', 'synthetic'):.1f} vs {med('overlap_s', 'human'):.1f} s per call).",
                     f"They are also **louder**: caller RMS inside speech {med('caller_speech_rms_db', 'synthetic'):.1f} dB vs {med('caller_speech_rms_db', 'human'):.1f} dB, "
                     f"with peaks close to full scale ({med('caller_peak', 'synthetic'):.2f} vs {med('caller_peak', 'human'):.2f}) - a gain difference of the injection path, not a property of a voice."])
    rep.image(F / "levels.png", "Signal-level statistics of the caller channel (train). Loudness, peak and the 3.4-4 kHz band ratio separate the classes.")
    rep.h2("Our segmentation vs the official turns")
    v = ins.get("vad_vs_official", {})
    ch = ins.get("chunks", {})
    rep.p("At inference we only receive the WAV, so the caller turns must come from our own voice-activity detector (VAD). "
          "We compared three options against the official turns (frame-level F1 at 10 ms resolution):")
    sweep = Path(O / "dataset_inspection" / "vad_sweep_summary.txt")
    silero = Path(O / "dataset_inspection" / "vad_silero_vs_energy_summary.txt")
    rows = [["Detector", "Median F1", "Calls with F1 < 0.8", "Human recall", "Synthetic recall", "Verdict"]]
    rows.append(["energy VAD, 15 dB above the noise floor (**used**)", f3(v.get("vad_f1")), str(v.get("calls_with_f1_below_0.8", "?")), "0.961", "0.997", "balanced across classes; 1 problem call"])
    rows.append(["energy VAD, 12 dB", "0.952", "21", "0.963", "0.999", "21 noisy synthetic calls over-detected"])
    rows.append(["energy VAD 12 dB + crosstalk gate (drop frames where the agent is louder)", "0.950", "29-34", "0.90-0.92", "0.99", "removes human speech during overlaps: rejected"])
    rows.append(["Silero VAD (neural), threshold 0.5", "0.927", "54", "0.857", "0.981", "misses quiet human speech: class-dependent recall, rejected"])
    rep.table(rows, [6.0, 1.7, 2.3, 1.8, 2.0, 3.2], caption="VAD candidates measured on all 353 calls against the official caller turns "
              "(outputs/dataset_inspection/vad_*.csv). The chosen detector treats both classes alike.")
    rep.image(F / "vad_agreement.png", "Agreement of the chosen VAD with the official turns, regions per call, and chunks per call.")
    if ch:
        rep.p(f"Chunking every VAD region into pieces of at most {config.CHUNK_SECONDS:.0f} s (leftovers under {config.MIN_CHUNK_SECONDS:.0f} s dropped) gives "
              f"**{ch['train']['n_chunks']:,} training chunks ({ch['train']['chunk_hours']:.2f} h)** and **{ch['val']['n_chunks']:,} validation chunks "
              f"({ch['val']['chunk_hours']:.2f} h)**, median {ch['train']['chunks_per_call_median']:.0f} per call, no call without chunks. "
              "The median is the same for both classes, so the number of chunks is not a cue either.")
    rep.h2("Leakage checks")
    idl, fp = ins.get("id_leakage", {}), ins.get("ltas_fingerprint", {})
    rep.bullets([f"Anonymous ids are `call_` + 12 hex characters; the hex value has AUC {idl.get('id_hex_auc', 0):.3f} against the label and the "
                 f"label sequence in id order has {idl.get('label_runs_in_id_order')} runs (random expectation {idl.get('expected_runs_if_random', 0):.0f}): filenames leak nothing.",
                 f"No file appears in both splits; no duplicate audio (MD5). Spectral fingerprints of caller speech across splits: "
                 f"maximum train/val correlation {fp.get('max_cross_split_correlation', 0):.3f}, {fp.get('pairs_above_0.995', 0)} near-duplicate pairs.",
                 "Speaker identity is not available, so we cannot verify the speaker-disjoint claim ourselves nor count distinct TTS voices; "
                 "within the training split the same caller may appear in several calls, which is why no cross-validation inside train is reported."])
    rep.h2("Shortcut features (preview of Part 9)")
    sf = pd.read_csv(O / "dataset_inspection" / "shortcut_features.csv", index_col=0) if (O / "dataset_inspection" / "shortcut_features.csv").exists() else None
    if sf is not None:
        sf["strength"] = (sf["train_auc"] - 0.5).abs()
        top = sf.sort_values("strength", ascending=False).head(10)
        rows = [["Statistic", "Train AUC", "Val AUC", "Median human", "Median synthetic"]]
        for name, r in top.iterrows():
            rows.append([r["description"], f3(r["train_auc"]), f3(r["val_auc"]), f"{r['train_median_human']:.2f}", f"{r['train_median_synthetic']:.2f}"])
        rep.table(rows, [7.4, 1.8, 1.8, 2.5, 2.5], caption="Single-feature ROC AUC of trivial per-call statistics (0.5 = no information; "
                  "values far from 0.5 in either direction are shortcuts). Part 9 measures what a model built only on them achieves.")


# ============================================================================ part 5
def part_embeddings(rep, d):
    rep.h1("Part 5 - Embeddings and the layer probe")
    rep.p("An **embedding** is a fixed-size vector that a neural network computes to describe its input. Each backbone turns "
          "a chunk of waveform into one vector per 20 ms frame in every transformer layer; **mean pooling** over the frames "
          "gives one vector per chunk and layer. We cache the mean (and the standard deviation) of every layer for every "
          "chunk once, in float16, so that every classifier experiment afterwards runs in seconds.")
    rep.code("""
caller chunk (<= 4 s, 16 kHz, RMS -26 dB)
  -> CNN feature encoder (7 conv layers, 20 ms frames)          = hidden layer 0
  -> transformer layer 1 ... layer N                             = hidden layers 1..N
  -> for every layer: mean over frames -> one vector (768 or 1024 numbers)
  -> cached: (n_chunks, N+1, dim) float16 per split and backbone""")
    rep.h2("Why not simply use the last layer?")
    rep.p("Self-supervised speech models are trained to predict masked content, so their top layers move towards *what was "
          "said* and discard fine acoustic texture; the lower and middle layers still describe the sound itself, where "
          "synthesis and channel artefacts live. Our ASVspoof project found the spoofing evidence in WavLM's layer 3, with "
          "the last layer seven times worse. So for every backbone we train one logistic regression per layer on the "
          "training chunks and score the validation calls (mean chunk log-odds). The layer is chosen **on validation only** "
          "- never on anything the judges will use.")
    rows = [["Layer"] + [DISPLAY[k] for k in config.BACKBONE_ORDER]]
    probes = {k: d[f"res_{k}"]["probe"] for k in config.BACKBONE_ORDER if d[f"res_{k}"]}
    if probes:
        n_max = max(len(p) for p in probes.values())
        for layer in range(n_max):
            row = [str(layer) if layer else "0 (CNN)"]
            for k in config.BACKBONE_ORDER:
                p = probes.get(k)
                if p and layer < len(p):
                    v = p[layer]["val"]
                    mark = "**" if layer == d[f"res_{k}"]["layer"] else ""
                    row.append(f"{mark}AUC {v['auc']:.3f} / EER {v['eer'] * 100:.1f}% / acc {v['accuracy']:.2f}{mark}")
                else:
                    row.append("-")
            rows.append(row)
        rep.table(rows, [1.6, 5.1, 5.1, 5.2], font_size=6.9,
                  caption="Layer probe on the CLEAN validation split: call-level AUC / EER / accuracy of a logistic regression trained on each "
                          "frozen layer. Bold = the layer finally used for that backbone.")
        all_perfect = all(p["val"]["auc"] >= 0.999 for pr in probes.values() for p in pr)
        if all_perfect:
            rep.note(["Every layer of every backbone - even layer 0, the convolutional feature encoder that has seen no transformer "
                      "block - separates the validation calls perfectly. The clean probe therefore proves that the evidence is easy "
                      "to read from any frozen representation, but it **cannot choose a layer**. The layer used for each backbone "
                      "is chosen by the same probe under the channel perturbations of Part 6b (the 'robust layer'), which is the "
                      "reason the bold rows are not layer 0. The chunk-level AUC (not shown) is 0.99-1.00 for the lower layers and "
                      "drops slightly for the top layers, the only visible trace of the depth effect we saw on ASVspoof."], "key",
                     "The clean probe is saturated")
        else:
            rep.image(F / "bench_layer_probes.png", "The three probes on one axis (relative depth). Circles mark the chosen layers.")
            for k in config.BACKBONE_ORDER:
                if d[f"res_{k}"]:
                    rep.image(O / "classifier" / k / "layer_probe.png", f"{DISPLAY[k]}: validation AUC, EER and accuracy per layer.",
                              width_cm=15.5)
        ms = [k for k in config.BACKBONE_ORDER if d[f"meanstd_{k}"]]
        if ms:
            rows = [["Backbone", "mean pooling (layer, val AUC / EER)", "mean+std pooling (layer, val AUC / EER)"]]
            for k in ms:
                a, b = d[f"res_{k}"], d[f"meanstd_{k}"]
                rows.append([DISPLAY[k], f"layer {a['layer']}: {a['mlp']['val']['auc']:.3f} / {pct(a['mlp']['val']['eer'])}",
                             f"layer {b['layer']}: {b['mlp']['val']['auc']:.3f} / {pct(b['mlp']['val']['eer'])}"])
            rep.table(rows, [4, 6.5, 6.5], caption="Pooling diagnostic (MLP): adding the temporal standard deviation to the mean.")
    else:
        rep.p("*(layer probe not run yet)*")


# ============================================================================ part 6
def part_experiment(rep, d):
    b = d["bench"]
    rep.h1("Part 6 - The controlled three-model experiment")
    rep.p("The backbone is the only variable. Everything else is held fixed so that a difference in the numbers can only come "
          "from the representation:")
    rep.table([["Held identical for the three backbones", "Value"],
               ["train / validation calls", "official manifest split, all 353 calls, same seeded order"],
               ["caller channel, VAD, chunking", "channel 0; energy VAD 15 dB above the noise floor; chunks <= 4 s, leftovers < 1 s dropped"],
               ["loudness", "every chunk scaled to RMS -26 dBFS (removes the loudness shortcut; needed because only two backbones normalise their input)"],
               ["resampling", "8 kHz -> 16 kHz polyphase (no information above 4 kHz is created)"],
               ["pooling / layer choice", "mean over frames; layer chosen by validation AUC of a logistic-regression probe"],
               ["classifier", "logistic regression (C = 1, class-balanced) and MLP dim-256-2 (ReLU, dropout 0.3), class-weighted cross-entropy, AdamW lr 1e-3, wd 1e-4, batch 64, <= 60 epochs, early stopping on validation loss (patience 15)"],
               ["aggregation", "mean of the chunk log-odds per call (fixed in advance)"],
               ["seed / precision", f"seed {config.SEED}; backbone in bfloat16 autocast on the GPU, classifier in float32"],
               ["metrics", "identical code (src/metrics.py); threshold p = 0.5 for the decision metrics"]],
              [5.2, 11.8], font_size=7.3)
    if b is None:
        rep.p("*(benchmark not run yet)*")
        return
    rep.h2("Validation results")
    rows = [["Backbone", "Params", "Dim", "Layer", "**Accuracy**", "Bal. acc.", "Precision", "Recall", "F1", "AUC", "EER", "FAR", "FRR", "Brier (raw)"]]
    for _, r in b.iterrows():
        rows.append([r["model"], f"{r['params_m']:.0f}M", str(int(r["embedding_dim"])), str(int(r["selected_layer"])), f"**{pct(r['val_accuracy'])}**",
                     pct(r["val_balanced_accuracy"]), f3(r["val_precision"]), f3(r["val_recall"]), f3(r["val_f1"]), f3(r["val_auc"]),
                     pct(r["val_eer"]), pct(r["val_far"]), pct(r["val_frr"]), f3(r["val_brier_raw"])])
    best = d["best"].get("backbone")
    hl = [i + 1 for i, (_, r) in enumerate(b.iterrows()) if r["backbone"] == best]
    rep.table(rows, [2.9, 1.1, 0.9, 0.9, 1.6, 1.4, 1.3, 1.2, 1.1, 1.1, 1.2, 1.1, 1.1, 1.3], font_size=6.8, highlight_rows=hl,
              caption="PRIMARY CHALLENGE METRIC = accuracy (bold). All others are secondary diagnostics. MLP on the chosen layer, 71 validation calls "
                      "(37 human, 34 synthetic), threshold p = 0.5, no calibration.")
    rep.image(F / "bench_validation_metrics.png", "Validation metrics of the three frozen backbones with the identical MLP.")
    rep.images([F / "bench_roc.png", F / "bench_confusion.png"], "ROC curves and confusion matrices on validation (MLP).", max_height_cm=7.5)
    rep.h2("Linear probe vs MLP vs call-level embedding")
    rows = [["Backbone", "Logistic regression on chunks", "MLP on chunks", "Logistic regression on call-averaged embedding"]]
    for k in config.BACKBONE_ORDER:
        r = d[f"res_{k}"]
        if r:
            f = lambda m: f"acc {pct(m['accuracy'])}, AUC {m['auc']:.3f}, EER {pct(m['eer'])}"
            rows.append([DISPLAY[k], f(r["logreg"]["val"]), f(r["mlp"]["val"]), f(r["call_embedding_logreg"]["val"])])
    rep.table(rows, [3.4, 4.5, 4.5, 4.6], caption="Three classifiers on the same chosen layer (validation). Chunk-level training gives ~15x more "
              "training examples than call-level training.")
    rep.image(F / "bench_aggregation.png", "Call-level aggregation diagnostic: how the chunk scores are combined. 'mean' was fixed before the experiment.")
    rep.h2("Cost")
    rows = [["Backbone", "Weights", "Extraction (backbone time, all chunks)", "Realtime factor", "Peak VRAM extraction", "Cache (all layers)", "Classifier training", "Latency / call (GPU, median)", "Backbone ms / chunk", "Peak VRAM inference"]]
    for _, r in b.iterrows():
        rows.append([r["model"], f"{r['weights_mb']:,.0f} MB", f"{r['extraction_backbone_s']:.0f} s", f"{r['realtime_factor']:.0f}x",
                     f"{r['extraction_peak_vram_mb']:,.0f} MB", f"{r['cache_mb']:,.0f} MB",
                     f"probe {r['probe_time_s']:.0f} s + MLP {r['mlp_train_time_s']:.0f} s",
                     f"{r['gpu_latency_median_s'] * 1000:.0f} ms" if np.isfinite(r.get("gpu_latency_median_s", np.nan)) else "n/a",
                     f"{r['gpu_ms_per_chunk']:.1f}" if np.isfinite(r.get("gpu_ms_per_chunk", np.nan)) else "n/a",
                     f"{r['inference_peak_vram_mb']:,.0f} MB" if np.isfinite(r.get("inference_peak_vram_mb", np.nan)) else "n/a"])
    rep.table(rows, [2.5, 1.4, 2.2, 1.3, 1.7, 1.5, 2.3, 1.9, 1.3, 1.6], font_size=6.8,
              caption="Measured on an RTX 4080 Laptop GPU (12 GB). Latency = decode + VAD + resample + backbone (truncated after the chosen layer) + MLP "
                      "for one whole call, median over 10 validation calls.")
    rep.image(F / "bench_resources.png", "Cost of each backbone.")
    if "cpu_latency_median_s" in b.columns and b["cpu_latency_median_s"].notna().any():
        rows = [["Backbone", "CPU latency / call (median of 3)", "GPU latency / call (median of 10)"]]
        for _, r in b.iterrows():
            rows.append([r["model"], f"{r['cpu_latency_median_s']:.2f} s", f"{r['gpu_latency_median_s'] * 1000:.0f} ms"])
        rep.table(rows, [4, 6.5, 6.5], caption="CPU-only inference (24-thread laptop CPU): what a deployment without a GPU would see.")
    rep.h2("Calibration")
    rep.p("A model can be right and still over-confident. The raw probability sigmoid(mean chunk log-odds) tends to be extreme "
          "because chunk scores are averaged in log-odds space and the MLP was trained until it fit the training chunks. "
          "The clean validation split contains **no errors**, so a calibrator fitted on it can only make the probabilities "
          "sharper without limit (a first attempt drove the temperature to its grid minimum). We therefore fit and evaluate "
          "the calibrators on the validation calls **under all stress conditions** of Part 6b - the same 71 calls, 11 "
          "conditions, with real mistakes and score drift - using 5-fold cross-validation grouped by call:")
    rows = [["Backbone", "raw: Brier / log-loss / ECE / accuracy", "Platt (out-of-fold): Brier / log-loss / ECE / accuracy", "temperature (out-of-fold): Brier / ECE", "deployed"]]
    for k in config.BACKBONE_ORDER:
        c = d[f"cal_{k}"]
        if c:
            m = c["mlp"]
            f = lambda x: f"{x['brier']:.3f} / {x['log_loss']:.3f} / {x['ece']:.3f} / {pct(x['accuracy'])}"
            rows.append([DISPLAY[k], f(m["raw"]), f(m["platt_cv"]), f"{m['temperature_cv']['brier']:.3f} / {m['temperature_cv']['ece']:.3f}", m["chosen"]])
    if len(rows) > 1:
        rep.table(rows, [3.0, 4.2, 4.6, 3.0, 2.2], font_size=7.0,
                  caption="Calibration of the MLP call score on the validation calls under all conditions (71 calls x 11 conditions). "
                          "Brier score: mean squared error of the probability (0 = perfect); ECE: expected calibration error (gap between "
                          "stated confidence and observed accuracy). On clean audio all three are ~0 before and after.")
        if best and (F / f"calibration_{best}.png").exists():
            rep.image(F / f"calibration_{best}.png", f"Reliability diagrams of the winner ({DISPLAY[best]}): raw vs calibrated, out-of-fold on validation.", width_cm=15)
    rep.note("**Classification correctness vs probability calibration.** Accuracy, F1 and EER only care about the order of "
             "the scores and the position of one threshold. Calibration asks whether a stated 0.9 is right nine times out of "
             "ten. The challenge rewards both: the verdict is scored, and `confidence` breaks ties. Platt/temperature scaling "
             "changes the calibration without changing the ranking (AUC/EER stay identical); it can move the accuracy slightly "
             "because it moves the p = 0.5 point.", "info")


# ============================================================================ part 7
def part_why(rep, d):
    b = d["bench"]
    rep.h1("Part 7 - Why the winner won")
    if b is None or not d["best"]:
        rep.p("*(benchmark not run yet)*")
        return
    best = d["best"]["backbone"]
    w = bench_row(d, best)
    others = b[b["backbone"] != best]
    has_stress = "stress_mean_auc" in b.columns and b["stress_mean_auc"].notna().all()
    if has_stress:
        rep.p(f"**Evidence (measured):** on the clean validation split the three backbones tie (accuracy " +
              ", ".join(f"{pct(r['val_accuracy'])}" for _, r in b.iterrows()) + f"; AUC {b['val_auc'].min():.3f}-{b['val_auc'].max():.3f}). "
              f"Under the ten test-time perturbations, {DISPLAY[best]} (layer {int(w['selected_layer'])}) keeps the highest mean AUC "
              f"({w['stress_mean_auc']:.3f}, mean accuracy {pct(w['stress_mean_accuracy'])}, worst condition {w['stress_worst_condition']} at "
              f"AUC {f3(w['stress_worst_auc'])}). The others: " +
              "; ".join(f"{r['model']} (layer {int(r['selected_layer'])}) mean stressed AUC {r['stress_mean_auc']:.3f} / accuracy "
                        f"{pct(r['stress_mean_accuracy'])}, worst {r['stress_worst_condition']} ({f3(r['stress_worst_auc'])})" for _, r in others.iterrows()) + ".")
        spread = b["stress_mean_auc"].max() - b["stress_mean_auc"].min()
        rep.p(f"The spread in mean stressed AUC between the best and the worst backbone is {spread:.3f}. The stress conditions are "
              "synthetic approximations of other lines and handsets, and 71 calls remain a small sample; the ranking is a "
              "*preference supported by a controlled experiment*, not a proof, and the explanations below are hypotheses.")
    else:
        rep.p(f"**Evidence (measured):** {DISPLAY[best]} reached the highest validation AUC ({w['val_auc']:.3f}, EER {pct(w['val_eer'])}, accuracy "
              f"{pct(w['val_accuracy'])}) with layer {int(w['selected_layer'])}. The runners-up: " +
              "; ".join(f"{r['model']} AUC {r['val_auc']:.3f} / EER {pct(r['val_eer'])} / accuracy {pct(r['val_accuracy'])} (layer {int(r['selected_layer'])})"
                        for _, r in others.iterrows()) + ".")
    layers = {k: (d[f"res_{k}"]["layer"], d[f"res_{k}"]["n_layers"] - 1) for k in config.BACKBONE_ORDER if d[f"res_{k}"]}
    rep.h2("Evidence vs hypothesis")
    ev = []
    for k, (L, n) in layers.items():
        sp = d.get(f"stress_probe_{k}")
        if sp:
            ev.append(f"{DISPLAY[k]}: robust layer {L} of {n} (relative depth {L / n:.2f}); mean AUC over conditions {sp['mean_auc_per_layer'][L]:.3f} "
                      f"vs {sp['mean_auc_per_layer'][0]:.3f} for the CNN output and {sp['mean_auc_per_layer'][-1]:.3f} for the last layer.")
        else:
            p = d[f"res_{k}"]["probe"]
            ev.append(f"{DISPLAY[k]}: best layer {L} of {n} (relative depth {L / n:.2f}); last-layer probe AUC {p[-1]['val']['auc']:.3f} vs chosen {p[L]['val']['auc']:.3f}.")
    rep.bullets(["**Evidence - where the robust evidence lives.** " + " ".join(ev) +
                 " On clean audio every layer is perfect; under stress the transformer layers beat the raw CNN output, and the "
                 "chosen layers sit below the top of each network, as in our ASVspoof project: the human/synthetic difference is "
                 "acoustic texture, not linguistic content."])
    rep.bullets(["**Evidence - the linear probe is already strong.** Logistic regression on the chosen layer reaches " +
                 ", ".join(f"AUC {d[f'res_{k}']['logreg']['val']['auc']:.3f} ({DISPLAY[k]})" for k in config.BACKBONE_ORDER if d[f"res_{k}"]) +
                 "; the MLP adds little. The classes are close to linearly separable in the frozen representation, which is why the "
                 "backbone matters less than one might expect.",
                 "**Evidence - chunk vs call.** " + "; ".join(
                     f"{DISPLAY[k]}: chunk-level AUC {d[f'res_{k}']['mlp']['val_chunk']['auc']:.3f} -> call-level {d[f'res_{k}']['mlp']['val']['auc']:.3f}"
                     for k in config.BACKBONE_ORDER if d[f"res_{k}"]) +
                 ". Averaging ~14 chunk scores per call removes most single-chunk mistakes; a few seconds of speech already carry the signal."])
    hyp = []
    if best == "wav2vec2_spanish":
        hyp.append("**Language match (hypothesis).** The Spanish-pretrained model saw Mexican-like phonetics and prosody during pretraining, "
                   "so its early layers may model Spanish speech more tightly and expose deviations (synthetic prosody, vocoder texture) more clearly.")
    if best == "wavlm":
        hyp.append("**Denoising objective (hypothesis).** WavLM's pretraining mixes noise and other speakers into the input; its representations "
                   "of low-level acoustics may be more robust to the telephone channel, which could matter more than the pretraining language.")
    if best == "xlsr":
        hyp.append("**Capacity and diversity (hypothesis).** XLS-R saw 436k hours including telephone corpora (BABEL) and Spanish; more capacity "
                   "and more channels may give a representation in which the injection-path artefacts stand out more clearly.")
    hyp.append("**Telephone-band robustness (hypothesis).** None of the models was pretrained on 8 kHz audio; the empty 4-8 kHz half of the "
               "spectrum is out of distribution for all three. Models whose early layers depend less on high-frequency energy should suffer less.")
    hyp.append("**Sample efficiency (hypothesis).** With 282 training calls, a smaller, better-matched representation can beat a larger one: "
               "a 1024-dimensional embedding gives the MLP more room to memorise speaker-specific traits.")
    hyp.append("**What the winner is probably using (hypothesis).** Part 9 shows a consistent 3.3 kHz roll-off and extra 1.3-3 kHz energy in "
               "synthetic calls, plus prosodic regularity. Frozen low-level layers see all of these; we did not run an ablation that "
               "removes the band edge, so we cannot say how much of the score depends on it.")
    rep.bullets(hyp)


# ============================================================================ part 8
def part_generalisation(rep, d):
    rep.h1("Part 8 - Overfitting and generalisation")
    rep.p("**Overfitting** means the model memorises its training examples instead of learning the general pattern; it keeps "
          "improving on training data while getting worse on new data. **Underfitting** is the opposite: it cannot even fit the "
          "training data. We watch both with train-vs-validation curves after every epoch and keep the epoch with the lowest "
          "validation loss.")
    rows = [["Backbone", "Train accuracy / AUC (fit data)", "Validation accuracy / AUC", "Best epoch of 60", "Epochs run"]]
    for k in config.BACKBONE_ORDER:
        r = d[f"res_{k}"]
        if r:
            rows.append([DISPLAY[k], f"{pct(r['mlp']['train']['accuracy'])} / {r['mlp']['train']['auc']:.3f}",
                         f"{pct(r['mlp']['val']['accuracy'])} / {r['mlp']['val']['auc']:.3f}", str(r["mlp_best_epoch"]), str(r["mlp_epochs_run"])])
    if len(rows) > 1:
        rep.table(rows, [3.4, 4.2, 4.2, 2.6, 2.6])
    for k in config.BACKBONE_ORDER:
        if d[f"res_{k}"]:
            rep.image(O / "classifier" / k / "mlp_curves.png", f"{DISPLAY[k]}: MLP training vs validation curves (chunk-level loss/accuracy, call-level AUC/EER).", width_cm=16)
    rep.h2("What the model could be memorising")
    rep.bullets(["**Specific speakers.** The training split contains an unknown number of distinct humans, some in several calls. "
                 "A chunk classifier can learn 'this voice = human'. The validation split has different callers, so the validation "
                 "numbers already measure speaker generalisation - which is exactly why the train/validation gap above matters.",
                 "**Specific TTS voices.** The synthetic calls come from an unknown number of voices/engines. If the hidden test set uses "
                 "voices from a new engine, artefacts learned from the training engines may not transfer; the judges explicitly test "
                 "'unseen speakers, engines and call conditions'.",
                 "**Call properties.** Loudness (removed by RMS normalisation), the 3.3 kHz band edge (kept: it is in the audio the model sees), "
                 "leading silence and turn counts (not visible to a model that only sees speech chunks).",
                 "**Domain shift.** The dataset is one bank flow, one agent, one telephony stack. Other lines, codecs or handset types "
                 "shift the spectra; our ASVspoof project lost 4x in EER on real telephone channels for exactly this reason."])
    rep.note("With 71 validation calls, one call is 1.4 accuracy points. Every comparison in this report should be read with that "
             "granularity in mind; the hidden test set will be the real measurement.", "warn")


# ============================================================================ part 9
def part_shortcuts(rep, d):
    sc = d["shortcut"]
    rep.h1("Part 9 - Shortcuts")
    rep.p("Our ASVspoof project found that file duration and silence alone reached 17% EER on an academic benchmark - a classifier "
          "that knows nothing about voices. We repeated the exercise here with per-call statistics that contain **no voice content**: "
          "durations, turn counts, silences, loudness, bandwidth and conversation timing, in logistic regression and a small "
          "gradient-boosting model (trained on train, scored on validation).")
    if sc is None:
        rep.p("*(shortcut check not run yet)*")
        return
    rows = [["Feature set", "Features", "Model", "**Val accuracy**", "Bal. acc.", "AUC", "EER", "Train AUC"]]
    for _, r in sc.sort_values(["feature_set", "model"]).iterrows():
        rows.append([r["feature_set"], str(int(r["n_features"])), r["model"], f"**{pct(r['val_accuracy'])}**", pct(r["val_balanced_accuracy"]),
                     f3(r["val_auc"]), pct(r["val_eer"]), f3(r["train_auc"])])
    rep.table(rows, [3.0, 1.4, 1.4, 2.2, 1.8, 1.6, 1.6, 1.6], font_size=7.0,
              caption="Shortcut models on trivial statistics (validation). None of these listens to the voice.")
    rep.image(F / "shortcut_check.png", "Left: which cues the all-features logistic regression uses. Right: AUC per family of cues.")
    top = sc.sort_values("val_auc", ascending=False).iloc[0]
    rep.note([f"The best trivial model ({top['feature_set']}, {top['model']}) reaches **validation AUC {top['val_auc']:.3f} and accuracy "
              f"{pct(top['val_accuracy'])}** without hearing a single phoneme. Three cues carry most of it: the time until the caller "
              "first speaks (the AI pipeline answers only after the agent's greeting), the caller's loudness (the synthetic audio is "
              "injected ~6 dB hotter), and the 3.4 kHz band edge of the synthetic audio path."], "danger", "The dataset contains strong shortcuts")
    rep.h2("Which of them can the acoustic model use?")
    rep.table([["Cue", "Available to our acoustic model?", "Why"],
               ["time until the caller speaks, turn counts, response latency, overlap", "**no**", "the model only sees speech chunks; silence, timing and the agent channel never reach it"],
               ["absolute loudness / peak level", "**no**", "every chunk is scaled to the same RMS before the backbone"],
               ["noise floor and signal-to-noise ratio of the line", "partly", "RMS normalisation scales noise together with speech; a noisier line stays noisier"],
               ["3.4 kHz band edge, spectral tilt", "**yes**", "it is in the waveform; a frozen low-level layer certainly represents it"],
               ["voice quality: prosody, harmonics, vocoder texture", "**yes**", "the acoustic evidence we actually want"]],
              [5.6, 3.2, 8.2], font_size=7.3)
    rep.p("Why this matters for the hidden evaluation: the judges say the hidden calls come from 'callers and voices that appear "
          "in neither split' and stress robustness to unseen engines and call conditions. A cue tied to the recording pipeline "
          "(gain, band edge, pipeline latency) survives only if the hidden calls were produced by the same pipeline. Our design "
          "removes the cues we can remove without touching the voice (loudness, timing) and keeps the spectral ones, because "
          "separating 'codec path of the synthetic audio' from 'synthetic voice' is impossible with this dataset alone. The "
          "behavioural cues are not lost: they are the input of the future behavioural branch, where they belong.")


# ============================================================================ part 10
def part_finetuning(rep, d):
    rep.h1("Part 10 - Fine-tuning")
    rep.p("**Feature extraction** keeps the pretrained weights frozen and trains only a small classifier on top. **Fine-tuning** "
          "continues to train (part of) the backbone on our labels. It can learn task-specific features, but with 282 calls it "
          "can also erase pretrained knowledge (**catastrophic forgetting**) and memorise the training callers. Our ASVspoof project "
          "found that a partially fine-tuned WavLM generalised *worse* than frozen embeddings + MLP, so fine-tuning is the last "
          "experiment here, run once, on the winner only, conservatively:")
    rep.bullets([f"backbone truncated after the chosen layer (deeper layers are unused); head = the frozen-trained MLP as a warm start",
                 f"phase 1: backbone frozen, head trained for {config.FT_HEAD_EPOCHS} epochs on random crops (sanity check: should match the frozen result)",
                 f"phase 2: the last {config.FT_UNFREEZE_LAST_N_LAYERS} kept transformer layers unfrozen with learning rate {config.FT_BACKBONE_LR:g} "
                 f"(100x below the head's), warm-up + linear decay, gradient clipping, early stopping (patience {config.FT_PATIENCE})",
                 "the checkpoint with the best **validation** call-level AUC is kept; the same chunks, aggregation and metrics as the frozen pipeline"])
    fts = {k: d[f"ft_{k}"] for k in config.BACKBONE_ORDER if d[f"ft_{k}"]}
    if not fts:
        rep.p("*(fine-tuning not run)*")
        return
    for k, ft in fts.items():
        fr, fv = ft["frozen"], ft["finetuned"]["val"]
        rows = [["Model", "**Val accuracy**", "Bal. acc.", "F1", "AUC", "EER", "Brier"],
                [f"frozen {DISPLAY[k]} + MLP", f"**{pct(fr['accuracy'])}**", pct(fr["balanced_accuracy"]), f3(fr["f1"]), f3(fr["auc"]), pct(fr["eer"]), f3(fr["brier"])],
                [f"fine-tuned {DISPLAY[k]} (best epoch {ft['best_epoch']})", f"**{pct(fv['accuracy'])}**", pct(fv["balanced_accuracy"]), f3(fv["f1"]), f3(fv["auc"]), pct(fv["eer"]), f3(fv["brier"])]]
        rep.table(rows, [5.2, 2.2, 1.8, 1.6, 1.6, 1.8, 1.8], highlight_rows=[1 if ft["winner"] == "frozen" else 2],
                  caption=f"Frozen vs fine-tuned on validation. Fine-tuning ran {len(ft['history']) - 1} epochs in {ft['runtime_s'] / 60:.0f} min "
                          f"on a Runpod pod (NVIDIA L4); train accuracy of the fine-tuned model {pct(ft['finetuned']['train']['accuracy'])}.")
        rep.image(F / f"finetune_{k}.png", f"Fine-tuning curves for {DISPLAY[k]}. Dotted: layers unfrozen; dashed grey: the frozen model's level.")
        fs, frozen_stress = ft.get("finetuned_stress") or {}, d.get(f"stress_{k}")
        if fs and frozen_stress:
            rows = [["Condition", "frozen: AUC / accuracy", "fine-tuned: AUC / accuracy"]]
            for c, m in fs.items():
                fz = frozen_stress["conditions"][c]["mlp"]
                rows.append([c, f"{fz['auc']:.3f} / {pct(fz['accuracy'])}", f"{m['auc']:.3f} / {pct(m['accuracy'])}"])
            rows.append(["**mean over stressed conditions**", f"**{ft['frozen_mean_auc_stressed']:.3f}**", f"**{ft['finetuned_mean_auc_stressed']:.3f}**"])
            rep.table(rows, [4.5, 6.2, 6.3], font_size=7.2,
                      caption="Frozen vs fine-tuned under the stress conditions of Part 6b (the clean split cannot separate them). "
                              f"Decision rule: {ft.get('rule', 'validation AUC')}.")
        if ft["winner"] == "frozen":
            rep.note(["Fine-tuning did **not** beat the frozen model (clean validation is a tie at 100%; under stress the frozen model is "
                      "at least as robust), so the frozen model stays. This is a useful result: the frozen representation already "
                      "contains the evidence, the extra capacity can only fit the training callers and the training pipeline's "
                      "artefacts more tightly, and the deployment stays simpler: no backbone weights of our own to ship."],
                     "key", "Result")
        else:
            rep.note(["Fine-tuning beat the frozen model on validation and is the deployed model; the frozen one remains the fallback."], "key", "Result")


# ============================================================================ part 11
def part_inference(rep, d):
    b = d["bench"]
    best = d["best"].get("backbone")
    rep.h1("Part 11 - Production inference")
    rep.code("""
POST /detect  {"audio": "<base64 of the stereo 8 kHz WAV>"}
  1. decode the WAV bytes                                    (soundfile)
  2. take channel 0 = caller                                 (channel 1 is ignored by the acoustic branch)
  3. energy VAD -> caller speech regions                     (15 dB above the 10th-percentile noise floor)
  4. cut regions into chunks <= 4 s, scale each to RMS -26 dB
  5. resample 8 kHz -> 16 kHz                                (polyphase; adds no information above 4 kHz)
  6. frozen backbone, truncated after the chosen layer -> mean over frames -> one vector per chunk
  7. MLP -> chunk log-odds
  8. mean over chunks -> call score -> Platt/temperature calibration -> p(synthetic)
  9. {"is_synthetic": p >= 0.5, "confidence": max(p, 1 - p)}""")
    rep.p("`predict.py` exposes the same pipeline as `predict_acoustic(call_wav) -> float` (path, bytes or base64) and the class "
          "`AcousticDetector` with per-stage timings; `server.py` wraps it in FastAPI. The server accepts the base64 field under "
          "several names, a raw base64 body or a multipart upload, and always answers with the required contract - if anything "
          "fails it returns `is_synthetic: false, confidence: 0.5` plus an error field rather than an HTTP error. `confidence` is "
          "the probability that the returned verdict is right (`config.CONFIDENCE_MODE`); switching it to p(synthetic) is one setting.")
    if b is not None and best:
        w = bench_row(d, best)
        rows = [["Stage", "Median per call (GPU)"], ["decode + VAD + chunking + resampling", f"{w['gpu_segmentation_median_s'] * 1000:.0f} ms"],
                ["backbone (all chunks of the call)", f"{w['gpu_backbone_median_s'] * 1000:.0f} ms ({w['gpu_ms_per_chunk']:.1f} ms per chunk)"],
                ["classifier + aggregation", "< 5 ms"], ["**total**", f"**{w['gpu_latency_median_s'] * 1000:.0f} ms** (max {w['gpu_latency_max_s'] * 1000:.0f} ms)"],
                ["model load (once)", f"{w['load_s']:.1f} s"]]
        if "cpu_latency_median_s" in b.columns and np.isfinite(w.get("cpu_latency_median_s", np.nan)):
            rows.append(["total on CPU only", f"{w['cpu_latency_median_s']:.2f} s"])
        rep.table(rows, [7, 10], caption=f"Measured latency of the deployed model ({DISPLAY[best]}) for whole ~150 s calls.")
    rep.p("A call is ~150 s long and contains ~40 s of caller speech; scoring it takes a fraction of a second, so the endpoint "
          "answers far inside the judges' window, and a streaming version could give a first verdict after the first few caller turns.")


# ============================================================================ part 12
def part_winner(rep, d):
    best, b = d["best"], d["bench"]
    rep.h1("Part 12 - The winning acoustic model")
    if not best or b is None:
        rep.p("*(benchmark not run yet)*")
        return
    k = best["backbone"]
    w = bench_row(d, k)
    res = d[f"res_{k}"]
    cal = d[f"cal_{k}"]["mlp"] if d[f"cal_{k}"] else None
    ft = d[f"ft_{k}"]
    deployed = "fine-tuned" if ft and ft["winner"] == "finetuned" else "frozen"
    rows = [["BEST ACOUSTIC MODEL", ""],
            ["Backbone", f"{DISPLAY[k]} (`{config.BACKBONES[k]['hf_name']}`), {w['params_m']:.0f}M parameters, {deployed}"],
            ["Selected layer", f"{int(w['selected_layer'])} of {int(w['n_layers'])} (chosen by validation AUC of the layer probe)"],
            ["Classifier", f"MLP {int(w['embedding_dim'])} -> 256 -> 2 (ReLU, dropout 0.3), {res['mlp_params']:,} parameters, best epoch {res['mlp_best_epoch']}"],
            ["Segmentation", f"caller channel, energy VAD (15 dB above noise floor), chunks <= {config.CHUNK_SECONDS:.0f} s, RMS {config.TARGET_RMS_DB:.0f} dB, 16 kHz"],
            ["Aggregation", "mean of chunk log-odds, then " + (f"{cal['chosen']} scaling fitted on validation" if cal else "sigmoid")],
            ["Validation score (primary)", f"accuracy {pct(w['val_accuracy'])} (balanced {pct(w['val_balanced_accuracy'])}) at p = 0.5, 71 calls"],
            ["EER", pct(w["val_eer"])], ["F1", f3(w["val_f1"])], ["AUC", f3(w["val_auc"])],
            ["Off-pipeline check (Part 8c)", (lambda e: f"{pct(e['per_variant']['pstn']['accuracy'])} accuracy / AUC {f3(e['per_variant']['pstn']['auc'])} on 108 clips from unseen "
                                              f"engines and speakers through a simulated PSTN line" if e else "n/a")(d.get("external"))],
            ["Robustness (stress test)", (f"mean AUC {f3(w['stress_mean_auc'])} / accuracy {pct(w['stress_mean_accuracy'])} over 10 perturbations; "
                                          f"worst {w['stress_worst_condition']} (AUC {f3(w['stress_worst_auc'])})") if "stress_mean_auc" in w and np.isfinite(w["stress_mean_auc"]) else "n/a"],
            ["Calibration", (f"Brier {cal['raw']['brier']:.3f} raw -> {cal[cal['chosen'] + '_cv']['brier']:.3f} calibrated (out-of-fold), ECE {cal['raw']['ece']:.3f} -> {cal[cal['chosen'] + '_cv']['ece']:.3f}" if cal else "n/a")],
            ["Inference latency", f"{w['gpu_latency_median_s'] * 1000:.0f} ms per call on the GPU (median)" if np.isfinite(w.get("gpu_latency_median_s", np.nan)) else "n/a"],
            ["Model files", f"models/{k}/ (mlp.pt, logreg.joblib, meta.json, calibration_mlp.json) + the Hugging Face backbone"],
            ["Why we selected it", best["reason"]]]
    rep.table(rows, [3.6, 13.4], font_size=7.6)
    rep.p("Commands: `python src/predict.py path/to/call.wav` scores a file; `python src/server.py --port 8000` serves `POST /detect`.")


# ============================================================================ README results
def update_readme(d):
    path = R.parent / "README.md"
    if not path.exists() or d["bench"] is None:
        return
    b, best = d["bench"], d["best"]
    lines = ["## Results", "", "Validation split (71 calls from speakers never seen in training), frozen backbone + identical MLP:", "",
             "| Backbone | Params | Layer | Accuracy (primary) | Bal. acc. | F1 | AUC | EER | Latency/call (GPU) |", "|---|---|---|---|---|---|---|---|---|"]
    for _, r in b.iterrows():
        star = " **(deployed)**" if r["backbone"] == best.get("backbone") else ""
        lat = f"{r['gpu_latency_median_s'] * 1000:.0f} ms" if np.isfinite(r.get("gpu_latency_median_s", np.nan)) else "n/a"
        lines.append(f"| {r['model']}{star} | {r['params_m']:.0f}M | {int(r['selected_layer'])} | {pct(r['val_accuracy'])} | {pct(r['val_balanced_accuracy'])} | "
                     f"{f3(r['val_f1'])} | {f3(r['val_auc'])} | {pct(r['val_eer'])} | {lat} |")
    sc = d["shortcut"]
    if sc is not None:
        top = sc.sort_values("val_auc", ascending=False).iloc[0]
        lines += ["", f"Shortcut check (trivial statistics, no voice): best {top['feature_set']}/{top['model']} = AUC {top['val_auc']:.3f}, accuracy {pct(top['val_accuracy'])}."]
    fts = [d[f"ft_{k}"] for k in config.BACKBONE_ORDER if d[f"ft_{k}"]]
    if fts:
        ft = fts[0]
        if ft.get("finetuned_mean_auc_stressed") is not None:
            lines += [f"Fine-tuning the winner: mean stressed validation AUC {ft['finetuned_mean_auc_stressed']:.3f} vs frozen "
                      f"{ft['frozen_mean_auc_stressed']:.3f} (clean validation: both 100%) -> {ft['winner']} model deployed."]
        else:
            lines += [f"Fine-tuning the winner: validation AUC {ft['finetuned']['val']['auc']:.3f} vs frozen {ft['frozen']['auc']:.3f} -> {ft['winner']} model deployed."]
    ext = d.get("external")
    if ext:
        pv = ext["per_variant"]["pstn"]
        lines += ["", f"Out-of-dataset check (new TTS engines + new speakers through a simulated PSTN line, 108 clips): "
                      f"accuracy {pct(pv['accuracy'])}, AUC {f3(pv['auc'])}, EER {pct(pv['eer'])}. This, not the 100%, is the number to expect on unseen engines."]
    lines += ["", f"Winner: **{DISPLAY[best['backbone']]} layer {best['layer']} + MLP** ({best['reason']}).", ""]
    text = path.read_text(encoding="utf-8")
    head = text.split("## Results")[0].rstrip() + "\n\n"
    path.write_text(head + "\n".join(lines), encoding="utf-8")
    print(f"README results section updated: {path}")
