"""Report Part 6b - the robustness stress test (imported by build_report.py)."""
import _bootstrap  # noqa: F401
import config
from build_report import DISPLAY, F, f3, pct


def part_stress(rep, d):
    rep.h1("Part 6b - The robustness stress test")
    rep.p("Every backbone reached AUC 1.000 on the clean validation split from **every** layer - including layer 0, the "
          "output of the convolutional feature encoder, which is little more than a learned filterbank. Together with the "
          "shortcut check (Part 9: trivial statistics reach AUC 1.0 too) this says that the two classes of this dataset are "
          "separated by gross acoustic properties of the recording pipeline, and that the clean split cannot rank the "
          "backbones, nor tell us anything about the judges' first criterion, robustness to unseen call conditions.")
    rep.p("So we built a harder validation: the **same 71 validation calls** with the caller channel perturbed at test time in "
          "ways the training data never contained (nothing is retrained). The perturbations simulate other lines, handsets "
          "and codecs, and one of them - a low-pass filter at the telephone band edge - directly removes the 3.3 kHz cue.")
    rows = [["Condition", "What it simulates", "Why"],
            ["gain -12 dB", "a quieter line", "sanity check: the loudness normalisation should make it a near no-op"],
            ["lowpass 3.4 kHz / 3.0 kHz", "a narrower line or codec", "removes the band edge that separates synthetic calls; a band-edge detector must fail here"],
            ["pstn 300-3400 Hz", "a classic PSTN band-pass", "narrow-band telephony"],
            ["white noise 20 / 10 dB SNR", "line noise, a bad connection", "additive noise relative to the caller's speech level"],
            ["pink noise 10 dB SNR", "room / traffic / crowd noise", "coloured noise closer to real backgrounds"],
            ["mu-law codec", "G.711 8-bit companding", "the most common real telephone codec"],
            ["reverb 0.3 s", "a speakerphone in a small room", "synthetic exponentially decaying impulse response"],
            ["spectral tilt -3 dB/octave", "a different handset / microphone", "changes the spectral balance without removing bands"]]
    rep.table(rows, [3.6, 5.0, 8.4], font_size=7.3)
    probes = {k: d.get(f"stress_probe_{k}") for k in config.BACKBONE_ORDER}
    if not any(probes.values()):
        rep.p("*(stress test not run yet)*")
        return
    rep.h2("Which layers survive?")
    rep.p("For every backbone the validation embeddings of every layer were re-extracted under every condition and scored by the "
          "logistic-regression probe trained on the **clean** training chunks. The **robust layer** of a backbone is the layer with "
          "the highest mean validation AUC over all conditions (clean included); this replaces the saturated clean-only choice.")
    rows = [["Backbone", "Robust layer", "Mean AUC (all conditions)", "Mean AUC (stressed only)", "Worst-condition AUC", "Clean-only choice"]]
    for k in config.BACKBONE_ORDER:
        p = probes.get(k)
        if p:
            L = p["robust_layer"]
            rows.append([DISPLAY[k], str(L), f3(p["mean_auc_per_layer"][L]), f3(p["stressed_mean_auc_per_layer"][L]),
                         f3(p["worst_auc_per_layer"][L]), "layer 0 (every layer tied at AUC 1.000)"])
    rep.table(rows, [3.2, 1.8, 3.0, 3.0, 2.6, 3.4])
    for k in config.BACKBONE_ORDER:
        if probes.get(k):
            rep.image(F / f"stress_probe_{k}.png", f"{DISPLAY[k]}: validation AUC and accuracy of every frozen layer under every "
                      "condition (logistic-regression probe trained on clean train chunks). * = robust layer.", width_cm=16.5)
    rep.h2("The MLPs under stress")
    st = {k: d.get(f"stress_{k}") for k in config.BACKBONE_ORDER}
    if not any(st.values()):
        rep.p("*(MLP stress evaluation not run yet)*")
        return
    conds = list(next(v for v in st.values() if v)["conditions"].keys())
    keys = [k for k in config.BACKBONE_ORDER if st.get(k)]
    rows = [["Condition"] + [f"{DISPLAY[k]} (layer {st[k]['layer']})" for k in keys]]
    for c in conds:
        row = [c]
        for k in keys:
            m = st[k]["conditions"][c]["mlp"]
            row.append(f"AUC {m['auc']:.3f} / acc {pct(m['accuracy'])} / EER {pct(m['eer'])}")
        rows.append(row)
    rows.append(["**mean over stressed conditions**"] + [f"**AUC {st[k]['mlp_mean_auc_stressed']:.3f} / acc {pct(st[k]['mlp_mean_accuracy_stressed'])}**" for k in keys])
    rows.append(["worst condition"] + [f"{st[k]['mlp_worst_condition']} (AUC {st[k]['mlp_worst_auc']:.3f})" for k in keys])
    widths = [3.4] + [13.6 / len(keys)] * len(keys)
    rep.table(rows, widths, font_size=7.0,
              caption="MLP trained on clean training chunks (robust layer), scored on the validation calls under each condition. "
                      "The mean over the stressed conditions is the selection score of the benchmark.")
    rep.image(F / "stress_mlp.png", "MLP validation AUC and accuracy per condition and backbone.")
    worst = ", ".join(f"{DISPLAY[k]}: {st[k]['mlp_worst_condition']}" for k in keys)
    rep.note([f"Worst conditions - {worst}. Conditions that remove or mask the upper band (low-pass, PSTN band-pass, noise) "
              "are the hardest; a gain change costs nothing (the loudness normalisation works as intended). Wherever a "
              "low-pass filter alone breaks a model, that model was partly a band-edge detector - a property of this dataset's "
              "synthetic pipeline that the hidden test set may or may not share."], "warn", "What the stress test says")
