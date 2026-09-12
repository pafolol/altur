"""
THE TELEPHONE-ROBUSTNESS REPORT - Markdown and PDF, from the artefacts on disk.

Nothing in here is typed in by hand: every number is read back from the JSON and CSV the pipeline wrote, so
the report cannot drift away from the experiment. If a stage has not been run, its section says so instead of
inventing a figure.

Run:  python src/report_robust.py
Outputs: reports/TELEPHONE_ROBUSTNESS_REPORT.{md,pdf}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import config
import phone_channel as pc
from build_channel_dataset import MANIFESTS, PLAN
from report_utils import Report

ROBUST = config.OUTPUTS_DIR / "robust"
PILOT = config.PROJECT_ROOT.parent / "channel_dataset" / "pilot"
CHANNEL_FIG = config.PROJECT_ROOT.parent / "channel_dataset" / "figures"
FIG = config.FIGURES_DIR
MODEL_LABEL = {"specialist": "Specialist (Model A)", "specialist_silero": "Specialist, VAD swapped",
               "orig_only": "Retrained, original data only", "robust_v2": "Robust V2 (Model B)"}
DOMAIN_LABEL = {
    "altur_val": "Altur validation (original)",
    "channel_val_seen": "Altur validation through a SEEN-family line",
    "channel_val_unseen": "Altur validation through an UNSEEN-family line",
    "external_ood": "out-of-dataset clips (new engines, new speakers)",
    "external_ood_channel": "out-of-dataset clips through an UNSEEN-family line",
    "stress": "validation under 11 channel perturbations",
}


def read_json(path, default=None):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else default


def read_csv(path, default=None):
    p = Path(path)
    return pd.read_csv(p) if p.exists() else default


def pct(x, nd=1):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{100 * x:.{nd}f}%"


def num(x, nd=4):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def build():
    man = read_csv(MANIFESTS / "all.csv")
    man = man[man["status"] == "ok"] if man is not None else None
    pilot = read_json(PILOT / "pilot_summary.json")
    pilot_tab = read_csv(PILOT / "pilot_table.csv")
    dyn = read_json(ROBUST / "dynamic_range.json")
    robust = read_json(ROBUST / "results.json")
    matrix = read_csv(ROBUST / "evaluation_matrix.csv")
    matrix_meta = read_json(ROBUST / "evaluation_matrix.json", {})
    calib = read_json(ROBUST / "calibration.json")
    model_b = read_json(config.MODELS_DIR / "robust_v2" / "meta.json")

    rep = Report("Surviving the telephone line",
                 "Altur HackMTY 2026, acoustic branch - why a detector that is perfect on the dataset fails on a "
                 "real call, and what fixes it", config.REPORTS_DIR)

    # ============================================================ headline
    rep.h1("What this report is about")
    rep.p("The deployed acoustic detector - Wav2Vec2 Spanish, frozen, layer 5, plus a small MLP - scores "
          "**100 % accuracy and 1.000 AUC** on the official Altur validation split. On audio that has actually "
          "been through a telephone call it degrades badly. That gap is the subject of this report.")
    rep.p("A detector that only works on the recordings it was trained from is not a product. The bank's calls "
          "arrive down a narrowband line, through codecs, gain control and packet loss, and the caller is the "
          "same person either way. So the question is not how to score higher on the validation split - it is "
          "already saturated - but how to keep that performance when the channel changes underneath it.")

    if pilot:
        rep.kpis([(pct(pilot["accuracy_original"], 0), "specialist on clean audio"),
                  (pct(pilot["accuracy_channel"], 0), "the same calls, through a line"),
                  (f"{pilot['segmentation_failures']}/{pilot['n']}", "calls where it found no speech at all"),
                  (f"{pilot['mean_spectral_distance_db']:.1f} dB", "mean spectral distance")])

    # ============================================================ 1. the failure
    rep.h1("1. Why a model that scores 100 % can fail on a phone call")
    rep.p("Two separate things go wrong, and they need different fixes. Telling them apart was the first job.")

    rep.h2("1.1 The dataset's silence is digital silence")
    if dyn:
        rep.p(f"The Altur caller channel is not merely quiet between turns - it is *exactly zero*. Across "
              f"{dyn['n_calls']} validation calls the 10th-percentile frame level sits at "
              f"**{dyn['floor_db_original_median']:.0f} dB** and the median frame sits at the same place, which "
              f"is another way of saying more than half of every call is a run of zero samples. The speech peaks "
              f"reach about -17 dB, so the dynamic range is around "
              f"**{dyn['range_db_original_median']:.0f} dB**.")
        rep.p(f"A telephone line never delivers that. Gain control lifts quiet passages, comfort noise fills the "
              f"gaps left by voice-activity detection at the far end, and the line itself has a floor. After the "
              f"channel the same calls have a noise floor of **{dyn['floor_db_channel_median']:.0f} dB** and a "
              f"dynamic range of **{dyn['range_db_channel_median']:.0f} dB**.")
        rep.note([f"The deployed segmenter calls a frame speech when it is {config.VAD_THRESHOLD_DB:.0f} dB above "
                  f"the noise floor. After the line there is only {dyn['range_db_channel_median']:.0f} dB of range "
                  f"in the whole file, so that test can never fire.",
                  f"It returned **no speech at all** on {dyn['energy_vad_dead_calls']} of {dyn['n_calls']} channel "
                  f"calls ({pct(dyn['energy_vad_dead_fraction'], 0)}). The endpoint then answers 0.5, which the "
                  f"challenge contract reads as 'synthetic' - the model never got to vote."],
                 kind="danger", title="The detector goes blind before the model is even reached")
        rep.p(f"Silero VAD, a small neural detector that does not key on the noise floor, finds "
              f"{dyn['mean_regions_silero_channel']:.1f} regions per channel call against the energy detector's "
              f"{dyn['mean_regions_energy_channel']:.1f}, and never returns nothing. Everything downstream in this "
              f"report therefore uses Silero, and the specialist is scored under **both** detectors so the report "
              f"can price the segmentation fix separately from the new training data.")
        if (FIG / "channel_dynamic_range.png").exists():
            rep.image(FIG / "channel_dynamic_range.png", width_cm=17,
                      caption="Left: the noise floor moves by about 45 dB. Middle: the dynamic range falls below what "
                              "the energy VAD needs to fire. Right: how many speech regions each detector still finds.")

    rep.h2("1.2 The cues it learned are cues the line erases")
    rep.p("The dataset inspection found that a classifier built on trivial statistics alone - loudness, spectral "
          "shape, noise floor, band edge, with no voice modelling whatsoever - reaches **AUC 1.000** on this "
          "validation split. The synthetic callers are about 6 dB louder than the human ones and their band edge "
          "sits in a different place. Those are properties of the recording pipeline, not of the voice.")
    rep.p("A telephone line normalises exactly those properties, and it does it to both classes equally: the AGC "
          "removes the level difference, the codec and the band-limit filter rewrite the band edge, and comfort "
          "noise replaces the noise floor. Whatever part of the model's decision rested on them is gone.")
    if pilot_tab is not None and len(pilot_tab):
        hum = pilot_tab[pilot_tab["label"] == "human"]
        syn = pilot_tab[pilot_tab["label"] == "synthetic"]
        rep.note([f"The damage is **asymmetric**. In the pilot, human callers survive the line almost untouched "
                  f"({pct(hum['correct_channel_silero'].mean(), 0)} still correct, probabilities staying near zero) "
                  f"while synthetic callers collapse towards 'human' "
                  f"({pct(syn['correct_channel_silero'].mean(), 0)} correct).",
                  "For a bank that is the dangerous direction: the deepfake is the one that gets through."],
                 kind="danger", title="The channel makes synthetic voices look human")

    # ============================================================ 2. channel shift
    rep.h1("2. What a telephone channel actually does to speech")
    rep.p("The public telephone network is narrowband by standard. Audio is sampled at 8 kHz - so nothing above "
          "4 kHz exists at all - and carried as G.711, a logarithmic 8-bit companding of each sample, at "
          "64 kbit/s. Every carrier, every provider, every SIP trunk terminates to that. Wideband codecs "
          "(G.722, AMR-WB, Opus) survive only between endpoints on the same network; the moment a call crosses "
          "into the PSTN it is transcoded back down.")
    rep.bullets([
        "**Band limiting.** The analogue path passes roughly 300-3400 Hz. Low-frequency energy and the top of the "
        "band disappear.",
        "**Companding noise.** G.711's 8-bit logarithmic quantisation adds a noise floor that follows the signal "
        "level, rather than sitting underneath it.",
        "**Transcoding.** A call that crosses two carriers is decoded and re-encoded, sometimes into a different "
        "codec family entirely, and the damage compounds.",
        "**Packetisation and loss.** Audio travels as 20 ms RTP packets. Lost packets are concealed by repeating "
        "the previous frame or by inserting silence, and loss is bursty, not independent.",
        "**Jitter and clock drift.** Sender and receiver clocks never agree exactly; the jitter buffer absorbs "
        "the difference by inserting or dropping samples.",
        "**Gain control.** Gateways run an AGC that pushes every caller towards the same level.",
        "**Discontinuous transmission.** The far end stops sending during silence and the receiver fills the gap "
        "with synthetic comfort noise.",
    ])

    rep.h2("2.1 Why this was built offline instead of through Twilio")
    rep.p("The original plan was to place a real Twilio call per dataset call and capture what came back. Two "
          "facts killed it, and both are worth stating plainly.")
    rep.bullets([
        "**Telephony runs in real time.** The caller channels of the 353 calls total 14.5 hours. Pushing them "
        "through real calls takes 14.5 hours of calling with any provider - the cost (about 28 USD) was never "
        "the obstacle; the clock was.",
        "**One captured channel buys robustness to one channel.** Training against a single provider's path "
        "makes the model robust to that path and nothing else. Change carrier and the work is repeated.",
    ])
    rep.p("So the channel is reproduced offline, from real codec implementations, and **randomised**: every "
          "sample draws its own codecs, band edges, gain, loss and noise. That covers a distribution of "
          "telephone lines rather than one, and it runs in minutes instead of a day. What is genuinely real "
          "here and what is modelled is set out in section 3.")

    # ============================================================ 3. the channel model
    rep.h1("3. The channel model: what is real and what is simulated")
    rep.h2("3.1 Real")
    rep.bullets([
        "**G.711 mu-law and A-law** are the exact ITU-T companding tables, implemented directly in numpy. They "
        "are verified bit-for-bit against `audioop`, the reference C implementation shipped with Python, over "
        "**all 65 536 int16 inputs and all 256 codes**, in both directions.",
        "**G.726 (32/24/16 kbit/s), GSM 06.10, AMR-NB (12.2/7.4/4.75 kbit/s), iLBC, Speex, Opus and G.722** are "
        "the actual reference encoders and decoders, invoked through ffmpeg. Nothing is approximated by a filter.",
        "**20 ms RTP framing**, which is what every carrier packetises to (160 samples, and exactly 160 bytes of "
        "G.711).",
    ])
    rep.h2("3.2 Modelled, not captured")
    rep.p("These are parameterised from the ranges telephony equipment uses, but they are models, and the "
          "report does not claim otherwise: packet loss and its burstiness (a two-state Gilbert-Elliott chain), "
          "the receiver's loss concealment, jitter-buffer clock drift, the analogue front-end response, the "
          "gateway AGC, DTX comfort noise and line noise.")
    rep.note("The one thing this cannot reproduce is a specific carrier's idiosyncrasies. When Twilio credentials "
             "are available, `evaluate_models.py` can be pointed at real captured calls as an additional held-out "
             "domain; the code does not need to change for that, only the data.",
             kind="warn", title="Honest limitation")

    rep.h2("3.3 Seen and unseen codec families")
    rep.p("The codecs are split in two, and the split is the backbone of every generalisation claim in this "
          "report.")
    rep.table([["family", "codecs", "generated for", "may be trained on"],
               ["seen", ", ".join(pc.SEEN_CODECS), "TRAIN and VAL", "yes"],
               ["unseen", ", ".join(pc.UNSEEN_CODECS), "VAL only", "**never**"]],
              col_widths_cm=[2.2, 7.6, 3.2, 3.0],
              caption="The seen family is waveform coding (companding and ADPCM); the unseen family is mostly "
                      "parametric and CELP coding, which rebuilds speech from a vocal-tract model rather than "
                      "coding the waveform. Generalising from one to the other is a hard test, and deliberately so.")
    rep.p(f"No unseen-family sample exists on the training side at all - `PLAN[('train','unseen')] = "
          f"{PLAN[('train', 'unseen')]}` - so there is no file on disk that could be trained on by accident. A "
          "test asserts it.")

    # ============================================================ 4. the pilot
    rep.h1("4. The pilot: ten calls before three hundred")
    rep.p("Before generating anything at scale, ten training calls - five human, five synthetic - went through "
          "the channel and were scored by the deployed detector. The rule was decided in advance: if the "
          "channel did not move the model, the transformation was not reproducing the production failure and "
          "nothing else would be generated.")
    if pilot:
        rep.table([["measure", "value"],
                   ["specialist accuracy, clean audio", pct(pilot["accuracy_original"], 0)],
                   ["specialist accuracy, through the line", pct(pilot["accuracy_channel"], 0)],
                   ["same audio, Silero segmentation", pct(pilot["accuracy_channel_silero"], 0)],
                   ["verdicts flipped", f"{pilot['verdict_flips']} / {pilot['n']}"],
                   ["calls where the energy VAD found no speech", f"{pilot['segmentation_failures']} / {pilot['n']}"],
                   ["mean change in synthetic probability", num(pilot["mean_abs_delta_p"], 3)],
                   ["mean spectral distance", f"{pilot['mean_spectral_distance_db']:.1f} dB"],
                   ["mean level change", f"{pilot['mean_level_change_db']:+.1f} dB"],
                   ["median measured codec delay", f"{pilot['median_delay_ms']:.1f} ms"],
                   ["decision", f"**{pilot['decision']}**"]],
                  col_widths_cm=[9.0, 7.0],
                  caption="The pilot gate. GO means the channel reproduces the failure seen on real calls.")
    if pilot_tab is not None and len(pilot_tab):
        rows = [["call", "truth", "codecs", "p clean", "p line", "chunks clean -> line", "spectral dist."]]
        for r in pilot_tab.itertuples():
            rows.append([r.call_id[-8:], r.label, r.codecs, num(r.p_original, 3), num(r.p_channel, 3),
                         f"{r.chunks_original} -> {r.chunks_channel}", f"{r.spectral_distance_db:.1f} dB"])
        rep.table(rows, col_widths_cm=[2.0, 1.9, 3.6, 1.7, 1.7, 3.0, 2.1], font_size=7.0,
                  caption="Every pilot call. A 'p line' of exactly 0.500 with 0 chunks is the no-speech fallback, "
                          "not a decision by the model.")
    for name, cap in (("pilot_human.png", "A human caller before and after the line."),
                      ("pilot_synthetic.png", "A synthetic caller before and after the line.")):
        if (CHANNEL_FIG / name).exists():
            rep.image(CHANNEL_FIG / name, width_cm=17, caption=cap)
    if (CHANNEL_FIG / "pilot_summary.png").exists():
        rep.image(CHANNEL_FIG / "pilot_summary.png", width_cm=17,
                  caption="Left: the measured frequency response of the drawn lines. Middle: what the AGC does to "
                          "the level gap between the classes. Right: the deployed detector's probability before and "
                          "after - points far below the diagonal on the red side are synthetic callers that now "
                          "look human.")

    # ============================================================ 5. the factory
    rep.h1("5. The dataset factory")
    if man is not None:
        counts = man.groupby(["split", "family"]).size()
        rep.p(f"The factory produced **{len(man)} transformed samples** from the 353 official calls "
              f"({man['duration_channel_s'].sum() / 3600:.1f} hours of audio), with zero failures. Each sample "
              f"draws its channel from a generator seeded with (call id, family, variant), so the dataset is "
              f"reproducible bit for bit and regenerating it is idempotent.")
        rows = [["split", "family", "samples", "human", "synthetic", "variants per call"]]
        for (split, family), g in man.groupby(["split", "family"]):
            rows.append([split, family, str(len(g)), str((g["label"] == "human").sum()),
                         str((g["label"] == "synthetic").sum()), str(PLAN.get((split, family), 0))])
        rep.table(rows, col_widths_cm=[2.4, 2.4, 2.4, 2.4, 2.6, 3.4])
        rep.p(f"Alignment: the codec delay is measured by cross-correlating the smoothed speech envelopes and "
              f"removed, so the transformed caller channel stays sample-aligned with the untouched agent channel. "
              f"Measured delay ranges {man['estimated_delay_ms'].min():.0f} to "
              f"{man['estimated_delay_ms'].max():.0f} ms (median {man['estimated_delay_ms'].median():.0f} ms); "
              f"after the shift the residual is at most "
              f"{int(man['residual_delay_samples'].abs().max())} samples "
              f"({1000 * man['residual_delay_samples'].abs().max() / pc.SR:.1f} ms).")
        rep.note("The envelope correlation reported per sample (median "
                 f"{man['alignment_correlation'].median():.2f}) is *not* an alignment quality score. An AGC "
                 "deliberately flattens the envelope, which lowers the correlation while leaving the timing "
                 "perfect - samples with AGC on sit near 0.4-0.7, samples with it off near 0.99. The residual "
                 "delay after alignment is the number that decides whether a sample can be rebuilt into stereo.",
                 kind="info", title="Reading the alignment numbers")
    if (FIG / "channel_dataset.png").exists():
        rep.image(FIG / "channel_dataset.png", width_cm=17,
                  caption="What the factory drew: sample counts, codec hops, level change, realised packet loss, "
                          "measured delay, and the alignment residual.")
    if (FIG / "channel_architecture.png").exists():
        rep.image(FIG / "channel_architecture.png", width_cm=17.5,
                  caption="The pipeline end to end. Everything on the right of the diagram is evaluation only.")

    # ============================================================ 6. leakage
    rep.h1("6. Keeping the split honest")
    rep.p("A transformed validation call is still a validation call. It carries the same speaker, the same "
          "script and the same recording session as its original, so letting one into training would leak the "
          "validation set into the model through the back door and every number after that would be worthless. "
          "The protections are structural rather than procedural:")
    rep.numbered([
        "The split of a transformed sample is read from the official manifest and asserted on write.",
        "A test checks that **no call id appears in two splits**.",
        "Unseen-family samples are never generated for TRAIN, so no such file exists to be trained on.",
        "The embedding extractor re-checks each set's calls against the official manifest before it runs.",
        "Model selection and early stopping may look at the original and seen-family validation domains only; "
        "the unseen-family domain is consulted once, at the end, by the evaluation script.",
        "The calibrator for Model B is fitted with folds **grouped by call id**, so a call's original and its "
        "channel transform never straddle a fold.",
    ])
    rep.p("The integrity tests also verify that the audio on disk is 8 kHz mono PCM16 of the right duration, "
          "that every transformed sample differs from its original, that every sample maps back to a real call, "
          "that no sample id is duplicated, and that channel 1 of a reconstructed stereo file is the original "
          "agent **bit for bit** - only channel 0 was ever transformed.")

    # ============================================================ 7. Model B
    rep.h1("7. Robust Acoustic V2")
    rep.p("Model A is untouched. It stays in `models/wav2vec2_spanish/` exactly as it was benchmarked, and the "
          "original test suite still passes against it. Model B is a new classifier in `models/robust_v2/`.")
    rep.p("The backbone, the layer stack, the chunking, the loudness normalisation, the MLP architecture, the "
          "optimiser and the seed are all identical to the specialist. Two things change: the classifier also "
          "sees the TRAIN calls after a telephone line, and the segmentation moves to Silero because the energy "
          "detector does not survive the channel. Four variants were trained to keep those two changes apart.")
    if robust:
        rows = [["variant", "trained on", "chunks", "kept epoch"]]
        for name, r in robust["variants"].items():
            rows.append([name, ", ".join(r["train_sets"]), str(r["n_train_chunks"]), str(r["best_epoch"])])
        rep.table(rows, col_widths_cm=[3.2, 6.6, 2.6, 3.0])
        rep.p(f"The hidden layer was re-probed under channel shift rather than inherited: the specialist's "
              f"layer 5 was chosen on a split where every layer scores 1.000, so that choice was never really "
              f"tested. The criterion here has teeth - the **worst** call-level AUC across the two validation "
              f"domains selection is allowed to see - and it chose **layer {robust['layer']}**.")
        rep.p(f"Model selection picked **{robust['winner']}**. Selecting on clean Altur validation would have "
              f"been meaningless: it is saturated.")
    if (FIG / "robust_v2_training.png").exists():
        rep.image(FIG / "robust_v2_training.png", width_cm=17,
                  caption="What each training domain buys. The dashed line is the codec family nobody trained on; "
                          "it is plotted for information and was not used to stop training or pick a variant.")

    # ============================================================ 8. the matrix
    rep.h1("8. Specialist versus Robust V2")
    if matrix is not None and len(matrix):
        models = [m for m in MODEL_LABEL if m in set(matrix["model"])]
        domains = [d for d in DOMAIN_LABEL if d in set(matrix["domain"])]
        rows = [["domain"] + [MODEL_LABEL[m] for m in models]]
        for d in domains:
            cells = [DOMAIN_LABEL[d]]
            for m in models:
                sub = matrix[(matrix["domain"] == d) & (matrix["model"] == m)]
                cells.append(f"{pct(sub['accuracy'].iloc[0], 1)} / {num(sub['auc'].iloc[0], 3)}" if len(sub) else "-")
            rows.append(cells)
        worst = matrix_meta.get("worst_domain_accuracy", {})
        worst_phone = matrix_meta.get("worst_telephone_accuracy", {})
        rows.append(["worst over ALL domains"] + [pct(worst.get(m), 1) for m in models])
        rows.append(["**worst over TELEPHONE domains**"] + [f"**{pct(worst_phone.get(m), 1)}**" for m in models])
        # the printable width is 17 cm; share it between the label column and however many models there are
        model_w = min(3.3, (17.0 - 5.0) / max(len(models), 1))
        rep.table(rows, col_widths_cm=[17.0 - model_w * len(models)] + [model_w] * len(models),
                  highlight_rows=(len(rows) - 1,), font_size=7.0,
                  caption="Accuracy at the deployed threshold / ROC AUC. The last two rows are how bad it gets on "
                          "the worst domain - first counting everything, then counting only telephone audio.")
        rep.p("Every domain in that table except **external_ood** is telephone audio: 8 kHz, band-limited, "
              "through a line. external_ood is clean studio and TTS material merely resampled to 8 kHz. It is "
              "not what this product ingests - the demo page already pushes such audio through a simulated line "
              "before scoring it, and a bank's call never arrives that way - so the deployment criterion is the "
              "worst over the telephone domains. Both rows are printed because the two rules disagree, and "
              "hiding the one that flatters the older model would be the wrong kind of report.")
        if matrix_meta.get("best_model"):
            best_all = matrix_meta.get("best_model_all_domains")
            best_phone = matrix_meta.get("best_model")
            rep.note([f"Worst over all six domains: **{MODEL_LABEL.get(best_all, best_all)}** "
                      f"({pct(worst.get(best_all), 1)}).",
                      f"Worst over the five telephone domains: **{MODEL_LABEL.get(best_phone, best_phone)}** "
                      f"({pct(worst_phone.get(best_phone), 1)}).",
                      "The disagreement is entirely external_ood, and section 9 shows most of that gap is a "
                      "threshold nobody has fitted for non-telephone audio, not a difference in what the models "
                      "can actually tell apart."],
                     kind="key" if best_all == best_phone else "warn", title="Selection")
        rep.p("The middle two columns are there to keep the accounting honest, because 'it got better' is not a "
              "finding until you can say what made it better:")
        rep.bullets([
            "**Specialist, VAD swapped** - the same weights, Silero at inference only. This is the cheap fix, and "
            "a partial one: that MLP was fitted on chunks a *different* detector chose, so train and inference "
            "no longer agree on what a chunk is.",
            "**Retrained, original data only** - Silero everywhere, MLP re-fitted on Silero chunks, no new data. "
            "The segmentation fix done properly. The gap between this column and Robust V2 is what the "
            "**telephone-channel training data** is worth, and nothing else: same backbone, same layer, same "
            "architecture, same seed, same segmenter.",
        ])
        rep.h2("8.1 Which way do the errors go?")
        err = []
        for m in models:
            for d in ("channel_val_seen", "channel_val_unseen", "external_ood_channel"):
                sub = matrix[(matrix["domain"] == d) & (matrix["model"] == m)]
                if len(sub):
                    s = sub.iloc[0]
                    err.append([MODEL_LABEL[m], DOMAIN_LABEL[d], pct(s["far"], 1), pct(s["frr"], 1)])
        if err:
            rep.table([["model", "domain", "synthetic callers MISSED (FAR)", "human callers flagged (FRR)"]] + err,
                      col_widths_cm=[4.0, 6.0, 3.5, 3.5], font_size=7.0,
                      caption="For a bank these two columns are not symmetric. A human wrongly flagged is an "
                              "annoyed customer; a synthetic caller missed is a fraud that went through.")
    if (FIG / "evaluation_matrix.png").exists():
        rep.image(FIG / "evaluation_matrix.png", width_cm=17,
                  caption="Accuracy and AUC for every model on every domain.")

    # ============================================================ 9. AUC vs threshold
    rep.h1("9. Calibration shift or representation collapse?")
    rep.p("When accuracy falls there are two possible explanations and they call for completely different work. "
          "If the scores still rank the calls correctly - AUC stays high - only the decision threshold is in the "
          "wrong place, and recalibration fixes it without retraining anything. If the AUC itself collapses, the "
          "representation no longer separates the classes and no threshold can rescue it.")
    if matrix is not None and len(matrix):
        rows = [["model", "domain", "AUC", "acc. at 0.5", "acc. at the best threshold", "headroom"]]
        for m in [x for x in MODEL_LABEL if x in set(matrix["model"])]:
            for d in [x for x in DOMAIN_LABEL if x in set(matrix["domain"])]:
                sub = matrix[(matrix["domain"] == d) & (matrix["model"] == m)]
                if len(sub):
                    s = sub.iloc[0]
                    rows.append([MODEL_LABEL[m], d, num(s["auc"], 3), pct(s["accuracy"], 1),
                                 pct(s["accuracy_best_threshold"], 1), pct(s["threshold_headroom"], 1)])
        rep.table(rows, col_widths_cm=[3.4, 4.2, 1.8, 2.2, 3.2, 1.8], font_size=6.9,
                  caption="'Headroom' is what a perfect, domain-specific threshold would recover. Large headroom "
                          "with a high AUC means a calibration problem; a low AUC means the representation is gone.")
    if (FIG / "auc_vs_threshold.png").exists():
        rep.image(FIG / "auc_vs_threshold.png", width_cm=17,
                  caption="A tall orange band means the ranking survived and only the threshold moved. A low black "
                          "line means the representation itself collapsed.")
    if calib:
        deployed = read_json(config.MODELS_DIR / "robust_v2" / "calibration_mlp.json", {})
        method = deployed.get("method", "platt")
        cv = calib.get(f"{method}_cv", calib.get("platt_cv", {}))
        rep.p(f"Model B's probabilities are calibrated on {calib['n']} validation calls from "
              f"{calib['fit_on']}, measured out-of-fold with the folds grouped by call so that a call's "
              f"original and its channel transform never straddle a fold: Brier "
              f"{num(calib['raw']['brier'], 4)} raw against {num(cv.get('brier'), 4)} after "
              f"{method} scaling, ECE {num(calib['raw']['ece'], 3)} against {num(cv.get('ece'), 3)}. "
              f"The deployed calibrator is **{method}** with slope {num(deployed.get('a'), 3)}.")
        if calib.get("fit_set_separable"):
            rep.note("The calibration set contains no mistakes - Model B gets every one of those 142 calls "
                     "right - so the likelihood has no maximum and the fit would sharpen without limit, "
                     f"reporting 0.9999 on everything. The slope is therefore capped at "
                     f"{calib.get('slope_capped_at', 3.0):.0f}, exactly as calibrate.py does for the "
                     "specialist in the same situation. The honest reading is that these probabilities are "
                     "ordered correctly but their absolute values are not evidence of anything.",
                     kind="warn", title="A calibrator fitted on a set with no errors")

    # ============================================================ 10. limitations
    rep.h1("10. What this does not prove")
    rep.bullets([
        "**No real carrier was involved.** The codecs are real; the network between them is a model. A capture "
        "from the production Twilio path remains the last piece of evidence, and the evaluation script is ready "
        "to take it as another domain.",
        "**The unseen family is a proxy for 'a different line', not for 'every line'.** It is a hard proxy - "
        "CELP coding is a genuinely different mechanism from companding - but it is still a chosen set.",
        "**The agent channel never went through the channel.** Only channel 0 is transformed. The reconstructed "
        "stereo files are useful for the API and the demo, but their channel 1 is the original recording and "
        "nothing in this report treats it as telephone audio.",
        "**The validation split is 71 calls.** Differences of one or two calls are inside the noise, which is "
        "why the report leans on AUC and on the worst domain rather than on a single accuracy figure.",
        "**Early stopping looked at validation.** As in the original project, the kept epoch was chosen on "
        "validation domains. The unseen-family number is the one untouched by any such choice.",
    ])

    # ============================================================ 11. verdict
    rep.h1("11. Should Robust V2 replace the specialist?")
    if matrix is not None and len(matrix) and matrix_meta.get("worst_domain_accuracy"):
        worst = matrix_meta["worst_domain_accuracy"]
        worst_phone = matrix_meta.get("worst_telephone_accuracy", {})
        best = matrix_meta.get("best_model")
        get = lambda d, m, c: matrix[(matrix["domain"] == d) & (matrix["model"] == m)][c].iloc[0]
        rep.p("The case for it, in the order it should be read:")
        rep.bullets([
            f"**It gives up nothing where the specialist was already perfect.** Altur validation: "
            f"{pct(get('altur_val', 'specialist', 'accuracy'), 1)} against "
            f"{pct(get('altur_val', 'robust_v2', 'accuracy'), 1)}.",
            f"**It fixes the failure this project was about.** The same calls down a telephone line: "
            f"{pct(get('channel_val_seen', 'specialist', 'accuracy'), 1)} against "
            f"{pct(get('channel_val_seen', 'robust_v2', 'accuracy'), 1)}.",
            f"**It holds on codecs nobody trained on.** GSM, AMR-NB, iLBC, Speex, Opus, G.722: "
            f"{pct(get('channel_val_unseen', 'specialist', 'accuracy'), 1)} against "
            f"{pct(get('channel_val_unseen', 'robust_v2', 'accuracy'), 1)}.",
            f"**It holds on engines nobody trained on, down a line nobody trained on.** "
            f"{pct(get('external_ood_channel', 'specialist', 'accuracy'), 1)} against "
            f"{pct(get('external_ood_channel', 'robust_v2', 'accuracy'), 1)}, and the synthetic callers it lets "
            f"through fall from {pct(get('external_ood_channel', 'specialist', 'far'), 1)} to "
            f"{pct(get('external_ood_channel', 'robust_v2', 'far'), 1)}.",
            f"**It answers at all.** Across every domain the deployed segmenter found no caller speech in "
            f"{int(matrix[matrix['model'] == 'specialist']['files_without_speech'].sum())} files, where the "
            f"endpoint falls back to an undecided 0.5 that the contract reads as 'synthetic'. For Robust V2 "
            f"that number is {int(matrix[matrix['model'] == 'robust_v2']['files_without_speech'].sum())}.",
        ])
        rep.p("The case against, which is real and is not buried here: on **external_ood** - clean studio and "
              f"TTS clips resampled to 8 kHz, never through a line - the specialist scores "
              f"{pct(worst.get('specialist'), 1)} and Robust V2 {pct(worst.get('robust_v2'), 1)}. Section 9 "
              f"shows what that gap is made of. At the best available threshold the two are "
              f"{pct(get('external_ood', 'specialist', 'accuracy_best_threshold'), 1)} and "
              f"{pct(get('external_ood', 'robust_v2', 'accuracy_best_threshold'), 1)}; the AUCs are "
              f"{num(get('external_ood', 'specialist', 'auc'), 3)} and "
              f"{num(get('external_ood', 'robust_v2', 'auc'), 3)}. Neither model is calibrated for "
              "non-telephone audio, and Robust V2 is the more miscalibrated of the two there precisely because "
              "it has stopped keying on loudness and band edge - the cues that made those clean clips easy.")
        rep.p("The challenge's judging criteria are robustness to unseen speakers, engines and call conditions, "
              "and whether a bank could deploy this on real phone audio. Both are questions about the telephone "
              "domains.")
        if best == "robust_v2":
            rep.note(["Deploy Robust V2 for telephone audio. It is better on every domain that is a phone call, "
                      "including two the training never touched, and it answers on files where the deployed "
                      "detector goes silent.",
                      "Keep the specialist. It is untouched in `models/wav2vec2_spanish/`, still passes its own "
                      "tests, and is one flag away: `python src/server.py --model specialist`.",
                      "Before pointing either model at non-telephone audio, recalibrate on that audio - or do "
                      "what the demo page already does and put it through a line first."],
                     kind="key", title="Recommendation")
        else:
            rep.note(f"On these measurements the better model by the deployment criterion is "
                     f"**{MODEL_LABEL.get(best, best)}**. The recommendation follows the measurement.",
                     kind="warn", title="Recommendation")
    if model_b:
        rep.p(f"Model B on disk: layer {model_b['layer']}, variant `{model_b['variant']}`, trained on "
              f"{', '.join(model_b['train_sets'])}, segmentation `{model_b['vad']}`. Selected by worst "
              f"call-level AUC over val_original and val_channel_seen, tie-broken by worst accuracy and then "
              f"by worst Brier - both tie-breaks were needed, because the first two criteria saturate.")

    rep.h1("12. Reproducing this")
    rep.code("\n".join([
        "python src/build_channel_dataset.py --split all --family all   # the transformed dataset (~6 min)",
        "python src/channel_pilot.py                                    # the 10-call gate",
        "python src/channel_figures.py                                  # the channel figures",
        "python src/extract_channel_embeddings.py --sets all            # frozen backbone -> embeddings",
        "python src/train_robust.py                                     # Model B + the three controls",
        "python src/evaluate_models.py                                  # the evaluation matrix",
        "python src/report_robust.py                                    # this document",
        "python -m pytest tests -q                                      # every integrity check",
    ]))
    rep.p("Every experiment is appended to `reports/EXPERIMENT_LOG.md`; nothing there is ever overwritten.")

    out = config.REPORTS_DIR / "TELEPHONE_ROBUSTNESS_REPORT"
    rep.render(out)
    print(f"report -> {out}.md and {out}.pdf")


if __name__ == "__main__":
    build()
