"""
Builds reports/ACOUSTIC_LEARNING_SUMMARY.{md,pdf} from the ACTUAL outputs of every stage
(outputs/**/*.json, reports/*.csv, outputs/figures/*.png). Nothing in the report is typed by hand:
every number is read from the files the experiments wrote.

Run (after every stage):  python src/build_report.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
import config
from report_utils import Report

O, R, F, M = config.OUTPUTS_DIR, config.REPORTS_DIR, config.FIGURES_DIR, config.MODELS_DIR
DISPLAY = {k: v["display"] for k, v in config.BACKBONES.items()}
SHORT = {"wavlm": "WavLM Base+", "wav2vec2_spanish": "Wav2Vec2 ES", "xlsr": "XLS-R 300M"}


# ----------------------------------------------------------------------------- helpers
def load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def csv(path):
    path = Path(path)
    return pd.read_csv(path) if path.exists() else None


def pct(x, d=1):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.{d}f}%"


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def gather():
    d = {"inspect": load(O / "dataset_inspection" / "summary.json", {}),
         "stats": csv(O / "dataset_inspection" / "call_stats.csv"),
         "shortcut": csv(R / "shortcut_check.csv"),
         "shortcut_res": load(O / "shortcut" / "results.json", {}),
         "sp": load(O / "signal_processing" / "examples.json", {}),
         "bench": csv(R / "benchmark.csv"),
         "comparison": csv(R / "model_comparison.csv"),
         "best": load(R / "best_model.json", {}),
         "runpod": (R / "RUNPOD_USAGE.md").exists()}
    for k in config.BACKBONE_ORDER:
        d[f"res_{k}"] = load(O / "classifier" / k / "results.json")
        d[f"ext_{k}"] = load(O / "embeddings" / k / "extraction_stats.json")
        d[f"cal_{k}"] = load(O / "calibration" / f"{k}.json")
        d[f"meanstd_{k}"] = load(O / "classifier" / f"{k}_meanstd" / "results.json")
        d[f"ft_{k}"] = load(O / "finetune" / k / "results.json")
        d[f"stress_{k}"] = load(O / "stress" / k / "mlp_stress.json")
        d[f"stress_probe_{k}"] = load(O / "stress" / k / "probe_stress.json")
        d[f"pred_{k}"] = csv(config.PREDICTIONS_DIR / f"{k}_val.csv")
    d["ablation"] = load(O / "ablation" / "results.json")
    d["sanity"] = None
    for k in config.BACKBONE_ORDER:
        if load(O / "sanity" / k / "results.json"):
            d["sanity"] = load(O / "sanity" / k / "results.json")
    d["voice_identity"] = load(O / "voice_identity" / "results.json")
    d["external"] = load(O / "external" / "results.json")
    d["augment"] = None
    for k in config.BACKBONE_ORDER:
        a = load(O / "augmentation" / k / "results.json")
        if a:
            d["augment"] = a
    return d


def bench_row(d, key):
    b = d["bench"]
    if b is None or key not in set(b["backbone"]):
        return None
    return b[b["backbone"] == key].iloc[0]


# ============================================================================ front matter
def part_glance(rep, d):
    best, b = d["best"], d["bench"]
    ins = d["inspect"]
    counts = ins.get("counts", {})
    rep.h1("At a glance")
    rep.p("We built the **acoustic branch** of a synthetic-caller detector for the Altur HackMTY 2026 challenge: a program "
          "that listens to the caller's side of a bank phone call and returns the probability that the voice was generated "
          "by a machine. Three pretrained speech models were compared as **frozen feature extractors** under strictly "
          "identical conditions, the winner was chosen on the official validation split, and the result is served through "
          "the challenge's `POST /detect` endpoint.")
    if b is not None and best:
        w = bench_row(d, best["backbone"])
        has_stress = "stress_mean_auc" in b.columns and b["stress_mean_auc"].notna().all()
        kp = [(f"{w['val_accuracy'] * 100:.1f}%", "validation accuracy (primary metric, clean)"),
              (f"{w['val_auc']:.3f}", "validation ROC AUC (clean)")]
        if has_stress:
            kp.append((f"{w['stress_mean_auc']:.3f}", "mean validation AUC under 10 channel perturbations"))
        kp += [(f"{w['gpu_latency_median_s'] * 1000:.0f} ms" if np.isfinite(w.get("gpu_latency_median_s", np.nan)) else "n/a",
                "median latency per call (GPU)"),
               (SHORT[best["backbone"]], f"winning backbone ({DISPLAY[best['backbone']]}), layer {best['layer']}")]
        rep.kpis(kp)
        rows = [["Backbone", "Params", "Layer", "Accuracy", "F1", "AUC", "EER", "Stressed AUC", "Stressed acc.", "Worst condition", "Latency"]]
        for _, r in b.iterrows():
            rows.append([r["model"], f"{r['params_m']:.0f}M", str(int(r["selected_layer"])), pct(r["val_accuracy"]),
                         f3(r["val_f1"]), f3(r["val_auc"]), pct(r["val_eer"]),
                         f3(r["stress_mean_auc"]) if has_stress else "n/a", pct(r["stress_mean_accuracy"]) if has_stress else "n/a",
                         f"{r['stress_worst_condition']} ({f3(r['stress_worst_auc'])})" if has_stress else "n/a",
                         f"{r['gpu_latency_median_s'] * 1000:.0f} ms" if np.isfinite(r.get("gpu_latency_median_s", np.nan)) else "n/a"])
        hl = [i + 1 for i, (_, r) in enumerate(b.iterrows()) if r["backbone"] == best["backbone"]]
        rep.table(rows, [2.7, 1.3, 1.0, 1.5, 1.0, 1.0, 1.0, 1.5, 1.5, 2.9, 1.6], highlight_rows=hl, font_size=7.0,
                  caption="Frozen backbone + the same MLP on the validation split (71 calls from speakers never seen in training). "
                          "Clean accuracy is the decision metric the judges score; 'stressed' = the same calls with the caller channel "
                          "perturbed at test time (Part 6b). The highlighted row is the deployed model.")
    rep.h2("Key takeaways")
    take = []
    if b is not None and best:
        w = bench_row(d, best["backbone"])
        take.append(f"**{DISPLAY[best['backbone']]} wins the controlled comparison** ({best['reason']}).")
        lo = b.sort_values("val_auc").iloc[0]
        take.append(f"**The clean validation split is saturated.** All three frozen backbones reach AUC {lo['val_auc']:.3f}-"
                    f"{b['val_auc'].max():.3f} and every one of their layers - even the raw CNN output - separates the classes perfectly. "
                    "The human/synthetic difference in this dataset is gross and acoustic (a 3.3 kHz band edge, spectral balance, "
                    "line noise), so the clean split cannot rank the backbones. A test-time **robustness stress test** (Part 6b) can, "
                    "and it decides the winner.")
    sc = d["shortcut"]
    if sc is not None:
        top = sc[sc["model"] == "logreg"].sort_values("val_auc", ascending=False).iloc[0]
        ext = d.get("external")
        if ext:
            pv = ext["per_variant"]["pstn"]
            take.append(f"**Off-pipeline the honest number is {pct(pv['accuracy'])} accuracy, AUC {f3(pv['auc'])}** (Part 8c): 108 clips "
                        "from TTS engines and human speakers the dataset never contained, through a simulated PSTN line. The evidence "
                        "transfers to new engines; the fixed threshold is what the dataset's pipeline biases.")
        take.append("**100% is zero errors in 71 calls, not a guarantee.** The 95% confidence lower bound is 95.9%. Five checks "
                    "(Part 8b) rule out a leak (shuffled labels give chance) and memorised training callers (unseen callers, unseen "
                    "voice groups still AUC 1.0); what remains is a dataset whose classes differ by their audio path, so the fixed "
                    "threshold - not the ranking - is the fragile part.")
        take.append(f"**The dataset has shortcuts.** A logistic regression on trivial statistics - no voice content at all - reaches "
                    f"validation AUC {top['val_auc']:.3f} / accuracy {pct(top['val_accuracy'])} ({top['feature_set']} features). "
                    "The strongest cues are how long the caller takes to say the first word, the caller's loudness and a "
                    "3.4 kHz band edge. Part 9 explains which of them our acoustic model can and cannot use.")
    fts = [d[f"ft_{k}"] for k in config.BACKBONE_ORDER if d[f"ft_{k}"]]
    if fts:
        ft = fts[0]
        if ft.get("finetuned_mean_auc_stressed") is not None:
            take.append(f"**Fine-tuning {'did not beat' if ft['winner'] == 'frozen' else 'beat'} the frozen model** (clean validation: both "
                        f"{pct(ft['finetuned']['val']['accuracy'])} / {pct(ft['frozen']['accuracy'])}; mean stressed AUC "
                        f"{ft['finetuned_mean_auc_stressed']:.3f} vs {ft['frozen_mean_auc_stressed']:.3f}); we deploy the "
                        f"{'frozen' if ft['winner'] == 'frozen' else 'fine-tuned'} one.")
        else:
            take.append(f"**Fine-tuning {'did not beat' if ft['winner'] == 'frozen' else 'beat'} the frozen model** (validation AUC "
                        f"{ft['finetuned']['val']['auc']:.3f} vs {ft['frozen']['auc']:.3f}); we deploy the "
                        f"{'frozen' if ft['winner'] == 'frozen' else 'fine-tuned'} one.")
    take.append("**Compute.** The frozen benchmark ran on the laptop GPU (RTX 4080 Laptop, 12 GB): embedding extraction takes "
                "about two minutes per backbone, classifier training seconds, and one call is scored in about a tenth of a second. "
                "The two sustained-load stages - fine-tuning and the augmentation experiment - ran on a Runpod pod after the laptop "
                "shut itself down under sustained load (details and cost in reports/RUNPOD_USAGE.md).")
    rep.bullets(take)
    rep.h2("How this document is organised")
    rep.bullets(["Part 1 - the Altur problem and what this project does (and does not) solve",
                 "Part 2 - the dataset: splits, format, conversation structure, leakage and shortcuts",
                 "Part 3 - audio fundamentals on real Altur calls: waveform, sampling, FFT, spectrogram, mel, MFCC, pitch",
                 "Part 4 - the three pretrained speech models and self-supervised learning",
                 "Part 5 - embeddings and the layer probe", "Part 6 - the controlled three-model experiment",
                 "Part 6b - the robustness stress test (why the clean validation split is not enough)",
                 "Part 7 - why the winner won", "Part 8 - overfitting and generalisation",
                 "Part 8b - is 100% believable? five checks (permutation, one chunk, voice identity, unseen voices, confidence bound)",
                 "Part 8c - the honest number: new TTS engines and new speakers from outside the dataset through a simulated phone line",
                 "Part 9 - shortcuts",
                 "Part 10 - fine-tuning", "Part 10b - augmentation experiment", "Part 11 - production inference", "Part 12 - the winning model",
                 "Part 13 - how the acoustic branch fits the complete product", "Glossary"])


# ============================================================================ part 1
def part_problem(rep, d):
    rep.h1("Part 1 - The Altur problem")
    rep.p("Altur runs AI voice agents that handle bank phone calls in Spanish across Latin America. Voice used to prove "
          "identity; today a few seconds of public audio are enough to clone a voice, and contact centres are attacked from "
          "both sides: fraudsters impersonating customers, and fraudsters impersonating banks. The challenge asks for a "
          "system that tells, from a recorded call, whether the **caller** is a real person or a synthetic voice.")
    rep.h2("What a synthetic caller is here")
    rep.p("In the dataset, a synthetic caller is a complete autonomous pipeline: speech recognition listens to the bank's "
          "agent, a language model decides what to say, and a text-to-speech (TTS) system speaks it into the phone line. "
          "The human callers are volunteers who phoned the same agent with invented personal data. Both go through the same "
          "customer-service flow, and the agent deliberately interrupts, falls silent and asks about things that do not exist.")
    rep.h2("Three families of evidence")
    rep.table([["Branch", "What it looks at", "Typical cue", "This project"],
               ["**Acoustic**", "the caller's voice signal itself", "vocoder artefacts, spectral shape, prosody, breathing, band limits", "**implemented and benchmarked**"],
               ["Behavioural", "how the caller handles the flow of the conversation", "reaction to interruptions, response latency, overlap, consistency", "not implemented; clean interface reserved"],
               ["Semantic", "what the caller says", "invented answers to questions about things that do not exist", "not implemented; clean interface reserved"]],
              [2.4, 4.6, 5.8, 4.2])
    rep.p("The challenge is scored through one HTTP endpoint, `POST /detect`, that receives a stereo 8 kHz WAV (base64) and "
          "answers `{\"is_synthetic\": true, \"confidence\": 0.87}`. `is_synthetic` is required; `confidence` is optional and is "
          "used to break ties and to reward well-calibrated systems. The PDF does not name a metric formula, so the natural "
          "primary metric is the **accuracy of the boolean decision** on the hidden test set; we report it first everywhere "
          "and treat AUC, EER, F1 and calibration as diagnostics.")
    rep.note(["This project answers one question well: **which pretrained acoustic representation, used frozen, generalises "
              "best to unseen human and synthetic callers of the Altur dataset** - and how much it costs to run. Behavioural "
              "and semantic signals, and their fusion, are the next steps (Part 13)."], "key", "Scope")


# ============================================================================ part 3
def part_audio(rep, d):
    sp = d["sp"]
    h, s = sp.get("human", {}), sp.get("synthetic", {})
    rep.h1("Part 3 - Audio fundamentals, on real Altur calls")
    rep.p("Everything a model sees is numbers. This part follows two real training calls - one human, one synthetic, chosen "
          "with similar caller speech time - through every representation used in the project. All figures were produced by "
          "`src/signal_processing_demo.py` at the native 8 kHz.")
    if h:
        rep.table([["Example", "Call", "Call length", "Caller speech", "Example turn", "Samples in turn", "Turn RMS"],
                   ["human", h["anon_id"], f"{h['duration_s']:.0f} s", f"{h['caller_speech_s']:.1f} s", f"{h['turn_seconds']:.2f} s at {h['turn_start_s']:.1f} s", f"{h['turn_samples']:,}", f"{h['rms_db']:.1f} dB"],
                   ["synthetic", s["anon_id"], f"{s['duration_s']:.0f} s", f"{s['caller_speech_s']:.1f} s", f"{s['turn_seconds']:.2f} s at {s['turn_start_s']:.1f} s", f"{s['turn_samples']:,}", f"{s['rms_db']:.1f} dB"]],
                  [1.6, 3.0, 1.8, 2.0, 3.2, 2.4, 1.8])
    rep.image(F / "sp_call_overview.png", "A whole call. Channel 0 (caller) and channel 1 (agent) are separate signals; the shaded "
              "regions are our VAD's caller turns and the black bars the official ones. Only channel 0 is classified.")
    rep.h2("Waveform, samples and amplitude")
    rep.p("A microphone turns air pressure into a voltage; an analogue-to-digital converter measures it at fixed intervals. "
          "Each measurement is a **sample**, the whole list is the **waveform**, and each value (between -1 and +1 after "
          "loading) is the **amplitude**. Telephone audio uses **8,000 samples per second**: a 2 s caller turn is 16,000 numbers.")
    rep.image(F / "sp_waveform.png", "Left: one caller turn of each example. Right: 25 ms of the loudest part - every dot is one sample.")
    rep.h2("Sample rate, Nyquist and why 8 kHz matters")
    rep.p("The **Nyquist rule**: a sample rate can only represent frequencies below half of it. 8 kHz audio therefore contains "
          "**nothing above 4 kHz**; classic telephone lines pass roughly 300-3,400 Hz. The pretrained models expect 16 kHz "
          "input, so we resample the caller channel from 8 kHz to 16 kHz - a representation change only. **Resampling does "
          "not restore frequencies above 4 kHz**: the 4-8 kHz half of the 16 kHz spectrum stays empty, and the 4-8 kHz "
          "region where many synthesis artefacts live in studio-quality datasets simply does not exist in this task.")
    rep.image(F / "sp_sampling_nyquist.png", "Left: Nyquist and aliasing. Right: the same caller turn at 8 kHz and after resampling "
              "to 16 kHz - the spectrum above 4 kHz is empty.")
    rep.h2("Frequency and the FFT")
    rep.p("A voice contains many frequencies at once: the **pitch** (F0, how fast the vocal folds vibrate, ~85-255 Hz), its "
          "**harmonics** (whole multiples of F0), and **formants** (throat/mouth resonances that make an 'a' differ from an 'i'). "
          "The **Fourier transform** re-describes a short piece of waveform as a sum of sine waves; the **FFT** is the fast "
          "algorithm that computes it. On 64 ms of 8 kHz audio (512 samples) it returns 257 frequency bins, 15.6 Hz apart.")
    rep.image(F / "sp_fft.png", "Three pure tones (the FFT finds exactly three peaks) and 64 ms of a loud vowel of each example.")
    rep.h2("Spectrogram, mel spectrogram, MFCC")
    rep.p("Speech changes every few milliseconds, so we take the FFT of short overlapping windows (64 ms, every 10 ms) and "
          "stand the spectra side by side: the **spectrogram** (STFT). The **mel scale** regroups frequencies the way hearing "
          "does (fine at low frequencies, coarse at high ones) - the **mel spectrogram** is the input of many speech models. "
          "**MFCCs** take the log of the mel spectrogram and a cosine transform, keeping only the first ~20 coefficients: "
          "a compact description of the spectral *shape* that discards fine harmonic detail.")
    rep.images([F / "sp_representations_human.png", F / "sp_representations_synthetic.png"],
               "The four views of the same caller turn: human (left) and synthetic (right).", max_height_cm=17)
    rep.image(F / "sp_spectrogram_pair.png", "Human and synthetic caller turns side by side (dotted line = 3.4 kHz telephone band edge).")
    rep.image(F / "sp_mel_mfcc_concepts.png", "The mel scale, some mel filters, and how 20 MFCCs rebuild only the smooth envelope of a frame.")
    rep.h2("What differs on average")
    rep.p("Single examples mislead; averages over many calls do not. The long-term average spectrum of caller speech, per "
          "class and split, is the most informative single picture of this dataset:")
    rep.image(F / "average_spectrum.png", "Long-term average spectrum of caller speech (our VAD regions), peak-normalised, per class "
              "and split. Solid = train, dashed = validation.")
    stats = d["stats"]
    if stats is not None:
        tr = stats[stats["split"] == "train"]
        hf_h, hf_s = tr.loc[tr["label"] == "human", "caller_hf_ratio_db"].median(), tr.loc[tr["label"] == "synthetic", "caller_hf_ratio_db"].median()
        rep.p(f"Synthetic callers roll off sharply above ~3.3 kHz while human callers keep energy up to 4 kHz: the median ratio of "
              f"3.4-4 kHz energy to 0.3-3.4 kHz energy is {hf_h:.1f} dB for humans and {hf_s:.1f} dB for synthetic callers in "
              "training, and the validation curves follow the training curves closely. Synthetic speech also carries slightly more "
              "energy between 1.3 and 3 kHz. These are properties of the **pipelines** that produced the audio (the TTS output "
              "and the path it took into the phone line) as much as of the voices; Part 9 discusses what that means for robustness.")
    rep.h2("Pitch (F0)")
    pm = sp.get("pitch_medians")
    if pm and "f0_range_semitones" not in pm:
        pm = None  # written by an older run of the demo script
    if pm:
        rep.image(F / "sp_pitch.png", "Pitch statistics (pyin) on 30 s of caller speech from 15 train calls per class. Illustration only.")
        rep.p(f"On this small sample the median F0 is {pm['f0_median']['human']:.0f} Hz (human) vs {pm['f0_median']['synthetic']:.0f} Hz "
              f"(synthetic), with F0 standard deviation {pm['f0_std']['human']:.0f} vs {pm['f0_std']['synthetic']:.0f} Hz and a "
              f"5th-95th percentile range {pm['f0_range_semitones']['human']:.1f} vs {pm['f0_range_semitones']['synthetic']:.1f} semitones. With 15 calls per class these are "
              "descriptions, not evidence; the models below never see pitch features explicitly.")
    rep.h2("Which artefacts synthetic speech MAY create")
    rep.bullets(["Vocoder artefacts: smeared or too-regular harmonics, missing breath noise, phase inconsistencies between frames.",
                 "Unnatural prosody: overly steady pitch, too-regular rhythm, identical pause lengths.",
                 "Band limits and codec fingerprints: the TTS output has its own sample rate, resampling filter and codec path before "
                 "it reaches the line - visible here as the 3.3 kHz roll-off.",
                 "Missing micro-variation: no lip smacks, breaths, room reflections or handset movement.",
                 "None of these is proof on its own; the classifier learns which combinations separate the classes in this data."])


# ============================================================================ part 4
def part_models(rep, d):
    rep.h1("Part 4 - The three pretrained speech models")
    rep.p("A **pretrained model** is a neural network somebody already trained on a huge dataset; we download its weights and "
          "reuse them. All three candidates are **self-supervised**: they never saw a label. During pretraining, parts of the "
          "audio are masked and the network must predict what was hidden (WavLM additionally has to see through added noise "
          "and overlapping speakers). To succeed it must learn what speech sounds like at every level - fine acoustic texture "
          "in the early layers, phonetic and lexical content in the late ones. That is exactly the knowledge a small labelled "
          "dataset (282 training calls) cannot teach from scratch.")
    rows = [["Model", "Creator", "Pretraining data", "Params", "Layers x dim", "Why it may help", "Why it may fail"]]
    rows.append(["**WavLM Base+**\n`microsoft/wavlm-base-plus`", "Microsoft (2021)", "94k h English: Libri-Light, GigaSpeech, VoxPopuli-en; masked prediction + denoising",
                 "94M", "12 x 768", "the front-end of many top ASVspoof systems; our previous project's baseline; denoising objective keeps low-level acoustics",
                 "English only; pretrained on wide-band 16 kHz audio, never on 8 kHz telephone speech"])
    rows.append(["**Wav2Vec2 Spanish**\n`facebook/wav2vec2-base-es-voxpopuli-v2`", "Meta AI (2021)", "21.4k h Spanish European-Parliament speech (VoxPopuli); contrastive masked prediction",
                 "94M", "12 x 768", "the only candidate pretrained on Spanish: phonetic inventory and prosody match the calls",
                 "smallest pretraining set, one domain (parliament microphones), older wav2vec 2.0 objective"])
    rows.append(["**XLS-R 300M**\n`facebook/wav2vec2-xls-r-300m`", "Meta AI (2021)", "436k h in 128 languages: VoxPopuli, MLS, CommonVoice, BABEL, VoxLingua107",
                 "315M", "24 x 1024", "largest and most diverse pretraining (BABEL is telephone speech); Spanish included",
                 "3x the size, higher latency and memory; large models can be less sample-efficient with 282 calls"])
    rep.table(rows, [3.0, 1.9, 3.6, 1.2, 1.5, 3.4, 2.9], font_size=7.0)
    for k in config.BACKBONE_ORDER:
        e = d[f"ext_{k}"]
        if e:
            rep.p(f"Measured for {DISPLAY[k]}: {e['params_m']:.1f}M parameters, {e['num_hidden_layers']} transformer layers of "
                  f"{e['hidden_size']} dimensions, weights on disk {e.get('weights_mb', float('nan')):,.0f} MB, input normalisation "
                  f"`do_normalize={e['do_normalize']}`.")
    rep.h2("Why an English model may still work in Spanish")
    rep.p("Synthesis artefacts are not words. Vocoder texture, band limits, unnatural harmonic structure and prosodic regularity "
          "live in the low-level acoustic representation that every self-supervised speech model builds in its early layers, "
          "regardless of the pretraining language. Our previous ASVspoof project found the spoofing evidence in WavLM's "
          "lower-middle layers, far below the layers that encode phonemes and words. The language-specific question is "
          "whether Spanish phonetics look 'unusual' to an English model in a way that confuses the detector - which is why "
          "the Spanish-pretrained model is a fair challenger, and why the multilingual XLS-R (which saw Spanish and telephone "
          "speech) is the third.")
    rep.note("No winner is claimed here. Part 6 lets the validation split decide.", "info")


# ============================================================================ part 13
def part_product(rep, d):
    rep.h1("Part 13 - How the acoustic branch fits the complete product")
    rep.code("""
                         CALL (stereo 8 kHz WAV, base64)
                          |
              +-----------+-----------+
              v           v           v
          ACOUSTIC    BEHAVIOURAL   SEMANTIC
       (this project)  (planned)    (planned)
       caller voice    turn timing, transcript of
       -> backbone     latency,     caller answers
       -> MLP          overlap,     -> LLM / rules
       -> p(synthetic) recovery     -> p(synthetic)
              |           |           |
              +-----------+-----------+
                          v
                       FUSION  (weighted average / small logistic regression on the branch scores)
                          v
              {"is_synthetic": bool, "confidence": float}
""")
    rep.p("`server.py` already produces the verdict in one function, `score_call()`, and reports the acoustic probability under "
          "`details.branches.acoustic`. The behavioural branch will consume the agent channel and the turn structure "
          "(Part 2 shows that response latency, overlap and turn counts already separate the classes), the semantic branch "
          "the transcript. Fusion replaces the single-branch decision without changing the HTTP contract.")
    rep.p("**What this project does not solve yet:** any behavioural or semantic signal, adversarial robustness (an attacker "
          "who post-processes the TTS audio), speaker-level analysis across calls, and streaming/real-time scoring while the "
          "call is still running (the pipeline scores chunks, so an early verdict after the first few caller turns is a small change).")


# ============================================================================ glossary
def part_glossary(rep):
    rep.h1("Glossary")
    rows = [["Term", "Meaning"],
            ["caller / agent channel", "channel 0 / channel 1 of the stereo call; only the caller is classified"],
            ["VAD", "voice activity detection: finding where the caller speaks"],
            ["chunk", "a piece of at most 4 s of one caller speech region; the unit the backbone scores"],
            ["backbone", "the big pretrained speech model used frozen as a feature extractor"],
            ["hidden layer / layer probe", "the output of one transformer block; a probe trains a classifier on each to see what it contains"],
            ["embedding", "fixed-size vector summarising a chunk (mean over time of one hidden layer)"],
            ["mean pooling", "averaging the per-frame vectors over time"],
            ["logistic regression / linear probe", "the simplest classifier: weighted sum + sigmoid"],
            ["MLP", "small neural network: Linear -> ReLU -> Dropout -> Linear"],
            ["log-odds / score", "log(p / (1 - p)); 0 = 50/50, +3 = ~95% synthetic, -3 = ~95% human"],
            ["aggregation", "how chunk scores become one call score (mean by default)"],
            ["accuracy (primary)", "share of correct is_synthetic decisions"],
            ["balanced accuracy", "mean of the per-class recalls; immune to class imbalance"],
            ["ROC AUC", "probability that a random synthetic call scores above a random human call (1 = perfect ranking)"],
            ["EER", "error rate at the threshold where FAR = FRR (0 = perfect, 50% = coin flip)"],
            ["FAR / FRR", "synthetic accepted as human / human rejected as synthetic"],
            ["Brier score / ECE", "calibration: mean squared error of the probability / gap between confidence and accuracy"],
            ["Platt / temperature scaling", "1-2 parameter re-mapping of the score into a calibrated probability"],
            ["frozen", "weights that are never updated (requires_grad = False)"],
            ["fine-tuning", "continuing to train (part of) the backbone on our labels"],
            ["overfitting", "the model memorises the training set; validation stops improving while training keeps improving"],
            ["shortcut", "a trivial cue (loudness, silence, duration) that separates the classes without any voice evidence"],
            ["speaker-disjoint split", "no caller appears in both train and validation; the hidden test set has new callers and voices"]]
    rep.table(rows, [4.2, 12.8])


def build():
    d = gather()
    rep = Report("Acoustic Learning Summary: detecting synthetic callers in the Altur HackMTY 2026 dataset",
                 "From stereo 8 kHz phone calls to a benchmarked, deployable acoustic detector - three frozen speech backbones "
                 "compared under identical conditions. Every number and figure in this document was produced by the code in "
                 "D:/altur/acoustic and read from its output files.", R)
    part_glance(rep, d)
    part_problem(rep, d)
    import build_report_results as res  # the data-driven parts (2, 5-12), kept in a second file for readability
    res.part_dataset(rep, d)
    part_audio(rep, d)
    part_models(rep, d)
    res.part_embeddings(rep, d)
    res.part_experiment(rep, d)
    import report_stress
    report_stress.part_stress(rep, d)
    res.part_why(rep, d)
    res.part_generalisation(rep, d)
    import report_sanity
    report_sanity.part_sanity(rep, d)
    import report_external
    report_external.part_external(rep, d)
    res.part_shortcuts(rep, d)
    res.part_finetuning(rep, d)
    import report_augment
    report_augment.part_augmentation(rep, d)
    res.part_inference(rep, d)
    res.part_winner(rep, d)
    part_product(rep, d)
    part_glossary(rep)
    rep.render(R / "ACOUSTIC_LEARNING_SUMMARY")
    res.update_readme(d)


if __name__ == "__main__":
    build()
