"""Report section 8c - the out-of-dataset check (imported by build_report.py)."""
import _bootstrap  # noqa: F401
import config
from build_report import F, f3, pct


def part_external(rep, d):
    ext = d.get("external")
    rep.h1("Part 8c - The honest number: audio from outside the dataset")
    rep.p("Validation cannot separate 'detects synthetic speech' from 'detects the pipeline the synthetic calls came through', "
          "because both splits came through the same pipeline. The only way to know is audio the dataset never contained: "
          "new TTS engines and new human speakers, pushed through the **same** simulated telephone path and scored by the "
          "deployed model exactly as the judges' endpoint would. Nothing here was used for training or selection.")
    rep.table([["Set", "Source", "Clips", "What it represents"],
               ["synthetic", "Microsoft Edge neural TTS: es-MX Dalia/Jorge, es-CO Salome/Gonzalo, es-AR Elena/Tomas, es-US Paloma/Alonso", "32", "modern neural voices, Mexican and other Latin-American Spanish - the kind of engine a fraudster would use"],
               ["synthetic", "Google TTS (es-MX)", "8", "a second neural engine"],
               ["synthetic", "Windows SAPI Sabina (es-MX), Helena (es-ES)", "8", "old concatenative desktop voices"],
               ["human", "FLEURS es_419 (read Latin-American Spanish, many speakers)", "40", "real people the dataset never recorded, same language"],
               ["human", "ASVspoof 2019 bonafide (English studio speech)", "20", "real people, different language and microphone"]],
              [1.6, 7.0, 1.2, 7.2], font_size=7.2)
    rep.p("Every clip goes through: resample to 8 kHz, 1.5 s of faint line noise before and after, a silent agent channel, "
          "and then one of two channel variants: **8k** (nothing else) or **pstn** (300-3400 Hz band-pass + G.711 mu-law, "
          "i.e. a classic telephone line). Sentences for the TTS clips are bank-call phrases in Spanish.")
    if not ext:
        rep.p("*(external check not run)*")
        return
    pv = ext["per_variant"]
    rows = [["Channel variant", "Accuracy", "Balanced acc.", "AUC", "EER", "Synthetic missed", "Humans flagged"]]
    for v, name in (("pstn", "pstn: band-pass + mu-law (closest to a real call)"), ("8k", "8k: resample only")):
        m = pv[v]
        rows.append([name, f"**{pct(m['accuracy'])}**", pct(m["balanced_accuracy"]), f3(m["auc"]), pct(m["eer"]), pct(m["far"]), pct(m["frr"])])
    rep.table(rows, [5.6, 1.8, 2.0, 1.4, 1.6, 2.2, 2.4], caption=f"Deployed model on {ext['n_clips']} out-of-dataset clips (48 synthetic, 60 human), threshold p = 0.5.")
    rows = [["Source", "Label", "n", "correct (pstn)", "median p(synthetic) (pstn)", "correct (8k)", "median p(synthetic) (8k)"]]
    for src, r in ext["per_source"].items():
        lab = "synthetic" if src.startswith(("edge", "gTTS", "Windows")) else "human"
        rows.append([src, lab, str(r["pstn"]["n"]), pct(r["pstn"]["correct"]), f3(r["pstn"]["median_p_synthetic"]),
                     pct(r["8k"]["correct"]), f3(r["8k"]["median_p_synthetic"])])
    rep.table(rows, [4.6, 1.6, 0.8, 2.2, 2.9, 2.0, 2.9], font_size=7.0, caption="Per source. 'correct' = share of clips on the right side of p = 0.5.")
    rep.image(F / "external_check.png", "Detector scores of every out-of-dataset clip (red = synthetic engines, blue = human speakers). "
              "Left: resample only. Right: through a simulated PSTN line.")
    rep.h2("Reading the result")
    rep.bullets([f"**The evidence transfers.** On engines and speakers it never saw, through a simulated phone line, the model reaches "
                 f"{pct(pv['pstn']['accuracy'])} accuracy and AUC {f3(pv['pstn']['auc'])}. Every Windows voice, Google TTS and 6 of 8 Edge "
                 "neural voices are caught; 90% of the Spanish speakers and 80% of the English speakers are accepted. This is real "
                 "synthetic-speech evidence, not only the dataset's pipeline.",
                 "**The threshold depends on the channel.** With resampling only (no band limit) the same model flags half of the clean "
                 "human recordings as synthetic while still catching every neural voice: clean, noise-free audio *looks synthetic* to "
                 "it, because in the training data the synthetic callers were the clean ones and the humans were on real, noisy lines. "
                 "The ranking (AUC 0.90) survives; the 0.5 cut-off does not.",
                 "**So the honest number is not 100%.** 100% is what happens on this dataset's own pipeline. On genuinely new voices "
                 "and lines, expect roughly **85-90% with a fixed threshold**, and a ranking quality of AUC 0.90-0.95. The judges test "
                 "'unseen speakers, engines and call conditions': this is the number to say out loud.",
                 "**Caveats.** 108 clips; a simulated line, not a real one; read sentences, not conversation; the neural-TTS clips passed "
                 "through MP3 (a cue of its own). Indicative, not definitive - but it is the only measurement here that does not "
                 "share the dataset's pipeline."])
    rep.note(["Not memorisation, not a leak - but also not 100%. The model learned something about synthetic speech that carries to "
              "new engines (AUC ~0.95 off-pipeline), *plus* a preference for 'clean audio = synthetic' inherited from how this "
              "dataset was recorded. The behavioural branch, which ignores the audio path entirely, is the natural fix for the second part."],
             "key", "Verdict")
