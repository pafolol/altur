"""Report Part 10b - the augmentation experiment (imported by build_report.py)."""
import _bootstrap  # noqa: F401
import config
from build_report import DISPLAY, F, f3, pct


def part_augmentation(rep, d):
    rep.h1("Part 10b - Augmentation experiment (model-specific optimisation)")
    rep.p("The stress test showed that under a channel change the score *ranking* survives (AUC stays close to 1) while the "
          "fixed p = 0.5 threshold does not (accuracy drops). A classifier that has only ever seen one line does not know "
          "how far a score may drift. The brief asked for augmentation to be treated as an experiment, after the baseline, "
          "and to make sure it does not destroy the artefacts we detect - so the design keeps two families of perturbations "
          "apart:")
    rep.bullets(["**Seen** families, used to perturb the training chunks: gain -12 dB, white noise 20 / 10 dB, pink noise 10 dB, mu-law codec.",
                 "**Unseen** families, never used in training: low-pass 3.4 / 3.0 kHz, PSTN band-pass, reverberation, spectral tilt.",
                 "Same backbone, layer, MLP, optimiser, seed and early-stopping rule; the only change is the training set "
                 "(clean chunks + one perturbed copy per seen family). Evaluated on the cached validation stress embeddings."])
    aug = d.get("augment")
    if not aug:
        rep.p("*(augmentation experiment not run)*")
        return
    k, L = aug["backbone"], aug["layer"]
    rows = [["Condition", "Family", "Baseline MLP: AUC / accuracy", "Augmented MLP: AUC / accuracy"]]
    for c in aug["baseline"]:
        fam = "seen" if c in aug["seen"] else ("**unseen**" if c in aug["unseen"] else "clean")
        b, a = aug["baseline"][c], aug["augmented"][c]
        rows.append([c, fam, f"{b['auc']:.3f} / {pct(b['accuracy'])}", f"{a['auc']:.3f} / {pct(a['accuracy'])}"])
    for g, label in (("seen", "mean over seen families"), ("unseen", "**mean over unseen families**"), ("all_stressed", "mean over all stressed")):
        rows.append([label, "", f"{aug[f'baseline_{g}_mean_auc']:.3f} / {pct(aug[f'baseline_{g}_mean_accuracy'])}",
                     f"{aug[f'augmented_{g}_mean_auc']:.3f} / {pct(aug[f'augmented_{g}_mean_accuracy'])}"])
    rep.table(rows, [3.4, 2.0, 5.8, 5.8], font_size=7.2,
              caption=f"{DISPLAY[k]} layer {L}: the same MLP trained on {aug['n_train_clean']:,} clean chunks vs "
                      f"{aug['n_train_augmented']:,} clean + perturbed chunks, scored on validation under every condition.")
    rep.image(F / f"augmentation_{k}.png", "Baseline vs augmented MLP per condition; shaded columns are the unseen families.")
    if aug["winner"] == "augmented":
        rep.note([f"The augmented head wins on the **unseen** families (mean AUC {aug['augmented_unseen_mean_auc']:.3f} vs "
                  f"{aug['baseline_unseen_mean_auc']:.3f}, accuracy {pct(aug['augmented_unseen_mean_accuracy'])} vs "
                  f"{pct(aug['baseline_unseen_mean_accuracy'])}) without losing on clean audio, so it is the deployed head "
                  f"(models/{k}/mlp_augmented.pt). Rule: {aug['rule']}."], "key", "Result")
    else:
        rep.note([f"The augmented head does not beat the baseline on the **unseen** families (mean AUC {aug['augmented_unseen_mean_auc']:.3f} vs "
                  f"{aug['baseline_unseen_mean_auc']:.3f}, accuracy {pct(aug['augmented_unseen_mean_accuracy'])} vs "
                  f"{pct(aug['baseline_unseen_mean_accuracy'])}), so the baseline head stays. Augmentation with noise, gain and "
                  f"codec teaches robustness to noise, gain and codec - not to band limits or spectral tilt. Rule: {aug['rule']}."],
                 "info", "Result")
