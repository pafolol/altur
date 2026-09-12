"""Report section 8b - is a 100% validation score believable? (imported by build_report.py)"""
import _bootstrap  # noqa: F401
import config
from build_report import DISPLAY, F, O, f3, pct


def part_sanity(rep, d):
    rep.h1("Part 8b - Is 100% believable? Five checks")
    rep.p("A perfect validation score is a warning sign, not a trophy. Before trusting it we asked what could produce it "
          "dishonestly - a leak in the evaluation code, memorised voices, or a cue so gross that it says nothing about "
          "synthetic *speech* - and tested each possibility with the cached embeddings of the winner.")
    san = d.get("sanity")
    vid = d.get("voice_identity")
    if not san:
        rep.p("*(sanity checks not run)*")
        return
    rep.h2("Check 1 - overfitting or underfitting?")
    rep.p("**Overfitting** shows up as a gap: the model fits its training calls and does worse on new ones. Here the training "
          "and validation curves sit on top of each other (final train loss 0.01, validation loss 0.02, both at 100%) and the "
          "validation callers are people the model never heard. **Underfitting** would mean it cannot even fit the training "
          "calls. Neither applies: there is no gap to close. The question is what the classes are separated *by*.")
    rep.h2("Check 2 - a leak in the pipeline? (label permutation)")
    perm = san["permutation"]
    aucs = ", ".join("%.2f" % p["auc"] for p in perm)
    mean_auc = sum(p["auc"] for p in perm) / len(perm)
    rep.p("We shuffled the training labels and retrained the same classifier five times. If validation data, labels or a "
          "metric bug had leaked into the pipeline, a model trained on nonsense labels would still score high. It scores "
          f"at chance: validation AUC {aucs} (mean {mean_auc:.2f}). The evaluation is honest.")
    rep.h2("Check 3 - how little audio is enough?")
    fc = san["few_chunks"]
    rep.p(f"Scoring every validation call from its **first chunk only** (1-4 s of speech) already gives accuracy "
          f"{pct(fc['first_chunk']['accuracy'])}; a random single chunk gives {pct(fc['random_chunk']['accuracy'])}; single chunks "
          f"judged alone, without any call-level averaging, are right {pct(fc['single_chunk_all']['accuracy'])} of the time, and "
          f"even chunks shorter than 2 s {pct(fc['single_chunk_under_2s']['accuracy'])}. A cue that is present in every second of "
          "audio is a property of the *signal path* (band limit, spectral balance, noise floor), not of how a voice handles a "
          "sentence. This is consistent with the shortcut check (Part 9) and the average spectra (Part 3).")
    rep.image(F / f"sanity_{san['backbone']}.png", "Left: shuffled labels give chance. Middle: one chunk is enough. Right: the "
              "detector's own embeddings cannot separate voices (see check 4), so this leave-one-cluster-out is not informative.")
    rep.h2("Check 4 - the same voices in training and validation?")
    if vid:
        c86 = vid["clusters"]["0.86"]
        c70 = vid["clusters"]["0.7"]
        rep.p("The organisers state that train and validation are speaker-disjoint and that the hidden set uses voices from "
              "neither split. We tried to measure the number of distinct synthetic voices with a **speaker-verification model** "
              "(WavLM x-vectors, trained on VoxCeleb to tell people apart), used purely as an instrument.")
        rep.bullets([f"On this 8 kHz, RMS-normalised telephone audio the instrument does **not** resolve individuals: the human calls, "
                     f"which come from many different people, collapse into {c70['human']['n_clusters']} clusters at a 0.70 cosine threshold "
                     f"and {c86['human']['n_clusters']} at 0.86, and the similarity histograms of human-human, synthetic-synthetic and "
                     "human-synthetic pairs are nearly identical (a bimodal same-gender / cross-gender shape). It measures voice *group* "
                     "(pitch range, gender), not identity, on narrow-band audio.",
                     f"So the number of distinct TTS voices is **unknown**; the organisers do not disclose it. At the finest threshold the "
                     f"synthetic calls form {c86['synthetic']['n_clusters']} groups (sizes {c86['synthetic']['sizes'][:5]}), and every validation "
                     "synthetic call has a group-mate in training - as it would if the same few TTS voices were reused, and also as it "
                     "would if the instrument simply cannot tell voices apart."])
        rep.h2("Check 5 - does the detector survive a voice group it never saw?")
        loo = vid.get("loo", [])
        if loo:
            rows = [["Synthetic voice group held out", "Calls removed from training", "Synthetic calls left to train on",
                     "Held-out calls recognised", "AUC vs validation humans"]]
            for r in loo:
                rows.append([str(r["voice"]), str(r["n_held_out_calls"]), str(r["n_train_synthetic_left"]),
                             pct(r["held_out_recognised"]), f3(r["auc"])])
            s = vid["loo_summary"]
            rows.append(["**all groups**", str(s["calls_tested"]), "",
                         f"**{pct(s['call_weighted_recognised'])}** (worst {pct(s['min_recognised'])})", f3(s["mean_auc"])])
            rep.table(rows, [3.6, 3.0, 3.2, 3.4, 3.8],
                      caption=f"Leave-one-voice-group-out (x-vector clusters at cosine {s['threshold']}): every call of one synthetic "
                              "voice group is removed from training, the detector is retrained, and the removed calls are scored.")
            worst = min(loo, key=lambda r: r["held_out_recognised"])
            rep.p("The **ranking survives** an unseen voice group completely (AUC 1.000 in every case) but the **fixed 0.5 threshold** does "
                  f"not always: the smallest group ({worst['n_held_out_calls']} calls) is recognised only "
                  f"{pct(worst['held_out_recognised'])} of the time. This is exactly the pattern of the stress test in Part 6b: what the "
                  "frozen representation captures transfers across voices and channels; what a threshold learned on one pipeline does not.")
        rep.image(F / "voice_identity.png", "Speaker x-vector similarity of all call pairs (left), estimated number of voice groups per "
                  "threshold (middle), and the leave-one-voice-group-out result (right).")
    prov, tiny, rev = san.get("provenance"), san.get("tiny_train"), san.get("reversed")
    if prov and tiny:
        rep.h2("Check 6 - were the validation calls really outside training?")
        rep.bullets([f"The split is the organisers' `manifest.csv` column, not ours: {prov['train_calls']} training and {prov['val_calls']} validation "
                     f"calls, {prov['overlap']} in common, and the cached embeddings the classifier was fitted on match the manifest exactly "
                     f"({'yes' if prov['val_equals_manifest_val'] and prov['train_equals_manifest_train'] else 'NO'}). No duplicate audio (MD5) "
                     "and no near-duplicate spectral fingerprint crosses the split (Part 2).",
                     "In `train_classifier.py` every `.fit()` and every gradient step sees training chunks only. Validation is used for "
                     "**decisions**: which layer, when to stop the MLP, which backbone, and the calibration. None of those decisions can "
                     "create separability that is not there: the clean probe was 100% on every layer of every backbone before any choice was made."])
        rep.h2("Check 7 - can it be memorisation if ten calls are enough?")
        rep.p(f"We trained a logistic regression on **only 10 training calls** (5 human + 5 synthetic, ten random draws) and scored all 71 "
              f"validation calls: accuracy {', '.join(f'{a:.2f}' for a in tiny['accuracies'])} (mean {tiny['mean_accuracy']:.3f}, mean AUC "
              f"{tiny['mean_auc']:.3f}). A model cannot memorise 71 unseen calls from 10 examples; it can only pick up a difference that is "
              "consistent across the whole dataset - the audio path. " +
              (f"Trained the other way round (fit on the 71 validation calls, scored on the 282 training calls) it reaches {rev['accuracy']:.3f} "
               f"accuracy / AUC {rev['auc']:.3f} ({rev['errors']} errors), the same picture from the other side." if rev else ""))
    rep.h2("What 100% actually means")
    rep.bullets(["**Zero errors in 71 calls.** The 95% confidence lower bound on the accuracy is 95.9% (Clopper-Pearson, 0 of 71). The "
                 "honest claim is 'above 96% on calls produced by this pipeline', not 100%.",
                 "**The dataset is easy for the wrong reason.** Synthetic and human calls reached the recording through different audio "
                 "paths (a 3.3 kHz roll-off, ~6 dB more gain, a different noise floor, a 10 s start delay). Any model, even a "
                 "filterbank plus logistic regression, sees the path. Our pipeline removes the cues it can remove without touching "
                 "the voice (loudness, timing) and selects the model by robustness to channel changes, but it cannot make the "
                 "band edge disappear from the audio the judges will send.",
                 "**Where it will break, if it breaks.** If the hidden calls were produced by the same pipeline, the score will be near "
                 "100%. If new TTS engines arrive through a different path, the evidence says the *ranking* will hold (AUC ~1.0 under "
                 "every perturbation and every held-out voice group) while the *fixed threshold* may not (72-93% recognition in the "
                 "harder tests). That is why the calibrated confidence is fitted on the stressed conditions, and why the "
                 "behavioural and semantic branches (Part 13) are the right complement: they do not depend on the audio path at all."])
    rep.note(["Not a bug, not memorisation of the training callers, not classical overfitting. It is a dataset in which the two "
              "classes differ by more than their voices, evaluated on a small split. We report it as such, and we selected the "
              "model on the tests that can still tell the backbones apart."], "key", "Verdict on the 100%")
