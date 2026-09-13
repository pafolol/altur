# 50 questions a judge could ask, and the answers

Every answer is grounded in this project's own artifacts. Where the honest answer is "we did not measure
that", it says so. Cross-references point at `PROJECT_TECHNICAL_REPORT.md`.

---

## The problem and the product

**1. What does your system actually do?**
It takes a stereo 8 kHz call, classifies the caller on channel 0 as human or synthetic, and returns
`{"is_synthetic": bool, "confidence": float}`. The verdict is a weighted vote of three independent
detectors: acoustic (the voice), behaviour (conversation timing) and semantic (the words). Acoustic and
behaviour decide every call 50/50; semantic is consulted only when those two do not settle it.

**2. Why three layers instead of one really good one?**
They fail for unrelated reasons. A codec degrades the acoustic layer but leaves timing untouched. A caller
who barely interacts silences the behaviour layer but not the voice. A new TTS engine can shift the
semantic layer's features without affecting either. On the held-out split the acoustic layer alone is
already perfect, so the fusion buys nothing there — it buys insurance for the domains we cannot see.

**3. Why is the third layer a verifier and not a third vote?**
Cost and privacy. It is the only layer that leaves the machine, the only one that costs money per call,
and the only one that sends customer audio to a third party. Gated at 0.80 joint primary confidence, it
ran on 5 of 71 calls — 7 % of what a flat vote would have spent. Accuracy is identical either way.

**4. What happens if the semantic service is down?**
The layer abstains, `combine()` redistributes its weight to the primaries, and the verdict is produced
normally. A layer that raises an exception is caught and converted to an abstention — one broken layer can
never take the endpoint down. There is a test for exactly that.

**5. What happens if the caller says nothing?**
The VAD returns no regions, the acoustic layer abstains with "no caller speech found", and if the
behaviour layer also has fewer than two interaction events, `combine()` returns an explicit **non-flag**:
`is_synthetic: false, confidence: 0.5, decisive: false`. We deliberately do not let `0.5 >= 0.5` accuse a
caller we know nothing about.

**6. Could a bank actually deploy this?**
As a risk signal, yes: it abstains rather than guessing, it is calibrated so a threshold is a business
decision, it produces a verdict with a median 133 seconds of call remaining, and it stores no audio. As
identity verification, no — see question 47.

---

## The data

**7. How much data do you have?**
353 calls: 282 train (113 human, 169 synthetic) and 71 validation (37 human, 34 synthetic). 14.5 hours of
audio, of which only **3.9 hours is the caller actually speaking**. All 353 files are 8 kHz, stereo,
16-bit PCM WAV with no exceptions.

**8. Is the split speaker-disjoint?**
The organisers document it as speaker-disjoint and we used it unchanged. We cannot verify that directly
because **the dataset contains no speaker ids, no TTS system ids and no codec fields**. That is precisely
why every leakage check we ran had to be done on the audio itself.

**9. Did you check for duplicates or leakage between train and validation?**
Yes, four ways. Zero duplicate MD5 hashes. Zero near-duplicate spectral fingerprints across the split
(max cross-split correlation 0.992, none above 0.995). The hex value in the filename has AUC 0.497 against
the label. The label sequence in id order has 174 runs where 173.5 are expected at random.

**10. What is channel 1 and why do you ignore it for the acoustic model?**
It is the Altur agent — which is **a synthetic voice on every single call in both classes**. It therefore
carries no label information, and any leakage of it into the caller channel would vary with how much the
agent talked, turning a conversational property into a spurious class cue. The behaviour and semantic
layers do use it, for timing and for grounding.

**11. Why is only a quarter of the audio useful?**
Because three quarters of a call is the agent talking or nobody talking. This is why we measure latency in
seconds of **caller speech** rather than seconds of call — the two differ by roughly a factor of four.

**12. Did the organisers give you turn annotations, and do you use them?**
They give one JSON per call with per-channel turn boundaries. We use them to **evaluate** our own
voice-activity detector and to compute dataset statistics. We never use them at inference, because a real
incoming call arrives as audio with no annotation.

**13. How balanced are the classes?**
Train is 60 % synthetic, validation 48 %. We handle the imbalance with inverse-frequency class weights in
the loss, and we report balanced accuracy alongside accuracy.

---

## Signal processing

**14. What is Nyquist and why does it matter here?**
A signal sampled at rate *f* can represent frequencies only up to *f*/2. At 8 kHz that is 4 kHz, so
**telephone audio contains no information above 4 kHz at all** — not attenuated, absent. Most of the
high-frequency artefacts that give away a vocoder on studio audio simply do not exist in our input.

**15. Why do you upsample 8 kHz audio to 16 kHz if that adds nothing?**
Because every pretrained speech backbone was trained at 16 kHz, and its convolutional front end has fixed
strides tuned for that rate. Feeding it 8 kHz would halve every effective receptive field and put the
input far outside the distribution the weights were learned on. We are explicit that **upsampling restores
nothing above 4 kHz** — it satisfies the model's expectations, it does not recover information.

**16. What is a spectrogram and do you use one?**
A spectrogram is the short-time Fourier transform magnitude: time on one axis, frequency on the other,
energy as brightness. We generate them for explanation, but **no deployed model sees one**. Wav2Vec2 takes
the raw waveform and learns its own front end.

**17. What is μ-law, and what is the classic mistake?**
μ-law is logarithmic companding: compress the dynamic range before quantising to 8 bits, expand after. It
is the base codec of the PSTN. The classic mistake is reading μ-law bytes directly as 16-bit PCM — they
are **codewords, not small linear samples**, and the result is noise of the right length. We have a
regression test that asserts exactly this failure mode is detectable.

**18. How do you know your G.711 implementation is correct?**
It is checked against Python's reference `audioop` implementation over **all 65 536 int16 values** for the
encoder and all 256 codes for the decoder, with exact array equality and no tolerance. Four tests, both
μ-law and A-law.

**19. How do you find the caller's speech?**
Frame the caller channel at 20 ms with a 10 ms hop, take the noise floor as the 10th percentile of frame
levels in dB, mark a frame as speech if it exceeds `max(floor + 15 dB, −55 dB)`, close gaps shorter than
0.30 s, keep runs of at least 0.30 s, pad by 0.10 s.

**20. Why 15 dB and not 12?**
We swept seven settings against the organisers' own turn annotations. 15 dB reduces the number of calls
where agreement collapses (F1 < 0.8) from **21 to 1**, at no meaningful cost to median F1 (0.953 vs 0.952).

**21. Why 4-second chunks?**
Long enough to contain several phonemes and syllable transitions, where synthesis artefacts live; short
enough that a median 37 seconds of caller speech still yields a median of 14 chunks per call. More
observations stabilises the call score — and, crucially, it is what makes the *prefix* of the running mean
a valid estimate, which is what makes our latency measurement exact rather than approximate.

**22. Why normalise every chunk to −26 dBFS?**
Because synthetic callers in this dataset are about 6 dB louder than humans, which is a trivial shortcut.
Normalising puts absolute loudness architecturally out of the acoustic model's reach. It does **not**
remove spectral shortcuts — a scalar gain does not change spectral shape.

---

## The model

**23. Why Wav2Vec2?**
Self-supervised speech pretraining gives a representation learned from tens of thousands of hours of
speech. We have three hours of caller audio. Learning speech representations from three hours is hopeless;
learning a decision boundary in a space where speech is already well represented is easy.

**24. Why the Spanish one specifically?**
We did not assume it — we ran a controlled three-backbone comparison against an English model (WavLM
Base+) and a 128-language model (XLS-R 300M), holding everything else fixed. The Spanish model won on mean
AUC under eleven stress conditions (0.9998), on stressed accuracy (97.2 % against 93.0 % and 86.2 %) and
on latency (119 ms against 255 and 310). The calls are LatAm Spanish, so a language match is a plausible
explanation — but we label that a hypothesis, because we never ran the ablation that would prove it.

**25. What is self-supervised pretraining, in one paragraph?**
The model is never shown a label. It is given a task it can grade itself: mask part of the latent speech
sequence, then identify from context which quantised latent belonged there, against distractors. To win
that game it has to model phonetics, coarticulation, speaker characteristics and channel properties.
Nobody told it about phonemes; representing them was the cheapest way to succeed.

**26. What is an embedding, concretely?**
A 4-second chunk becomes a `(T, 768)` matrix of hidden states — measured on a real chunk, `(156, 768)`,
about one step per 20 ms. Mean-pooling over time collapses that to a single 768-dimensional vector, where
each dimension is that dimension's average value across the chunk. Geometric closeness in that space means
acoustic similarity.

**27. Why an intermediate layer instead of the last one?**
Transformer layers specialise: lower layers stay close to the signal, upper layers drift toward what the
pretraining objective rewarded, which for a speech model means *what was said* rather than *how it
sounded*. A synthetic-voice detector wants the artefacts of generation, not the words. We use layer 5 of
12 — 42 % depth.

**28. How did you choose layer 5?**
Not from the clean split: **every layer of every backbone reached AUC 1.000 there**, including layer 0, the
raw convolutional encoder. A metric that returns the same answer for every candidate cannot rank
candidates. We re-probed every layer under eleven channel perturbations and chose on stressed performance.

**29. If layer 0 was already perfect, isn't the transformer pointless?**
On this dataset, for this split, yes — and that is a finding about the dataset, not about the model. A
convolutional front end reaching AUC 1.000 was one of the first strong hints that the separating signal
was gross and signal-level rather than linguistic.

**30. What does the classifier look like?**
Input standardisation using train statistics stored inside the checkpoint, then `Linear(768→256)`, ReLU,
Dropout 0.3, `Linear(256→2)`. The score is `logit[1] − logit[0]`, the log-odds of synthetic.
**197 378 trainable parameters** against 94 371 712 frozen — 0.21 %.

**31. Why freeze the backbone? Did you try fine-tuning?**
We tried it. Unfreezing the last four transformer layers at `lr = 1e-5` produced a mean stressed AUC of
**0.9997615262321146** against the frozen model's **0.9997615262321146** — identical to sixteen digits.
Meanwhile validation loss rose from 0.018 to 0.169 while training loss fell to 0.0003. Frozen ships.

**32. Why not train from scratch?**
Three hours of narrowband audio. A speech representation needs orders of magnitude more, and we have
direct evidence that even *fine-tuning* an existing one on this data overfits.

---

## Training and calibration

**33. Walk me through training.**
Extract mean- and std-pooled hidden states for every layer once, cached as float16. Probe every layer.
Train the MLP with AdamW, `lr = 1e-3`, weight decay 1e-4, batch 64, class-weighted cross-entropy, up to 60
epochs with early stopping on validation loss at patience 15. It stopped at epoch 18 and kept **epoch 3**.
Seed 42 throughout.

**34. Why cross-entropy?**
Because it punishes confident mistakes disproportionately: predicting 0.99 when the answer is 1 costs
0.01, predicting 0.01 costs **4.6**. For a bank, a detector that is wrong is bad and one that is
*confidently* wrong is dangerous, so that asymmetry is exactly right.

**35. How do chunk scores become one verdict?**
The arithmetic **mean of the chunk log-odds**, then the calibrator. Log-odds because independent evidence
adds in that space; averaging probabilities would let one saturated chunk dominate. We fixed `mean` in
config **before seeing results** and report the six alternatives as diagnostics only — choosing an
aggregation on a saturated split would be self-deception.

**36. What is calibration and why does it matter?**
A score of 0.858 is a ranking signal. Calling it a probability asserts that 85.8 % of calls scoring that
really are synthetic, which training does not guarantee. We fit Platt scaling, `σ(a·s + b)` with
`a = 0.9025, b = 0.8366`. The brief explicitly rewards well-calibrated systems.

**37. Where did you fit the calibrator, and why not on the validation split?**
Fitting on the clean split was **degenerate** — with zero errors the likelihood has no interior maximum,
the optimiser sharpened without limit and produced Brier 0.0000, a meaningless number. We refit on the 71
calls under all eleven stress conditions, 781 call-conditions, grouped by call so no call spans folds.
Brier 0.0200 → 0.0196, ECE 0.0190 → 0.0150.

**38. Does calibration change your accuracy?**
No. It is monotone, so it cannot change any ranking and therefore cannot change AUC or EER. It changes
Brier, log loss, ECE, and what a threshold means.

---

## The 100 %, and the shortcuts

**39. You report 100 % on validation. Why should we believe that?**
You should not take it at face value, and we do not. 71/71 has an exact **one-sided** 95 % lower bound of
**95.9 %** — `0.05^(1/71)`, or 94.9 % if you want the two-sided interval — and
one call is 1.4 accuracy points. More importantly the split is **saturated**: every backbone, every layer,
every classifier ties at 1.000, so the metric cannot rank anything. We treated that as a warning and went
looking for the cause.

**40. How do you know it is not data leakage?**
Label permutation. Shuffle the training labels and refit: AUC came back **0.471, 0.532, 0.510, 0.394,
0.592** — mean 0.500, chance. A pipeline with validation data in training, a label encoded in the features
or a metric bug scores well on shuffled labels. Ours collapses, which is what an honest pipeline does.

**41. So what *is* the explanation?**
The two classes in this dataset differ by more than their voices. A logistic regression on **24 call
statistics with no voice content whatsoever** — silence lengths, turn counts, loudness, bandwidth
summaries — reaches validation **AUC 1.000, accuracy 98.6 %**. You can saturate this split without any
voice modelling at all.

**42. Name the shortcuts.**
Leading silence before the caller's first word is the strongest single feature in the dataset: median
1.16 s for humans against **10.38 s** for synthetic, AUC 0.948. Then the high-band energy ratio (−30.7 dB
against −36.6 dB), loudness (−28.6 against −22.1 dBFS), median response latency (1.70 s against 2.79 s),
turn length and talk-over.

**43. Which of those can your acoustic model actually use?**
Timing shortcuts, none — the model only ever sees chunks *inside* detected caller speech. Absolute
loudness, none — every chunk is RMS-normalised. Noise-floor ratio, partly. **Band edge and spectral tilt,
yes** — normalisation is a scalar gain and does not change spectral shape. That last one is the real
exposure, and §13 measures its cost behaviourally.

**44. Did the model memorise the training voices?**
No. Ten training calls — five per class — already reach 98.5 % mean accuracy on all 71 validation calls,
and training on validation and testing on train gives 95.0 % with AUC 0.9998. A task learnable from ten
examples is not one that requires memorising voices.

**45. But your leave-one-voice-out accuracy dropped to 0.175. Isn't that catastrophic?**
Look at the AUC in the same row: **0.996**. The model still ranks every held-out synthetic call above
every held-out human call; only the fixed 0.5 threshold has moved off the data. That is a recalibration
problem, not a representation failure — and it is the single most repeated pattern in this whole project.
Separately, we showed the clustering instrument does not resolve individual speakers on 8 kHz audio, so
those clusters are recording conditions rather than people.

**46. How do you know the clusters aren't speakers?**
We brought in an independent VoxCeleb speaker-verification model. Within-class median pairwise similarity
was 0.760 for humans and 0.774 for synthetic; **cross-class was 0.750** — essentially the same. Calls from
dozens of different real people collapsed into two clusters. The instrument measures voice group, not
identity.

---

## Robustness and the telephone channel

**47. What happens on a real phone line?**
The specialist falls from 100 % to **70 %** on the first ten calls put through a simulated line, with
three verdict flips and four segmentation failures. Worse, **the failure is asymmetric**: humans held near
100 % while synthetic callers dropped to 40 %. After a codec, a deepfake starts to look human — which is
the failure direction that costs a bank money.

**48. Why does a codec break it?**
Because it removes the cues the model was partly keying on. The caller channel in this dataset is
**digitally silent between turns** — median frame level exactly −72.247 dB, the 16-bit floor, in 266 of 353
calls. A real line never delivers digital zero. After a channel the noise floor rises to −37.5 dB and total
dynamic range falls from 52 dB to **16 dB**.

**49. What does that do to your voice-activity detector?**
Kills it. The energy VAD needs speech 15 dB above the floor; after a line there are 16 dB of range in the
entire file. It found **no speech at all on 5 of 40 calls** — and a call with no chunks returns probability
0.5, which under the challenge contract reads as "synthetic". The model never even got to vote. That is why
the deployed model uses Silero VAD.

**50. Why did you simulate a telephone channel instead of capturing one?**
Two reasons. Time: 14.5 hours of audio needs 14.5 hours of real-time calling. And generalisation: one
captured carrier buys robustness to one carrier, whereas a parameterised simulator produces a
*distribution* of channels and lets us hold an entire codec family out for honest testing. We say plainly
in the report that no carrier was ever involved.

**51. What is in the simulator?**
Bit-exact G.711 μ-law and A-law in numpy, eleven more codecs through ffmpeg, band-pass filtering, spectral
tilt, Gilbert-Elliott bursty packet loss with concealment over 20 ms RTP frames, clock drift, DTX comfort
noise, line noise with optional mains hum, AGC, and cross-correlation re-alignment. 706 samples,
29.0 hours, 4.7 GB.

**52. What is the seen/unseen codec split and why does it matter?**
Five codecs are used in training — G.711 μ-law and A-law, G.726 at three bitrates. Eight are **never**
trained on: GSM, AMR-NB at three bitrates, iLBC, Speex, Opus narrowband, G.722. Seen is waveform coding;
unseen is mostly **parametric** coding, which analyses the voice and re-synthesises it. The held-out family
is therefore not just different codecs — it is a category of transformation that makes every voice partly
synthetic. A test enforces that no unseen codec ever reaches the training side.

**53. How do you avoid augmentation becoming a new shortcut?**
The channel is drawn **without ever consulting the label**. The RNG seed derives from
`(anon_id, family, variant)` only; the class travels alongside as metadata. If the channel distribution
differed by class, the model would learn "codec means synthetic" and the resulting accuracy would be pure
self-deception.

**54. How much did Robust V2 help?**
Through a line with seen codecs, 77.5 % → **100 %**. Unseen codecs, 81.7 % → **100 %**. Off-pipeline TTS
down a line, 62.0 % → **81.5 %**. Clean validation stays at 100 %.

**55. Your deployed model is worse than the old one on one domain. Explain.**
On clean, non-telephone audio Robust V2 scores 54.6 % against the specialist's 68.5 %. That is the intended
consequence of giving up a shortcut: those clean clips were easy precisely *because* of loudness and band
edge, and V2 has stopped keying on them. We selected on **worst-case accuracy across the telephone
domains**, because a bank's phone line cannot physically produce clean studio audio. We print both
criteria, name the runner-up, and record the exclusion in the deployment artifact itself.

**56. What is the number you actually stand behind?**
**88.0 % accuracy, AUC 0.945** — 108 out-of-dataset clips from three TTS engines and two human corpora,
band-passed through a simulated line. Not the 100 %.

---

## Metrics

**57. What is AUC and why do you quote it so much?**
The probability that a randomly chosen synthetic call scores above a randomly chosen human call. It
measures **ranking only**. That is why it can stay at 1.000 while accuracy collapses — a domain shift that
moves every score in the same direction leaves the ordering intact and strands the fixed threshold. We saw
exactly that at 0.175 accuracy with 0.996 AUC, and again under reverb at 0.887 with 1.000.

**58. What is EER?**
The error rate at the threshold where false acceptance equals false rejection. It summarises the trade-off
without arguing about where the threshold should sit. Ours is 0.000 on the held-out split and 0.129
off-pipeline.

**59. FAR or FRR — which matters more here?**
For a bank, **FRR** — a missed synthetic caller is a potential account takeover, while a false alarm costs
one extra verification step. A deployment would push the threshold toward lower FRR and accept more false
alarms, and because the model is calibrated that adjustment can be made in terms of an actual probability.

**60. Would you deploy at the EER threshold?**
No. The costs are wildly asymmetric, so the operating point should be chosen from those costs, not from a
symmetry point.

**61. What is Brier score and why report it?**
Mean squared error of the predicted probability. It is the metric that still had resolution when accuracy
and AUC had both saturated — it is literally what broke the tie between the four Robust V2 variants and
selected the model we shipped.

**62. What does your confusion matrix look like?**
On the 71 held-out calls, fused: 37 true human, 34 true synthetic, **zero false positives, zero false
negatives**. Which, per question 39, is a statement about the split rather than about the world.

---

## Fusion, latency and the product

**63. How exactly do you combine three layers?**
Each layer returns a probability, an evidence-quality score and an abstention flag. Effective weight is
zero for an abstaining layer under the default `renormalise` policy, so its share is redistributed rather
than replaced by a neutral 0.5. Primaries are mixed first; if `max(p, 1−p) >= 0.80` that is the verdict.
Otherwise verifiers are consulted and everything is re-mixed.

**64. Why 50/50/0.15 and why a gate at 0.80?**
They are stated design choices exposed as runtime knobs, **not fitted values**. The acoustic layer alone is
already at 100 % on these 71 calls, so there is no signal in this split with which to justify any
particular weighting — every setting that keeps the acoustic layer influential scores identically. Our own
evaluation output says so: "a sweep maximum describes this split, it is not a tuned setting you can promise
on the hidden test set."

**65. Give me a concrete case where the gate earned its keep.**
`call_569ffb0869eb`, a human caller. Acoustic said 0.000067, confidently human. Behaviour said **0.963752
— confidently, wrongly, synthetic**. The 50/50 landed at 0.482, 52 % confidence, below the gate. The
verifier said 0.046 and the fused verdict came out at 0.425: human, correct.

**66. Is the behaviour layer just "machines answer faster"?**
No, and we explicitly refuse that claim. Median within-call response-latency spread is 0.502 s for humans
and **0.635 s for synthetic** — the synthetic callers are *more* variable. Removing the consistency
features costs only 0.991 → 0.979 AUC. It is a 24-feature logistic regression on turn shape and interaction
events, not a slogan.

**67. Is the semantic layer just trap detection?**
No. The two planted conversational traps scored **AUC 0.45 and 0.35** — worse than chance — and the
module's own plan records the hypothesis as "false, corrected". The layer's signal is ASR confidence and
text style, and we also removed two features that correlated 0.94 with turn length because they were
double-counting the behaviour layer.

**68. How fast is it?**
Model inference is **119 ms** median on GPU, 850 ms on CPU. End to end, a settled call is about 1.0 s and
an escalated one about 3.9 s.

**69. How early can you decide?**
Median **2.33 seconds of caller speech**, 12.32 seconds into the call, with **132.7 seconds of lead time**
before the call would have ended. All 71 calls became confident, 100 % before the halfway point, and the
early verdict agreed with the final verdict **100 %** of the time.

**70. Is that real-time streaming?**
No, and we say so on the page and in `/api/health`, which returns `streaming: false`. It is a **prefix
evaluation**: the model scores a finished recording, and we ask what it would have concluded after *k*
chunks. That is exact rather than approximate because the aggregation is a mean of chunk log-odds, so the
mean of the first *k* is literally what the deployed model would have answered. There is a test asserting
the walk equals the production aggregation function.

**71. Where does the audio go?**
Acoustic and behaviour run entirely locally. The semantic verifier sends audio to ElevenLabs and the
transcript to Google. There is no offline mode for it, which is the main reason it is gated rather than
consulted on every call.

**72. What do you store?**
Scores, verdicts, timings and per-layer contributions in a SQLite file. **No audio.** The dataset terms
forbid redistributing the recordings and a database of them would be exactly that.

**73. What is `decided_at_s`?**
The second after which the running verdict never flips again. We walk the chunks, track whether the running
mean agrees with the final verdict, and **reset the marker whenever it disagrees** — a call that looks
decided at chunk 2 and flips at chunk 5 was never decided at chunk 2. It is `None` when the layer abstained
or never settled, never a placeholder.

**74. Two fields in your health endpoint look suspiciously honest.**
`streaming: false` and `queue_depth: 0`. The detector scores complete calls and serves requests
synchronously. We could have invented numbers there; we describe them in the code as "two fields the server
refuses to fake."

**75. The live phone demo — how does that work without a public URL?**
The usual way to collect a Twilio recording is a webhook, which means Twilio must reach your machine. We
issue inline TwiML with **no `action` attribute**, so Twilio ends the call after recording, and we collect
the audio by **polling Twilio's REST API**. No tunnel, no public URL, nothing to fail on stage.

**76. Which model scores the live call, and why?**
Robust V2. That is the one path where the audio really has been through a codec, and V1 falls to 77.5 %
there while V2 holds 100 %. Using the specialist would be knowingly deploying the wrong model.

**77. A Twilio recording is mono. What happens to the behaviour layer?**
It abstains, correctly — there is no separate agent track to time an interaction against. Channel 1 is
filled with zeros, which is the truth, and the page displays "No signal" rather than a fabricated number.

---

## The hard questions

**78. What would break this system?**
A TTS engine whose artefacts differ from the eleven voices in our external set; a carrier whose codec chain
is outside our simulated distribution; an attacker who deliberately adds noise or adjusts level; a caller
who says almost nothing. We have measured none of these adversarially.

**79. Is this adversarially robust?**
No. Nobody has tried to evade it. That is an honest gap, not a solved problem.

**80. What if the hidden test set is clean studio audio rather than telephone audio?**
Then Robust V2 is the wrong choice and the specialist would score better — 68.5 % against 54.6 % on
exactly that kind of data. Both models are on disk and the acoustic slot is a one-flag change
(`--acoustic-model`). We bet on telephone audio because the brief says telephone audio.

**81. What is the biggest limitation?**
That 71 calls cannot establish banking error rates, and that no audio in this project has been through a
real carrier. Everything telephone-related rests on a simulator — a detailed one with a bit-exact G.711,
but a model of a line nonetheless.

**82. What would you do next, with a week?**
Get real carrier audio, even a few hundred calls, which converts the entire simulated-channel argument from
plausible to demonstrated. Then ablate the band edge — low-pass the training data to a fixed cut-off,
retrain, and measure the drop; that is the one number we never measured and it would quantify exactly how
much of V1 was shortcut.

**83. Is there anything in the repository you would flag against yourselves?**
Three things. The layer-selection provenance: the clean-probe tie-break code as it stands today would pick
layer 0, not the layer 5 recorded in the checkpoints — the layers came from the stress probe, and the
deployed artifacts are unambiguous, but the clean-probe path would not reproduce them. The semantic
module's audit document describes a 10-feature model while 15 are served. And API keys were committed to
git history on a merged branch and **have not been rotated**.

**84. Why is there a second FastAPI app in your repository?**
It arrived with a merged teammate branch and is marked superseded at the top of its own integration
document. Nothing starts it. Its only surviving advantage is serving hygiene the main server lacks — a
request-size limit, a readiness endpoint, request ids, typed errors — which we list as future work rather
than quietly delete.

**85. Give me the one-sentence version of what makes this project different.**
Most teams would have stopped at 100 % on validation; we treated that as a defect, proved the dataset could
be solved without hearing a voice, built a telephone channel to break our own model, and shipped the model
that wins where a bank actually operates rather than the one with the better headline number.
