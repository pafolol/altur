# The semantic layer — what arrived, and what is still missing

This directory is the merge of the `fusion` branch (`061c93f`, "semantica"). **The module's source code is
not in it, because it was never pushed.** What follows is exactly what the branch contained, how that was
verified, and what has to happen for the layer to run.

## What is here

| path | what it is |
| --- | --- |
| `cache/asr/` | ElevenLabs Scribe transcripts, 353 calls × 2 channels — word-level, with timings and per-word logprobs |
| `cache/asr_scribe_v2/` | the same, from a second Scribe configuration |
| `cache/gemini/` | the 7-dimension Gemini rubric per call, in four model/prompt variants (`gemini-3.5-flash-lite` × prompts `7f24a2` / `fcb51f`, `…_scribe_v2`, and `gemini-3.6-flash`) — 353 calls each |
| `cache/*.log` | the batch logs of those two runs |
| `bytecode/` | the nine `.pyc` files that were pushed instead of the sources (`asr, audit, config, features, model, rubric, server, traps, vad`, CPython 3.14) |

The caches are the expensive part — they are two full passes of a paid ASR API and four full passes of an LLM
over every call in the dataset. They are laid out the way the module expects (`<root>/cache/asr`,
`<root>/cache/gemini`), so dropping the sources in beside them should hit the cache rather than the APIs.

## What is missing

- **every `.py` file.** Verified against the GitHub API on the branch head `061c93f3438e…`, not a local clone:
  `__pycache__/` holds nine `.pyc`, `reports/` holds only a `.DS_Store`, `tests/` holds only a `__pycache__/`.
  Zero `.py` blobs on that branch — and zero on any other branch of the repository.
- **`model.pkl`** — the fitted logistic + Platt calibration. Without it there is no probability to contribute.
- **`scores_semantico.csv`** — the out-of-fold score per call, which would have let the console show the
  semantic column for the 71 held-out calls without running anything.

A directory whose only surviving file is `.DS_Store` is the signature of a **global** gitignore (something
like `~/.gitignore_global` with a broad rule) rather than a repository one: the branch has no `.gitignore` at
all. Whoever owns the module should check `git check-ignore -v model.py` on their machine, then push the
sources and `model.pkl`.

## ⚠ Two things that need action

1. **The branch committed a `.env` with live API keys** (ElevenLabs and Gemini). It has been removed from the
   working tree here, but it remains reachable in Git history through the merged commit, and it was already
   pushed. **Those keys must be rotated** — deleting the file does not undo the exposure.
2. The ASR caches are **verbatim transcripts of the challenge audio**. The dataset terms say not to
   redistribute it, and a transcript is the dataset in another form. Keep this repository private.

## How the layer plugs in

The fusion does not import this module; `SemanticLayer` in `src/fusion.py` reaches it over HTTP. That is
deliberate — it needs two third-party API keys, it is the only layer that leaves the machine, and its cost is
a network budget rather than a model's compute.

```
python src/server.py --semantic-url http://127.0.0.1:8100/detect
# or: set SEMANTIC_URL=http://127.0.0.1:8100/detect
```

The service must answer:

```
GET  /health
POST /detect  {"audio": "<base64 stereo 8 kHz wav>"}
  -> {"is_synthetic", "confidence", "score", "abstain", "reason", "used", "ms"}
```

`score` is the calibrated P(synthetic) and is what the fusion reads. `confidence` is P(the verdict is right),
which is a different quantity and would be wrong to feed a weighted vote. `used` names the path that
answered — `"f4+f5"` (Scribe + rubric), `"f4"` (text features only, recorded at half evidence quality) or
`"abstain"`.

Until something answers, the layer abstains on every call, its 15 % is shared out among the layers that did
answer, and the console shows the channel greyed out with the reason. Nothing waits for it.

## What the module does, as far as it can be reconstructed

Recovered from the strings in `bytecode/` — enough to write the adapter against, not enough to run:

```
Fase 1  vad.py       energy VAD + compaction of the caller channel
Fase 2  traps.py     trap bank by template matching on the AGENT channel, no ASR: the scripted moments
                     (saludo, pide_referencia, permitame_confirmar, asi_es, interrupcion, producto,
                     reconfirmacion) located by sliding cosine similarity over log-band spectra
Fase 3  asr.py       ElevenLabs Scribe on the compacted audio, chunked on segment boundaries, concurrent,
                     word times mapped back to the original timeline, cached per call
Fase 4  features.py  F4 text features with no LLM: logprob mean/std/p10/min/low-fraction/first-30,
                     disfluency rate, mexicanism rate, false-start rate, sentence length, words per turn,
                     guion rate, plus f_agent — the share of the caller's vocabulary the agent also said
                     ("humans echo the agent to confirm; bots follow their script")
Fase 5  rubric.py    one Gemini call per conversation, structured JSON, 7 dimensions scored 0–1 with a short
                     quote each: sobrecompletitud, registro_formal, reparacion_conversacional, iniciativa,
                     manejo_producto_ambiguo, fidelidad_repeticion, consistencia_memoria
Fase 6  model.py     logistic regression on F4 + F5, class_weight balanced, Platt-calibrated by CV on train
Fase 7  server.py    the /detect contract above, degrading f4+f5 -> f4 -> abstain within a hard deadline
```

For reference, a logistic fit on the **cached rubric alone** (the 7 dimensions, nothing else) reaches
val ROC-AUC 0.83 and accuracy 77.5 % on the official split — a real signal, and consistent with the 15 %
weight the fusion gives this layer. The module's own model adds the F4 text features on top of that.
