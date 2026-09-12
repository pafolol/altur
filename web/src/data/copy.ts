/**
 * Every line of copy on the page except the transcripts (calls.ts) and the
 * acronym (acronym.ts). Headings and labels are plain text. Paragraph-level
 * fields (`lede`, `text`, `note`, list items) may carry inline HTML —
 * &nbsp;, <i>, <code> — exactly as in the prototype.
 * The <title> and <meta name="description"> live in /index.html.
 *
 * Every number here is read from the repository's own reports: README.md (Results),
 * reports/ACOUSTIC_LEARNING_SUMMARY.md, behaviour/reports/, semantic/AUDIT.md.
 * The long version, with sources, is reports/ISISI_PRESENTER_GUIDE.pdf.
 */

export const nav = {
  brand: 'ISISI',
  /** Accessible name of the logo link, which goes back to the top. */
  brandLabel: 'Back to the start',
  sub: 'Altur · HackMTY26 · Defend the bank against voice deepfakes',
  link: { label: 'The endpoint', href: '#api' },
}

export const hero = {
  title: 'Your voice stopped being a password.',
  sub: 'A few seconds of public audio is enough to clone anyone well enough to pass — for a human listener, and for a machine.',
  tapHint: 'Tap the lock',
  /** Accessible name of the tap zone. */
  tapLabel: 'Break the lock',
  scrollHint: 'SCROLL',
  /** Shown under the lock skeleton while the turntable frames decode. */
  loading: 'Loading the lock',
}

/** The name act, after the lock breaks. Each letter is a button that jumps to its word. */
export const nameAct = {
  hint: 'Scroll to read the name',
}

export type Diagram = 'spec' | 'overlap' | 'sem'

export const signals = {
  heading: 'Three ways a machine gives itself away on a phone call.',
  lede: 'Three independent detectors read the same call and fail in different places. The acoustic and behaviour layers decide every call, half and half. The semantic layer is the expensive one, so it is asked only when the first two do not settle it.',
  /** Labels drawn inside the card diagrams. */
  diagramLabels: {
    overlap: { bargeIn: 'BARGE-IN' },
    sem: { asked: 'ASKED', missing: 'NO HESITATION' },
  },
  cards: [
    {
      diagram: 'spec' as Diagram,
      what: 'SIGNAL 01 — ACOUSTIC',
      title: 'What the voice is made of',
      text: 'Only the caller&rsquo;s channel, only where a voice activity detector hears speech, cut into chunks of at most 4&nbsp;s. Each chunk goes through a frozen Spanish speech model, Wav2Vec2 pretrained on VoxPopuli, and a small classifier reads its fifth hidden layer. Nothing in the backbone was trained on this data.',
      points: [
        'Chunk scores are averaged in log-odds, so the running verdict can be read at any point in the call',
        'Chosen on a stress test, not the clean split: 10 channel perturbations, mean AUC 0.9998',
        'A second model, hardened on real telephone codecs, is one flag away',
      ],
    },
    {
      diagram: 'overlap' as Diagram,
      what: 'SIGNAL 02 — BEHAVIOUR',
      title: 'When the caller speaks, yields and answers',
      text: 'Both channels through a voice activity detector; turns, pauses and overlaps become 24 timing features, scored by a logistic regression with 40 parameters. It never hears a word. In this data the machines were not the steady ones: their reply timing varied <i>more</i> than the humans&rsquo;.',
      points: [
        'Response latency after the agent stops, and its spread, not just its mean',
        'What happens when the agent barges in: does the caller stop, and how fast',
        'Who speaks after a long silence',
      ],
    },
    {
      diagram: 'sem' as Diagram,
      what: 'SIGNAL 03 — SEMANTIC',
      title: 'How the answer is spoken',
      text: 'The caller&rsquo;s words, transcribed with a confidence per word, plus a rubric an LLM fills in from the transcript. The traps the agent plants turned out to be weak evidence. What carries this layer is how a caller talks: a clean synthetic voice is transcribed with unnatural certainty, and its answers arrive complete, formal and without a single filler.',
      points: [
        'Transcription confidence per word: the strongest family of features',
        'Fillers, false starts and colloquialisms a person produces and a pipeline does not',
        'Over-completeness and formal register, the two rubric dimensions that survived validation',
      ],
    },
  ],
}

export const demo = {
  heading: 'Two real calls from the held-out set.',
  lede: 'Both callers reach Marina, the same agent, with the same script: a digit read back wrong, an interruption, a question about a product they never mentioned. One caller is a volunteer; the other is a text-to-speech pipeline. The bars are the scores the three layers actually produced. The verdict is the fused one.',
  switchLabel: 'Choose caller',
  buttons: { human: 'Real caller', synthetic: 'Cloned voice' },
}

export const pipeline = {
  heading: 'Between the audio arriving and the verdict leaving.',
  lede: 'Two stages. The acoustic and behaviour layers score every call, half and half, in about a second on a CPU. If together they reach 80% confidence, that is the verdict. If not, the semantic service is asked, and its vote is added at 0.15.',
  steps: [
    {
      label: 'STEP 1',
      title: 'Split and segment',
      text: 'Channel&nbsp;0 is the caller, channel&nbsp;1 is the agent. Voice activity is marked on both. The caller&rsquo;s speech is cut into chunks of at most 4&nbsp;s, level-normalised and resampled to 16&nbsp;kHz for the speech model.',
    },
    {
      label: 'STEP 2',
      title: 'Two primaries, in parallel',
      text: 'Acoustic: frozen Wav2Vec2 Spanish, layer 5, a small MLP. Behaviour: 24 timing features from the turns, a logistic regression. Each returns a calibrated probability, or abstains when it has nothing to go on.',
    },
    {
      label: 'STEP 3',
      title: 'Gate at 80%',
      text: 'The two are averaged 50/50. If the confidence in the result is at least 0.80, the answer leaves now. On the held-out set that is 66 calls of 71.',
    },
    {
      label: 'STEP 4',
      title: 'Verifier, only when needed',
      text: 'The rest go to the semantic service: transcription with per-word confidence, text features, an LLM rubric. Its vote is added at 0.15 and the three re-combine. On the held-out set, 5 calls.',
    },
  ],
  timeline: {
    heading: 'A confident call answers in about a second. An escalated one, in about four.',
    /** Position of the verdict line, as a percent of the axis below (5 s wide). */
    verdictAt: 20,
    markers: [
      { at: 20, label: 'verdict · 1.0 s' },
      { at: 78, label: 'with verifier · 3.9 s' },
    ],
    axis: ['0 s', '2.5 s', '5 s'],
    note: 'The acoustic layer scores a call in about 120&nbsp;ms on a GPU. The behaviour layer needs about 860&nbsp;ms on a CPU, almost all of it voice activity detection. The semantic layer is a paid transcription and a paid LLM call, about 2.6&nbsp;s of network, so it is spent only on the calls the primaries could not settle. Nothing is streamed: the detector scores a complete call.',
  },
}

export const endpoint = {
  heading: 'One endpoint, exactly as the brief asks.',
  method: 'POST',
  path: '/detect',
  contentType: 'application/json',
  request: { audio: '<base64 stereo WAV, 8 kHz>' },
  status: '→ 200 OK',
  copy: 'Copy',
  copied: 'Copied',
  /** The real answer for call_0847d7417bb1, one of the five held-out escalations (README.md, Results). */
  response: {
    is_synthetic: true,
    confidence: 0.748,
    decisive: true,
    settled_by_primaries: false,
    verifiers_consulted: true,
    details: {
      branches: { acoustic: 1.0, behaviour: 0.435, semantic: 0.952 },
      weights: { acoustic: 0.5, behaviour: 0.5, semantic: 0.15 },
      acoustic_model: 'wav2vec2_spanish',
    },
  },
  specs: [
    {
      label: 'REQUIRED',
      text: '<code>is_synthetic</code> and <code>confidence</code>, the challenge contract. <code>confidence</code> is the probability that the verdict is right. Everything under <code>details</code> is extra: each layer&rsquo;s probability, its weight, and whether it was even asked.',
    },
    {
      label: 'INPUT',
      text: 'Stereo WAV, 8&nbsp;kHz, base64. Channel&nbsp;0 is classified; channel&nbsp;1 is the agent and gives the timing its context. Mono still works: the behaviour layer abstains and its weight goes to the others.',
    },
    {
      label: 'LATENCY',
      text: 'About 1.0&nbsp;s when the primaries settle it, which they did on 66 of 71 held-out calls. About 3.9&nbsp;s when the semantic verifier is consulted. No streaming: it scores a complete call.',
    },
    {
      label: 'CALIBRATION',
      text: 'Every layer returns a calibrated probability: Platt scaling for the acoustic score, fitted under 11 channel conditions; a sigmoid on out-of-fold logits for behaviour; Platt for semantic. So 0.75 means about 75%, not &ldquo;quite sure&rdquo;.',
    },
    {
      label: 'ABSTAINING',
      text: 'A layer with no evidence says so instead of guessing: no audible caller speech, fewer than two turn events, fewer than five transcribed words. Its weight goes to the layers that answered. If none did, the answer is <code>false</code> at 0.5, flagged <code>decisive: false</code>, which is not a vote for human.',
    },
  ],
}

export const closing = {
  heading: 'The people with the most to lose are the ones who bank by phone because they cannot bank any other way.',
  lede: 'They do not have the app. They will not be enrolled in a voiceprint. If the line stops being trustworthy they do not move to another channel — they just stop being served. That is the reason to build this, and the reason it has to work on an old handset over a bad connection.',
  footer: ['ISISI — Intelligent System Identifying Synthetic Interactions', 'HackMTY26 · Altur challenge track'],
}
