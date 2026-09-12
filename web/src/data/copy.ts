/**
 * Every line of copy on the page except the transcripts (calls.ts) and the
 * acronym (acronym.ts). Headings and labels are plain text. Paragraph-level
 * fields (`lede`, `text`, `note`, list items) may carry inline HTML —
 * &nbsp;, <i>, <code> — exactly as in the prototype.
 * The <title> and <meta name="description"> live in /index.html.
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
  sub: 'Six seconds of public audio is enough to clone anyone well enough to pass — for a human listener, and for a machine.',
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
  lede: 'The brief says depth beats breadth, so each of these is built to stand alone and be argued for. They are fused at the end, but any one of them can carry a verdict when the other two are unsure.',
  /** Labels drawn inside the card diagrams. */
  diagramLabels: {
    overlap: { bargeIn: 'BARGE-IN' },
    sem: { asked: 'ASKED', missing: 'DOES NOT EXIST' },
  },
  cards: [
    {
      diagram: 'spec' as Diagram,
      what: 'SIGNAL 01 — ACOUSTIC',
      title: 'What the codec cannot hide',
      text: 'An 8&nbsp;kHz line throws away everything above 4&nbsp;kHz, which is exactly where most deepfake detectors do their work. So we look lower: at band ratios a vocoder cannot get right, and at what is missing between the words.',
      points: [
        'Breath, lip noise and room tone in the gaps — synths leave them digitally clean',
        'Jitter and shimmer in sustained vowels, measured per turn',
        'Phase coherence across the 300–3400&nbsp;Hz passband',
      ],
    },
    {
      diagram: 'overlap' as Diagram,
      what: 'SIGNAL 02 — CONVERSATIONAL',
      title: 'Recovery is a fingerprint',
      text: 'Every call in the set has moments where the agent cuts in, goes quiet, or talks over the caller. A person recovers instantly and messily. A pipeline recovers the same way every time, and sameness is the tell.',
      points: [
        'Barge-in latency: how long until the caller yields the floor',
        'Variance of turn-taking gaps, not their mean',
        'Whether a restarted sentence restarts from the top or mid-phrase',
      ],
    },
    {
      diagram: 'sem' as Diagram,
      what: 'SIGNAL 03 — SEMANTIC',
      title: 'The question with no answer',
      text: 'The agent asks the caller to confirm a product that does not exist. A person pushes back — confused, slightly annoyed. A language model, trained to be helpful, fills the hole with something plausible.',
      points: [
        "Trap questions planted in the agent's own turns",
        'Confabulation on details never mentioned in the call',
        'Fluent recital of digits a real customer would fumble',
      ],
    },
  ],
}

export const demo = {
  heading: 'The same trap, asked twice.',
  lede: 'One exchange from the middle of a collections call. The agent asks about the <i>Platino&nbsp;Plus</i> card, a product that does not exist. Switch the caller and watch the three signals move.',
  meta: 'Call 0412 · 00:47 → 01:02 · channel 0 isolated',
  switchLabel: 'Choose caller',
  buttons: { human: 'Real caller', synthetic: 'Cloned voice' },
}

export const pipeline = {
  heading: 'Between the audio arriving and the verdict leaving.',
  lede: 'The stereo file is split at the door. Channel&nbsp;1 — the agent — is never classified; it is used to find the moments worth looking at on channel&nbsp;0, which is the whole reason both sides are in the file.',
  steps: [
    {
      label: 'STEP 1',
      title: 'Split and align',
      text: 'Channels separated, voice activity marked on both, turn boundaries built from the overlap between them.',
    },
    {
      label: 'STEP 2',
      title: 'Find the pressure points',
      text: "The agent's track shows where it interrupted, went silent, or asked a trap question. Those windows get the attention.",
    },
    {
      label: 'STEP 3',
      title: 'Score in parallel',
      text: 'The three heads run at the same time, each returning a probability and a reliability weight.',
    },
    {
      label: 'STEP 4',
      title: 'Fuse and commit',
      text: 'A calibrated layer weights each head by how much evidence it actually saw, then emits a confidence that means something.',
    },
  ],
  timeline: {
    heading: 'Confidence crosses the commit threshold before the call is five seconds old.',
    /** Position of the verdict line, as a percent of the axis below. */
    verdictAt: 34, // TODO: replace with real eval numbers
    markers: [
      { at: 12, label: 'first caller turn' },
      { at: 34, label: 'verdict · 4.2 s' }, // TODO: replace with real eval numbers
      { at: 60, label: 'trap answered' },
    ],
    axis: ['0 s', '6 s', '12 s'],
    note: 'Acoustic scoring needs roughly 1.8&nbsp;s of caller speech. The conversational head needs one interruption, which the agent can provoke deliberately if the score is still ambiguous. The semantic head is the slowest and the most certain, so it is allowed to overturn an early verdict rather than delay one.',
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
  response: {
    is_synthetic: true,
    confidence: 0.87, // TODO: replace with real eval numbers
    evidence: {
      acoustic: 0.71, // TODO: replace with real eval numbers
      conversational: 0.83, // TODO: replace with real eval numbers
      semantic: 0.95, // TODO: replace with real eval numbers
      decided_at_s: 4.2, // TODO: replace with real eval numbers
    },
  },
  specs: [
    {
      label: 'REQUIRED',
      text: '<code>is_synthetic</code> and <code>confidence</code>. Everything under <code>evidence</code> is extra — it is there so a fraud analyst can see which signal made the call.',
    },
    {
      label: 'INPUT',
      text: 'Stereo WAV, 8 kHz, base64. Channel 0 is classified, channel 1 is context. Mono still works; the conversational head abstains and the fusion re-weights.',
    },
    {
      label: 'LATENCY',
      // TODO: replace with real eval numbers
      text: 'Median 310 ms on a 15-second clip, CPU only. Streaming mode emits a running verdict every 500 ms.',
    },
    {
      label: 'CALIBRATION',
      text: 'Confidence is isotonic-fitted on held-out calls, so 0.87 means right about 87% of the time at that score — not just "quite sure".',
    },
    {
      label: 'ABSTAINING',
      text: 'Under 1.2 s of caller speech it returns the prior rather than guessing. A wrong confident answer costs a bank more than a slow one.',
    },
  ],
}

export const closing = {
  heading: 'The people with the most to lose are the ones who bank by phone because they cannot bank any other way.',
  lede: 'They do not have the app. They will not be enrolled in a voiceprint. If the line stops being trustworthy they do not move to another channel — they just stop being served. That is the reason to build this, and the reason it has to work on an old handset over a bad connection.',
  footer: ['ISISI — Intelligent System Identifying Synthetic Interactions', 'HackMTY26 · Altur challenge track'],
}
