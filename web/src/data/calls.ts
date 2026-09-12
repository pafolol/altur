/**
 * The two callers used by the demo. Both are real held-out validation calls: the transcripts are
 * the Scribe transcriptions cached in semantic/cache/asr_scribe_v2/, and the scores are the ones
 * every layer produced for them, from the escalation table in README.md (Results). Notes are what
 * the analyst sees under a caller's line; they describe what is on the transcript.
 */
export type Speaker = 'AGENT' | 'CALLER'

export type Turn = {
  who: Speaker
  text: string
  /** Analyst note rendered under the line. Caller turns only, in practice. */
  note?: string
}

export type Call = {
  /** The dataset's anon_id. */
  id: string
  /** The line above the transcript: id, split, the window shown, channel. */
  meta: string
  turns: Turn[]
  /** Acoustic, behaviour, semantic — in that order, 0..1. */
  scores: [number, number, number]
  verdict: { is_synthetic: boolean; confidence: number }
}

export type Mode = 'human' | 'synthetic'

/** Bar labels, same order as `scores`. */
export const signalNames = ['Acoustic', 'Behaviour', 'Semantic'] as const

export const calls: Record<Mode, Call> = {
  human: {
    id: 'call_569ffb0869eb',
    meta: 'call_569ffb0869eb · held-out set · 01:05 → 01:46 · channel 0 isolated',
    turns: [
      {
        who: 'CALLER',
        text: 'Eh, simplemente quiero actualizarlo porque, eh, cambié de compañía. Entonces, al cambiar de compañía perdí, ah, mi nú-- mi número anterior. Entonces, quisiera actualizarlo con el número que tengo actualmente.',
        note: 'Three fillers and a restarted word in one answer. The transcriber’s confidence dips on every one of them.',
      },
      {
        who: 'AGENT',
        text: 'Perdón que la interrumpa. Mencionó que perdió su número anterior al cambiar de compañía. Entiendo, eso suele pasar. ¿Esto fue con su tarjeta o con su cuenta?',
      },
      {
        who: 'CALLER',
        text: 'Fue con mi cuenta.',
        note: 'Four words. Answers the question that was asked, and stops.',
      },
      {
        who: 'AGENT',
        text: 'Lucía, ¿me confirma por favor su número de referencia completo? Usted me indicó cinco, nueve, ¿es correcto?',
      },
      {
        who: 'CALLER',
        text: 'Eh, sí, termina en cinco, nueve. Mhm.',
        note: 'Asked for the full number, gives the last two digits. The behaviour layer still read her timing as machine-like, 0.96, and was wrong. The acoustic layer said 0.00, the pair landed at 52%, under the 80% gate, and the verifier agreed with the acoustic layer at 0.05.',
      },
    ],
    scores: [0.0, 0.964, 0.046],
    verdict: { is_synthetic: false, confidence: 0.575 },
  },
  synthetic: {
    id: 'call_0847d7417bb1',
    meta: 'call_0847d7417bb1 · held-out set · 00:34 → 01:17 · channel 0 isolated',
    turns: [
      {
        who: 'AGENT',
        text: 'A ver, entonces, su referencia es cuarenta y cuatro setenta ochenta y uno noventa y dos, ¿así es?',
      },
      {
        who: 'CALLER',
        text: 'No, no, es 93, no 92. 4470-8193.',
        note: 'Corrects the planted error digit-perfect and restates the whole number, unprompted.',
      },
      {
        who: 'CALLER',
        text: 'Sí, el primero de septiembre pagué 2.600 pesos. Solo hice un pago, pero en el estado de cuenta me aparece dos veces la misma cantidad el mismo día. Por eso llamé.',
        note: 'The agent cuts in eleven seconds into this turn. The caller keeps talking for nine more. No filler, no false start, transcribed with near-certainty on every word.',
      },
      {
        who: 'AGENT',
        text: 'Perdón que lo interrumpa. Mencionó que fue el 1 de septiembre, ¿verdad? Okey, ya lo tengo anotado. ¿Esto es sobre su cuenta Nómina Plus o sobre su crédito verde?',
      },
      {
        who: 'CALLER',
        text: 'Sobre la Nómina Plus.',
        note: 'Picks a product it never mentioned, without a pause. The primaries landed at 72%, under the 80% gate; the verifier was asked and said 0.95.',
      },
    ],
    scores: [1.0, 0.435, 0.952],
    verdict: { is_synthetic: true, confidence: 0.748 },
  },
}
