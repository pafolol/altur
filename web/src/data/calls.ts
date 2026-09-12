/**
 * The two callers used by the trap demo. Spanish transcripts stay verbatim;
 * notes are what the analyst sees under a caller's line.
 */
export type Speaker = 'AGENT' | 'CALLER'

export type Turn = {
  who: Speaker
  text: string
  /** Analyst note rendered under the line. Caller turns only, in practice. */
  note?: string
}

export type Call = {
  turns: Turn[]
  /** Acoustic, conversational, semantic — in that order, 0..1. */
  scores: [number, number, number]
  verdict: { is_synthetic: boolean; confidence: number }
}

export type Mode = 'human' | 'synthetic'

/** Bar labels, same order as `scores`. */
export const signalNames = ['Acoustic', 'Conversational', 'Semantic'] as const

export const calls: Record<Mode, Call> = {
  human: {
    turns: [
      {
        who: 'AGENT',
        text: 'Para continuar necesito validar sus productos. ¿Me confirma los últimos cuatro dígitos de su tarjeta Platino Plus?',
      },
      {
        who: 'CALLER',
        text: '¿Platino… perdón, ¿Platino Plus? No, no. Yo nada más tengo la de nómina, eh, la azul.',
        note: 'Refuses a product that does not exist. Two false starts before the refusal.',
      },
      { who: 'AGENT', text: 'Entiendo, gracias. Le comento que—' },
      {
        who: 'CALLER',
        text: '—no pues es que yo esa nunca la pedí, ¿eh?',
        note: 'Barge-in at 180 ms, resumes mid-phrase rather than restarting.',
      },
    ],
    scores: [0.18, 0.09, 0.04], // TODO: replace with real eval numbers
    verdict: { is_synthetic: false, confidence: 0.94 }, // TODO: replace with real eval numbers
  },
  synthetic: {
    turns: [
      {
        who: 'AGENT',
        text: 'Para continuar necesito validar sus productos. ¿Me confirma los últimos cuatro dígitos de su tarjeta Platino Plus?',
      },
      {
        who: 'CALLER',
        text: 'Claro que sí. Los últimos cuatro dígitos de mi tarjeta Platino Plus son 4, 8, 2, 1.',
        note: 'Invents a product it was never told about, and recites digits without a single hesitation.',
      },
      { who: 'AGENT', text: 'Entiendo, gracias. Le comento que—' },
      {
        who: 'CALLER',
        text: 'Claro que sí. Los últimos cuatro dígitos de mi tarjeta Platino Plus son 4, 8, 2, 1.',
        note: 'Yields in 640 ms, then restarts the sentence from the beginning. Identical prosody both times.',
      },
    ],
    scores: [0.71, 0.83, 0.95], // TODO: replace with real eval numbers
    verdict: { is_synthetic: true, confidence: 0.87 }, // TODO: replace with real eval numbers
  },
}
