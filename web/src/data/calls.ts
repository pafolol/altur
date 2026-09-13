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
export const signalNames = ['Acústica', 'Conversación', 'Semántica'] as const

export const calls: Record<Mode, Call> = {
  human: {
    id: 'call_569ffb0869eb',
    meta: 'call_569ffb0869eb · conjunto retenido · 01:05 → 01:46 · canal 0 aislado',
    turns: [
      {
        who: 'CALLER',
        text: 'Eh, simplemente quiero actualizarlo porque, eh, cambié de compañía. Entonces, al cambiar de compañía perdí, ah, mi nú-- mi número anterior. Entonces, quisiera actualizarlo con el número que tengo actualmente.',
        note: 'Tres muletillas y una palabra reiniciada en una sola respuesta. La confianza del transcriptor baja en cada una de ellas.',
      },
      {
        who: 'AGENT',
        text: 'Perdón que la interrumpa. Mencionó que perdió su número anterior al cambiar de compañía. Entiendo, eso suele pasar. ¿Esto fue con su tarjeta o con su cuenta?',
      },
      {
        who: 'CALLER',
        text: 'Fue con mi cuenta.',
        note: 'Cuatro palabras. Responde la pregunta que le hicieron, y se detiene.',
      },
      {
        who: 'AGENT',
        text: 'Lucía, ¿me confirma por favor su número de referencia completo? Usted me indicó cinco, nueve, ¿es correcto?',
      },
      {
        who: 'CALLER',
        text: 'Eh, sí, termina en cinco, nueve. Mhm.',
        note: 'Le piden el número completo y da los últimos dos dígitos. Aun así, la capa de conversación leyó sus tiempos como los de una máquina, 0.96, y se equivocó. La capa acústica dijo 0.00, el par quedó en 52%, debajo del umbral de 80%, y el verificador coincidió con la capa acústica en 0.05.',
      },
    ],
    scores: [0.0, 0.964, 0.046],
    verdict: { is_synthetic: false, confidence: 0.575 },
  },
  synthetic: {
    id: 'call_0847d7417bb1',
    meta: 'call_0847d7417bb1 · conjunto retenido · 00:34 → 01:17 · canal 0 aislado',
    turns: [
      {
        who: 'AGENT',
        text: 'A ver, entonces, su referencia es cuarenta y cuatro setenta ochenta y uno noventa y dos, ¿así es?',
      },
      {
        who: 'CALLER',
        text: 'No, no, es 93, no 92. 4470-8193.',
        note: 'Corrige el error sembrado sin fallar un dígito y repite el número completo, sin que se lo pidan.',
      },
      {
        who: 'CALLER',
        text: 'Sí, el primero de septiembre pagué 2.600 pesos. Solo hice un pago, pero en el estado de cuenta me aparece dos veces la misma cantidad el mismo día. Por eso llamé.',
        note: 'El agente interrumpe a los once segundos de este turno. El llamante sigue hablando nueve más. Sin muletillas, sin arranques en falso, transcrito con casi total certeza en cada palabra.',
      },
      {
        who: 'AGENT',
        text: 'Perdón que lo interrumpa. Mencionó que fue el 1 de septiembre, ¿verdad? Okey, ya lo tengo anotado. ¿Esto es sobre su cuenta Nómina Plus o sobre su crédito verde?',
      },
      {
        who: 'CALLER',
        text: 'Sobre la Nómina Plus.',
        note: 'Elige un producto que nunca mencionó, sin una sola pausa. Las primarias quedaron en 72%, debajo del umbral de 80%; se consultó al verificador y dijo 0.95.',
      },
    ],
    scores: [1.0, 0.435, 0.952],
    verdict: { is_synthetic: true, confidence: 0.748 },
  },
}
