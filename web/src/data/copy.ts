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
  brandLabel: 'Volver al inicio',
  sub: 'Altur · HackMTY26 · Defender al banco de las voces sintéticas',
  link: { label: 'El endpoint', href: '#api' },
}

export const hero = {
  title: 'La voz dejó de ser una contraseña.',
  sub: 'Bastan unos segundos de audio público para clonar a cualquiera lo bastante bien como para pasar — ante una persona y ante una máquina.',
  tapHint: 'Toca el candado',
  /** Accessible name of the tap zone. */
  tapLabel: 'Romper el candado',
  scrollHint: 'DESLIZA',
  /** Shown under the lock skeleton while the turntable frames decode. */
  loading: 'Cargando el candado',
}

/** The name act, after the lock breaks. Each letter is a button that jumps to its word. */
export const nameAct = {
  hint: 'Desliza para leer el nombre',
}

export type Diagram = 'spec' | 'overlap' | 'sem'

export const signals = {
  heading: 'Tres formas en que una máquina se delata en una llamada telefónica.',
  lede: 'Tres detectores independientes leen la misma llamada y fallan en lugares distintos. Las capas acústica y de conversación deciden cada llamada, mitad y mitad. La capa semántica es la cara, así que solo se le pregunta cuando las dos primeras no resuelven.',
  /** Labels drawn inside the card diagrams. */
  diagramLabels: {
    overlap: { bargeIn: 'INTERRUPCIÓN' },
    sem: { asked: 'PREGUNTA', missing: 'SIN TITUBEO' },
  },
  cards: [
    {
      diagram: 'spec' as Diagram,
      what: 'SEÑAL 01 — ACÚSTICA',
      title: 'De qué está hecha la voz',
      text: 'Solo el canal del llamante, solo donde un detector de actividad de voz escucha habla, cortado en fragmentos de 4&nbsp;s como máximo. Cada fragmento pasa por un modelo de voz en español congelado, Wav2Vec2 preentrenado con VoxPopuli, y un clasificador pequeño lee su quinta capa oculta. Nada del backbone se entrenó con estos datos.',
      points: [
        'Los puntajes de cada fragmento se promedian en log-odds, así que el veredicto en curso se puede leer en cualquier punto de la llamada',
        'Elegido en una prueba de estrés, no en la partición limpia: 10 perturbaciones de canal, AUC media 0.9998',
        'Un segundo modelo, endurecido con códecs telefónicos reales, está a una bandera de distancia',
      ],
    },
    {
      diagram: 'overlap' as Diagram,
      what: 'SEÑAL 02 — CONVERSACIÓN',
      title: 'Cuándo el llamante habla, cede y responde',
      text: 'Ambos canales pasan por un detector de actividad de voz; turnos, pausas y traslapes se vuelven 24 características de tiempo, evaluadas por una regresión logística de 40 parámetros. Nunca escucha una sola palabra. En estos datos las máquinas no fueron las estables: sus tiempos de respuesta variaron <i>más</i> que los de las personas.',
      points: [
        'La latencia de respuesta después de que el agente se calla, y su dispersión, no solo su media',
        'Qué pasa cuando el agente interrumpe: si el llamante se detiene, y qué tan rápido',
        'Quién habla después de un silencio largo',
      ],
    },
    {
      diagram: 'sem' as Diagram,
      what: 'SEÑAL 03 — SEMÁNTICA',
      title: 'Cómo se dice la respuesta',
      text: 'Las palabras del llamante, transcritas con una confianza por palabra, más una rúbrica que un LLM llena a partir de la transcripción. Las trampas que planta el agente resultaron ser evidencia débil. Lo que sostiene esta capa es cómo habla un llamante: una voz sintética limpia se transcribe con una certeza poco natural, y sus respuestas llegan completas, formales y sin una sola muletilla.',
      points: [
        'La confianza de transcripción por palabra: la familia de características más fuerte',
        'Muletillas, arranques en falso y coloquialismos que una persona produce y un pipeline no',
        'El exceso de completitud y el registro formal, las dos dimensiones de la rúbrica que sobrevivieron a la validación',
      ],
    },
  ],
}

export const demo = {
  heading: 'Dos llamadas reales del conjunto retenido.',
  lede: 'Los dos llamantes hablan con Marina, la misma agente, con el mismo guion: un dígito leído mal, una interrupción, una pregunta sobre un producto que nunca mencionaron. Un llamante es una persona voluntaria; el otro es un pipeline de texto a voz. Las barras son los puntajes que de verdad produjeron las tres capas. El veredicto es el fusionado.',
  switchLabel: 'Elegir llamante',
  buttons: { human: 'Llamante real', synthetic: 'Voz clonada' },
}

export const pipeline = {
  heading: 'Entre que llega el audio y sale el veredicto.',
  lede: 'Dos etapas. Las capas acústica y de conversación evalúan cada llamada, mitad y mitad, en cerca de un segundo en CPU. Si juntas alcanzan 80% de confianza, ese es el veredicto. Si no, se consulta el servicio semántico y su voto se suma con 0.15.',
  steps: [
    {
      label: 'PASO 1',
      title: 'Separar y segmentar',
      text: 'El canal&nbsp;0 es el llamante, el canal&nbsp;1 es el agente. Se marca la actividad de voz en ambos. La voz del llamante se corta en fragmentos de 4&nbsp;s como máximo, se normaliza de nivel y se remuestrea a 16&nbsp;kHz para el modelo de voz.',
    },
    {
      label: 'PASO 2',
      title: 'Dos primarias, en paralelo',
      text: 'Acústica: Wav2Vec2 español congelado, capa 5, un MLP pequeño. Conversación: 24 características de tiempo sacadas de los turnos, una regresión logística. Cada una devuelve una probabilidad calibrada, o se abstiene cuando no tiene nada en qué apoyarse.',
    },
    {
      label: 'PASO 3',
      title: 'Umbral en 80%',
      text: 'Las dos se promedian 50/50. Si la confianza en el resultado es de al menos 0.80, la respuesta sale ya. En el conjunto retenido eso son 66 llamadas de 71.',
    },
    {
      label: 'PASO 4',
      title: 'Verificador, solo cuando hace falta',
      text: 'El resto va al servicio semántico: transcripción con confianza por palabra, características de texto, una rúbrica de LLM. Su voto se suma con 0.15 y las tres se recombinan. En el conjunto retenido, 5 llamadas.',
    },
  ],
  timeline: {
    heading: 'Una llamada resuelta con confianza responde en cerca de un segundo. Una escalada, en cerca de cuatro.',
    /** Position of the verdict line, as a percent of the axis below (5 s wide). */
    verdictAt: 20,
    markers: [
      { at: 20, label: 'veredicto · 1.0 s' },
      { at: 78, label: 'con verificador · 3.9 s' },
    ],
    axis: ['0 s', '2.5 s', '5 s'],
    note: 'La capa acústica evalúa una llamada en cerca de 120&nbsp;ms en GPU. La capa de conversación necesita cerca de 860&nbsp;ms en CPU, y casi todo se va en la detección de actividad de voz. La capa semántica es una transcripción de pago y una llamada de pago a un LLM, cerca de 2.6&nbsp;s de red, así que solo se gasta en las llamadas que las primarias no pudieron resolver. Nada va en streaming: el detector evalúa una llamada completa.',
  },
}

export const endpoint = {
  heading: 'Un solo endpoint, exactamente como lo pide el reto.',
  method: 'POST',
  path: '/detect',
  contentType: 'application/json',
  request: { audio: '<base64 stereo WAV, 8 kHz>' },
  status: '→ 200 OK',
  copy: 'Copiar',
  copied: 'Copiado',
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
      label: 'OBLIGATORIO',
      text: '<code>is_synthetic</code> y <code>confidence</code>, el contrato del reto. <code>confidence</code> es la probabilidad de que el veredicto sea correcto. Todo lo que va bajo <code>details</code> es extra: la probabilidad de cada capa, su peso y si siquiera se le preguntó.',
    },
    {
      label: 'ENTRADA',
      text: 'WAV estéreo, 8&nbsp;kHz, base64. Se clasifica el canal&nbsp;0; el canal&nbsp;1 es el agente y le da contexto a los tiempos. Mono también funciona: la capa de conversación se abstiene y su peso pasa a las demás.',
    },
    {
      label: 'LATENCIA',
      text: 'Cerca de 1.0&nbsp;s cuando las primarias lo resuelven, que fue en 66 de las 71 llamadas retenidas. Cerca de 3.9&nbsp;s cuando se consulta al verificador semántico. Sin streaming: evalúa una llamada completa.',
    },
    {
      label: 'CALIBRACIÓN',
      text: 'Cada capa devuelve una probabilidad calibrada: escalamiento de Platt para el puntaje acústico, ajustado bajo 11 condiciones de canal; una sigmoide sobre logits fuera de pliegue para conversación; Platt para semántica. Así que 0.75 significa cerca de 75%, no &ldquo;bastante seguro&rdquo;.',
    },
    {
      label: 'ABSTENCIÓN',
      text: 'Una capa sin evidencia lo dice en vez de adivinar: no hay voz audible del llamante, hay menos de dos eventos de turno, hay menos de cinco palabras transcritas. Su peso pasa a las capas que sí respondieron. Si ninguna respondió, la respuesta es <code>false</code> con 0.5, marcada <code>decisive: false</code>, y eso no es un voto por Human.',
    },
  ],
}

export const closing = {
  heading: 'Quienes más tienen que perder son las personas que usan el banco por teléfono porque no pueden usarlo de otra forma.',
  lede: 'No tienen la app. No las van a inscribir en una huella de voz. Si la línea deja de ser confiable no se cambian a otro canal: simplemente dejan de ser atendidas. Esa es la razón para construir esto, y la razón por la que tiene que funcionar en un teléfono viejo con una mala conexión.',
  footer: ['ISISI — un sistema inteligente que identifica interacciones Synthetic', 'HackMTY26 · reto Altur'],
}
