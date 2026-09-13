/**
 * Every line of copy on the page except the transcripts (calls.ts) and the
 * acronym (acronym.ts). Headings and labels are plain text. Paragraph-level
 * fields (`lede`, `text`, `note`, list items) may carry inline HTML —
 * &nbsp;, <i>, <code> — exactly as in the prototype.
 * The <title> and <meta name="description"> live in /index.html.
 *
 * POSITIONING: this page sells a bank-security product. The differentiator is that the
 * telephony integration already exists (outbound Twilio calling, scored end to end) and that
 * the acoustic model was trained across codec families rather than one, so a carrier change
 * is a configuration change and not a retraining project.
 *
 * Every number here is read from the repository's own reports: README.md (Results),
 * reports/TELEPHONE_ROBUSTNESS_REPORT.md (the codec matrix), reports/ACOUSTIC_LEARNING_SUMMARY.md,
 * behaviour/reports/, semantic/AUDIT.md. Nothing on this page is aspirational: where a claim has
 * a limit, the limit is printed next to it.
 */

export const nav = {
  brand: 'ISISI',
  /** Accessible name of the logo link, which goes back to the top. */
  brandLabel: 'Volver al inicio',
  sub: 'Detección de voz sintética para la línea telefónica de un banco',
  link: { label: 'El endpoint', href: '#api' },
}

export const hero = {
  title: 'La voz dejó de ser una contraseña.',
  sub: 'Bastan unos segundos de audio público para clonar a un cliente lo bastante bien como para pasar el filtro de un ejecutivo. El centro de contacto es la puerta de entrada del banco, y hoy es la más barata de forzar.',
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
  lede: 'Un solo detector se rompe cuando cambia una cosa: el códec, el ruido, el motor de voz. Aquí tres detectores independientes leen la misma llamada y fallan en lugares distintos. Las capas acústica y de conversación deciden cada llamada, mitad y mitad, sin salir de tu infraestructura. La capa semántica cuesta dinero por llamada, así que solo se le pregunta cuando las dos primeras no resuelven: 5 de cada 71 llamadas.',
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
      text: 'Solo el canal del llamante, solo donde hay habla, cortado en fragmentos de 4&nbsp;s. Cada fragmento pasa por un modelo de voz en español congelado y un clasificador lee su quinta capa oculta. Lo que lo distingue no es la arquitectura: es <b>con qué se entrenó</b>. El modelo desplegado vio 29 horas de las mismas llamadas después de pasar por líneas telefónicas simuladas, con códec, pérdida de paquetes, deriva de reloj, ruido de comfort y control automático de ganancia.',
      points: [
        'Entrenado con 5 familias de códec; otras 8 se dejaron <b>fuera del entrenamiento a propósito</b>',
        'En esas 8 que nunca vio — GSM, AMR, iLBC, Speex, Opus, G.722 — acierta 100% de las llamadas retenidas',
        'El modelo anterior, entrenado solo con audio limpio, caía a 81.7% en esas mismas líneas',
      ],
    },
    {
      diagram: 'overlap' as Diagram,
      what: 'SEÑAL 02 — CONVERSACIÓN',
      title: 'Cuándo el llamante habla, cede y responde',
      text: 'Ambos canales pasan por un detector de actividad de voz; turnos, pausas y traslapes se vuelven 24 características de tiempo, evaluadas por una regresión logística de 40 parámetros. Nunca escucha una sola palabra, así que no le afecta el códec ni el idioma. En estos datos las máquinas no fueron las estables: sus tiempos de respuesta variaron <i>más</i> que los de las personas.',
      points: [
        'La latencia de respuesta después de que el agente se calla, y su dispersión, no solo su media',
        'Qué pasa cuando el agente interrumpe: si el llamante se detiene, y qué tan rápido',
        '97.2% de exactitud por sí sola, corriendo en CPU, sin enviar audio a ningún lado',
      ],
    },
    {
      diagram: 'sem' as Diagram,
      what: 'SEÑAL 03 — SEMÁNTICA',
      title: 'Cómo se dice la respuesta',
      text: 'Las palabras del llamante, transcritas con una confianza por palabra, más una rúbrica que un LLM llena a partir de la transcripción. Es la única capa que sale de la máquina y la única que cuesta por llamada, así que está detrás de una compuerta: si las dos primarias ya llegaron a 80% de confianza, nunca se le pregunta y nunca se gasta.',
      points: [
        'La confianza de transcripción por palabra: la familia de características más fuerte',
        'Muletillas, arranques en falso y coloquialismos que una persona produce y un pipeline no',
        'Se consultó en 5 de 71 llamadas retenidas: 7% del gasto de una votación plana',
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
  heading: 'Ya está conectado a una telefonía automatizada.',
  lede: 'No es una demostración sobre archivos subidos a mano. El sistema marca un número por Twilio, graba lo que contesta quien responde, lo convierte al formato que espera el detector y lo evalúa de punta a punta. Y no necesita que tu carrier alcance nuestros servidores: el TwiML va en línea, sin <code>action</code>, y la grabación se recoge consultando la API de Twilio. Cero túneles, cero URL pública, cero puertos abiertos hacia adentro.',
  steps: [
    {
      label: 'PASO 1',
      title: 'Entra por la línea que ya tienes',
      text: 'Llamada saliente por Twilio, o un WAV estéreo de 8&nbsp;kHz desde la grabadora del centro de contacto. El canal&nbsp;0 es el llamante, el canal&nbsp;1 es el agente. Si solo hay mono — como devuelve una grabación de Twilio — el sistema lo dice y la capa de conversación se abstiene en lugar de inventar.',
    },
    {
      label: 'PASO 2',
      title: 'Dos primarias, en paralelo',
      text: 'Acústica: el modelo endurecido para teléfono, capa 5, un MLP pequeño. Conversación: 24 características de tiempo sacadas de los turnos, una regresión logística. Ambas corren localmente. Cada una devuelve una probabilidad calibrada, o se abstiene cuando no tiene nada en qué apoyarse.',
    },
    {
      label: 'PASO 3',
      title: 'Umbral en 80%',
      text: 'Las dos se promedian 50/50. Si la confianza en el resultado es de al menos 0.80, la respuesta sale ya, sin que un solo byte de audio haya salido de tu infraestructura. En el conjunto retenido eso son 66 llamadas de 71.',
    },
    {
      label: 'PASO 4',
      title: 'Verificador, solo cuando hace falta',
      text: 'El resto va al servicio semántico: transcripción con confianza por palabra, características de texto, una rúbrica de LLM. Su voto se suma con 0.15 y las tres se recombinan. En el conjunto retenido, 5 llamadas — y es la única ruta en la que el audio sale.',
    },
  ],
  timeline: {
    heading: 'El veredicto llega con la llamada todavía abierta: mediana de 2.3 s de voz del llamante, 133 s antes de que la llamada termine.',
    /** Position of the verdict line, as a percent of the axis below (5 s wide). */
    verdictAt: 20,
    markers: [
      { at: 20, label: 'veredicto · 1.0 s' },
      { at: 78, label: 'con verificador · 3.9 s' },
    ],
    axis: ['0 s', '2.5 s', '5 s'],
    note: 'Eso es lo que separa un control de una autopsia: da tiempo de escalar a verificación reforzada, avisar a un supervisor o limitar qué puede ejecutar el ejecutivo, con el cliente todavía en la línea. En las 71 llamadas retenidas el veredicto temprano coincidió con el final el 100% de las veces. La capa acústica evalúa una llamada en cerca de 120&nbsp;ms en GPU; la de conversación, cerca de 860&nbsp;ms en CPU. Nada va en streaming: el detector evalúa una llamada completa.',
  },
}

export const endpoint = {
  heading: 'Un solo endpoint. Un proceso. Sin dependencias hacia afuera para decidir.',
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
      label: 'PORTABILIDAD DE CARRIER',
      text: 'Entrenado con G.711 (μ-law y A-law) y G.726. Se dejaron fuera del entrenamiento <b>ocho familias completas</b> — GSM&nbsp;06.10, AMR-NB, iLBC, Speex, Opus y G.722 — justamente para medir si la robustez se generaliza o se memoriza. En esas ocho: <b>100% de exactitud, AUC 1.000</b>. Cambiar de carrier es cambiar una configuración, no reentrenar un modelo.',
    },
    {
      label: 'INTEGRACIÓN',
      text: 'Llamada saliente por Twilio ya implementada: marca, graba, convierte y evalúa. <b>Sin URL pública ni túnel</b> — el TwiML va en línea y la grabación se recoge por la API. Tu red no necesita aceptar una sola conexión entrante.',
    },
    {
      label: 'DÓNDE VIVE EL AUDIO',
      text: 'Las dos capas que deciden corren en tu infraestructura. El registro guarda puntajes, veredictos y tiempos — <b>nunca audio</b>. Solo la capa verificadora sale hacia terceros, y solo en las llamadas que las primarias no resolvieron: 5 de 71.',
    },
    {
      label: 'CALIBRACIÓN',
      text: 'Cada capa devuelve una probabilidad calibrada, no un puntaje: escalamiento de Platt para lo acústico, ajustado bajo 11 condiciones de canal. Así que 0.75 significa cerca de 75%, y el umbral de bloqueo se vuelve una decisión de negocio con un costo calculable, no un número arbitrario.',
    },
    {
      label: 'ABSTENCIÓN',
      text: 'Una capa sin evidencia lo dice en vez de adivinar: no hay voz audible, hay menos de dos eventos de turno, hay menos de cinco palabras. Su peso pasa a las que sí respondieron. Si ninguna respondió, la respuesta es <code>false</code> con 0.5 y <code>decisive: false</code> — el sistema <b>no acusa a un cliente sobre el que no sabe nada</b>.',
    },
  ],
}

export const closing = {
  heading: 'Lo que puedes verificar, y lo que todavía no.',
  lede: 'Sobre 71 llamadas retenidas el sistema no se equivoca ni una vez, pero 71 llamadas no fijan tasas de error bancarias: la cota inferior honesta de ese 100% es 95.9%. Contra motores de voz que nunca vio, pasados por una línea, la cifra que sostenemos es <b>88.0% de exactitud, AUC 0.945</b>. Y el canal telefónico del entrenamiento está <b>simulado, no capturado</b>: nuestro G.711 es idéntico bit a bit a la implementación de referencia en los 65,536 valores posibles, pero ninguna llamada de entrenamiento pasó por un carrier real. La integración con Twilio sí es real, y el siguiente paso es medir en tu línea antes de fijar un umbral. Preferimos decirlo aquí que en la junta posterior al incidente.',
  footer: ['ISISI — un sistema inteligente que identifica interacciones Synthetic', 'HackMTY26 · reto Altur'],
}
