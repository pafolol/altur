/**
 * Every line of copy on the page except the transcripts (calls.ts) and the
 * acronym (acronym.ts). Headings and labels are plain text. Paragraph-level
 * fields (`lede`, `text`, `note`, list items) may carry inline HTML —
 * &nbsp;, <i>, <b>, <code> — exactly as in the prototype.
 * The <title> and <meta name="description"> live in /index.html.
 *
 * VOICE: this sells a bank-security product, so it leads with the stake and the claim, not with
 * the architecture. Short lines. No jargon the buyer does not need. The technical detail that
 * survived is the detail that IS the sales argument.
 *
 * BUT EVERY NUMBER IS REAL. Nothing on this page is rounded in our favour or invented for effect:
 * README.md (Results), outputs/robust/evaluation_matrix.csv (the codec table and the chart),
 * outputs/fusion/latency_benchmark.json (the timings), reports/TELEPHONE_ROBUSTNESS_REPORT.md,
 * behaviour/reports/, semantic/AUDIT.md. Where a claim has a limit, the limit is on the page.
 */

export const nav = {
  brand: 'ISISI',
  /** Accessible name of the logo link, which goes back to the top. */
  brandLabel: 'Volver al inicio',
  sub: 'Seguridad de voz para la línea telefónica de un banco',
  link: { label: 'El endpoint', href: '#api' },
  /** Always visible, lock broken or not: the demo is the point of the page. */
  demo: { label: 'Ver la demo', href: '/demo/' },
}

export const hero = {
  title: 'La voz dejó de ser una contraseña.',
  sub: 'Clonar a tu cliente cuesta menos que un café y toma menos de un minuto. Tu centro de contacto es la puerta principal del banco — y hoy cualquiera puede tocar con la voz de quien quiera.',
  tapHint: 'Toca el candado',
  /** Accessible name of the tap zone. */
  tapLabel: 'Romper el candado',
  scrollHint: 'DESLIZA',
  /** Shown under the lock skeleton while the turntable frames decode. */
  loading: 'Cargando el candado',
  /** The primary call to action, under the sub, above the lock. */
  cta: { label: 'Pruébalo con tu voz', href: '/demo/' },
}

/** The name act, after the lock breaks. Each letter is a button that jumps to its word. */
export const nameAct = {
  hint: 'Desliza para leer el nombre',
}

export type Diagram = 'spec' | 'overlap' | 'sem'

export const signals = {
  heading: 'Una voz clonada no se delata en un solo lugar.',
  lede: 'Se delata en tres. Y quien construye un detector solo se rompe el día que cambia una cosa: el códec, el ruido, el motor de voz. Nosotros pusimos tres detectores que no comparten ni una suposición. Dos deciden cada llamada dentro de tu casa. Al tercero, que cuesta dinero, solo se le pregunta cuando de verdad hace falta.',
  /** Labels drawn inside the card diagrams. */
  diagramLabels: {
    overlap: { bargeIn: 'INTERRUPCIÓN' },
    sem: { asked: 'PREGUNTA', missing: 'SIN TITUBEO' },
  },
  cards: [
    {
      diagram: 'spec' as Diagram,
      what: 'SEÑAL 01 — LA VOZ',
      title: 'De qué está hecha',
      text: 'Una voz sintética deja huellas en el material del sonido, no en lo que dice. Nuestro detector escucha solo al llamante, solo donde hay habla. Y aquí está lo que nadie más hizo: <b>lo entrenamos dentro del teléfono</b>, no antes de él. 29 horas de llamadas pasadas por líneas con códec, pérdida de paquetes, deriva de reloj y control automático de ganancia.',
      points: [
        'Aprendió con 5 familias de códec y le escondimos otras 8',
        'En esas 8 que nunca escuchó: <b>100% de aciertos</b>',
        'El mismo modelo sin ese entrenamiento: 81.7%',
      ],
    },
    {
      diagram: 'overlap' as Diagram,
      what: 'SEÑAL 02 — EL RITMO',
      title: 'Cómo se comporta',
      text: 'Una conversación tiene una coreografía: cuándo respondes, cuándo te callas, cuándo te lanzas encima del otro. Esta capa no escucha <i>ni una sola palabra</i> — mide el baile. Por eso le da exactamente igual el códec, el idioma o el acento. Y por eso sigue viva cuando el audio ya se degradó demasiado para todo lo demás.',
      points: [
        '97.2% de aciertos ella sola, en CPU, sin mandar audio a ningún lado',
        'Lee el silencio: cuánto tarda en contestar, y si esa tardanza es siempre igual',
        'Sorpresa del proyecto: las máquinas resultaron <i>menos</i> consistentes que las personas',
      ],
    },
    {
      diagram: 'sem' as Diagram,
      what: 'SEÑAL 03 — LAS PALABRAS',
      title: 'Cómo lo dice',
      text: 'Nadie habla con puntos y comas. Una persona duda, se corrige, mete un «este…». Un guion generado contesta completo, formal y perfecto a la primera. Esta capa lee eso — y es la única que sale de tu red y la única que cuesta por llamada, así que la pusimos detrás de una puerta con llave.',
      points: [
        'Solo se le pregunta cuando las otras dos no se ponen de acuerdo',
        'En 71 llamadas reales eso pasó 5 veces: pagamos el 7%',
        'Si el servicio se cae, el sistema sigue decidiendo sin él',
      ],
    },
  ],
}

export const demo = {
  heading: 'Dos llamadas reales. Adivina cuál es la máquina.',
  lede: 'Misma agente, mismo guion, mismo día: un dígito mal leído, una interrupción, una pregunta sobre un producto que nunca mencionaron. Una es una persona. La otra no. Las barras no son un ejemplo bonito — son los puntajes que las tres capas produjeron de verdad sobre estas dos llamadas.',
  switchLabel: 'Elegir llamante',
  buttons: { human: 'Llamante real', synthetic: 'Voz clonada' },
}

/**
 * The differentiator, as a chart. Numbers are call-level accuracy at the deployed threshold,
 * straight out of outputs/robust/evaluation_matrix.csv:
 *   altur_val            specialist 1.0000  robust_v2 1.0000
 *   channel_val_seen     specialist 0.7746  robust_v2 1.0000
 *   channel_val_unseen   specialist 0.8169  robust_v2 1.0000
 *   external_ood_channel specialist 0.6204  robust_v2 0.8148
 * The fifth row of that table (clean studio audio, where ours is worse) is named in `note`.
 */
export const carrier = {
  heading: 'Cambia de carrier un lunes. El modelo ni se entera.',
  lede: 'Todo detector de voz aprende el sonido de la línea por la que lo entrenaste. Cambias de proveedor, cambia el códec, y de pronto tu detector se volvió tonto sin avisarte. Nosotros lo probamos al revés: entrenamos con cinco familias de códec y <b>le escondimos ocho</b> — GSM, AMR, iLBC, Speex, Opus, G.722 — para ver si de verdad aprendió a oír, o solo se había aprendido una línea de memoria.',
  legend: { before: 'Detector entrenado con audio limpio', after: 'ISISI, entrenado dentro del teléfono' },
  rows: [
    { label: 'Llamada limpia', note: 'sin línea de por medio', before: 100.0, after: 100.0 },
    { label: 'Por una línea que sí conoce', note: 'G.711, G.726', before: 77.5, after: 100.0 },
    { label: 'Por 8 códecs que nunca escuchó', note: 'GSM, AMR, iLBC, Speex, Opus, G.722', before: 81.7, after: 100.0 },
    { label: 'Voces clonadas nuevas, por teléfono', note: 'motores que no estaban en el entrenamiento', before: 62.0, after: 81.5 },
  ],
  tableCaption: 'Los mismos datos, en tabla',
  tableCols: { domain: 'Escenario' },
  note: 'Aciertos sobre las llamadas retenidas, medidos al umbral que sale desplegado — no al que nos conviene. <b>La fila que no está en la gráfica:</b> sobre audio de estudio limpio, que no pasó por ningún teléfono, el modelo viejo gana 68.5% contra nuestro 54.6%. Dejó de apoyarse en la pista fácil — el volumen y el borde de banda — y esa pista es justo la que un códec destruye. Preferimos perder ahí, porque una línea telefónica no produce audio de estudio.',
}

export const pipeline = {
  heading: 'No es un demo. Ya marca teléfonos.',
  lede: 'Esto no vive de archivos que alguien sube a mano. El sistema levanta el teléfono, marca, escucha a quien conteste y da un veredicto — de punta a punta, hoy. Y no le pide nada a tu red: cero túneles, cero URL pública, cero puertos abiertos hacia adentro. Tu equipo de seguridad no tiene que aprobar una sola conexión entrante.',
  steps: [
    {
      label: 'PASO 1',
      title: 'Entra por donde ya entras',
      text: 'Una llamada saliente por Twilio, o la grabación que tu centro de contacto ya produce. Si el audio viene incompleto, el sistema lo dice en lugar de inventar un número.',
    },
    {
      label: 'PASO 2',
      title: 'Dos jueces, en paralelo',
      text: 'La voz y el ritmo opinan al mismo tiempo, mitad y mitad, sin salir de tu infraestructura. Cada uno responde con una probabilidad — o admite que no tiene nada en qué apoyarse.',
    },
    {
      label: 'PASO 3',
      title: 'Si están seguros, ya está',
      text: 'Cuando los dos coinciden con 80% de confianza, el veredicto sale ahí mismo, sin que un solo byte de audio haya salido de tu casa. En 71 llamadas reales, eso resolvió 66.',
    },
    {
      label: 'PASO 4',
      title: 'La duda cuesta, y solo se paga cuando toca',
      text: 'Las 5 que quedaron se van al tercer juez, el que lee las palabras. Es el único que sale hacia afuera, y solo lo hace cuando de verdad hay algo que decidir.',
    },
  ],
  timeline: {
    heading: 'Te avisa con la llamada todavía abierta.',
    /** Position of the verdict line, as a percent of the axis below (5 s wide). */
    verdictAt: 20,
    markers: [
      { at: 20, label: 'veredicto · 1.0 s' },
      { at: 78, label: 'con verificador · 3.9 s' },
    ],
    axis: ['0 s', '2.5 s', '5 s'],
    note: 'Mediana de 2.3 segundos de voz del llamante, y <b>133 segundos antes de que la llamada termine</b>. Esa es la diferencia entre un control y una autopsia: da tiempo de pedir una verificación extra, avisar a un supervisor o limitar qué puede ejecutar el ejecutivo, con el cliente todavía en la línea. En las 71 llamadas retenidas, el veredicto temprano coincidió con el final <b>el 100% de las veces</b>.',
  },
}

export const endpoint = {
  heading: 'Una llamada HTTP. Eso es toda la integración.',
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
      label: 'CAMBIA DE CARRIER',
      text: 'Ocho familias de códec que el modelo nunca vio en el entrenamiento, y aun así: <b>100% de aciertos</b>. Cambiar de proveedor es cambiar una línea de configuración, no abrir un proyecto de reentrenamiento de tres meses.',
    },
    {
      label: 'YA ESTÁ CONECTADO',
      text: 'Llamada saliente por Twilio funcionando hoy: marca, graba, evalúa. <b>Sin URL pública ni túnel.</b> Tu red no acepta una sola conexión entrante.',
    },
    {
      label: 'EL AUDIO NO SE VA',
      text: 'Las dos capas que deciden corren en tu infraestructura, y el registro guarda veredictos y tiempos — <b>nunca audio</b>. Solo la tercera sale, y solo en las llamadas dudosas: 5 de 71.',
    },
    {
      label: 'UN NÚMERO QUE SIGNIFICA ALGO',
      text: '0.75 quiere decir cerca de 75%, no «bastante seguro». Está calibrado bajo 11 condiciones de canal, así que tu umbral de bloqueo se vuelve una decisión de negocio con un costo calculable.',
    },
    {
      label: 'SABE CUÁNDO CALLARSE',
      text: 'Sin evidencia, lo dice. No hay voz audible, la llamada no tuvo interacción, no hubo palabras suficientes. <b>El sistema no acusa a un cliente del que no sabe nada</b> — y esa es la falla que le cuesta cara a un banco.',
    },
  ],
}

export const closing = {
  heading: 'Lo que puedes verificar, y lo que todavía no.',
  lede: 'Sobre 71 llamadas retenidas no fallamos ni una vez — y 71 llamadas no fijan las tasas de error de un banco: la cota honesta de ese 100% es 95.9%. Contra motores de voz que el modelo nunca había escuchado, por teléfono, la cifra que sostenemos es <b>88.0% de aciertos</b>. Y el teléfono con el que entrenamos está <b>simulado</b>: nuestro G.711 es idéntico bit a bit al de referencia en los 65,536 valores posibles, pero ninguna llamada de entrenamiento pasó por un carrier real. La integración telefónica sí es real, y el siguiente paso obvio es medir en tu línea antes de fijar un umbral. Preferimos decírtelo aquí y no en la junta de después del incidente.',
  footer: ['ISISI — un sistema inteligente que identifica interacciones Synthetic', 'HackMTY26 · reto Altur'],
}
