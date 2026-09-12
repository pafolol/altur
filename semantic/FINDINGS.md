# Hallazgos medidos sobre el dataset (2026-09-12)

Correcciones a PLAN.md, medidas sobre las 353 llamadas del release v1.0.

## El audio del agente NO es reutilizado (contradice §2.4)

Correlación cruzada de la forma de onda del saludo (`call_acda78859d0d`, 0.34–3.74 s)
contra los primeros 8 s del canal 1 de otras 5 llamadas: **r máx = 0.09–0.18**.
Audio idéntico daría r ≈ 1. El TTS de Marina se regenera en cada llamada.
La r ≥ 0.58 de §2.4 es correlación de espectrogramas: misma voz y texto parecido,
no el mismo audio.

Consecuencia: el template matching exacto (Fase 2) no funciona. Se probó:
- NCC espectral (32 bandas log, CMN): 7/29 anclas correctas en las 6 llamadas del kit.
- DTW de subsecuencia con el saludo (frase fija, 3.4 s): sí lo ubica en 5/5 llamadas,
  pero margen costo 2º-mejor − mejor = 0.06–0.13. Frases cortas/variables saldrían peor.

## El guion del agente NO es fijo palabra por palabra (matiza §2.3)

Transcripciones del kit (6 llamadas): el saludo es idéntico; todo lo demás lo redacta
un LLM y cambia entre llamadas ("me puede proporcionar" / "me puede indicar" /
"necesito que me proporciones"; "seis y ocho dígitos" / "siete o ocho dígitos").
Las trampas no aparecen en todas las llamadas (interrupción en 3/6, producto en 4/6).
Por eso "el texto de la agente ya lo sabes" no aplica para el transcript anotado.

## Decisión pendiente (Fase 2 → Fase 3)

Recomendación: **transcribir también el canal del agente** con Scribe (compactado,
en trozos concurrentes) y ubicar las trampas por búsqueda de texto. Voz por canal
(VAD, media / p95 / máx): cliente 43 / 77 / 131 s; agente 76 / 104 / 155 s.
Alternativa sin red: DTW keyword spotting (margen medido arriba, precisión dudosa).
Decidir midiendo la latencia real de Scribe en la Fase 3.

## Las 6 llamadas del kit están en el dataset

`kit_id` = primeros 6 hex del `anon_id` (`e618b7` → `call_e618b7b1fa5b`). Sus timelines
coinciden exactamente con `turns/`. Sirven como transcripts de referencia timestamp a timestamp.

## VAD (§2.5) confirmado

20 llamadas, 562 fronteras: mediana |err| 50 ms, 98.6 % dentro de ±100 ms, ~10 ms/llamada.
Voz compactada del cliente: 41 s de voz + separadores de 0.5 s ≈ 49 s de audio.

## Latencias reales (medidas 2026-09-12, 1 core, red doméstica)

Scribe (`scribe_v1`, el default de `asr.py`; v2 solo se probó en una medición aislada), audio compactado, trozos concurrentes, 5 llamadas:

| configuración                  | trozos/llamada | media  | máx    |
|--------------------------------|---------------:|-------:|-------:|
| solo cliente, trozos de 10 s   | 6.6            | 1.22 s | 2.16 s |
| solo cliente, trozos de 20 s   | 3.4            | 1.08 s | 1.30 s |
| ambos canales, trozos de 20 s  | 8.0            | 1.24 s | 1.76 s |
| ambos canales, trozos de 30 s  | 5.8            | 1.82 s | 2.56 s |

La latencia por petición es ~1 s casi fija (0.94 s para 3 segmentos cortos); trocear no la baja, solo evita que crezca. Sin 429 con 21 peticiones concurrentes. La respuesta trae `logprob` por palabra.

Gemini (`thinkingLevel: minimal`; con thinking por defecto 3.6-flash tarda 4.5 s), una llamada de 148 s:

| modelo               | texto (~1.1k tokens in) | audio cliente (44 s) | audio ambos canales |
|----------------------|------------------------:|---------------------:|--------------------:|
| gemini-3.6-flash     | 0.93–1.24 s             | 1.7–2.5 s            | 2.3–2.5 s           |
| gemini-3.5-flash-lite| 0.65–0.68 s             | 1.6–1.9 s            | 1.9–2.7 s           |
| gemini-3.1-flash-lite| 0.96 s                  | 1.3 s                | 1.9–2.3 s           |
| gemini-3.8-flash     | no acepta `minimal`     |                      |                     |

Con la rúbrica completa (7 dimensiones + cita) la salida son ~270 tokens y flash-lite sube a 1.2–1.4 s; solo scores (~40 tokens) baja a ~0.6–1.0 s. La salida manda en la latencia.

**Decisión: Opción A** (Scribe → texto → Gemini). Opción B (audio directo) no es más rápida y mezcla lo acústico en un módulo que debe ser semántico. Ambos canales se transcriben (20 s); el agente ya no se conoce de antemano (ver arriba). El servicio degrada por plazo: `f4+f5` → `f4` → abstención.

## Resultados finales offline (val, 71 llamadas, 2026-09-12)

Fase 4, features de texto sin LLM (AUC individual en val): logprob medio 0.88, desv. logprob 0.88, logprob p10 0.86, palabras por turno 0.82, n palabras 0.81, disfluencias 0.77, arranques falsos 0.71, TTR 0.64, mexicanismos 0.62, longitud de oración 0.61. Regresión logística F4 sola: **AUC 0.935**.

Fase 5, rúbrica Gemini (AUC por dimensión en val, gemini-3.5-flash-lite / gemini-3.6-flash):

| dimensión                  | flash-lite | 3.6-flash |
|----------------------------|-----------:|----------:|
| sobrecompletitud           | 0.72       | 0.71      |
| registro_formal            | 0.73       | 0.69      |
| iniciativa                 | 0.59       | 0.55      |
| reparacion_conversacional  | 0.54       | 0.55      |
| consistencia_memoria       | 0.48       | 0.53      |
| fidelidad_repeticion       | 0.45       | 0.47      |
| manejo_producto_ambiguo    | 0.35       | 0.38      |

Solo sobrecompletitud y registro_formal aportan; las trampas A y B no discriminan (confirma §2.3). El modelo grande no mejora y tarda el doble (1.30 s vs 0.69 s). Pedir citas (`evidencia`) baja la AUC de F5 (0.78 vs 0.83) y suma ~0.4 s: **apagado por defecto** (`RUBRIC_EVIDENCE=0`).

Fase 6 (calibración Platt, 5-fold en train):

| config | AUC val | acc val | gap medio de calibración |
|--------|--------:|--------:|-------------------------:|
| F4     | 0.935   | 0.887   | 0.070                    |
| F5     | 0.832   | 0.718   | 0.146                    |
| F4+F5  | **0.955** | 0.873 | 0.106                    |

F4+F5 gana en AUC por 0.02; F4 sola es más precisa en accuracy y mejor calibrada. El servicio usa F4+F5 y cae a F4 si Gemini no llega a tiempo, así que el backend recibe siempre un score calibrado del mismo módulo.

## Servicio por HTTP real (check_server.py, 71 de val, peticiones en serie, 2 workers)

| métrica                          | valor |
|----------------------------------|------:|
| latencia pared p50               | 1.96 s |
| latencia pared p95               | 2.24 s |
| latencia pared máx               | 2.24 s |
| camino f4+f5 / f4 / abstención   | 51 / 20 / 0 |
| AUC de los scores devueltos      | 0.937 |
| llamada más larga (273 s, 11.7 MB base64) | 200, camino f4 |

Dos causas de abstención que se corrigieron:
- **Cola de latencia de Scribe**: p50 ≈ 1.0 s pero ~8 % de peticiones tardan 1.7–2.5 s sin relación con el largo del audio. Solución: *hedging*, toda petición pendiente a los 1.1 s se duplica y gana la primera (`asr.HEDGE_S`). Cuesta ~15 % más de audio facturado.
- **Carrera de plazos**: el `wait_for` exterior y el timeout interior de Gemini vencían al mismo instante y ganaba el exterior (abstención en vez de degradar a F4). Ahora el plazo de red es 2.2 s y el corte exterior 2.4 s solo atrapa errores.

El 28 % de llamadas cae a F4 porque Gemini no llega dentro del plazo. Si el backend puede dar 3.0 s, casi todas irían por f4+f5 (AUC 0.955 vs 0.937).

## Presupuesto subido a 2.8 s (plazo de red) / 3.0 s (corte exterior), 2026-09-12

71 de val por HTTP: p50 2.08 s, p95 2.83 s, máx 2.84 s. Camino f4+f5 65 / f4 6 / abstención 0. **AUC 0.961.** La llamada más larga ya va por f4+f5.

## Configuración servida desde 2026-09-12 (ver AUDIT.md)

Se quitaron `words_per_turn` y `n_words` (proxies de estadísticas de turnos, PLAN §1.1, r = 0.94 con la
longitud media de turno) y las 5 dimensiones de rúbrica con AUC < 0.6. Modelo servido: 8 features de texto
+ sobrecompletitud + registro_formal. Val: **AUC 0.921, acc 0.887, gap de calibración 0.053**. Por HTTP:
AUC 0.907, 0 abstenciones. El 0.955 anterior contaba señal del módulo de comportamiento.

## Corrección documental (2026-09-12, /autoplan)

La caché de las 353 llamadas, `model.pkl` y el servidor usan **`scribe_v1`** (default de `asr.MODEL`; `.env` no fija
`SCRIBE_MODEL`). Las menciones a "Scribe v2" en versiones anteriores de este archivo y de la guía eran incorrectas:
v2 solo se ejecutó en el proceso de medición inicial. Entrenamiento e inferencia son consistentes entre sí. El
servidor ahora guarda `scribe_model` en `model.pkl` y se niega a arrancar si el env difiere. Probar v2 es un TODO
(353 transcripciones + rúbrica + reentreno).

## Prueba con TTS ajeno (Apple `say`, 10 llamantes, ver AUDIT §7)

Guion bot-like: 0.92–0.96. Guion human-like con muletillas: 0.28–0.79 (media 0.58). El módulo mide contenido y
estilo, y un bot coloquial lo engaña a medias. `samples/call.wav` sale de esta prueba (Paulina + Mónica), no del dataset.

## Concurrencia: el techo real es la suscripción de ElevenLabs (2026-09-12, tras /autoplan)

Ráfagas directas a Scribe con trozos reales de 20 s: 16 concurrentes → 16×200; 32 → 20×200 + 12×429
`concurrent_limit_exceeded`; 64 → 40×200 + 24×429. **La llave admite ~20 peticiones simultáneas.** Una llamada
completa son ~8 trozos (dos canales), así que el servicio soporta **2-3 llamadas simultáneas a calidad completa**;
por encima, degrada (f4, abstención con `reason`) en vez de fallar.

Cambios: semáforo por proceso (`SCRIBE_MAX_INFLIGHT`, default 9; con `--workers N` mantener N×9 < 20), un reintento
en 429, el error real de upstream llega al campo `reason`, y **el canal del agente cede el turno** (espera ≤ 0.6 s por
un slot; si no hay, se salta la rúbrica → `f4` con `reason=degraded:Contended`) para que los trozos del cliente de
otras llamadas nunca hagan cola detrás de él.

| `check_server.py` | p50 | p95 | f4+f5 / f4 / abstain | AUC |
|---|--:|--:|---|--:|
| serie | 2.10 s | 2.81 s | 65 / 6 / 0 | 0.944 |
| `--parallel 3` | 1.99 s | 2.74 s | 68 / 3 / 0 | 0.922 |
| `--parallel 8` (antes del semáforo) | 0.95 s | 1.67 s | 3 / 14 / 54 (429) | 0.662 |
| `--parallel 8` (semáforo 18 × 2 workers) | 2.27 s | 2.83 s | 24 / 35 / 12 | 0.831 |
| `--parallel 8` (semáforo 9 × 2, sin prioridad) | 2.83 s | 2.91 s | 0 / 11 / 60 (timeout) | 0.597 |
| **`--parallel 3` (final: 9 × 2 + agente cede)** | 1.80 s | 2.36 s | 47 / 24 / 0 | **0.949** |
| **`--parallel 8` (final)** | 1.73 s | 2.33 s | 5 / 65 / 1 | **0.897** |

Lección: el semáforo solo, sin prioridad, es peor que el 429 (todo espera y todo vence). Con prioridad al cliente,
el servicio degrada a texto bajo carga y nunca se cae. El techo de calidad completa lo pone la suscripción, no el código:
si el benchmark corre en paralelo, subir el plan de ElevenLabs o pedir a los organizadores que corran en serie.

## Más precisión: regla de medición, features extra de logprob, scribe_v2 (2026-09-12)

Con 71 llamadas de val la AUC tiene ±0.04 de ruido y val está balanceada 50/50 (más fácil que train). Nueva regla
para comparar experimentos: `model.py cv`, 5-fold estratificado × 10 repeticiones sobre las 353 llamadas, mismas
particiones para todo (`random_state=0`), comparación pareada fold a fold.

| experimento | F4 (CV) | F4+F5 (CV) | folds a favor (de 50) |
|---|--:|--:|--:|
| 8 features + 2 dims (servido hasta hoy) | 0.874 ± 0.039 | 0.893 ± 0.036 | — |
| + `logprob_min`, `logprob_low_frac`, `logprob_first30` | 0.892 ± 0.035 | **0.904 ± 0.034** | 43 |
| lo mismo con `scribe_v2` | 0.891 | 0.896 | 20 |

- **Features extra de logprob** (`features.EXTRA`): fracción de palabras con logprob < −0.05 (AUC sola 0.79),
  logprob medio de los primeros 30 s (0.79, sigue a AUDIT §3) y mínimo (0.68). +0.011 de AUC, consistente
  (43/50 folds). Se quedan; `SEMANTIC_EXTRA=0` reproduce el modelo de 8. En val (train→val) F4+F5 pasa de 0.921 a 0.934.
- **scribe_v2**: devuelve `logprob` por palabra, pero no separa mejor (p10 sube 0.846→0.860, el resto baja) y tarda
  3.04 s por llamada (dos canales, media) contra 2.19 s de v1; p95 4.08 s vs 3.13 s. Rechazado; caché en
  `cache/asr_scribe_v2` y `cache/gemini/*_scribe_v2.json` por si se retoma. El caché de ASR ahora se llama por modelo.
- **Modelo final**: `model.pkl` se ajusta con las 353 llamadas (antes 282); `scores_semantico.csv` es fuera de fold
  para todas las filas (AUC OOF 0.911). La tabla train→val de `model.py` y `model.py cv` siguen siendo la medición;
  la AUC que reporte `check_server.py` sobre val a partir de ahora es optimista porque val ya está en el ajuste.
- **Lo que no cambia**: los errores siguen siendo bots con voz sucia y humanos que hablan como asistente
  (AUDIT §4); la ganancia grande sigue estando en la fusión con acústica y comportamiento.
- **Presupuesto 3.0 s** (`BUDGET_S=3.0 HARD_S=3.2`, 71 de val en serie, modelo nuevo): 68 / 3 / 0 contra 65 / 6 / 0 con
  2.8 s; p95 2.63 s vs 2.48 s. Compra 3 llamadas de `f4` a `f4+f5`; la AUC devuelta (0.949 vs 0.956) es ruido, y esos
  números son optimistas porque val ya está en el ajuste. Las 3 que quedan en `f4` a 3.0 s son `degraded:Contended`:
  llamadas largas cuyos dos canales suman más trozos que el semáforo por proceso (9), así que el agente cede aunque no
  haya otra llamada. Se mantiene 2.8 / 3.0; subir el plazo no es donde está la precisión.

## Vocabulario compartido con el agente y el tic de "guion" (2026-09-12, noche)

Leyendo los 8 errores de val: tres humanos que hablan como asistente, dos humanos al filo (0.55), un bot con muletillas
de guion, un bot mal transcrito y un bot escueto. Dos observaciones se convirtieron en features:

- **`overlap_agent`** (`features.f_agent`): fracción del vocabulario del cliente que el agente también dijo. Los humanos
  repiten lo que oyen para confirmar ("noventa, cincuenta y siete", "sobre el crédito, ¿verdad?"); el bot sigue su libreto.
  Media 0.61 en humanos, 0.51 en bots; AUC sola 0.78. Correlación con el número de turnos 0.04 (no es una estadística de
  turnos disfrazada; con n_palabras −0.43). Necesita la transcripción del agente, que ya se pide para la rúbrica, así que
  vive en el bloque F5: entra en `f4+f5` y el camino `f4` no la usa.
- **`guion_rate`** (en F4): el bot lee en voz alta el guion de su número de referencia ("tres tres cero seis guion nueve
  dos ocho uno"). 72 de 203 bots lo dicen, 14 de 150 humanos. Es un tic del generador de este dataset: sirve en el set
  escondido si lo hicieron igual, y no cuesta nada si no.

| modelo (CV 5×10 sobre 353) | F4 | F4+F5 |
|---|--:|--:|
| 11 features + 2 dims (mañana) | 0.892 | 0.904 |
| + guion_rate, + overlap_agent | **0.906 ± 0.032** | **0.955 ± 0.023** |

Probadas y descartadas: logprob del cliente relativo al del agente (+0.005, 39/50 folds), tasa de palabras numéricas
(+0.013, 41/50; la de "guion" la cubre). Fuera de fold sobre las 353: AUC 0.957, errores a 0.5 de 57 a 36.
En val (train→val) F4+F5 0.978, acc 0.915; la página de demo pasa de 63 a 65 aciertos de 71. Los tres humanos que hablan
como asistente siguen mal clasificados: eso no lo arregla el texto, lo arregla la fusión.
Por HTTP (71 de val en serie, 2.8 s): p50 2.08 s, p95 2.79 s, camino f4+f5 60 / f4 11 / abstención 0 (6 por plazo de
Gemini, 5 por contención del canal del agente en llamadas largas); la AUC devuelta (0.989) es optimista porque val está en
el ajuste. La feature nueva no añade latencia: usa la transcripción del agente que ya se pedía.
