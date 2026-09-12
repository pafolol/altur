# Auditoría metodológica del módulo semántico (2026-09-12)

Misma escalera que el prompt del módulo de comportamiento exige: dummy → atajos → modelo → ablación →
errores → prefijos. Todo en val (71 llamadas, speaker-disjoint), clase positiva = sintético, umbral 0.5,
preprocesado ajustado solo en train. Se reproduce con `python audit.py` (solo lee cachés, no llama APIs).

## 1. Escalera

| modelo | k | AUC train / val | acc | bal. acc | P / R / F1 | Brier | FPR / FNR | (tn, fp, fn, tp) |
|---|--:|---|--:|--:|---|--:|---|---|
| dummy (clase mayoritaria de train) | 0 | 0.500 / 0.500 | 0.479 | 0.500 | — | — | 1.00 / 0.00 | (0, 37, 0, 34) |
| atajo: duración | 1 | 0.531 / 0.517 | 0.479 | 0.500 | 0.48 / 1.00 / 0.65 | 0.263 | 1.00 / 0.00 | (0, 37, 0, 34) |
| atajo: segundos de voz del cliente | 1 | 0.562 / 0.451 | 0.437 | 0.456 | — | 0.273 | 1.00 / 0.09 | (0, 37, 3, 31) |
| atajo: estadísticas de turnos (n, media, share) | 3 | 0.697 / 0.734 | 0.662 | 0.665 | 0.62 / 0.74 / 0.68 | 0.215 | 0.41 / 0.26 | (22, 15, 9, 25) |
| F4 limpio (sin proxies de turnos) | 8 | 0.883 / 0.912 | 0.859 | 0.861 | 0.82 / 0.91 / 0.86 | 0.126 | 0.19 / 0.09 | (30, 7, 3, 31) |
| F4 completo (lo que se servía antes) | 10 | 0.956 / 0.935 | 0.887 | 0.886 | 0.91 / 0.85 / 0.88 | 0.095 | 0.08 / 0.15 | (34, 3, 5, 29) |
| F5, 7 dimensiones | 7 | 0.786 / 0.832 | 0.718 | 0.724 | 0.66 / 0.85 / 0.74 | 0.190 | 0.41 / 0.15 | (22, 15, 5, 29) |
| F5, 2 dimensiones útiles | 2 | 0.730 / 0.795 | 0.761 | 0.763 | 0.72 / 0.82 / 0.77 | 0.192 | 0.30 / 0.18 | (26, 11, 6, 28) |
| **F4 limpio + F5 (2 dims) — lo que se sirve ahora** | 10 | 0.908 / 0.921 | 0.887 | 0.888 | 0.86 / 0.91 / 0.89 | 0.106 | 0.14 / 0.09 | (32, 5, 3, 31) |
| F4 limpio + F5 (7 dims) | 15 | 0.929 / 0.933 | 0.831 | 0.833 | 0.79 / 0.88 / 0.83 | 0.109 | 0.22 / 0.12 | (29, 8, 4, 30) |
| F4 completo + F5 (7 dims) — lo de antes | 17 | 0.966 / 0.955 | 0.873 | 0.872 | 0.88 / 0.85 / 0.87 | 0.084 | 0.11 / 0.15 | (33, 4, 5, 29) |

**Atajos.** La duración y los segundos de voz no separan nada (AUC ≈ 0.5): el dataset está balanceado a
propósito. Las estadísticas de turnos sí (0.73), y eso es exactamente lo que el módulo de comportamiento
va a explotar.

**Fuga hacia el módulo de comportamiento.** Dos features del F4 original eran proxies de turnos:

| feature | r con n_turns | r con mean_turn_len | r con longest_turn_share |
|---|--:|--:|--:|
| words_per_turn | 0.58 | **0.94** | 0.55 |
| n_words | 0.03 | **0.62** | 0.08 |
| mean_sentence_len | 0.15 | 0.51 | 0.20 |
| resto de F4 y F5 | ≤ 0.30 | ≤ 0.35 | ≤ 0.23 |

`words_per_turn` es casi la longitud media de turno con otro nombre (r = 0.94). Solas, las dos proxies dan
AUC 0.84. Se quitaron del modelo servido (PLAN §1.1): el módulo pierde 0.03 de AUC en solitario, pero la
fusión deja de contar la misma señal dos veces. `mean_sentence_len` (r 0.51) se queda: es contenido
(puntuación del ASR), no cronometraje, y su AUC individual es 0.61.

**Sobreajuste.** Gap train/val negativo o ≈ 0 en todas las configuraciones (val es más fácil que train
por su balance 50/50). No hay señal de sobreajuste; con 10 features y 282 llamadas, tampoco cabía.

## 2. Ablación por familia (sobre F4 limpio + F5 2 dims, AUC val 0.921)

| se quita | AUC | acc | comentario |
|---|--:|--:|---|
| logprob de Scribe (3) | 0.882 | 0.831 | la familia más fuerte |
| estilo/léxico (5) | 0.918 | 0.775 | la AUC casi no cae pero la accuracy sí: aporta calibración |
| rúbrica (2) | 0.912 | 0.859 | Gemini suma 0.01 de AUC y 0.03 de accuracy |

Familias solas: logprob 0.882, estilo/léxico 0.863, rúbrica 0.795, proxies (descartadas) 0.842.

## 3. Prefijos de llamada (F4 limpio, solo texto)

| se usa | AUC | acc | FNR |
|---|--:|--:|--:|
| primeros 30 s | **0.937** | 0.873 | 0.06 |
| primeros 60 s | 0.915 | 0.845 | 0.03 |
| primeros 90 s | 0.883 | 0.803 | 0.09 |
| llamada completa | 0.912 | 0.859 | 0.09 |

Los primeros 30 s bastan y hasta son mejores: ahí el humano dice "me hicieron un cobro dos veces" y el bot
suelta el párrafo completo. No se cambió el servicio (la rúbrica necesita la llamada entera), pero si el
presupuesto aprieta, transcribir solo los primeros 60 s del cliente es una degradación válida.

## 4. Errores en val (F4 limpio + F5, 8 de 71)

| llamada | real | p | palabras | logprob μ / σ | disfl. | sobrecomp. | formal | turnos |
|---|---|--:|--:|---|--:|--:|--:|--:|
| call_743448b8a861 | sintético | 0.10 | 88 | −0.022 / 0.085 | 0.000 | 1.0 | 0.0 | 14 |
| call_5e4539a471f6 | sintético | 0.20 | 64 | −0.011 / 0.038 | 0.062 | 0.0 | 0.0 | 37 |
| call_0294f969f98b | sintético | 0.42 | 57 | −0.008 / 0.046 | 0.000 | 0.0 | 0.0 | 15 |
| call_fa26d1186721 | humano | 0.90 | 73 | −0.003 / 0.010 | 0.000 | 1.0 | 0.0 | 10 |
| call_e9bd33cdb226 | humano | 0.81 | 81 | −0.015 / 0.085 | 0.000 | 1.0 | 1.0 | 32 |
| call_1c1c0cce9fb5 | humano | 0.78 | 83 | −0.007 / 0.058 | 0.024 | 1.0 | 0.0 | 30 |
| call_9bb847dfc83e | humano | 0.69 | 63 | −0.013 / 0.044 | 0.000 | 0.0 | 0.5 | 12 |
| call_6971b2685c1d | humano | 0.67 | 48 | −0.013 / 0.059 | 0.021 | 1.0 | 0.0 | 10 |

Patrones: (a) los falsos negativos son bots con confianza de ASR baja o con disfluencias, es decir, bots
con voz "sucia" o que el ASR entendió mal; (b) los falsos positivos son humanos que hablan limpio (sin
muletillas, ASR seguro) y que además entregaron todo de un jalón (sobrecompletitud 1.0). El módulo mide
"habla como asistente", y algunos humanos hablan así. Esto es lo esperado y por eso se fusiona con acústica
y comportamiento en vez de decidir solo.

## 5. Lo que este módulo no controla

- Las tres features de `logprob` dependen del ASR (Scribe v2) y del TTS del atacante: un motor nuevo con
  voz más natural puede subir la confianza del ASR en humanos o bajarla en bots. Es la señal más fuerte y
  la más frágil ante el set escondido.
- Los datos personales son inventados por los humanos (README del reto): `consistencia_memoria` no
  discrimina (AUC 0.48) porque los humanos también se trabucan poco con ellos.
- Audio del cliente y del agente sale a ElevenLabs; el texto sale a Google. Decisión del equipo tomada al
  inicio; no hay versión sin red de este módulo.

## 6. Servicio con la configuración limpia (71 de val, HTTP real, presupuesto 2.8 s)

p50 2.06 s, p95 2.83 s. Camino f4+f5 66 / f4 5 / abstención 0. AUC de los scores devueltos 0.907.
Calibración offline de F4+F5: gap medio 0.053 (antes 0.106); scores_semantico.csv regenerado con este modelo.

## 7. Prueba de robustez con un motor de TTS ajeno al dataset (tts_probe.py, 2026-09-12)

Diez llamantes sintetizados con Apple TTS (`say`, 5 voces es_MX) que no existe en el dataset; Marina también
sintetizada. Dos guiones con el mismo contenido factual: **bot-like** (párrafos completos, formal, entrega
todo de un jalón) y **human-like** (fragmentos, "este", "eh", "¿mande?", "sale"). Pipeline real (Scribe + Gemini).

| estilo | Paulina | Eddy | Flo | Grandma | Grandpa | media |
|---|--:|--:|--:|--:|--:|--:|
| bot-like | 0.96 | 0.92 | 0.94 | 0.95 | 0.95 | **0.94** |
| human-like | 0.57 | 0.59 | 0.79 | 0.28 | 0.66 | **0.58** |

Lectura: (1) el módulo reacciona al contenido, no solo a la voz: misma voz, mismo motor, 0.36 de diferencia
por el guion; (2) con un motor nuevo, la señal de confianza del ASR sigue empujando hacia "sintético"
(las voces human-like promedian > 0.5): el bot adversarial que habla coloquial baja a ~0.6, no a 0; (3) la
dispersión entre voces del guion human-like (0.28 a 0.79) es la fragilidad de `logprob` ante voces nuevas
que AUDIT §5 anticipaba. Diez muestras: es una prueba de humo, no una evaluación.
