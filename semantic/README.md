# altur-semantico — módulo semántico (HackMTY26, reto Altur)

Uno de tres módulos. Entrada: WAV estéreo 8 kHz (canal 0 llamante, canal 1 agente). Salida: `P(llamante sintético)` en [0,1], calibrada, en < 3 s (plazo de red 2.8 s).
Spec: `PLAN.md`. Lo que se midió y contradice al plan: `FINDINGS.md`.

## Correr

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
cp .env.example .env        # ELEVENLABS_API_KEY, GEMINI_API_KEY (el servidor no arranca sin ellas)
.venv/bin/uvicorn server:app --workers 4
```

```bash
.venv/bin/python detect.py samples/call.wav       # codifica, llama y muestra el JSON (WAV sintético de Apple TTS, no del dataset)
```

Página de demo: con el servidor arriba, abre <http://localhost:8000/>. Lista las 71 llamadas de validación con su audio,
la verdad y el puntaje fuera de fold (el modelo que las calificó no las vio), y deja subir audios nuevos en cualquier
formato que lea `afconvert` (macOS): `POST /check` los convierte a estéreo 8 kHz y corre el mismo camino que `/detect`.
Mono se toma como cliente con el agente en silencio.

```bash
curl -s localhost:8000/detect -H 'Content-Type: application/json' -d '{"audio": "<wav estéreo 16-bit 8 kHz en base64>"}'
# -> {"is_synthetic": true, "confidence": 0.91, "score": 0.91, "abstain": false, "reason": "", "used": "f4+f5", "ms": 1980}
```

| campo | significado |
|---|---|
| `is_synthetic` | `score >= 0.5` (lo que pide el contrato del reto) |
| `confidence` | igual a `score`: P(sintético) calibrada, no "certeza". El backend aprende los pesos de fusión sobre esto |
| `abstain` | `true` = no hubo evidencia (`score` es 0.5 y **no es una opinión**; el backend no debe promediarlo) |
| `reason` | vacío si todo bien; `no_speech`, `asr_error:<tipo>`, `degraded:<tipo>` (Gemini o canal del agente no llegaron), `internal:<tipo>` |
| `used` | camino alcanzado: `f4+f5` (Scribe + rúbrica Gemini), `f4` (solo texto), `abstain` |

Entrada inválida (no es WAV, mono, 16 kHz, 0 frames, > 400 s) devuelve **400** con el formato esperado; cuerpo > 20 MB de base64 devuelve **413**. `GET /health` devuelve config, procedencia del modelo y qué llaves están presentes. Cada petición deja una línea JSON en stderr (`used`, `reason`, `ms`, trozos, hedges).

## Pipeline

```
decode + VAD (vad.py)  →  Scribe (scribe_v1) por canal, compactado, trozos de 20 s en paralelo con hedging (asr.py)
  → F4 features de texto sin LLM (features.py)  →  regresión logística F4  ─┐
  → transcript anotado → rúbrica Gemini, JSON estructurado (rubric.py)    → regresión logística F4+F5 (model.py)
  → vocabulario del cliente compartido con el agente (features.f_agent)   ─┘  (solo en f4+f5: necesita el canal del agente)
```

## Reproducir el entrenamiento

```bash
.venv/bin/python check_dataset.py     # Fase 0: manifest + 353 audios
.venv/bin/python vad.py               # Fase 1: VAD vs turns.json
.venv/bin/python asr.py all           # Fase 3: transcribe y cachea (cache/asr)
.venv/bin/python features.py          # Fase 4: AUC por feature en val
.venv/bin/python rubric.py all        # Fase 5: rúbrica en las 353 (cache/gemini), AUC por dimensión
.venv/bin/python model.py             # Fase 6: F4 / F5 / F4+F5, calibración, model.pkl, scores_semantico.csv
.venv/bin/python model.py cv          # regla para comparar experimentos: 5-fold x10 sobre las 353 (±0.01 en la media)
.venv/bin/python check_server.py      # Fase 7: HTTP real, 71 de val, p50/p95, AUC  (--parallel 8 para carga)
.venv/bin/pytest -q                   # contrato y degradación con Scribe/Gemini simulados, sin red
.venv/bin/python tts_probe.py         # robustez con un TTS ajeno (Apple, macOS); regenera samples/call.wav
```

## Perillas e invariantes

| qué | dónde | se puede tocar |
|---|---|---|
| `BUDGET_S` / `HARD_S` (2.8 / 3.0 s) | env o server.py | sí, sin reentrenar |
| `HEDGE_S`, `CHUNK_S` | asr.py | `HEDGE_S` sí; `CHUNK_S` está guardado en model.pkl: cambiarlo exige `asr.py all` + `model.py` |
| `SEMANTIC_CONFIG` (`F4+F5` / `F4`) | env | sí; `F4` sirve sin Gemini |
| `SCRIBE_MAX_INFLIGHT` (9) | env | sí; la suscripción de ElevenLabs admite ~20 peticiones simultáneas: con `--workers N`, N × 9 debe quedar por debajo. Techo real: 2-3 llamadas simultáneas a calidad completa |
| `SCRIBE_MODEL`, `GEMINI_MODEL`, `RUBRIC_EVIDENCE`, el prompt de la rúbrica | env / rubric.py | **no sin reentrenar**: el servidor se niega a arrancar si model.pkl fue entrenado con otros valores. Cambiarlos: `asr.py all` (si ASR) o `rubric.py all` (si rúbrica), luego `model.py` |
| lista de features (`model.F4`, `model.AGENT`, `model.DIMS`, `SEMANTIC_EXTRA`) | model.py / env | solo con `model.py` después: model.pkl guarda la lista y el servidor la comprueba al arrancar |
| versión de scikit-learn | requirements.txt | el servidor avisa si difiere de la que pickleó model.pkl |

`AUDIT.md` tiene la auditoría de atajos, ablación, prefijos y errores. `scores_semantico.csv` (`anon_id,split,score`, todas las filas fuera de fold) es la entrega al backend para aprender los pesos de fusión. `model.pkl` está ajustado con las 353 llamadas; las cifras de val y de CV son la medición.
Filas de train = predicción out-of-fold (5-fold aleatorio: no hay id de hablante, ligeramente optimista). Filas de val = modelo entrenado en train.
