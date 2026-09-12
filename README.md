# Altur Voice Deepfake Backend

Backend ligero para analizar el canal del caller de llamadas del reto Altur HackMTY 2026. Recibe WAV estéreo en Base64 y devuelve la probabilidad de voz sintética. El canal 0 es caller y el canal 1 es agente.

La extracción está separada en features conversacionales (`app/behavioral_features.py`), acústicas (`app/acoustic_features.py`) y una interfaz semántica reservada (`app/semantic_features.py`) sin Whisper ni LLM. `app/fusion.py` prepara `single_model`, `weighted` y `meta_model`; `single_model` es el flujo predeterminado compatible con el detector actual.

## Instalación

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

## Ejecutar

```bash
uvicorn app.main:app --reload
```

Swagger: `http://127.0.0.1:8000/docs`  
Health: `GET http://127.0.0.1:8000/health`  
Readiness: `GET http://127.0.0.1:8000/ready` (`503` si falta el modelo requerido)

## Detect

```json
{"audio": "BASE64_DEL_WAV"}
```

Respuesta pública:

```json
{"is_synthetic": false, "confidence": 0.5}
```

`audio_base64` también es aceptado. En caso de recibir ambos campos se usa `audio`.

`confidence` expresa confianza en la decisión final: si el resultado es sintético es la probabilidad sintética; si es humano es `1 - synthetic_probability`.

## Configuración

Las variables principales están en `.env.example`: `DETECTOR_MODE` (`mock` o `model`), `MODEL_PATH`, `MODEL_THRESHOLD`, `ENABLE_DEBUG_ENDPOINT`, `MAX_AUDIO_DURATION_SECONDS`, `MAX_REQUEST_SIZE_MB`, `MIN_AUDIO_DURATION_SECONDS`, `MIN_CALLER_SPEECH_SECONDS`, `ENABLE_PITCH_FEATURES`, `ENABLE_FEATURE_WARMUP`, `PORT` y `LOG_LEVEL`. `FUSION_MODE` admite `single_model`, `weighted` y `meta_model`; los pesos se configuran mediante `FUSION_WEIGHTS`. El modo mock siempre devuelve `0.5` y está marcado en logs; no es una detección real.

Al iniciar, `ENABLE_FEATURE_WARMUP=true` ejecuta una extracción determinista sobre una señal sintética antes de aceptar tráfico. `/ready` responde listo después del warm-up y del warm-up del modelo. Si el warm-up falla, `/ready` responde `503`. Para desactivarlo se puede usar `ENABLE_FEATURE_WARMUP=false`.

## Modelo real

Coloca `detector.joblib` en `models/` y usa `DETECTOR_MODE=model`. El bundle debe ser:

```python
{"model": trained_model, "feature_names": FEATURE_NAMES, "threshold": 0.53, "feature_schema_version": "1.0", "model_version": "1.0", "modalities": ["behavior", "acoustic"]}
```

El modelo debe implementar `predict_proba` (clase 1 = synthetic) o `predict`, y `feature_names` debe coincidir exactamente y en el mismo orden con `app.feature_extractor.FEATURE_NAMES`. Se carga una vez al iniciar.

## Pruebas

```bash
pytest -q
python send_test_request.py audio.wav
python benchmark.py carpeta_con_wavs
```

El WAV debe ser estéreo; si el sample rate no es 8000 Hz se remuestrea en memoria.

## Demo Twilio Voice

Configura el webhook de voz entrante como `POST https://TU_HOST_PUBLICO/twilio/voice`. La demo responde con TwiML, graba hasta que se presiona `#` y envía la grabación a `/twilio/playback`, que la reproduce y finaliza la llamada. No guarda ni analiza la grabación. La URL pública debe ser la de tu túnel o deployment, no está hardcodeada en el backend. La validación de firma de Twilio debe agregarse antes de producción.
