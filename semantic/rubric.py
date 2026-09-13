"""Fase 5: Gemini rubric. One call per conversation, structured JSON: score 0-1 + short quote per dimension.
Gemini emits features; the logistic regression decides.

score(transcript) -> {"scores": {dim: {"score", "evidencia"}}, "latency_s", "usage"}
score_cached(cid, transcript)   cache/gemini/<cid>_<model>_<prompt hash>.json
"""
import hashlib, json, os, pathlib, sys, time
import httpx
try:
    from . import config, asr  # noqa: F401  (.env loader)
except ImportError:  # standalone scripts run from semantic/
    import config  # noqa: F401  (.env loader)
    import asr

MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
EVIDENCE = os.environ.get("RUBRIC_EVIDENCE", "0") == "1"   # quotes cost ~230 output tokens ≈ +0.6 s; fixed for train+infer
URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent"
CACHE = pathlib.Path(__file__).parent / "cache" / "gemini"
DIMS = ["sobrecompletitud", "registro_formal", "reparacion_conversacional", "iniciativa",
        "manejo_producto_ambiguo", "fidelidad_repeticion", "consistencia_memoria"]

PROMPT = """Eres analista de conversaciones. Abajo va la transcripción de una llamada a Banco Altur. AGENTE es Marina, un agente de IA del banco; CLIENTE es quien llama. Evalúa SOLO el contenido de lo que dice el CLIENTE, en 7 dimensiones, cada una con score de 0 a 1 y una evidencia: cita textual corta del CLIENTE, máximo 12 palabras. No juzgues la voz ni la calidad del audio: solo qué dice y cómo lo estructura.

El guion de la agente es siempre el mismo flujo: saluda; pide el nombre; pide el número de referencia de cliente (6 a 8 dígitos); repite la referencia A PROPÓSITO con un dígito mal; pide detalles del problema; interrumpe al cliente ("perdón que lo interrumpa"); pregunta si es sobre "cuenta nómina plus" o "crédito verde" (productos que el cliente puede no tener); pide reconfirmar la referencia completa; cierra con folio.

Dimensiones:
1. sobrecompletitud: 1 si el CLIENTE entrega información que todavía no le han pedido (nombre, referencia, monto, fechas, opciones) en bloques completos y organizados; 0 si contesta a pedazos solo lo que le preguntan y hay que sacarle los datos uno por uno.
2. registro_formal: 1 si hay cortesía excesiva, frases completas y bien formadas, sin coloquialismos ni muletillas, estructura de asistente; 0 si habla coloquial, con muletillas, frases cortadas.
3. reparacion_conversacional: tras la interrupción de la agente, 1 si el CLIENTE reinicia su frase completa palabra por palabra; 0 si hace meta-conversación natural ("¿bueno?", "¿me escucha?", "le decía que...") o sigue donde iba.
4. iniciativa: 1 si el CLIENTE se queja, divaga, cuestiona ("¿y eso para qué lo necesita?"), mete contexto personal no solicitado o cambia de tema; 0 si solo responde lo que se le pregunta.
5. manejo_producto_ambiguo: ante "nómina plus o crédito verde", 1 si acepta la premisa y elige uno sin cuestionar; 0 si cuestiona, duda o no reconoce el producto.
6. fidelidad_repeticion: cuando la agente repite mal la referencia, 1 si el CLIENTE detecta y corrige el dígito; 0 si lo deja pasar.
7. consistencia_memoria: 1 si los datos que da el CLIENTE (nombre, referencia, montos, fechas) son idénticos cada vez que los repite; 0 si se contradice o se trabuca.

Si el momento que evalúa una dimensión no ocurre en la llamada, pon score 0.5 y evidencia vacía.

TRANSCRIPCIÓN:
"""
_DIM = ({"type": "OBJECT", "required": ["score", "evidencia"], "propertyOrdering": ["score", "evidencia"],
         "properties": {"score": {"type": "NUMBER"}, "evidencia": {"type": "STRING"}}} if EVIDENCE
        else {"type": "NUMBER"})
SCHEMA = {"type": "OBJECT", "required": DIMS, "propertyOrdering": DIMS, "properties": {d: _DIM for d in DIMS}}
if not EVIDENCE:
    PROMPT = PROMPT.replace(" y una evidencia: cita textual corta del CLIENTE, máximo 12 palabras", "").replace(" y evidencia vacía", "")
PROMPT_ID = hashlib.sha1((PROMPT + json.dumps(SCHEMA)).encode()).hexdigest()[:6]
_client = None


def score(transcript, timeout=3.0):
    global _client
    if _client is None:
        _client = httpx.Client(headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    t0 = time.perf_counter()
    r = _client.post(URL, timeout=timeout, json={"contents": [{"parts": [{"text": PROMPT + transcript}]}],
                                "generationConfig": {"responseMimeType": "application/json", "responseSchema": SCHEMA,
                                                     "temperature": 0, "thinkingConfig": {"thinkingLevel": "minimal"}}})
    r.raise_for_status()
    d = r.json()
    out = json.loads(d["candidates"][0]["content"]["parts"][0]["text"])
    out = {k: (v if isinstance(v, dict) else {"score": v, "evidencia": ""}) for k, v in out.items()}
    for v in out.values():
        v["score"] = float(min(max(v["score"], 0.0), 1.0))
    return {"scores": out, "latency_s": time.perf_counter() - t0, "usage": d.get("usageMetadata"), "model": MODEL}


def cache_path(cid):   # keyed by rubric model, prompt and (for non-default ASR) the ASR model the transcript came from
    suffix = "" if asr.MODEL == "scribe_v1" else "_" + asr.MODEL
    return CACHE / f"{cid}_{MODEL}_{PROMPT_ID}{suffix}.json"


def score_cached(cid, transcript):
    p = cache_path(cid)
    if p.exists():
        return json.load(open(p))
    for attempt in range(3):                       # offline batch only: retry slow/flaky calls
        try:
            out = score(transcript, timeout=15.0); break
        except (httpx.HTTPError, KeyError, json.JSONDecodeError):
            if attempt == 2: raise
    json.dump(out, open(p, "w"), ensure_ascii=False)
    return out


if __name__ == "__main__" and sys.argv[1:2] == ["all"]:
    # Fase 5 batch: rubric for every call with both channels transcribed (cached); per-dimension AUC on val.
    import csv
    from concurrent.futures import ThreadPoolExecutor
    from sklearn.metrics import roc_auc_score
    root = pathlib.Path(__file__).parent
    rows = [r for r in csv.DictReader(open(root / "data/hackmty26/manifest.csv")) if asr.annotated_cached(r["anon_id"]) is not None]
    rows = rows[:int(sys.argv[2])] if len(sys.argv) > 2 else rows
    with ThreadPoolExecutor(4) as ex:
        outs = list(ex.map(lambda r: score_cached(r["anon_id"], asr.annotated_cached(r["anon_id"])), rows))
    y = [r["label"] == "synthetic" for r in rows]; va = [r["split"] == "val" for r in rows]
    lat = [o["latency_s"] for o in outs]
    print(f"model={MODEL} evidence={EVIDENCE} prompt={PROMPT_ID}  calls={len(rows)} (val {sum(va)})  latency mean={sum(lat)/len(lat):.2f}s")
    import numpy as np
    y, va = np.array(y), np.array(va)
    for d in DIMS:
        v = np.array([o["scores"][d]["score"] for o in outs])
        a = roc_auc_score(y[va], v[va]) if va.sum() > 1 and len(set(y[va])) == 2 else float("nan")
        print(f"  {d:26s} val AUC={a:.3f}  human mean={v[~y].mean():.2f} synth mean={v[y].mean():.2f}  0.5-share={(v == 0.5).mean():.2f}")
    sys.exit(0)

if __name__ == "__main__":
    # latency + first look at separation, on the 6 kit transcripts (3 human, 3 synthetic)
    import glob
    print(f"model={MODEL} evidence={EVIDENCE} prompt={PROMPT_ID}")
    for f in sorted(glob.glob(os.path.expanduser("~/Downloads/altur_transcripts/*.txt"))):
        t = open(f).read()
        out = score(t)
        u = out["usage"] or {}
        print(f"{pathlib.Path(f).stem:18s} {out['latency_s']:.2f}s out_tok={u.get('candidatesTokenCount')}  " +
              " ".join(f"{d[:6]}={out['scores'][d]['score']:.1f}" for d in DIMS))
