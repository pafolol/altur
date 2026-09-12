"""Robustness probe with a TTS engine that is NOT in the dataset (Apple TTS via macOS `say`, es_MX voices).
Ten synthetic callers: 5 voices x 2 speaking styles (bot-like over-complete paragraphs vs human-like fragments
with fillers). Marina is synthesized too. Scores each call through the real pipeline (Scribe + Gemini) offline.
Also writes samples/call.wav (agent greeting + one caller turn) for the README hello-world.
Run: python tts_probe.py   (macOS only; costs ~10 Scribe + 10 Gemini calls)"""
import pathlib, subprocess, sys, tempfile, time, wave
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import server  # noqa: loads model.pkl, checks keys
from vad import SR

VOICES = ["Paulina", "Eddy (Spanish (Mexico))", "Flo (Spanish (Mexico))", "Grandma (Spanish (Mexico))", "Grandpa (Spanish (Mexico))"]
AGENT_VOICE = "Mónica"
AGENT = ["Buen día, gracias por llamar a Banco Altur, le atiende Marina. Le comento que esta llamada se graba para fines de calidad. ¿En qué le puedo ayudar?",
         "Claro, entiendo. ¿Me puede indicar su nombre, por favor, y su número de referencia de cliente? Es un número de entre seis y ocho dígitos.",
         "Perfecto, permítame confirmar: cuatro, cuatro, siete, cero, ocho, uno, cero, tres. ¿Así es?",
         "Gracias. Ahora, ¿me puede platicar con detalle qué fue lo que pasó, en qué fecha y en qué establecimiento?",
         "Sólo para continuar, ¿esto es sobre su cuenta nómina plus o sobre su crédito verde?",
         "Queda registrado su reporte. Su folio es tres, siete, cuatro, uno, dos, seis. Le deseo muy buena tarde."]
BOT = ["Buenas tardes. Le llamo porque tengo un cobro duplicado en mi tarjeta de crédito. La compra fue el primero de septiembre en una tienda departamental por dos mil seiscientos pesos y aparece dos veces en mi estado de cuenta. Quisiera solicitar la aclaración y saber en cuántos días se resuelve.",
       "Con gusto. Mi nombre es Adrián Cortés Lima y mi número de referencia de cliente es cuatro, cuatro, siete, cero, ocho, uno, nueve, tres.",
       "No, disculpe, el penúltimo dígito es nueve, no cero. La referencia correcta es cuatro, cuatro, siete, cero, ocho, uno, nueve, tres.",
       "Como le comentaba, el cargo fue el primero de septiembre en una tienda departamental, por un monto de dos mil seiscientos pesos, y se refleja duplicado. Ya verifiqué con el comercio y me indicaron que sólo procesaron un cargo.",
       "Es sobre mi crédito verde.",
       "Perfecto, muchas gracias por su atención. Quedo al pendiente del folio. Hasta luego."]
HUMAN = ["Hola, buenas tardes, eh, es que me hicieron un cobro dos veces.",
         "Sí, este, Adrián Cortés. ¿Mande? Ah, la referencia, sí, es cuatro cuatro siete cero, ocho uno nueve tres.",
         "No, no, el... es nueve, uno nueve tres.",
         "Pues fue el primero, este, de septiembre, en una tienda, y me cobraron dos mil seiscientos dos veces.",
         "Eh, ¿crédito verde? Creo que sí, el de la tarjeta.",
         "Ah, ok, sale, gracias."]


def say(text, voice, tmp):
    aiff = tmp / "t.aiff"; wav = tmp / "t.wav"
    subprocess.run(["say", "-v", voice, "-o", str(aiff), text], check=True)
    subprocess.run(["afconvert", "-f", "WAVE", "-d", f"LEI16@{SR}", "-c", "1", str(aiff), str(wav)], check=True)
    with wave.open(str(wav)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0


def build_call(caller_lines, voice, tmp, n_turns=None):
    gap = np.zeros(int(0.7 * SR), dtype=np.float32); ch0, ch1 = [], []
    for a, c in list(zip(AGENT, caller_lines))[:n_turns]:
        ya = say(a, AGENT_VOICE, tmp); yc = say(c, voice, tmp)
        ch1 += [ya, gap, np.zeros_like(yc), gap]; ch0 += [np.zeros_like(ya), gap, yc, gap]
    x = np.stack([np.concatenate(ch0), np.concatenate(ch1)], axis=1)
    return x


def write_stereo(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


if __name__ == "__main__":
    tmp = pathlib.Path(tempfile.mkdtemp())
    write_stereo("samples/call.wav", build_call(HUMAN, "Paulina", tmp, n_turns=1))
    print("wrote samples/call.wav")
    rows = []
    for style, lines in (("bot-like", BOT), ("human-like", HUMAN)):
        for v in VOICES:
            x = build_call(lines, v, tmp); stats = {}
            t0 = time.perf_counter(); p, used, reason = server.score_call(x, t0 + 10.0, stats); dt = time.perf_counter() - t0
            rows.append((style, v, p, used)); print(f"{style:10s} {v:28s} score={p:.2f} used={used:6s} words={stats.get('n_words')} {dt:.1f}s {reason}")
    for style in ("bot-like", "human-like"):
        ps = [r[2] for r in rows if r[0] == style]
        print(f"{style:10s} mean score={np.mean(ps):.2f}  min={min(ps):.2f} max={max(ps):.2f}")
