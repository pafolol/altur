"""
THE LIVE-CALL DEMO - dial a real number, record what the person says, and score it.

Everything else in this repository is fed a file. This dials out, and it is the only path where the audio
has actually been through a telephone network rather than a simulation of one - which is why it scores
with the ROBUST V2 acoustic model rather than V1. That is not a preference, it is the whole finding of
reports/TELEPHONE_ROBUSTNESS_REPORT: down a real line V1 falls to 77.5 % while V2 holds at 100 %, because
the recording pipeline cues V1 keys on (a digitally silent noise floor, the loudness, the band edge) do
not survive a codec. A live call is exactly the domain V2 exists for.

    POST /twilio/call {"to": "+52..."}   ->  call_sid, and a job that runs in the background
    GET  /twilio/status/{call_sid}       ->  ringing / recording / scoring / done, then the verdict

NO PUBLIC URL IS NEEDED. The obvious way to do this is a <Record action="..."> webhook, which means
Twilio has to reach your machine, which on a laptop means a tunnel and one more thing to fail on stage.
Instead the call carries inline TwiML and the recording is collected by POLLING Twilio's REST API for
recordings belonging to the call. Nothing has to reach in.

WHAT THE OTHER LAYERS DO HERE, honestly: a Twilio recording is MONO - one mixed channel, no separate
agent track. The behaviour layer needs both channels to time an interaction and abstains, and it says so
rather than guessing. So a live call is decided by the acoustic layer, with the semantic verifier
consulted if the acoustic layer is unsure. The page shows that.

Configure with TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER. Without them the endpoint
reports itself unconfigured and the button explains what is missing; nothing else in the server changes.
"""
import io
import os
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np

import _bootstrap  # noqa: F401
import config

SAMPLE_RATE = 8000
POLL_INTERVAL_S = 2.0
RECORDING_TIMEOUT_S = 240.0       # a call plus the moments Twilio takes to finalise the recording
DEFAULT_PROMPT = ("Hola. Esta es una demostración del detector de voz sintética de Altur. "
                  "Después del tono, habla durante unos segundos y luego cuelga.")


@dataclass
class CallJob:
    """One demo call, from dialling to verdict. Polled by the page."""
    call_sid: str
    to: str
    state: str = "queued"          # queued | ringing | recording | scoring | done | failed
    detail: str = ""
    started: float = field(default_factory=time.time)
    call_status: str = ""
    duration_s: float = 0.0
    recording_sid: str = ""
    verdict: dict = None
    error: str = ""
    logged: bool = False           # the call log records a finished demo call exactly once
    wav: bytes = None              # the stereo-converted recording, served back for playback
    scene: dict = None             # the visualiser's payload for it (envelopes, regions, running verdict)

    def to_dict(self):
        return {"call_sid": self.call_sid, "to": _mask(self.to), "state": self.state,
                "detail": self.detail, "elapsed_s": round(time.time() - self.started, 1),
                "call_status": self.call_status, "duration_s": self.duration_s,
                "recording_sid": self.recording_sid, "verdict": self.verdict, "error": self.error,
                "logged": self.logged}


def _mask(number):
    """Never echo a full phone number back to a browser; the last four are enough to recognise it."""
    n = "".join(ch for ch in (number or "") if ch.isdigit())
    return f"…{n[-4:]}" if len(n) >= 4 else "…"


def mono_to_stereo_wav(mono_wav_bytes):
    """
    Twilio hands back one mixed channel. Every detector here expects stereo 8 kHz PCM with the caller on
    channel 0, so the recording goes there and channel 1 is left silent - which is the truth: there is no
    separate agent track to put in it. The behaviour layer will correctly find no interaction to time.
    """
    with wave.open(io.BytesIO(mono_wav_bytes), "rb") as w:
        channels, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width != 2:
        raise ValueError(f"expected 16-bit PCM from Twilio, got {width * 8}-bit")
    samples = np.frombuffer(raw, dtype="<i2").reshape(-1, channels)
    caller = samples.mean(axis=1).astype("<i2") if channels > 1 else samples[:, 0]
    if rate != SAMPLE_RATE:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(int(rate), SAMPLE_RATE)
        caller = resample_poly(caller.astype(np.float32), SAMPLE_RATE // g, int(rate) // g).astype("<i2")
    stereo = np.stack([caller, np.zeros_like(caller)], axis=1)
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(stereo.tobytes())
    return out.getvalue()


class TwilioDemo:
    """Dials, waits for the recording, scores it. One instance per server."""

    ACOUSTIC_MODEL = "robust_v2"        # the phone-hardened model; see the module docstring

    def __init__(self, semantic_url=None):
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
        self.from_number = os.getenv("TWILIO_PHONE_NUMBER", "").strip()
        self.default_to = os.getenv("TWILIO_TO_NUMBER", "").strip()
        self.semantic_url = semantic_url
        self.jobs = {}
        self._lock = threading.Lock()
        self._fusion = None

    # ------------------------------------------------------------------ configuration
    @property
    def configured(self):
        return bool(self.account_sid and self.auth_token and self.from_number)

    def missing(self):
        return [name for name, v in (("TWILIO_ACCOUNT_SID", self.account_sid),
                                     ("TWILIO_AUTH_TOKEN", self.auth_token),
                                     ("TWILIO_PHONE_NUMBER", self.from_number)) if not v]

    def describe(self):
        d = {"configured": self.configured, "missing": self.missing(),
             "acoustic_model": self.ACOUSTIC_MODEL,
             "from": _mask(self.from_number), "default_to": _mask(self.default_to),
             # The demo's one button calls TWILIO_TO_NUMBER and asks for no number, so the page has to
             # know whether that variable is set - `configured` alone does not say, it is optional there.
             "has_default_to": bool(self.default_to),
             "model_available": (config.MODELS_DIR / self.ACOUSTIC_MODEL / "meta.json").exists()}
        try:
            import twilio  # noqa: F401
            d["sdk"] = True
        except ImportError:
            d["sdk"] = False
            d["missing"] = d["missing"] + ["the twilio package (pip install twilio)"]
            d["configured"] = False
        return d

    def client(self):
        from twilio.rest import Client
        return Client(self.account_sid, self.auth_token)

    def fusion(self):
        """A fusion whose acoustic slot is V2. Built once, on the first call."""
        if self._fusion is None:
            import fusion as fusion_mod
            self._fusion = fusion_mod.FusionDetector(
                layers=fusion_mod.build_layers(self.ACOUSTIC_MODEL, include_unavailable=True,
                                               semantic_url=self.semantic_url))
        return self._fusion

    # ------------------------------------------------------------------ the call
    def twiml(self, prompt=DEFAULT_PROMPT, max_seconds=30):
        """
        Say a line, then record. No `action` attribute on purpose: with none, Twilio finishes the call
        after the recording instead of asking our machine what to do next, and the recording is still
        created and reachable through the REST API. That is what removes the need for a public URL.
        """
        from xml.sax.saxutils import escape
        return ('<?xml version="1.0" encoding="UTF-8"?><Response>'
                f'<Say language="es-MX" voice="alice">{escape(prompt)}</Say>'
                f'<Record playBeep="true" maxLength="{int(max_seconds)}" finishOnKey="#" trim="do-not-trim"/>'
                '<Say language="es-MX" voice="alice">Gracias. Analizando.</Say>'
                '</Response>')

    def start(self, to=None, prompt=DEFAULT_PROMPT, max_seconds=30):
        to = (to or self.default_to).strip()
        if not self.configured:
            raise RuntimeError(f"Twilio is not configured: missing {', '.join(self.missing())}")
        if not to:
            raise ValueError("no destination number: pass 'to', or set TWILIO_TO_NUMBER")
        from twilio.base.exceptions import TwilioRestException
        try:
            call = self.client().calls.create(twiml=self.twiml(prompt, max_seconds),
                                              from_=self.from_number, to=to)
        except TwilioRestException as exc:
            raise RuntimeError(f"Twilio rejected the call (code {exc.code}): {exc.msg}") from exc
        job = CallJob(call.sid, to, state="ringing", detail="marcando")
        with self._lock:
            self.jobs[call.sid] = job
        threading.Thread(target=self._run, args=(job, max_seconds), daemon=True).start()
        return job

    def status(self, call_sid):
        with self._lock:
            job = self.jobs.get(call_sid)
        return job.to_dict() if job else None

    def warm(self):
        """
        Load and exercise the live-call fusion while the phone is still ringing.

        The acoustic slot here is Robust V2, which nothing else in the server uses, so without this the
        FIRST live call pays its backbone load *after* the caller has hung up - and that load is inside
        the layer's own measured latency, so it reaches the page as "voice latency: 7 s" and reads as the
        model being slow when it was only cold. A ringing phone is seconds of wall clock that are being
        spent anyway, so the load is free there.

        PRIMARIES ONLY. The semantic verifier is a paid API call, and warming up is not a reason to spend
        one on a test tone. It is never loaded here; the first call that actually needs it will load it.
        """
        try:
            det = self.fusion()
            t = np.arange(4 * SAMPLE_RATE, dtype=np.float32) / SAMPLE_RATE
            tone = (0.18 * np.sin(2 * np.pi * 220 * t) * 32767).astype("<i2")
            out = io.BytesIO()
            with wave.open(out, "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(SAMPLE_RATE)
                w.writeframes(np.stack([tone, np.zeros_like(tone)], axis=1).tobytes())
            wav = out.getvalue()
            for layer in det.layers:
                if det.config.role_of(layer) == "primary":
                    layer.score(wav)          # never raises: Layer.score() turns a failure into an abstention
        except Exception:
            return                            # a warm-up that fails costs latency, never a call

    # ------------------------------------------------------------------ the background job
    def _run(self, job, max_seconds):
        try:
            self.warm()                       # the phone is ringing; the load is free here. See warm().
            recording = self._await_recording(job, max_seconds)
            job.state, job.detail = "scoring", f"analizando {job.duration_s:.0f} s con {self.ACOUSTIC_MODEL}"
            wav = mono_to_stereo_wav(self._download(recording.sid))
            job.wav = wav
            try:
                import latency
                job.scene = latency.scene(self.fusion(), wav, run_fusion_prefixes=False)
                final = job.scene["final"]
                result = {"is_synthetic": final["is_synthetic"], "confidence": final["confidence"],
                          "synthetic_probability": final["probability"], "note": final.get("note", ""),
                          "layers": final["layers"]}
            except Exception as exc:                   # the verdict stands even if the scene cannot be drawn
                job.scene = {"error": f"{type(exc).__name__}: {exc}"}
                result = self.fusion().score(wav)
            job.verdict = {
                "is_synthetic": result["is_synthetic"],
                "confidence": result["confidence"],
                "synthetic_probability": result["synthetic_probability"],
                "acoustic_model": self.ACOUSTIC_MODEL,
                "note": result.get("note", ""),
                "layers": [{"key": c["key"], "display": c["display"], "probability": c["probability"],
                            "abstained": c["abstained"], "scored": c.get("scored", True),
                            "consulted": c.get("consulted", True), "role": c.get("role"),
                            "share": c["share"], "reason": c.get("reason", "")}
                           for c in result["layers"]],
            }
            job.state, job.detail = "done", ""
        except Exception as exc:
            job.state, job.error = "failed", f"{type(exc).__name__}: {exc}"
            job.detail = ""

    def _await_recording(self, job, max_seconds):
        client = self.client()
        deadline = time.time() + min(RECORDING_TIMEOUT_S, 60 + max_seconds * 3)
        while time.time() < deadline:
            call = client.calls(job.call_sid).fetch()
            job.call_status = call.status
            if call.status in ("ringing", "queued"):
                job.state, job.detail = "ringing", "esperando respuesta"
            elif call.status == "in-progress":
                job.state, job.detail = "recording", "la llamada está en curso"
            elif call.status in ("busy", "no-answer", "failed", "canceled"):
                raise RuntimeError(f"the call ended without a recording: {call.status}")
            recordings = client.recordings.list(call_sid=job.call_sid, limit=1)
            if recordings:
                rec = recordings[0]
                if rec.status == "completed":
                    job.recording_sid = rec.sid
                    job.duration_s = float(rec.duration or 0)
                    return rec
                job.state, job.detail = "recording", "terminando la grabación"
            elif call.status == "completed":
                job.detail = "la llamada terminó, esperando la grabación"
            time.sleep(POLL_INTERVAL_S)
        raise TimeoutError("no recording arrived before the deadline")

    def _download(self, recording_sid):
        """Twilio serves the media itself, not through the SDK object, so this is a plain authed GET."""
        import base64 as b64
        import urllib.request
        url = (f"https://api.twilio.com/2010-04-01/Accounts/{self.account_sid}"
               f"/Recordings/{recording_sid}.wav")
        auth = b64.b64encode(f"{self.account_sid}:{self.auth_token}".encode()).decode()
        req = urllib.request.Request(url, headers={"Authorization": f"Basic {auth}"})
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()


_DEMO = None


def get_demo(semantic_url=None):
    global _DEMO
    if _DEMO is None:
        _DEMO = TwilioDemo(semantic_url)
    return _DEMO


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Dial a number, record the answer, score it with Robust V2.")
    parser.add_argument("to", nargs="?", help="destination in E.164, e.g. +5218112345678")
    parser.add_argument("--seconds", type=int, default=30, help="maximum recording length")
    parser.add_argument("--status", action="store_true", help="only print whether Twilio is configured")
    args = parser.parse_args()
    demo = TwilioDemo()
    if args.status or not (args.to or demo.default_to):
        print(json.dumps(demo.describe(), indent=2))
        return
    job = demo.start(args.to, max_seconds=args.seconds)
    print(f"calling {_mask(job.to)}  call_sid={job.call_sid}")
    last = None
    while job.state not in ("done", "failed"):
        if (job.state, job.detail) != last:
            print(f"  [{job.state}] {job.detail}")
            last = (job.state, job.detail)
        time.sleep(1)
    if job.state == "failed":
        raise SystemExit(f"failed: {job.error}")
    v = job.verdict
    print(f"\n{job.duration_s:.0f} s recorded -> {'SYNTHETIC' if v['is_synthetic'] else 'HUMAN'} "
          f"(confidence {v['confidence']:.3f}, p={v['synthetic_probability']:.3f})")
    for l in v["layers"]:
        state = ("not asked" if not l["scored"] else
                 f"abstained - {l['reason']}" if l["abstained"] else f"p={l['probability']:.3f}")
        print(f"  {l['key']:10s} {state}")


if __name__ == "__main__":
    main()
